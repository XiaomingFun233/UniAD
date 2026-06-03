import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..dense_heads.occ_head_plugin import IntersectionOverUnion, PanopticMetric
from ..dense_heads.planning_head_plugin import PlanningMetric

import mmcv
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    rank, world_size = get_dist_info()

    # Occ eval init
    eval_occ = hasattr(model.module, 'with_occ_head') \
                and model.module.with_occ_head
    if eval_occ and rank == 0:
        # 30mx30m, 100mx100m at 50cm resolution
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            iou_metrics[key] = IntersectionOverUnion(n_classes).to('musa')
            iou_metrics[key].sync_on_compute = False
            iou_metrics[key].dist_sync_on_step = False
        panoptic_metrics = {}
        for key in EVALUATION_RANGES.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to('musa')
            panoptic_metrics[key].sync_on_compute = False
            panoptic_metrics[key].dist_sync_on_step = False
    elif eval_occ:
        # Keep torchmetrics state local to rank0 to avoid distributed sync
        # barriers that can fail on MUSA backend.
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        iou_metrics = {}
        panoptic_metrics = {}

    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head
    planning_metrics = None
    if eval_planning and rank == 0:
        # Keep planning metric accumulation on rank0 to avoid distributed
        # sync barrier failures in torchmetrics on MUSA.
        planning_metrics = PlanningMetric().to('cpu')
        planning_metrics.sync_on_compute = False
        planning_metrics.dist_sync_on_step = False
        
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']
                if planning_metrics is not None:
                    planning_metrics(
                        pred_sdc_traj[:, :6, :2].detach().cpu(),
                        sdc_planning[0][0, :, :6, :2].detach().cpu(),
                        sdc_planning_mask[0][0, :, :6, :2].detach().cpu(),
                        segmentation[0][:, [1, 2, 3, 4, 5, 6]].detach().cpu(),
                    )

            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    if rank == 0:
                        num_occ += 1
                        for key, grid in EVALUATION_RANGES.items():
                            limits = slice(grid[0], grid[1])
                            iou_metrics[key](result[0]['occ']['seg_out'][..., limits, limits].contiguous(),
                                            result[0]['occ']['seg_gt'][..., limits, limits].contiguous())
                            panoptic_metrics[key](result[0]['occ']['ins_seg_out'][..., limits, limits].contiguous().detach(),
                                                    result[0]['occ']['ins_seg_gt'][..., limits, limits].contiguous())

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:
                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid', 'seg_out', 'ins_seg_out']:
                    if k in result[0]['occ']:
                        result[0]['occ'][k] = result[0]['occ'][k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    planning_results = None
    if eval_planning and planning_metrics is not None:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ and rank == 0:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if eval_planning and planning_results is not None:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # Use filesystem synchronization to avoid MUSA collectives timing out
    # at the end of distributed evaluation.
    if tmpdir is None:
        mmcv.mkdir_or_exist('.dist_test')
        run_id = os.environ.get('TORCHELASTIC_RUN_ID') or os.environ.get('MASTER_PORT') or 'default'
        tmpdir = osp.join('.dist_test', f'collect_{run_id}')
        mmcv.mkdir_or_exist(tmpdir)
    else:
        mmcv.mkdir_or_exist(tmpdir)

    part_file = osp.join(tmpdir, f'part_{rank}.pkl')
    mmcv.dump(result_part, part_file)

    # Non-zero ranks return once local shard is persisted.
    if rank != 0:
        return None

    # Rank0 polls for all shard files instead of calling dist.barrier().
    timeout_s = int(os.environ.get('UNIAD_COLLECT_TIMEOUT', '1200'))
    poll_interval_s = float(os.environ.get('UNIAD_COLLECT_POLL', '1.0'))
    deadline = time.time() + timeout_s
    expected_files = [osp.join(tmpdir, f'part_{i}.pkl') for i in range(world_size)]

    while True:
        ready = [osp.exists(path) for path in expected_files]
        if all(ready):
            break
        if time.time() > deadline:
            missing = [path for path, ok in zip(expected_files, ready) if not ok]
            raise TimeoutError(
                f'collect_results_cpu timeout waiting shard files in {tmpdir}. missing={missing}'
            )
        time.sleep(poll_interval_s)

    part_list = [mmcv.load(path) for path in expected_files]
    # sort the results
    ordered_results = []
    '''
    bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
    '''
    #for res in zip(*part_list):
    for res in part_list:
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)