from __future__ import division

import argparse
import cv2
import torch
import sklearn
import mmcv
import copy
import os
import time
import warnings
from os import path as osp

try:
    from mmcv import Config, DictAction
    from mmcv.runner import get_dist_info, init_dist
    from mmcv.utils import TORCH_VERSION, digit_version
except Exception:
    from mmcv_wrapper import Config, DictAction
    from mmcv_wrapper import get_dist_info, init_dist
    from mmengine.utils import digit_version
    TORCH_VERSION = torch.__version__

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version

try:
    from mmdet3d.datasets import build_dataset
except Exception:
    build_dataset = None
try:
    from mmdet3d.models import build_model
except Exception:
    build_model = None
try:
    from mmdet3d.utils import collect_env, get_root_logger
except Exception:
    from mmdet3d.utils import collect_env
    from mmengine.logging import MMLogger

    def get_root_logger(log_file=None, log_level='INFO', name='mmdet'):
        return MMLogger.get_instance(name, log_file=log_file, log_level=log_level)
try:
    from mmdet.apis import set_random_seed
except Exception:
    from mmengine.runner import set_random_seed
try:
    from mmseg import __version__ as mmseg_version
except Exception:
    mmseg_version = 'unknown'

warnings.filterwarnings("ignore")


def _build_dataset_compat(cfg):
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg, dict) and 'classes' in cfg:
        classes = cfg.pop('classes')
        metainfo = dict(cfg.get('metainfo', {}))
        metainfo.setdefault('classes', tuple(classes))
        cfg['metainfo'] = metainfo
    # mmdet3d 1.x may join data_root + ann_file; convert known local ann files to absolute paths.
    if isinstance(cfg, dict) and 'ann_file' in cfg:
        ann_file = cfg['ann_file']
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))

        def _abspath_if_exists(path):
            if not isinstance(path, str) or osp.isabs(path):
                return path
            local_path = osp.join(repo_root, path.lstrip('./'))
            if osp.exists(local_path):
                return local_path
            return path

        if isinstance(ann_file, str):
            cfg['ann_file'] = _abspath_if_exists(ann_file)
        elif isinstance(ann_file, (list, tuple)):
            cfg['ann_file'] = [_abspath_if_exists(p) for p in ann_file]
    if build_dataset is not None:
        return build_dataset(cfg)
    from mmdet3d.registry import DATASETS
    return DATASETS.build(cfg)


def _build_model_compat(model_cfg, train_cfg=None, test_cfg=None):
    if build_model is not None:
        return build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    from mmdet3d.registry import MODELS
    model_cfg = copy.deepcopy(model_cfg)
    if train_cfg is not None and 'train_cfg' not in model_cfg:
        model_cfg['train_cfg'] = train_cfg
    if test_cfg is not None and 'test_cfg' not in model_cfg:
        model_cfg['test_cfg'] = test_cfg
    try:
        return MODELS.build(model_cfg)
    except KeyError:
        # Some UniAD custom modules are registered into mmdet::MODELS in mmcv2 stacks.
        from mmdet.registry import MODELS as MMDET_MODELS
        return MMDET_MODELS.build(model_cfg)


def _bootstrap_legacy_stack_shims():
    """Reuse compatibility shims from tools/test.py when running on mmcv2/mmdet3."""
    try:
        from test import (
            _bootstrap_mmcv_shims,
            _bootstrap_mmdet_core_shim,
            _bootstrap_mmdet_models_shim,
            _bootstrap_mmdet_models_utils_shim,
            _bootstrap_mmdet_datasets_shim,
            _bootstrap_mmdet3d_datasets_pipelines_shim,
            _bootstrap_mmdet3d_core_shim,
        )
        _bootstrap_mmcv_shims()
        _bootstrap_mmdet_core_shim()
        _bootstrap_mmdet_models_shim()
        _bootstrap_mmdet_models_utils_shim()
        _bootstrap_mmdet_datasets_shim()
        _bootstrap_mmdet3d_datasets_pipelines_shim()
        _bootstrap_mmdet3d_core_shim()
    except Exception as e:
        warnings.warn(f"Compat shims bootstrap skipped: {e}")


def _sync_custom_model_registries():
    """Mirror model registrations from mmdet to mmdet3d for mixed old/new stacks."""
    try:
        from mmdet.registry import MODELS as MMDET_MODELS
        from mmdet3d.registry import MODELS as MMDET3D_MODELS
    except Exception:
        return

    for name, module in MMDET_MODELS.module_dict.items():
        if name in MMDET3D_MODELS.module_dict:
            continue
        try:
            MMDET3D_MODELS.register_module(module=module, name=name, force=True)
        except Exception:
            pass


def _sync_registry_modules(src_registry, dst_registry):
    try:
        items = src_registry.module_dict.items()
    except Exception:
        return
    for name, module in items:
        if name in dst_registry.module_dict:
            continue
        try:
            dst_registry.register_module(module=module, name=name, force=True)
        except Exception:
            pass


def _sync_cross_stack_registries():
    """Sync core registries between mmdet/mmdet3d and mmengine."""
    try:
        from mmdet.registry import MODELS as MMDET_MODELS, TRANSFORMS as MMDET_TRANSFORMS
        from mmdet3d.registry import MODELS as MMDET3D_MODELS, TRANSFORMS as MMDET3D_TRANSFORMS
        from mmengine.registry import MODELS as MMENGINE_MODELS, TRANSFORMS as MMENGINE_TRANSFORMS
    except Exception:
        return

    _sync_registry_modules(MMDET_MODELS, MMDET3D_MODELS)
    _sync_registry_modules(MMDET_MODELS, MMENGINE_MODELS)
    _sync_registry_modules(MMDET3D_MODELS, MMENGINE_MODELS)

    _sync_registry_modules(MMDET_TRANSFORMS, MMENGINE_TRANSFORMS)
    _sync_registry_modules(MMDET3D_TRANSFORMS, MMENGINE_TRANSFORMS)


def _register_legacy_pipeline_transforms():
    """Register UniAD pipeline transforms into new registries."""
    try:
        import projects.mmdet3d_plugin.datasets.pipelines  # noqa: F401
        from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
        from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
        from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
        from projects.mmdet3d_plugin.datasets.pipelines.loading import (
            LoadMultiViewImageFromFilesInCeph, LoadAnnotations3D_E2E)
        from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
            PadMultiViewImage, NormalizeMultiviewImage, CustomCollect3D,
            PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage,
            ObjectRangeFilterTrack, ObjectNameFilterTrack)
        from projects.mmdet3d_plugin.datasets.pipelines.occflow_label import GenerateOccFlowLabels
        from mmdet3d.datasets.transforms import MultiScaleFlipAug3D
        from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
    except Exception:
        return

    for cls in [
            LoadMultiViewImageFromFilesInCeph,
            LoadAnnotations3D_E2E,
            PadMultiViewImage,
            NormalizeMultiviewImage,
            CustomCollect3D,
            PhotoMetricDistortionMultiViewImage,
            RandomScaleImageMultiViewImage,
            ObjectRangeFilterTrack,
            ObjectNameFilterTrack,
            GenerateOccFlowLabels,
            MultiScaleFlipAug3D,
            DefaultFormatBundle3D]:
        for registry in (MMENGINE_TRANSFORMS, MMDET_TRANSFORMS, MMDET3D_TRANSFORMS):
            if cls.__name__ not in registry.module_dict:
                registry.register_module(module=cls, force=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed') 
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    mmcv_major = 0
    try:
        mmcv_major = int(str(mmcv.__version__).split('.')[0])
    except Exception:
        pass
    if mmcv_major >= 2:
        _bootstrap_legacy_stack_shims()
    try:
        from mmdet3d.utils import register_all_modules
        register_all_modules(init_default_scope=False)
    except Exception:
        pass

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        try:
            from mmcv.utils import import_modules_from_strings
        except Exception:
            from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            _sync_custom_model_registries()
            _sync_cross_stack_registries()
            _register_legacy_pipeline_transforms()

            from projects.mmdet3d_plugin.uniad.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # NOTE: We have adopted a more elegant way to set the logger_name
    logger_name = cfg.get('logger_name', 'mmdet')
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = _build_model_compat(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [_build_dataset_compat(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(_build_dataset_compat(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    # NOTE: To fix the serialization issue in nuScenes-dev-kit, we adopt this method to skip the pickle steps
    torch.multiprocessing.set_start_method('fork')
    main()
