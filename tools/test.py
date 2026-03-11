import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import sys
import types
import logging
import warnings
from mmcv_wrapper import Config, DictAction
from mmcv_wrapper import fuse_conv_bn
from mmcv_wrapper import MMDataParallel, MMDistributedDataParallel
from mmcv_wrapper import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

try:
    from mmdet.apis import set_random_seed
except Exception:
    from mmengine.runner import set_random_seed

try:
    from mmdet.datasets import replace_ImageToTensor
except Exception:
    def replace_ImageToTensor(pipeline):
        return pipeline
import time
import os.path as osp

warnings.filterwarnings("ignore")


def _get_device():
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device(f"musa:{torch.musa.current_device()}"), "musa"
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}"), "cuda"
    return torch.device("cpu"), "cpu"


def _bootstrap_mmcv_shims():
    import mmcv as mmcv_pkg
    import mmcv.cnn as mmcv_cnn
    import mmcv.cnn.bricks.transformer as mmcv_transformer
    from mmengine.registry import MODELS as MMENGINE_MODELS

    if not hasattr(mmcv_cnn, 'bias_init_with_prob'):
        import math

        def bias_init_with_prob(prior_prob):
            return float(-math.log((1 - prior_prob) / prior_prob))

        mmcv_cnn.bias_init_with_prob = bias_init_with_prob
    if not hasattr(mmcv_cnn, 'constant_init'):
        from mmengine.model import constant_init
        mmcv_cnn.constant_init = constant_init
    if not hasattr(mmcv_cnn, 'xavier_init'):
        from mmengine.model import xavier_init
        mmcv_cnn.xavier_init = xavier_init

    if 'mmcv.utils' in sys.modules:
        utils_mod = sys.modules['mmcv.utils']
    else:
        utils_mod = types.ModuleType('mmcv.utils')
        sys.modules['mmcv.utils'] = utils_mod

    if not hasattr(utils_mod, 'Registry') or not hasattr(utils_mod, 'build_from_cfg'):
        from mmengine import Registry
        from mmengine.registry import build_from_cfg
        utils_mod.Registry = Registry
        utils_mod.build_from_cfg = build_from_cfg
    if not hasattr(utils_mod, 'digit_version'):
        from mmengine.utils import digit_version
        utils_mod.digit_version = digit_version
    if not hasattr(utils_mod, 'TORCH_VERSION'):
        utils_mod.TORCH_VERSION = torch.__version__
    if not hasattr(utils_mod, 'ConfigDict'):
        from mmengine.config import ConfigDict
        utils_mod.ConfigDict = ConfigDict
    if not hasattr(mmcv_pkg, 'ConfigDict'):
        mmcv_pkg.ConfigDict = utils_mod.ConfigDict
    if not hasattr(mmcv_pkg, 'FileClient'):
        from mmengine.fileio import FileClient
        mmcv_pkg.FileClient = FileClient
    if not hasattr(mmcv_pkg, 'load'):
        from mmengine.fileio import load
        mmcv_pkg.load = load
    if not hasattr(mmcv_pkg, 'dump'):
        from mmengine.fileio import dump
        mmcv_pkg.dump = dump
    if not hasattr(mmcv_pkg, 'mkdir_or_exist'):
        from mmengine.utils import mkdir_or_exist
        mmcv_pkg.mkdir_or_exist = mkdir_or_exist
    if not hasattr(mmcv_pkg, 'ProgressBar'):
        from mmengine.utils import ProgressBar
        mmcv_pkg.ProgressBar = ProgressBar
    if not hasattr(utils_mod, 'deprecated_api_warning'):
        from mmengine.utils import deprecated_api_warning
        utils_mod.deprecated_api_warning = deprecated_api_warning
    if not hasattr(utils_mod, 'to_2tuple'):
        from mmengine.utils import to_2tuple
        utils_mod.to_2tuple = to_2tuple
    if 'mmcv.utils.registry' not in sys.modules:
        registry_mod = types.ModuleType('mmcv.utils.registry')
        registry_mod.Registry = utils_mod.Registry
        registry_mod.build_from_cfg = utils_mod.build_from_cfg
        sys.modules['mmcv.utils.registry'] = registry_mod

    if 'mmcv.runner' not in sys.modules:
        runner_mod = types.ModuleType('mmcv.runner')
        from mmengine.model import BaseModule
        from torch.nn import ModuleList, Sequential
        from mmengine.registry import HOOKS

        runner_mod.get_dist_info = get_dist_info
        runner_mod.init_dist = init_dist
        runner_mod.load_checkpoint = load_checkpoint
        runner_mod.wrap_fp16_model = wrap_fp16_model
        runner_mod.HOOKS = HOOKS
        runner_mod.BaseModule = BaseModule
        runner_mod.ModuleList = ModuleList
        runner_mod.Sequential = Sequential

        def _noop_decorator(*args, **kwargs):
            def _wrap(func):
                return func
            return _wrap

        runner_mod.force_fp32 = _noop_decorator
        runner_mod.auto_fp16 = _noop_decorator
        sys.modules['mmcv.runner'] = runner_mod
        fp16_utils_mod = types.ModuleType('mmcv.runner.fp16_utils')
        fp16_utils_mod.force_fp32 = _noop_decorator
        fp16_utils_mod.auto_fp16 = _noop_decorator
        sys.modules['mmcv.runner.fp16_utils'] = fp16_utils_mod
        base_module_mod = types.ModuleType('mmcv.runner.base_module')
        base_module_mod.BaseModule = BaseModule
        base_module_mod.ModuleList = ModuleList
        base_module_mod.Sequential = Sequential
        sys.modules['mmcv.runner.base_module'] = base_module_mod
        from mmengine.hooks import Hook
        hooks_pkg = types.ModuleType('mmcv.runner.hooks')
        hook_mod = types.ModuleType('mmcv.runner.hooks.hook')
        hook_mod.HOOKS = HOOKS
        hook_mod.Hook = Hook
        sys.modules['mmcv.runner.hooks'] = hooks_pkg
        sys.modules['mmcv.runner.hooks.hook'] = hook_mod

    if 'mmcv.parallel' not in sys.modules:
        parallel_mod = types.ModuleType('mmcv.parallel')

        class DataContainer:
            def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
                self.data = data
                self.stack = stack
                self.padding_value = padding_value
                self.cpu_only = cpu_only

        def collate(batch, samples_per_gpu=1):
            if not batch:
                return batch
            elem = batch[0]
            if isinstance(elem, DataContainer):
                if elem.cpu_only:
                    return [sample.data for sample in batch]
                if elem.stack:
                    return torch.stack([sample.data for sample in batch], dim=0)
                return [sample.data for sample in batch]
            if isinstance(elem, dict):
                return {k: collate([d[k] for d in batch], samples_per_gpu) for k in elem}
            if isinstance(elem, (list, tuple)):
                transposed = list(zip(*batch))
                return [collate(list(items), samples_per_gpu) for items in transposed]
            try:
                from torch.utils.data._utils.collate import default_collate
                return default_collate(batch)
            except Exception:
                return batch

        parallel_mod.DataContainer = DataContainer
        parallel_mod.collate = collate
        parallel_mod.MMDataParallel = MMDataParallel
        parallel_mod.MMDistributedDataParallel = MMDistributedDataParallel
        sys.modules['mmcv.parallel'] = parallel_mod

    if 'mmcv.cnn.bricks.registry' not in sys.modules:
        registry_mod = types.ModuleType('mmcv.cnn.bricks.registry')
        base_registry = mmcv_transformer.MODELS
        registry_mod.ATTENTION = base_registry
        registry_mod.TRANSFORMER_LAYER = base_registry
        registry_mod.TRANSFORMER_LAYER_SEQUENCE = base_registry
        registry_mod.POSITIONAL_ENCODING = base_registry
        registry_mod.FEEDFORWARD_NETWORK = base_registry
        sys.modules['mmcv.cnn.bricks.registry'] = registry_mod

    # mmcv<2 style transformer components were moved to mmdet in new stacks.
    try:
        from mmdet.models.layers.transformer import detr_layers
        from mmdet.models.layers.transformer import deformable_detr_layers
        from mmdet.models.layers import positional_encoding as mmdet_positional

        class CompatDetrTransformerDecoderLayer(mmcv_transformer.BaseTransformerLayer):
            def __init__(self,
                         attn_cfgs=None,
                         feedforward_channels=None,
                         ffn_dropout=0.0,
                         operation_order=None,
                         norm_cfg=dict(type='LN'),
                         act_cfg=dict(type='ReLU', inplace=True),
                         init_cfg=None,
                         **kwargs):
                if attn_cfgs is None:
                    raise ValueError(
                        'CompatDetrTransformerDecoderLayer expects attn_cfgs with two configs.')
                # mmengine config may pass attn_cfgs as a single ConfigDict.
                if isinstance(attn_cfgs, dict):
                    attn_cfgs = [attn_cfgs, dict(attn_cfgs)]
                elif not isinstance(attn_cfgs, (list, tuple)):
                    attn_cfgs = list(attn_cfgs)
                else:
                    attn_cfgs = list(attn_cfgs)
                if len(attn_cfgs) == 1:
                    attn_cfgs = [attn_cfgs[0], dict(attn_cfgs[0])]
                if len(attn_cfgs) < 2:
                    raise ValueError(
                        'CompatDetrTransformerDecoderLayer expects attn_cfgs with two configs.')
                embed_dims = attn_cfgs[0].get(
                    'embed_dims', attn_cfgs[1].get('embed_dims', 256))
                ffn_cfgs = dict(
                    type='FFN',
                    embed_dims=embed_dims,
                    feedforward_channels=feedforward_channels or embed_dims * 4,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                    act_cfg=act_cfg)
                super().__init__(
                    attn_cfgs=attn_cfgs,
                    ffn_cfgs=ffn_cfgs,
                    operation_order=operation_order,
                    norm_cfg=norm_cfg,
                    init_cfg=init_cfg,
                    batch_first=False)

        class CompatDetrTransformerEncoder(mmcv_transformer.TransformerLayerSequence):
            def __init__(self,
                         num_layers,
                         transformerlayers=None,
                         layer_cfg=None,
                         **kwargs):
                layers_cfg = transformerlayers if transformerlayers is not None else layer_cfg
                super().__init__(transformerlayers=layers_cfg, num_layers=num_layers)

        class CompatDetrTransformerDecoder(mmcv_transformer.TransformerLayerSequence):
            def __init__(self,
                         num_layers,
                         transformerlayers=None,
                         layer_cfg=None,
                         return_intermediate=True,
                         post_norm_cfg=None,
                         **kwargs):
                layers_cfg = transformerlayers if transformerlayers is not None else layer_cfg
                super().__init__(transformerlayers=layers_cfg, num_layers=num_layers)
                self.return_intermediate = return_intermediate
                self.post_norm_cfg = post_norm_cfg

        class CompatDeformableDetrTransformerEncoder(
                mmcv_transformer.TransformerLayerSequence):
            def __init__(self,
                         num_layers,
                         transformerlayers=None,
                         layer_cfg=None,
                         **kwargs):
                layers_cfg = transformerlayers if transformerlayers is not None else layer_cfg
                super().__init__(transformerlayers=layers_cfg, num_layers=num_layers)

        class CompatDeformableDetrTransformerDecoder(
                mmcv_transformer.TransformerLayerSequence):
            def __init__(self,
                         num_layers,
                         transformerlayers=None,
                         layer_cfg=None,
                         return_intermediate=True,
                         post_norm_cfg=None,
                         **kwargs):
                layers_cfg = transformerlayers if transformerlayers is not None else layer_cfg
                super().__init__(transformerlayers=layers_cfg, num_layers=num_layers)
                self.return_intermediate = return_intermediate
                self.post_norm_cfg = post_norm_cfg

            def forward(self,
                        query,
                        key=None,
                        value=None,
                        query_pos=None,
                        key_padding_mask=None,
                        reference_points=None,
                        spatial_shapes=None,
                        level_start_index=None,
                        valid_ratios=None,
                        reg_branches=None,
                        **kwargs):
                output = query
                intermediate = []
                intermediate_references = []
                for lid, layer in enumerate(self.layers):
                    output = layer(
                        output,
                        key=key,
                        value=value,
                        query_pos=query_pos,
                        key_padding_mask=key_padding_mask,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        **kwargs)
                    if isinstance(output, tuple):
                        output = output[0]
                    if self.return_intermediate:
                        intermediate.append(output)
                        if reference_points is not None:
                            intermediate_references.append(reference_points)
                if self.return_intermediate:
                    inter_states = torch.stack(intermediate)
                    if intermediate_references:
                        inter_refs = torch.stack(intermediate_references)
                    else:
                        inter_refs = None
                    return inter_states, inter_refs
                return output, reference_points

        for name in (
                'DetrTransformerDecoderLayer',
                'DetrTransformerEncoderLayer',
                'DetrTransformerDecoder',
                'DetrTransformerEncoder'):
            cls = getattr(detr_layers, name, None)
            if name == 'DetrTransformerDecoderLayer':
                cls = CompatDetrTransformerDecoderLayer
            elif name == 'DetrTransformerEncoder':
                cls = CompatDetrTransformerEncoder
            elif name == 'DetrTransformerDecoder':
                cls = CompatDetrTransformerDecoder
            if cls is None:
                continue
            if not hasattr(mmcv_transformer, name):
                setattr(mmcv_transformer, name, cls)
            MMENGINE_MODELS.register_module(module=cls, name=name, force=True)

        for name in (
                'DeformableDetrTransformerEncoder',
                'DeformableDetrTransformerDecoder',
                'DeformableDetrTransformerEncoderLayer',
                'DeformableDetrTransformerDecoderLayer'):
            cls = getattr(deformable_detr_layers, name, None)
            if name == 'DeformableDetrTransformerEncoder':
                cls = CompatDeformableDetrTransformerEncoder
            elif name == 'DeformableDetrTransformerDecoder':
                cls = CompatDeformableDetrTransformerDecoder
            if cls is None:
                continue
            if not hasattr(mmcv_transformer, name):
                setattr(mmcv_transformer, name, cls)
            MMENGINE_MODELS.register_module(module=cls, name=name, force=True)

        for name in ('LearnedPositionalEncoding', 'SinePositionalEncoding'):
            cls = getattr(mmdet_positional, name, None)
            if cls is None:
                continue
            if not hasattr(mmcv_transformer, name):
                setattr(mmcv_transformer, name, cls)
            if name not in MMENGINE_MODELS.module_dict:
                MMENGINE_MODELS.register_module(module=cls, name=name, force=True)
    except Exception:
        pass


def _bootstrap_mmdet_core_shim():
    if 'mmdet.core' in sys.modules:
        return

    from mmdet.registry import TASK_UTILS
    from mmdet.utils import reduce_mean
    from mmdet.models.utils import multi_apply
    from mmdet.models.task_modules import (
        BaseAssigner, AssignResult, BaseBBoxCoder,
        build_assigner, build_sampler, build_bbox_coder)
    from mmdet.models.task_modules.samplers.base_sampler import BaseSampler
    from mmdet.structures.bbox import (
        bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps)
    import mmdet.structures.mask as mask_mod

    core_mod = types.ModuleType('mmdet.core')
    core_mod.multi_apply = multi_apply
    core_mod.reduce_mean = reduce_mean
    core_mod.build_assigner = build_assigner
    core_mod.build_sampler = build_sampler
    core_mod.build_bbox_coder = build_bbox_coder
    core_mod.bbox_overlaps = bbox_overlaps
    core_mod.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
    core_mod.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh

    bbox_mod = types.ModuleType('mmdet.core.bbox')
    bbox_mod.BaseBBoxCoder = BaseBBoxCoder
    bbox_mod.bbox_overlaps = bbox_overlaps
    bbox_mod.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
    bbox_mod.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh

    bbox_builder_mod = types.ModuleType('mmdet.core.bbox.builder')
    bbox_builder_mod.BBOX_ASSIGNERS = TASK_UTILS
    bbox_builder_mod.BBOX_SAMPLERS = TASK_UTILS
    bbox_builder_mod.BBOX_CODERS = TASK_UTILS
    bbox_builder_mod.MATCH_COST = TASK_UTILS

    assigners_mod = types.ModuleType('mmdet.core.bbox.assigners')
    assigners_mod.BaseAssigner = BaseAssigner
    assigners_mod.AssignResult = AssignResult

    match_costs_mod = types.ModuleType('mmdet.core.bbox.match_costs')
    match_costs_mod.build_match_cost = TASK_UTILS.build
    match_costs_builder_mod = types.ModuleType('mmdet.core.bbox.match_costs.builder')
    match_costs_builder_mod.MATCH_COST = TASK_UTILS

    transforms_mod = types.ModuleType('mmdet.core.bbox.transforms')
    transforms_mod.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
    transforms_mod.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh

    eval_hooks_mod = types.ModuleType('mmdet.core.evaluation.eval_hooks')

    class DistEvalHook:
        def __init__(self, *args, **kwargs):
            pass

    eval_hooks_mod.DistEvalHook = DistEvalHook

    core_eval_mod = types.ModuleType('mmdet.core.evaluation')
    core_eval_mod.eval_hooks = eval_hooks_mod

    core_mod.bbox = bbox_mod
    core_mod.evaluation = core_eval_mod
    core_mod.mask = mask_mod

    sys.modules['mmdet.core'] = core_mod
    sys.modules['mmdet.core.bbox'] = bbox_mod
    sys.modules['mmdet.core.bbox.builder'] = bbox_builder_mod
    sys.modules['mmdet.core.bbox.assigners'] = assigners_mod
    sys.modules['mmdet.core.bbox.match_costs'] = match_costs_mod
    sys.modules['mmdet.core.bbox.match_costs.builder'] = match_costs_builder_mod
    sys.modules['mmdet.core.bbox.transforms'] = transforms_mod
    sys.modules['mmdet.core.evaluation'] = core_eval_mod
    sys.modules['mmdet.core.evaluation.eval_hooks'] = eval_hooks_mod
    sys.modules['mmdet.core.mask'] = mask_mod

    assigners_base_mod = types.ModuleType('mmdet.core.bbox.assigners.base_assigner')
    assigners_base_mod.BaseAssigner = BaseAssigner
    sys.modules['mmdet.core.bbox.assigners.base_assigner'] = assigners_base_mod

    assigners_result_mod = types.ModuleType('mmdet.core.bbox.assigners.assign_result')
    assigners_result_mod.AssignResult = AssignResult
    sys.modules['mmdet.core.bbox.assigners.assign_result'] = assigners_result_mod

    samplers_base_mod = types.ModuleType('mmdet.core.bbox.samplers.base_sampler')
    samplers_base_mod.BaseSampler = BaseSampler
    sys.modules['mmdet.core.bbox.samplers.base_sampler'] = samplers_base_mod


def _bootstrap_mmdet_models_utils_shim():
    from mmdet.registry import MODELS
    from mmdet.models.layers import inverse_sigmoid

    if 'mmdet.models.utils.transformer' not in sys.modules:
        transformer_mod = types.ModuleType('mmdet.models.utils.transformer')
        transformer_mod.inverse_sigmoid = inverse_sigmoid
        sys.modules['mmdet.models.utils.transformer'] = transformer_mod

    if 'mmdet.models.utils.builder' not in sys.modules:
        builder_mod = types.ModuleType('mmdet.models.utils.builder')
        builder_mod.TRANSFORMER = MODELS
        sys.modules['mmdet.models.utils.builder'] = builder_mod

    import mmdet.models.utils as utils_mod
    if not hasattr(utils_mod, 'build_transformer'):
        def build_transformer(cfg):
            return MODELS.build(cfg)
        utils_mod.build_transformer = build_transformer

    if not hasattr(utils_mod, 'Transformer'):
        import torch.nn as nn
        from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

        class Transformer(nn.Module):
            def __init__(self, encoder=None, decoder=None, init_cfg=None, **kwargs):
                super().__init__()
                self.encoder = None
                self.decoder = None
                if encoder is not None:
                    self.encoder = build_transformer_layer_sequence(encoder)
                if decoder is not None:
                    self.decoder = build_transformer_layer_sequence(decoder)
        utils_mod.Transformer = Transformer


def _bootstrap_mmdet_models_shim():
    import mmdet.models as mmdet_models
    from mmdet.registry import MODELS

    for name in ['HEADS', 'LOSSES', 'BACKBONES', 'NECKS', 'DETECTORS']:
        if not hasattr(mmdet_models, name):
            setattr(mmdet_models, name, MODELS)

    if not hasattr(mmdet_models, 'build_loss'):
        mmdet_models.build_loss = MODELS.build
    if not hasattr(mmdet_models, 'build_head'):
        mmdet_models.build_head = MODELS.build
    if not hasattr(mmdet_models, 'build_detector'):
        mmdet_models.build_detector = MODELS.build

    if 'mmdet.models.builder' not in sys.modules:
        builder_mod = types.ModuleType('mmdet.models.builder')
        builder_mod.HEADS = getattr(mmdet_models, 'HEADS')
        builder_mod.LOSSES = getattr(mmdet_models, 'LOSSES')
        builder_mod.BACKBONES = getattr(mmdet_models, 'BACKBONES')
        builder_mod.NECKS = getattr(mmdet_models, 'NECKS')
        builder_mod.DETECTORS = getattr(mmdet_models, 'DETECTORS')
        builder_mod.build_loss = mmdet_models.build_loss
        builder_mod.build_head = mmdet_models.build_head
        builder_mod.build_detector = mmdet_models.build_detector
        sys.modules['mmdet.models.builder'] = builder_mod

    import mmdet.utils as mmdet_utils
    if not hasattr(mmdet_utils, 'util_mixins'):
        util_mixins_mod = types.ModuleType('mmdet.utils.util_mixins')

        class NiceRepr:
            def __repr__(self):
                try:
                    body = self.__nice__()
                except Exception:
                    body = ''
                return f'<{self.__class__.__name__}({body})>'

        util_mixins_mod.NiceRepr = NiceRepr
        sys.modules['mmdet.utils.util_mixins'] = util_mixins_mod
        mmdet_utils.util_mixins = util_mixins_mod


def _bootstrap_mmdet_datasets_shim():
    import mmdet.datasets as mmdet_datasets
    from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
    from mmdet3d.registry import DATASETS as M3D_DATASETS

    if not hasattr(mmdet_datasets, 'DATASETS'):
        mmdet_datasets.DATASETS = M3D_DATASETS

    if 'mmdet.datasets.builder' not in sys.modules:
        builder_mod = types.ModuleType('mmdet.datasets.builder')

        def _concat_dataset(cfg, default_args=None):
            raise NotImplementedError(
                '_concat_dataset is not implemented in this compatibility mode.')

        builder_mod._concat_dataset = _concat_dataset
        builder_mod.PIPELINES = MMDET_TRANSFORMS
        sys.modules['mmdet.datasets.builder'] = builder_mod

    if 'mmdet.datasets.pipelines' not in sys.modules:
        pipelines_mod = types.ModuleType('mmdet.datasets.pipelines')

        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data)

        pipelines_mod.to_tensor = to_tensor
        sys.modules['mmdet.datasets.pipelines'] = pipelines_mod


def _bootstrap_mmdet3d_datasets_pipelines_shim():
    if 'mmdet3d.datasets.pipelines' in sys.modules and \
            'mmdet3d.datasets.pipelines.transforms_3d' in sys.modules:
        return

    import mmdet3d.datasets.transforms as m3d_transforms

    pipelines_mod = types.ModuleType('mmdet3d.datasets.pipelines')

    if hasattr(m3d_transforms, 'LoadAnnotations3D'):
        pipelines_mod.LoadAnnotations3D = m3d_transforms.LoadAnnotations3D

    class DefaultFormatBundle3D:
        def __init__(self, class_names=None, with_gt=True, with_label=True):
            self.class_names = class_names
            self.with_gt = with_gt
            self.with_label = with_label

        def __call__(self, results):
            return results

    pipelines_mod.DefaultFormatBundle3D = DefaultFormatBundle3D

    transforms3d_mod = types.ModuleType('mmdet3d.datasets.pipelines.transforms_3d')
    transforms3d_mod.ObjectRangeFilter = m3d_transforms.ObjectRangeFilter
    transforms3d_mod.ObjectNameFilter = m3d_transforms.ObjectNameFilter

    sys.modules['mmdet3d.datasets.pipelines'] = pipelines_mod
    sys.modules['mmdet3d.datasets.pipelines.transforms_3d'] = transforms3d_mod


def _bootstrap_mmdet3d_core_shim():
    if 'mmdet3d.core' in sys.modules:
        return

    from mmdet3d import structures as m3d_struct
    from mmdet3d.registry import TASK_UTILS
    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS

    core_mod = types.ModuleType('mmdet3d.core')
    core_mod.bbox3d2result = m3d_struct.bbox3d2result
    core_mod.xywhr2xyxyr = m3d_struct.xywhr2xyxyr

    bbox_mod = types.ModuleType('mmdet3d.core.bbox')
    bbox_mod.BaseInstance3DBoxes = m3d_struct.BaseInstance3DBoxes
    bbox_mod.LiDARInstance3DBoxes = m3d_struct.LiDARInstance3DBoxes
    bbox_mod.CameraInstance3DBoxes = m3d_struct.CameraInstance3DBoxes
    bbox_mod.DepthInstance3DBoxes = m3d_struct.DepthInstance3DBoxes
    bbox_mod.Box3DMode = m3d_struct.Box3DMode
    bbox_mod.Coord3DMode = m3d_struct.Coord3DMode

    points_mod = types.ModuleType('mmdet3d.core.points')
    points_mod.BasePoints = m3d_struct.BasePoints

    coders_mod = types.ModuleType('mmdet3d.core.bbox.coders')

    def build_bbox_coder(cfg):
        try:
            return TASK_UTILS.build(cfg)
        except Exception:
            return MMDET_TASK_UTILS.build(cfg)

    coders_mod.build_bbox_coder = build_bbox_coder

    iou_mod = types.ModuleType('mmdet3d.core.bbox.iou_calculators')
    iou3d_mod = types.ModuleType('mmdet3d.core.bbox.iou_calculators.iou3d_calculator')
    BboxOverlaps3D = TASK_UTILS.get('BboxOverlaps3D')
    BboxOverlapsNearest3D = TASK_UTILS.get('BboxOverlapsNearest3D')
    iou_mod.BboxOverlaps3D = BboxOverlaps3D
    iou_mod.BboxOverlapsNearest3D = BboxOverlapsNearest3D

    def bbox_overlaps_nearest_3d(bboxes1, bboxes2, coordinate='lidar'):
        return BboxOverlapsNearest3D(coordinate=coordinate)(bboxes1, bboxes2)

    iou3d_mod.bbox_overlaps_nearest_3d = bbox_overlaps_nearest_3d
    iou3d_mod.BboxOverlaps3D = BboxOverlaps3D
    iou3d_mod.BboxOverlapsNearest3D = BboxOverlapsNearest3D

    core_mod.bbox = bbox_mod
    core_mod.points = points_mod

    sys.modules['mmdet3d.core'] = core_mod
    sys.modules['mmdet3d.core.bbox'] = bbox_mod
    sys.modules['mmdet3d.core.points'] = points_mod
    sys.modules['mmdet3d.core.bbox.coders'] = coders_mod
    sys.modules['mmdet3d.core.bbox.iou_calculators'] = iou_mod
    sys.modules['mmdet3d.core.bbox.iou_calculators.iou3d_calculator'] = iou3d_mod


def _build_dataset(cfg_data_test):
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))

    def _fix_ann_file(node):
        ann_file = node.get('ann_file')
        data_root = node.get('data_root', '')

        def _resolve_one(path):
            if not isinstance(path, str):
                return path
            if osp.isabs(path):
                return path
            joined = osp.join(data_root, path) if data_root else path
            if osp.exists(joined):
                return path
            alt = osp.join(repo_root, path.lstrip('./'))
            if osp.exists(alt):
                return alt
            return path

        if isinstance(ann_file, str):
            node['ann_file'] = _resolve_one(ann_file)
        elif isinstance(ann_file, (list, tuple)):
            node['ann_file'] = [_resolve_one(p) for p in ann_file]

    def _normalize_dataset_cfg(node):
        if isinstance(node, dict):
            if 'classes' in node and 'metainfo' not in node:
                classes = list(node.pop('classes'))
                node['metainfo'] = {'classes': tuple(classes)}
            if 'ann_file' in node:
                _fix_ann_file(node)
            if 'dataset' in node and isinstance(node['dataset'], dict):
                _normalize_dataset_cfg(node['dataset'])
            if 'datasets' in node and isinstance(node['datasets'], (list, tuple)):
                for child in node['datasets']:
                    if isinstance(child, dict):
                        _normalize_dataset_cfg(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                if isinstance(child, dict):
                    _normalize_dataset_cfg(child)

    _normalize_dataset_cfg(cfg_data_test)

    try:
        from mmdet3d.datasets import build_dataset
        return build_dataset(cfg_data_test)
    except Exception:
        from mmdet3d.registry import DATASETS
        return DATASETS.build(cfg_data_test)


def _build_model(model_cfg, test_cfg=None):
    try:
        from mmdet3d.models import build_model
        return build_model(model_cfg, test_cfg=test_cfg)
    except Exception:
        from mmdet3d.registry import MODELS as M3D_MODELS
        try:
            return M3D_MODELS.build(model_cfg)
        except Exception:
            from mmdet.registry import MODELS as MMDET_MODELS
            return MMDET_MODELS.build(model_cfg)


def _sync_model_registries():
    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet3d.registry import MODELS as M3D_MODELS
    for name, module in MMDET_MODELS.module_dict.items():
        if name in M3D_MODELS.module_dict:
            continue
        try:
            M3D_MODELS.register_module(module=module, name=name, force=False)
        except Exception:
            continue


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    _bootstrap_mmcv_shims()
    _bootstrap_mmdet_core_shim()
    _bootstrap_mmdet_models_shim()
    _bootstrap_mmdet_models_utils_shim()
    _bootstrap_mmdet_datasets_shim()
    _bootstrap_mmdet3d_datasets_pipelines_shim()
    _bootstrap_mmdet3d_core_shim()
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=False)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
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
                importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if isinstance(cfg.model, dict) and 'pretrained' in cfg.model:
        cfg.model.pop('pretrained')
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
    # Ensure custom dataset pipelines are registered under mmengine TRANSFORMS.
    import projects.mmdet3d_plugin.datasets.pipelines  # noqa: F401
    from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
    from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS
    from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
    from projects.mmdet3d_plugin.datasets.pipelines.loading import (
        LoadMultiViewImageFromFilesInCeph, LoadAnnotations3D_E2E)
    from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
        PadMultiViewImage, NormalizeMultiviewImage, CustomCollect3D,
        PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage)
    from projects.mmdet3d_plugin.datasets.pipelines.occflow_label import GenerateOccFlowLabels
    from mmdet3d.datasets.transforms import MultiScaleFlipAug3D
    from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

    for _cls in [
            LoadMultiViewImageFromFilesInCeph,
            LoadAnnotations3D_E2E,
            PadMultiViewImage,
            NormalizeMultiviewImage,
            CustomCollect3D,
            PhotoMetricDistortionMultiViewImage,
            RandomScaleImageMultiViewImage,
            GenerateOccFlowLabels,
            MultiScaleFlipAug3D,
            DefaultFormatBundle3D]:
        for _registry in (MMENGINE_TRANSFORMS, MMDET_TRANSFORMS, MMDET3D_TRANSFORMS):
            if _cls.__name__ not in _registry.module_dict:
                _registry.register_module(module=_cls, force=True)
    _sync_model_registries()

    # build the dataloader
    dataset = _build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = _build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        raise RuntimeError('Only distributed test is supported by this entrypoint. Use torchrun.')
    else:
        device, backend = _get_device()
        device_ids = None
        if backend in ('musa', 'cuda'):
            device_ids = [device.index]
        model = MMDistributedDataParallel(
            model.to(device),
            device_ids=device_ids,
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            #assert False
            mmcv.dump(outputs, args.out)
            #outputs = mmcv.load(args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    # NOTE: To fix the serialization issue in nuScenes-dev-kit, we adopt this method to skip the pickle steps
    torch.multiprocessing.set_start_method('fork')
    main()
