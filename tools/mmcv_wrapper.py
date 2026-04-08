#!/usr/bin/env python3
"""
mmcv 版本兼容性包装器 - 完整版
支持 mmcv 1.x (1.3.0+) 和 2.x (2.0.0+)
"""

import warnings
import sys
import logging
import argparse
import os
import torch

__all__ = [
    # 核心类
    'Config', 'Registry', 'Runner', 'DictAction',
    'HOOKS', 'DATASETS', 'MODELS', 'OPTIMIZERS', 'LOSSES', 'METRICS',
    # 构建函数
    'build_from_cfg', 'build_runner',
    # 工具函数
    'ProgressBar', 'Timer', 'get_logger', 'print_log',
    'track_iter_progress',
    'check_file_exist', 'mkdir_or_exist', 'is_filepath',
    # 图像处理
    'imread', 'imwrite', 'imrescale', 'imresize', 
    'imflip', 'impad', 'imnormalize', 'imshear', 'imrotate',
    # 视频处理
    'VideoReader', 'frames2video',
    # 可视化
    'color_val', 'imshow', 'imshow_bboxes', 'imshow_det_bboxes',
    # CNN 模块
    'fuse_conv_bn',
    # 并行模块
    'MMDataParallel', 'MMDistributedDataParallel',
    # 分布式训练
    'get_dist_info', 'init_dist', 'master_only',
    # 检查点
    'load_checkpoint', 'save_checkpoint',
    # FP16
    'wrap_fp16_model', 'auto_fp16', 'Fp16OptimizerHook',
    # 版本信息
    'MMCV_VERSION', 'MMCV_MAJOR',
]


def _check_mmcv_version():
    """检测 mmcv 版本"""
    try:
        import mmcv
        version = mmcv.__version__
        major = int(version.split('.')[0])
        return version, major
    except ImportError:
        return None, 0


MMCV_VERSION, MMCV_MAJOR = _check_mmcv_version()

if MMCV_MAJOR == 0 or MMCV_VERSION is None:
    raise ImportError("mmcv 未安装，请先安装: pip install mmcv>=1.3.0")


# ==================== DictAction ====================

class _DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', 1)
            if ',' in val:
                val = val.split(',')
            options[key] = val
        setattr(namespace, self.dest, options)


DictAction = _DictAction
try:
    if MMCV_MAJOR >= 2:
        from mmengine import DictAction
    else:
        from mmcv import DictAction
except ImportError:
    pass


# ==================== FP16 模块 ====================

wrap_fp16_model = None
auto_fp16 = None
Fp16OptimizerHook = None


def _get_amp_device_type():
    if hasattr(torch, 'musa') and torch.musa.is_available():
        return 'musa'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

try:
    if MMCV_MAJOR >= 2:
        # mmcv 2.x: FP16 移到 mmengine.runner.amp 或 mmengine.model
        try:
            from mmengine.runner.amp import auto_fp16
        except ImportError:
            try:
                from mmengine.model import auto_fp16
            except ImportError:
                def auto_fp16(func):
                    """Decorator for automatic mixed precision."""
                    def wrapper(*args, **kwargs):
                        with torch.autocast(device_type=_get_amp_device_type(), enabled=True):
                            return func(*args, **kwargs)
                    return wrapper
        
        # wrap_fp16_model
        try:
            from mmengine.runner import wrap_fp16_model
        except ImportError:
            try:
                from mmengine.model import wrap_fp16_model
            except ImportError:
                # 简单实现：将模型转换为半精度
                def wrap_fp16_model(model):
                    """Convert model to FP16."""
                    model.half()
                    return model
        
        # Fp16OptimizerHook
        try:
            from mmengine.runner import Fp16OptimizerHook
        except ImportError:
            try:
                from mmengine.hooks import Fp16OptimizerHook
            except ImportError:
                # 创建占位类
                class Fp16OptimizerHook:
                    def __init__(self, *args, **kwargs):
                        warnings.warn("Fp16OptimizerHook not available in this version")
                    def before_run(self, runner):
                        pass
    else:
        # mmcv 1.x
        from mmcv.runner import wrap_fp16_model, auto_fp16, Fp16OptimizerHook
except ImportError as e:
    warnings.warn(f"FP16 模块导入失败: {e}")


# ==================== 分布式训练模块 ====================

get_dist_info = None
init_dist = None
master_only = None

try:
    if MMCV_MAJOR >= 2:
        try:
            from mmengine.dist import get_dist_info, init_dist, master_only
        except ImportError:
            try:
                from mmengine.dist.utils import get_dist_info, init_dist
                from mmengine.dist.utils import master_only
            except ImportError:
                import torch.distributed as dist
                
                def get_dist_info():
                    if dist.is_available() and dist.is_initialized():
                        return dist.get_rank(), dist.get_world_size()
                    return 0, 1
                
                def init_dist(launcher, backend=None, **kwargs):
                    if backend is None:
                        backend = os.getenv('UNIAD_DIST_BACKEND', 'mccl')
                    if launcher == 'pytorch':
                        torch.distributed.init_process_group(backend=backend)
                    elif launcher == 'slurm':
                        import os
                        rank = int(os.environ.get('RANK', 0))
                        world_size = int(os.environ.get('WORLD_SIZE', 1))
                        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
                
                def master_only(func):
                    def wrapper(*args, **kwargs):
                        rank, _ = get_dist_info()
                        if rank == 0:
                            return func(*args, **kwargs)
                    return wrapper
    else:
        from mmcv.runner import get_dist_info, init_dist
        try:
            from mmcv.runner import master_only
        except ImportError:
            def master_only(func):
                def wrapper(*args, **kwargs):
                    from mmcv.runner import get_dist_info
                    rank, _ = get_dist_info()
                    if rank == 0:
                        return func(*args, **kwargs)
                return wrapper
except ImportError as e:
    warnings.warn(f"分布式训练模块导入失败: {e}")


# ==================== 检查点模块 ====================

load_checkpoint = None
save_checkpoint = None

try:
    if MMCV_MAJOR >= 2:
        try:
            from mmengine.runner import load_checkpoint, save_checkpoint
        except ImportError:
            try:
                from mmengine.checkpoint import load_checkpoint, save_checkpoint
            except ImportError:
                import torch
                
                def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
                    if map_location is None:
                        map_location = 'cpu'
                    checkpoint = torch.load(filename, map_location=map_location)
                    
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    model.load_state_dict(state_dict, strict=strict)
                    return checkpoint
                
                def save_checkpoint(model, filename, optimizer=None, meta=None):
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'meta': meta or {}
                    }
                    if optimizer is not None:
                        checkpoint['optimizer'] = optimizer.state_dict()
                    torch.save(checkpoint, filename)
    else:
        from mmcv.runner import load_checkpoint, save_checkpoint
except ImportError as e:
    warnings.warn(f"检查点模块导入失败: {e}")


# ==================== 并行模块 ====================

MMDataParallel = None
MMDistributedDataParallel = None

try:
    if MMCV_MAJOR >= 2:
        try:
            from mmengine.parallel import MMDataParallel, MMDistributedDataParallel
        except ImportError:
            from torch.nn.parallel import DataParallel as MMDataParallel
            from torch.nn.parallel import DistributedDataParallel as MMDistributedDataParallel
    else:
        from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
except ImportError as e:
    warnings.warn(f"并行模块导入失败: {e}")


# ==================== CNN 模块 ====================

fuse_conv_bn = None

try:
    if MMCV_MAJOR >= 2:
        try:
            from mmcv.cnn import fuse_conv_bn
        except ImportError:
            try:
                from mmengine.model.utils import fuse_conv_bn
            except ImportError:
                import torch
                def _fuse_conv_bn(conv, bn):
                    conv_weight = conv.weight
                    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                    conv.weight.data = conv_weight * factor.reshape([conv.out_channels, 1, 1, 1])
                    conv.bias.data = (conv_bias - bn.running_mean) * factor + bn.bias
                    return conv
                fuse_conv_bn = _fuse_conv_bn
    else:
        from mmcv.cnn import fuse_conv_bn
except ImportError as e:
    warnings.warn(f"fuse_conv_bn 导入失败: {e}")


# ==================== 核心类导入 ====================

if MMCV_MAJOR >= 2:
    from mmengine import Config, Registry
    from mmengine import build_from_cfg
    
    try:
        from mmengine.runner import Runner
    except ImportError:
        from mmengine import Runner
    
    try:
        from mmengine.registry import DATASETS, MODELS, OPTIMIZERS, LOSSES, METRICS, HOOKS
    except ImportError:
        try:
            from mmengine import DATASETS, MODELS, OPTIMIZERS
            LOSSES = Registry('loss')
            METRICS = Registry('metric')
            HOOKS = Registry('hook')
        except ImportError:
            DATASETS = Registry('dataset')
            MODELS = Registry('model')
            OPTIMIZERS = Registry('optimizer')
            LOSSES = Registry('loss')
            METRICS = Registry('metric')
            HOOKS = Registry('hook')
    
    def build_runner(cfg, default_args=None):
        if hasattr(Runner, 'from_cfg'):
            # Check if cfg has all required mmengine parameters
            # If it's mmcv1-style config (has 'type' but lacks 'model'), need to convert
            if isinstance(cfg, dict) or hasattr(cfg, '_cfg_dict'):
                cfg_dict = cfg._cfg_dict_dict if hasattr(cfg, '_cfg_dict_dict') else (dict(cfg) if isinstance(cfg, dict) else cfg)

                # Check if this is mmcv1-style config (has type but missing required mmengine params)
                if 'type' in cfg_dict and 'model' not in cfg_dict:
                    # mmcv1-style config detected
                    # Don't try to convert to mmengine, let compat runner handle it
                    # This avoids issues with train_dataloader, train_cfg, optim_wrapper mismatch
                    return None

            return Runner.from_cfg(cfg)
        return Runner(cfg, **(default_args or {}))
        
else:
    from mmcv import Config, Registry
    from mmcv import DATASETS, MODELS, OPTIMIZERS, LOSSES, METRICS
    from mmcv.runner import Runner, HOOKS
    from mmcv import build_from_cfg
    
    def build_runner(cfg, default_args=None):
        if default_args is not None:
            cfg.merge_from_dict(default_args)
        return Runner.from_cfg(cfg)


# ==================== 工具函数 ====================

if MMCV_MAJOR >= 2:
    from mmengine.utils import ProgressBar, Timer
    from mmengine.utils import check_file_exist, mkdir_or_exist, is_filepath
    
    try:
        from mmengine.logging import MMLogger
        def get_logger(name, log_file=None, log_level=logging.INFO):
            return MMLogger.get_instance(name, log_file=log_file, log_level=log_level)
        def print_log(msg, logger=None, level=logging.INFO):
            if logger is None:
                print(msg)
            else:
                logger.log(level, msg)
    except ImportError:
        def get_logger(name, log_file=None, log_level=logging.INFO):
            logger = logging.getLogger(name)
            logger.setLevel(log_level)
            if log_file:
                handler = logging.FileHandler(log_file)
                handler.setLevel(log_level)
                logger.addHandler(handler)
            return logger
        
        def print_log(msg, logger=None, level=logging.INFO):
            if logger is None:
                print(msg)
            else:
                logger.log(level, msg)
    
else:
    from mmcv.utils import ProgressBar, Timer
    from mmcv.utils import check_file_exist, mkdir_or_exist, is_filepath
    try:
        from mmcv.utils import get_logger, print_log
    except ImportError:
        def get_logger(name, log_file=None, log_level=logging.INFO):
            logger = logging.getLogger(name)
            logger.setLevel(log_level)
            if log_file:
                handler = logging.FileHandler(log_file)
                handler.setLevel(log_level)
                logger.addHandler(handler)
            return logger
        
        def print_log(msg, logger=None, level=logging.INFO):
            if logger is None:
                print(msg)
            else:
                logger.log(level, msg)

# track_iter_progress compatibility
def track_iter_progress(iterable):
    """Iterate with a progress bar for mmcv1/mmcv2 compatibility."""
    total = len(iterable) if hasattr(iterable, '__len__') else None
    prog_bar = ProgressBar(total) if total is not None else None
    for item in iterable:
        yield item
        if prog_bar is not None:
            prog_bar.update()


# ==================== 图像处理 ====================

try:
    from mmcv.image import imread, imwrite, imrescale, imresize
    from mmcv.image import imflip, impad, imnormalize
    try:
        from mmcv.image import imshear, imrotate
    except ImportError:
        imshear = None
        imrotate = None
except ImportError:
    from mmcv import imread, imwrite, imrescale, imresize
    from mmcv import imflip, impad, imnormalize
    try:
        from mmcv import imshear, imrotate
    except ImportError:
        imshear = None
        imrotate = None


# ==================== 视频处理 ====================

try:
    from mmcv.video import VideoReader, frames2video
except ImportError:
    from mmcv import VideoReader, frames2video


# ==================== 可视化 ====================

try:
    from mmcv.visualization import color_val, imshow
    try:
        from mmcv.visualization import imshow_bboxes, imshow_det_bboxes
    except ImportError:
        imshow_bboxes = imshow
        imshow_det_bboxes = imshow
except ImportError:
    from mmcv import color_val, imshow
    try:
        from mmcv import imshow_bboxes, imshow_det_bboxes
    except ImportError:
        imshow_bboxes = imshow
        imshow_det_bboxes = imshow


# ==================== 兼容性函数 ====================

if MMCV_MAJOR >= 2:
    def imread_compat(img_or_path, flag='color', channel_order='bgr', backend=None):
        try:
            return imread(img_or_path, flag=flag, channel_order=channel_order, backend=backend)
        except TypeError:
            return imread(img_or_path, flag=flag, channel_order=channel_order)
    
    def imwrite_compat(img, file_path, params=None, auto_mkdir=True):
        try:
            return imwrite(img, file_path, params=params, auto_mkdir=auto_mkdir)
        except TypeError:
            return imwrite(img, file_path, params=params)
else:
    imread_compat = imread
    imwrite_compat = imwrite


# 版本信息
def print_mmcv_info():
    print(f"MMCV Version: {MMCV_VERSION}")
    print(f"MMCV Major: {MMCV_MAJOR}")
    if MMCV_MAJOR >= 2:
        try:
            import mmengine
            print(f"MMEngine Version: {mmengine.__version__}")
        except:
            pass
    print(f"Python: {sys.version}")


# Expose common compat symbols on mmcv module for legacy call-sites.
try:
    import mmcv as _mmcv_pkg
    if not hasattr(_mmcv_pkg, 'ProgressBar'):
        _mmcv_pkg.ProgressBar = ProgressBar
    if not hasattr(_mmcv_pkg, 'track_iter_progress'):
        _mmcv_pkg.track_iter_progress = track_iter_progress
except Exception:
    pass
