import random
import warnings
import os
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
try:
    from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_optimizer,
                             build_runner, get_dist_info)
except Exception:
    from mmcv_wrapper import HOOKS, Fp16OptimizerHook, build_runner, get_dist_info
    from mmengine.hooks import DistSamplerSeedHook

    class EpochBasedRunner:  # compatibility marker for isinstance checks
        pass

    class OptimizerHook:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def build_optimizer(model, optimizer_cfg):
        cfg = dict(optimizer_cfg)
        opt_type = cfg.pop('type')
        # Old-style config keys not supported by torch optimizer constructor.
        cfg.pop('paramwise_cfg', None)
        cfg.pop('constructor', None)
        if opt_type == 'AdamW2':
            opt_type = 'AdamW'
        opt_cls = getattr(torch.optim, opt_type, None)
        if opt_cls is None:
            raise ValueError(f'Unsupported optimizer type in compat mode: {opt_type}')
        params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
        return opt_cls(params, **cfg)

try:
    from mmcv.utils import build_from_cfg
except Exception:
    from mmengine.registry import build_from_cfg

try:
    from mmdet.core import EvalHook
except Exception:
    class EvalHook:
        def __init__(self, *args, **kwargs):
            pass

try:
    from mmdet.datasets import (build_dataset, replace_ImageToTensor)
except Exception:
    build_dataset = None

    def replace_ImageToTensor(pipeline):
        return pipeline

try:
    from mmdet.utils import get_root_logger
except Exception:
    import logging

    def get_root_logger(log_level='INFO'):
        level = getattr(logging, str(log_level).upper(), logging.INFO)
        logger = logging.getLogger('uniad_train')
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(level)
        return logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


class _CompatEpochRunner:
    """Minimal mmcv1-style runner for mmengine/mmcv2 mixed stacks."""

    def __init__(self, model, optimizer, work_dir, logger, meta=None, max_epochs=1, eval_model=None):
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.meta = meta or {}
        self.eval_model = eval_model
        self.max_epochs = int(max_epochs)
        self.epoch = 0
        self.iter = 0
        self._hooks = []
        self.timestamp = None
        self.rank, _ = get_dist_info()

    def register_training_hooks(self, *args, **kwargs):
        # Keep signature compatibility; detailed hook behavior is skipped in compat mode.
        return

    def register_hook(self, hook, priority='NORMAL'):
        self._hooks.append(hook)

    def load_checkpoint(self, filename):
        from mmcv_wrapper import load_checkpoint as compat_load_checkpoint
        compat_load_checkpoint(self.model, filename, map_location='cpu', strict=False, logger=self.logger)

    def resume(self, filename):
        ckpt = torch.load(filename, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
        self.model.load_state_dict(state_dict, strict=False)
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
        self.epoch = int(ckpt.get('meta', {}).get('epoch', self.epoch))
        self.iter = int(ckpt.get('meta', {}).get('iter', self.iter))

    def _call_hook(self, fn_name):
        for hook in self._hooks:
            fn = getattr(hook, fn_name, None)
            if callable(fn):
                fn(self)

    def run(self, data_loaders, workflow):
        if not data_loaders:
            raise RuntimeError('No dataloader provided to runner.')
        data_loader = data_loaders[0]

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
                data_loader.sampler.set_epoch(epoch)
            self._call_hook('before_train_epoch')
            self.model.train()
            for i, data_batch in enumerate(data_loader):
                self.iter += 1
                self._call_hook('before_train_iter')
                self.optimizer.zero_grad(set_to_none=True)

                outputs = None
                if hasattr(self.model, 'train_step'):
                    try:
                        outputs = self.model.train_step(data_batch, self.optimizer)
                    except TypeError:
                        outputs = self.model.train_step(data_batch, optim_wrapper=self.optimizer)
                if outputs is None:
                    outputs = self.model(**data_batch, return_loss=True)

                loss = None
                if isinstance(outputs, dict):
                    if torch.is_tensor(outputs.get('loss', None)):
                        loss = outputs['loss']
                    else:
                        losses = []
                        for key, value in outputs.items():
                            if 'loss' not in key:
                                continue
                            if torch.is_tensor(value):
                                losses.append(value)
                            elif isinstance(value, (list, tuple)):
                                losses.extend([v for v in value if torch.is_tensor(v)])
                        if losses:
                            loss = sum(losses)
                elif torch.is_tensor(outputs):
                    loss = outputs

                if loss is None:
                    raise RuntimeError('Could not parse loss from model outputs in compat runner.')

                loss.backward()
                self.optimizer.step()
                self._call_hook('after_train_iter')

                if self.rank == 0 and i % 20 == 0:
                    self.logger.info(f'[CompatRunner] epoch={epoch + 1}/{self.max_epochs} iter={i} loss={float(loss.detach().cpu()):.6f}')

            self._call_hook('after_train_epoch')
def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #assert len(dataset)==1s
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        ) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.musa(),
            device_ids=[torch.musa.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.musa(),
                device_ids=[torch.musa.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.musa(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.musa(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner_args = dict(
            model=model,
            eval_model=eval_model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
    else:
        runner_args = dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)

    try:
        runner = build_runner(cfg.runner, default_args=runner_args)
    except Exception as e:
        logger.warning(f'Falling back to compat runner due to build_runner failure: {e}')
        runner = _CompatEpochRunner(
            max_epochs=cfg.runner.get('max_epochs', cfg.get('total_epochs', 1)),
            **runner_args)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        try:
            from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
        except Exception as e:
            logger.warning(f'Skip validation hook due to compatibility issue: {e}')
            CustomDistEvalHook = None
        from projects.mmdet3d_plugin.datasets import custom_build_dataset
        if CustomDistEvalHook is not None:
            # Support batch_size > 1 in validation
            val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
            if val_samples_per_gpu > 1:
                assert False
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline)
            val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
                shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
                nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            )
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
            eval_hook = CustomDistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
