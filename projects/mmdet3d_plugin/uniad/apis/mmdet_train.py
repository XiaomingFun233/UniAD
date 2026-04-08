import copy
import os
import warnings

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

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def _maybe_to_dict(cfg_obj):
    """Convert config nodes to plain dict when possible."""
    if cfg_obj is None:
        return None
    if isinstance(cfg_obj, dict):
        return copy.deepcopy(cfg_obj)
    try:
        return copy.deepcopy(dict(cfg_obj))
    except Exception:
        return copy.deepcopy(cfg_obj)


def _convert_lr_config_to_param_scheduler(cfg, max_epochs):
    """Translate mmcv-style lr_config to mmengine param_scheduler configs."""
    lr_cfg = cfg.get('lr_config', None)
    if lr_cfg is None:
        return None
    lr_cfg = _maybe_to_dict(lr_cfg)
    if not isinstance(lr_cfg, dict):
        return None

    schedulers = []
    warmup = lr_cfg.get('warmup', None)
    warmup_iters = int(lr_cfg.get('warmup_iters', 0) or 0)
    if warmup == 'linear' and warmup_iters > 0:
        schedulers.append(
            dict(
                type='LinearLR',
                by_epoch=False,
                begin=0,
                end=warmup_iters,
                start_factor=float(lr_cfg.get('warmup_ratio', 1.0 / 3)),
            ))
    elif warmup is not None and warmup_iters > 0:
        warnings.warn(
            f'Unsupported warmup type "{warmup}" in mmengine conversion, skip warmup.',
            UserWarning)

    policy = str(lr_cfg.get('policy', '')).lower()
    if policy == 'cosineannealing':
        main_scheduler = dict(
            type='CosineAnnealingLR',
            by_epoch=True,
            begin=0,
            end=int(max_epochs),
            T_max=int(max_epochs),
        )
        if 'min_lr_ratio' in lr_cfg:
            main_scheduler['eta_min_ratio'] = float(lr_cfg['min_lr_ratio'])
        elif 'min_lr' in lr_cfg:
            main_scheduler['eta_min'] = float(lr_cfg['min_lr'])
        schedulers.append(main_scheduler)
    elif policy in ('step', 'steplr'):
        steps = lr_cfg.get('step', [])
        if isinstance(steps, int):
            steps = [steps]
        schedulers.append(
            dict(
                type='MultiStepLR',
                by_epoch=True,
                begin=0,
                end=int(max_epochs),
                milestones=[int(s) for s in steps],
                gamma=float(lr_cfg.get('gamma', 0.1)),
            ))
    elif policy:
        warnings.warn(
            f'Unsupported lr policy "{lr_cfg.get("policy")}" in mmengine conversion, '
            'use fixed learning rate.',
            UserWarning)

    return schedulers or None


def _build_mmengine_runner(model,
                           data_loader,
                           cfg,
                           logger,
                           distributed=False,
                           validate=False):
    """Build mmengine.Runner directly from existing mmcv-style cfg."""
    from mmengine.runner import Runner

    runner_cfg = cfg.get('runner', None)
    max_epochs = int(cfg.get('total_epochs', 1))
    if runner_cfg is not None:
        max_epochs = int(runner_cfg.get('max_epochs', max_epochs))

    optimizer_cfg = _maybe_to_dict(cfg.optimizer)
    if isinstance(optimizer_cfg, dict):
        paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        constructor = optimizer_cfg.pop('constructor', None)
        if optimizer_cfg.get('type') == 'AdamW2':
            optimizer_cfg['type'] = 'AdamW'
    else:
        paramwise_cfg = None
        constructor = None

    optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=optimizer_cfg)
    if paramwise_cfg is not None:
        optim_wrapper_cfg['paramwise_cfg'] = paramwise_cfg
    if constructor is not None:
        optim_wrapper_cfg['constructor'] = constructor

    optimizer_config = cfg.get('optimizer_config', {}) or {}
    grad_clip = optimizer_config.get('grad_clip', None)
    if grad_clip is not None:
        optim_wrapper_cfg['clip_grad'] = _maybe_to_dict(grad_clip)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optim_wrapper_cfg['type'] = 'AmpOptimWrapper'
        fp16_cfg = _maybe_to_dict(fp16_cfg)
        if isinstance(fp16_cfg, dict):
            optim_wrapper_cfg.update(fp16_cfg)

    checkpoint_cfg = _maybe_to_dict(cfg.get('checkpoint_config', {})) or {}
    checkpoint_cfg.setdefault('by_epoch', True)
    checkpoint_cfg.setdefault('interval', 1)

    log_cfg = _maybe_to_dict(cfg.get('log_config', {})) or {}
    log_interval = int(log_cfg.get('interval', 50))
    default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=log_interval),
        checkpoint=dict(type='CheckpointHook', **checkpoint_cfg),
        sampler_seed=dict(type='DistSamplerSeedHook'),
    )

    param_scheduler = _convert_lr_config_to_param_scheduler(cfg, max_epochs)
    if param_scheduler is not None:
        default_hooks['param_scheduler'] = dict(type='ParamSchedulerHook')

    vis_backends = [dict(type='LocalVisBackend')]
    for hook_cfg in log_cfg.get('hooks', []) or []:
        if isinstance(hook_cfg, dict) and hook_cfg.get('type') == 'TensorboardLoggerHook':
            vis_backends.append(dict(type='TensorboardVisBackend'))
            break
    visualizer = dict(type='Visualizer', vis_backends=vis_backends, name='uniad_visualizer')

    eval_cfg = _maybe_to_dict(cfg.get('evaluation', {})) or {}
    val_interval = int(eval_cfg.get('interval', max_epochs + 1))
    train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)

    if validate:
        logger.warning(
            'MMEngine runner path currently skips legacy eval-hook conversion. '
            'Recommend adding --no-validate for this stack.')

    env_cfg = _maybe_to_dict(cfg.get('env_cfg', None))
    if env_cfg is None:
        env_cfg = dict(dist_cfg=dict(backend=os.getenv('UNIAD_DIST_BACKEND', 'mccl')))

    randomness = dict(
        seed=cfg.get('seed', None),
        deterministic=bool(cfg.get('deterministic', False)))

    resume_from = cfg.get('resume_from', None)
    load_from = resume_from if resume_from is not None else cfg.get('load_from', None)

    runner_meta_cfg = dict(
        find_unused_parameters=bool(cfg.get('find_unused_parameters', False)))

    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=data_loader,
        train_cfg=train_cfg,
        optim_wrapper=optim_wrapper_cfg,
        param_scheduler=param_scheduler,
        default_hooks=default_hooks,
        custom_hooks=_maybe_to_dict(cfg.get('custom_hooks', None)),
        load_from=load_from,
        resume=bool(resume_from),
        launcher='pytorch' if distributed else 'none',
        env_cfg=env_cfg,
        log_level=cfg.log_level,
        visualizer=visualizer,
        default_scope='mmdet3d',
        randomness=randomness,
        cfg=runner_meta_cfg,
    )
    return runner


def custom_train_detector(model,
                          dataset,
                          cfg,
                          distributed=False,
                          validate=False,
                          timestamp=None,
                          eval_model=None,
                          meta=None):
    """MMEngine-only training entry for UniAD."""
    logger = get_root_logger(cfg.log_level)
    _ = timestamp
    _ = meta

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if len(dataset) == 0:
        raise ValueError('Empty dataset list is not supported.')
    if len(dataset) > 1:
        logger.warning(
            'MMEngine-only path will use only the first dataset from `dataset` list.')

    if eval_model is not None:
        logger.warning('MMEngine-only path ignores eval_model argument.')

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

    train_dataloader = build_dataloader(
        dataset[0],
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    logger.info('Using mmengine.Runner path for training.')
    runner = _build_mmengine_runner(
        model=model,
        data_loader=train_dataloader,
        cfg=cfg,
        logger=logger,
        distributed=distributed,
        validate=validate)
    runner.train()
