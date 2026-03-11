from .mmdet_train import custom_train_detector
try:
    from mmseg.apis import train_segmentor
except Exception:
    train_segmentor = None
try:
    from mmdet.apis import train_detector
except Exception:
    train_detector = None

def custom_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                eval_model=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        assert False
    else:
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            eval_model=eval_model,
            meta=meta)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        if train_segmentor is None:
            raise RuntimeError('train_segmentor is unavailable in current MM* stack.')
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        if train_detector is None:
            raise RuntimeError('train_detector is unavailable in current MM* stack.')
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
