import warnings

try:
    from .motion_optimization import MotionNonlinearSmoother
except Exception as e:
    MotionNonlinearSmoother = None
    warnings.warn(f"MotionNonlinearSmoother is unavailable: {e}")
from .modules import MotionTransformerDecoder
from .motion_deformable_attn import MotionTransformerAttentionLayer, MotionDeformableAttention
from .motion_utils import *
