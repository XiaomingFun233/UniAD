import warnings

try:
    from .collision_optimization import *
except Exception as e:
    warnings.warn(f"collision_optimization is unavailable: {e}")
from .planning_metrics import *
