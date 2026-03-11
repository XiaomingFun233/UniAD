import importlib
import warnings


def _safe_import(module_name):
    try:
        importlib.import_module(module_name)
    except Exception as e:
        warnings.warn(f"Skip optional import {module_name}: {e}")


_safe_import('projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d')
_safe_import('projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder')
_safe_import('projects.mmdet3d_plugin.core.bbox.match_costs')
_safe_import('projects.mmdet3d_plugin.datasets.pipelines')
_safe_import('projects.mmdet3d_plugin.models.backbones.vovnet')
_safe_import('projects.mmdet3d_plugin.models.opt.adamw')
_safe_import('projects.mmdet3d_plugin.uniad')
_safe_import('projects.mmdet3d_plugin.losses')
