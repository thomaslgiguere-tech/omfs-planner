"""OMFS Planner: virtual surgical planning and guide design for oral and maxillofacial surgery."""

from . import roles, locking, scene_setup, ops_import, landmarks, reference_frame, view, ui

# Each module exposes register() / unregister(); order matters for dependencies.
modules = (roles, locking, scene_setup, ops_import, landmarks, reference_frame, view, ui)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()
