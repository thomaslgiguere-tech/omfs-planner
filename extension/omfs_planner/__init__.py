"""OMFS Planner: virtual surgical planning and guide design for oral and maxillofacial surgery."""

from . import roles, locking, scene_setup, ops_import, registration, landmarks, reference_frame, movement, osteotomy, osteotomy_gizmos, movement_gizmos, occlusion, occlusion_seating, text_labels, splint, section, section_contours, analysis, distance_map, snapshot, guide_workshop, view, ui

# Each module exposes register() / unregister(); order matters for dependencies.
modules = (roles, locking, scene_setup, ops_import, registration, landmarks, reference_frame, movement, osteotomy, osteotomy_gizmos, movement_gizmos, occlusion, occlusion_seating, text_labels, splint, section, section_contours, analysis, distance_map, snapshot, guide_workshop, view, ui)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()
