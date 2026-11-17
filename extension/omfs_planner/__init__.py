"""OMFS Planner: virtual surgical planning and guide design for oral and maxillofacial surgery."""

from . import roles, locking, scene_setup, ops_import, registration, landmarks, reference_frame, movement, osteotomy, osteotomy_gizmos, movement_gizmos, occlusion, occlusion_seating, occlusion_segments, text_labels, splint, splint_holes, section, section_contours, analysis, distance_map, snapshot, cylinders, screws, pins, guide_workshop, sleeves, trim, model_base, tidy, view, print_chain, print_profiles, print_scene, ui, print_ui

# Each module exposes register() / unregister(); order matters for dependencies.
modules = (roles, locking, scene_setup, ops_import, registration, landmarks, reference_frame, movement, osteotomy, osteotomy_gizmos, movement_gizmos, occlusion, occlusion_seating, occlusion_segments, text_labels, splint, splint_holes, section, section_contours, analysis, distance_map, snapshot, cylinders, screws, pins, guide_workshop, sleeves, trim, model_base, tidy, view, print_scene, ui, print_ui)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()
