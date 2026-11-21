"""OMFS Planner: virtual surgical planning and guide design for oral and maxillofacial surgery."""

from . import roles, locking, scene_setup, ops_import, new_case, registration, crown_fusion, landmarks, reference_frame, movement, osteotomy, osteotomy_gizmos, movement_gizmos, occlusion, occlusion_prepare, occlusion_segments, occlusion_seating, occlusion_segment_seat, text_labels, splint, splint_holes, section, section_contours, analysis, distance_map, snapshot, measure, cylinders, screws, pins, guide_workshop, sleeves, trim, model_base, tidy, view, print_chain, print_profiles, print_scene, volume_scene, ortho_osteotomy, ortho_movement, point_balls, ui, print_ui, volume_ui, ortho_ui, slice_core, slice_viewer

# Each module exposes register() / unregister(); order matters for dependencies.
modules = (roles, locking, scene_setup, ops_import, new_case, registration, crown_fusion, landmarks, reference_frame, movement, osteotomy, osteotomy_gizmos, movement_gizmos, occlusion, occlusion_prepare, occlusion_segments, occlusion_seating, occlusion_segment_seat, text_labels, splint, splint_holes, section, section_contours, analysis, distance_map, snapshot, measure, cylinders, screws, pins, guide_workshop, sleeves, trim, model_base, tidy, view, print_scene, volume_scene, ortho_osteotomy, ortho_movement, point_balls, ui, print_ui, volume_ui, ortho_ui, slice_viewer)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()
