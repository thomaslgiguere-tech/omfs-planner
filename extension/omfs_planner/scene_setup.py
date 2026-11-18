"""Scene in millimetres (1 Blender unit = 1 mm) and a viewport suited to bone and scans."""

import bpy


def set_millimetres(scene):
    units = scene.unit_settings
    units.system = 'METRIC'
    units.scale_length = 0.001        # 1 Blender unit is shown as 1 mm
    units.length_unit = 'MILLIMETERS'


def set_local_orientation(scene):
    """Move, rotate and scale along the object's own axes by default: G / R / S then X = the object's
    X (e.g. a tilted cylinder's axis). Pressing the axis key twice switches to global."""
    slots = scene.transform_orientation_slots   # default, then move, rotate, scale
    slots[0].type = 'LOCAL'
    for slot in slots[1:]:
        slot.type = 'DEFAULT'                    # follow the default


def setup_viewport(space):
    """Configure one 3D viewport (a SpaceView3D)."""
    space.clip_start = 0.1      # mm: don't cut away anatomy close to the camera
    space.clip_end = 10000.0    # mm
    shading = space.shading
    shading.type = 'SOLID'
    shading.light = 'STUDIO'
    shading.color_type = 'OBJECT'           # each object shows its role colour
    shading.show_cavity = True              # darkens grooves, brightens ridges: reads bone detail
    shading.cavity_type = 'BOTH'
    shading.show_specular_highlight = True
    shading.show_backface_culling = False   # see the inside of cut or open meshes
    space.overlay.show_relationship_lines = False   # no dotted lines from a point to the point it hangs from


def view3d_areas(context):
    """Yield (window, area, region, space) for every 3D viewport."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next(r for r in area.regions if r.type == 'WINDOW')
                yield window, area, region, area.spaces.active


def frame_all(context):
    for window, area, region, _space in view3d_areas(context):
        with context.temp_override(window=window, area=area, region=region):
            bpy.ops.view3d.view_all()


def setup_scene(context):
    set_millimetres(context.scene)
    set_local_orientation(context.scene)
    for _window, _area, _region, space in view3d_areas(context):
        setup_viewport(space)


class OMFS_OT_setup_scene(bpy.types.Operator):
    """Set units to millimetres and configure the viewport for bone and scans"""
    bl_idname = "omfs.setup_scene"
    bl_label = "Set Up Scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        setup_scene(context)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(OMFS_OT_setup_scene)


def unregister():
    bpy.utils.unregister_class(OMFS_OT_setup_scene)
