"""Clinical views (face, profiles, submental, vertex) and Meshmixer-like navigation.

C centres the view on the point under the mouse; orbit and zoom then turn around that centre.
With the Navigate tool active, click-drag orbits and Shift + click-drag pans (mouse buttons,
so it also works through remote-desktop apps that mangle trackpad gestures).

Once the reference frame exists, the views follow its axes. Before that, they assume the
meshes are in patient LPS coordinates, as exported by 3D Slicer: +X = patient left,
+Y = posterior, +Z = superior.
"""

import bpy
from mathutils import Quaternion

from . import reference_frame, scene_setup

# (id, label, Blender view axis). Blender's "Left" view looks from -X, i.e. at the patient's right.
VIEWS = (
    ('FACE', "Face", 'FRONT'),
    ('PROFILE_R', "Profile R", 'LEFT'),
    ('PROFILE_L', "Profile L", 'RIGHT'),
    ('SUBMENTAL', "Submental", 'BOTTOM'),
    ('VERTEX', "Vertex", 'TOP'),
)
AXIS = {view_id: axis for view_id, _label, axis in VIEWS}

# Blender's view rotations for each view axis (as set by View > Viewpoint), in LPS-like axes.
VIEW_ROTATION = {
    'FRONT': Quaternion((0.7071068, 0.7071068, 0, 0)),
    'LEFT': Quaternion((0.5, 0.5, -0.5, -0.5)),
    'RIGHT': Quaternion((0.5, 0.5, 0.5, 0.5)),
    'TOP': Quaternion((1, 0, 0, 0)),
    'BOTTOM': Quaternion((0, 1, 0, 0)),
}

CENTRE_KEY = 'C'   # as in Meshmixer: centre the view on the point under the mouse


addon_keymaps = []


class OMFS_OT_view(bpy.types.Operator):
    """Look at the patient from a standard clinical direction"""
    bl_idname = "omfs.view"
    bl_label = "Clinical View"

    view: bpy.props.EnumProperty(items=[(v, label, "") for v, label, _axis in VIEWS])

    def execute(self, context):
        area = context.area if context.area and context.area.type == 'VIEW_3D' else None
        if area is None:
            self.report({'ERROR'}, "No 3D viewport")
            return {'CANCELLED'}
        area.spaces.active.lock_object = None   # "Lock to Object" stops the view from framing
        region = next(r for r in area.regions if r.type == 'WINDOW')
        rotation = VIEW_ROTATION[AXIS[self.view]]
        frame = reference_frame.find_frame(context.scene)
        if frame is not None:   # turn the view with the patient's frame
            rotation = frame.matrix_world.to_quaternion() @ rotation
        region.data.view_rotation = rotation
        region.data.view_perspective = 'ORTHO'
        with context.temp_override(area=area, region=region):
            bpy.ops.view3d.view_all()
        return {'FINISHED'}


def navigation_preferences_ok(preferences):
    inputs = preferences.inputs
    return (not inputs.use_mouse_depth_navigate and not inputs.use_zoom_to_mouse
            and not inputs.use_rotate_around_active and inputs.use_emulate_numpad)


def set_navigation_preferences(preferences):
    """Meshmixer-like navigation: orbit and zoom around the view centre (set with C)."""
    inputs = preferences.inputs
    inputs.use_mouse_depth_navigate = False   # Auto Depth would orbit around the point under the mouse
    inputs.use_zoom_to_mouse = False          # zoom towards the view centre, not the mouse
    inputs.use_rotate_around_active = False
    inputs.use_emulate_numpad = True          # number row = numpad views (no numpad on most Macs)


class OMFS_OT_navigation_preferences(bpy.types.Operator):
    """Orbit and zoom around the view centre, number row as numpad (Blender preferences, all files)"""
    bl_idname = "omfs.navigation_preferences"
    bl_label = "Meshmixer-Like Navigation"

    def execute(self, context):
        set_navigation_preferences(context.preferences)
        self.report({'INFO'}, "Navigation preferences set (Preferences > Navigation / Input)")
        return {'FINISHED'}


class OMFS_NavigateTool(bpy.types.WorkSpaceTool):
    """Active tool whose left-drag navigates instead of box-selecting.

    The active tool's shortcuts win over the 3D View keymap, so a drag binding has to live here.
    """
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = "omfs.navigate"
    bl_label = "Navigate"
    bl_description = "Drag: orbit around the view centre. Shift + drag: pan. Click: select"
    bl_icon = "ops.generic.cursor"
    bl_keymap = (
        ("view3d.rotate", {"type": 'LEFTMOUSE', "value": 'CLICK_DRAG'}, None),
        ("view3d.move", {"type": 'LEFTMOUSE', "value": 'CLICK_DRAG', "shift": True}, None),
        ("view3d.select", {"type": 'LEFTMOUSE', "value": 'CLICK'}, None),
    )


def activate_navigate_tool(context):
    for window, area, region, _space in scene_setup.view3d_areas(context):
        with context.temp_override(window=window, area=area, region=region):
            bpy.ops.wm.tool_set_by_id(name=OMFS_NavigateTool.bl_idname)


classes = (OMFS_OT_view, OMFS_OT_navigation_preferences)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.utils.register_tool(OMFS_NavigateTool, separator=True)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig:   # None in background mode
        keymap = keyconfig.keymaps.new(name="3D View", space_type='VIEW_3D')
        item = keymap.keymap_items.new("view3d.view_center_pick", CENTRE_KEY, 'PRESS')
        addon_keymaps.append((keymap, item))


def unregister():
    for keymap, item in addon_keymaps:
        keymap.keymap_items.remove(item)
    addon_keymaps.clear()
    bpy.utils.unregister_tool(OMFS_NavigateTool)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
