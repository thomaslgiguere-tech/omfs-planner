"""Clinical views (face, profiles, submental, vertex) and the C shortcut to centre the view.

The views assume the meshes are in patient LPS coordinates, as exported by 3D Slicer:
+X = patient left, +Y = posterior, +Z = superior. Once the reference frame exists,
the views will follow it instead.
"""

import bpy

# (id, label, Blender view axis). Blender's "Left" view looks from -X, i.e. at the patient's right.
VIEWS = (
    ('FACE', "Face", 'FRONT'),
    ('PROFILE_R', "Profile R", 'LEFT'),
    ('PROFILE_L', "Profile L", 'RIGHT'),
    ('SUBMENTAL', "Submental", 'BOTTOM'),
    ('VERTEX', "Vertex", 'TOP'),
)
AXIS = {view_id: axis for view_id, _label, axis in VIEWS}

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
        region = next(r for r in area.regions if r.type == 'WINDOW')
        with context.temp_override(area=area, region=region):
            bpy.ops.view3d.view_axis(type=AXIS[self.view])
            bpy.ops.view3d.view_all()
        return {'FINISHED'}


def set_navigation_preferences(preferences):
    """Trackpad-friendly navigation: orbit and zoom around the bone under the mouse."""
    inputs = preferences.inputs
    inputs.use_mouse_depth_navigate = True   # Auto Depth
    inputs.use_zoom_to_mouse = True          # zoom towards the mouse, not the screen centre
    inputs.use_emulate_numpad = True         # number row = numpad views (no numpad on most Macs)


class OMFS_OT_navigation_preferences(bpy.types.Operator):
    """Turn on Auto Depth, Zoom to Mouse Position and Emulate Numpad (Blender preferences, all files)"""
    bl_idname = "omfs.navigation_preferences"
    bl_label = "Trackpad-Friendly Navigation"

    def execute(self, context):
        set_navigation_preferences(context.preferences)
        self.report({'INFO'}, "Navigation preferences set (Preferences > Navigation / Input)")
        return {'FINISHED'}


classes = (OMFS_OT_view, OMFS_OT_navigation_preferences)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig:   # None in background mode
        keymap = keyconfig.keymaps.new(name="3D View", space_type='VIEW_3D')
        item = keymap.keymap_items.new("view3d.view_center_pick", CENTRE_KEY, 'PRESS')
        addon_keymaps.append((keymap, item))


def unregister():
    for keymap, item in addon_keymaps:
        keymap.keymap_items.remove(item)
    addon_keymaps.clear()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
