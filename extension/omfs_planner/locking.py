"""Lock anatomy that must not move: still visible and measurable, but not selectable or movable."""

import bpy


def is_locked(obj):
    return obj.hide_select


def set_locked(obj, locked):
    if locked:
        obj.select_set(False)
    obj.hide_select = locked          # can't be clicked in the viewport or outliner
    obj.lock_location = (locked,) * 3  # and can't be moved, rotated or scaled
    obj.lock_rotation = (locked,) * 3
    obj.lock_scale = (locked,) * 3


class OMFS_OT_toggle_lock(bpy.types.Operator):
    """Lock or unlock this object (locked objects can't be selected or moved)"""
    bl_idname = "omfs.toggle_lock"
    bl_label = "Toggle Lock"
    bl_options = {'REGISTER', 'UNDO'}

    object_name: bpy.props.StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.object_name)
        if obj is None:
            self.report({'ERROR'}, f"No object named {self.object_name!r}")
            return {'CANCELLED'}
        set_locked(obj, not is_locked(obj))
        return {'FINISHED'}


def register():
    bpy.utils.register_class(OMFS_OT_toggle_lock)


def unregister():
    bpy.utils.unregister_class(OMFS_OT_toggle_lock)
