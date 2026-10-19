"""Sidebar tab "OMFS" in the 3D viewport (press N to show the sidebar)."""

import bpy

from . import locking


class OMFS_PT_case(bpy.types.Panel):
    bl_label = "Case"
    bl_idname = "OMFS_PT_case"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        layout.operator("omfs.import_case", icon='IMPORT')
        layout.operator("omfs.setup_scene", icon='SETTINGS')

        meshes = [obj for obj in context.scene.objects if obj.type == 'MESH']
        if not meshes:
            return
        box = layout.box()
        for obj in meshes:
            row = box.row(align=True)
            row.label(text=obj.name)
            row.prop(obj.omfs, "role", text="")
            locked = locking.is_locked(obj)
            op = row.operator("omfs.toggle_lock", text="", icon='LOCKED' if locked else 'UNLOCKED',
                              depress=locked)
            op.object_name = obj.name


classes = (OMFS_PT_case,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
