"""Sidebar tab "OMFS" in the 3D viewport (press N to show the sidebar)."""

import bpy

from . import landmarks, locking


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


class OMFS_PT_landmarks(bpy.types.Panel):
    bl_label = "Landmarks"
    bl_idname = "OMFS_PT_landmarks"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        group = None
        for landmark_id, name, _description, lm_group in landmarks.LANDMARKS:
            if lm_group != group:
                group = lm_group
                col = layout.box().column(align=True)
                col.label(text=group)
            obj = landmarks.find(context.scene, landmark_id)
            row = col.row(align=True)
            row.label(text=name, icon='CHECKMARK' if obj else 'BLANK1')
            op = row.operator("omfs.place_landmark", text="Redo" if obj else "Place",
                              icon='RESTRICT_SELECT_OFF')
            op.landmark_id = landmark_id
            if obj:
                row.operator("omfs.delete_landmark", text="", icon='X').landmark_id = landmark_id


classes = (OMFS_PT_case, OMFS_PT_landmarks)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
