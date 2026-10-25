"""Import Case: import every mesh in a folder, name, colour and lock them, set mm units."""

from pathlib import Path

import bpy

from . import locking, mesh_io, roles, scene_setup, view


def name_objects(objects, stem):
    """Name objects (and their mesh data) after the file; numbered if a file holds several."""
    for i, obj in enumerate(objects):
        name = stem if len(objects) == 1 else f"{stem}_{i + 1}"
        obj.name = name
        if obj.data is not None:
            obj.data.name = name


class OMFS_OT_import_case(bpy.types.Operator):
    """Import every STL, OBJ and PLY mesh in a folder"""
    bl_idname = "omfs.import_case"
    bl_label = "Import Case"
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(name="Case folder", subtype='DIR_PATH')
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)   # opens the file browser
        return {'RUNNING_MODAL'}

    def execute(self, context):
        folder = Path(self.directory)
        if not folder.is_dir():
            self.report({'ERROR'}, f"Not a folder: {folder}")
            return {'CANCELLED'}
        files = mesh_io.mesh_files(folder)
        if not files:
            self.report({'ERROR'}, f"No STL, OBJ or PLY files in {folder}")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        imported = []
        for path in files:
            objects = mesh_io.import_mesh(path)
            name_objects(objects, path.stem)
            role = roles.role_from_filename(path.name)
            for obj in objects:
                obj.omfs.role = role
                roles.apply_role_color(obj)
                obj.select_set(False)
                locking.set_locked(obj, role in roles.LOCKED_BY_DEFAULT)
            imported += objects

        scene_setup.setup_scene(context)
        scene_setup.frame_all(context)
        if not bpy.app.background:
            view.activate_navigate_tool(context)
        self.report({'INFO'}, f"Imported {len(imported)} meshes from {folder.name}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(OMFS_OT_import_case)


def unregister():
    bpy.utils.unregister_class(OMFS_OT_import_case)
