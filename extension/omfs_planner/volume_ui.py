"""Panels of the Segmentation workspace: an entry in the OMFS tab, and the "Segmentation" tab of its scene."""

import bpy

from . import params, volume_scene
from .ui import draw_text


class OMFS_PT_volume_entry(bpy.types.Panel):
    bl_label = "Segmentation (.nrrd)"
    bl_idname = "OMFS_PT_volume_entry"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return not context.scene.omfs_volume.is_volume_scene and not context.scene.omfs_print.is_print_scene

    def draw(self, context):
        layout = self.layout
        draw_text(layout, context, "Glissez un fichier de DentalSeg (…_probabilites.nrrd) dans la vue 3D, ou:")
        row = layout.row()
        row.scale_y = 1.3
        row.operator("omfs.volume_open", icon='FILE_VOLUME')
        if volume_scene.volume_scene() is not None:
            layout.operator("omfs.volume_show", icon='RESTRICT_VIEW_OFF')


class VolumePanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Segmentation"

    @classmethod
    def poll(cls, context):
        return context.scene.omfs_volume.is_volume_scene


def pieces_text(obj):
    """(icon, text) for the piece count of a structure, with a warning when there are more than expected."""
    found = volume_scene.piece_volumes(obj)
    if found is None:
        return 'SORTTIME', "calcul…"
    if not found:
        return 'INFO', "vide"
    expected = params.VOL_EXPECTED_PIECES.get(volume_scene.structure_name(obj))
    text = f"{len(found)} morceau{'x' if len(found) > 1 else ''}"
    if expected is not None and len(found) > expected:
        sizes = ", ".join(f"{v / 1000:.1f}" for v in found[:4])
        return 'ERROR', f"{text}: {sizes} cm³"
    return ('CHECKMARK' if expected is not None else 'BLANK1'), f"{text} · {sum(found) / 1000:.1f} cm³"


class OMFS_PT_volume_structures(VolumePanel):
    bl_label = "Structures"
    bl_idname = "OMFS_PT_volume_structures"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        volume = volume_scene.volume_of(scene)
        if volume is None:
            draw_text(layout, context, "Les voxels ne sont plus en mémoire (Blender a été relancé): rouvrez le "
                                       "fichier pour corriger. Les surfaces affichées peuvent encore être importées.")
            layout.operator("omfs.volume_open", icon='FILE_VOLUME').filepath = scene.omfs_volume.path
        else:
            kind = "probabilités" if volume.probabilities else "étiquettes"
            layout.label(text=f"{bpy.path.basename(scene.omfs_volume.path)[-34:]} · {kind}", icon='FILE_VOLUME')
        for obj in volume_scene.structures(scene):
            box = layout.box()
            row = box.row(align=True)
            row.prop(obj, "hide_viewport", text="", emboss=False, icon='HIDE_ON' if obj.hide_viewport else 'HIDE_OFF')
            row.prop(obj, "color", text="")
            row.label(text=volume_scene.structure_name(obj))
            icon, text = pieces_text(obj)
            box.label(text=text, icon=icon)
            if obj.hide_viewport:
                continue
            props = obj.omfs_vol
            col = box.column(align=True)
            if volume is None or volume.probabilities:
                col.prop(props, "threshold")
            col.prop(props, "min_piece")
            row = col.row(align=True)
            row.prop(props, "smooth")
            row.prop(props, "simplify")
            col.prop(props, "cavities")


class OMFS_PT_volume_correct(VolumePanel):
    bl_label = "Corriger un morceau"
    bl_idname = "OMFS_PT_volume_correct"

    def draw(self, context):
        layout = self.layout
        draw_text(layout, context, "Un morceau mal étiqueté (ex. un bout de « maxillaire » collé à la mandibule): "
                                   "choisissez à qui le donner, puis cliquez-le dans la vue.")
        layout.prop(context.scene.omfs_volume, "target")
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("omfs.volume_reassign", icon='EYEDROPPER')
        layout.operator("omfs.volume_undo", icon='LOOP_BACK')


class OMFS_PT_volume_finish(VolumePanel):
    bl_label = "Importer"
    bl_idname = "OMFS_PT_volume_finish"

    def draw(self, context):
        layout = self.layout
        warnings = [volume_scene.structure_name(o) for o in volume_scene.structures(context.scene) if pieces_text(o)[0] == 'ERROR']
        if warnings:
            draw_text(layout, context, "À vérifier: " + ", ".join(warnings) + " en plusieurs morceaux.")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("omfs.volume_import", icon='IMPORT')
        layout.operator("omfs.volume_back", icon='LOOP_BACK')


classes = (OMFS_PT_volume_entry, OMFS_PT_volume_structures, OMFS_PT_volume_correct, OMFS_PT_volume_finish)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
