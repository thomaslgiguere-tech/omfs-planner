"""Panel "Ostéotomies orthognathiques" (OMFS tab): Le Fort I and BSSO cuts, then the separation of the segments."""

import bpy

from . import ortho_osteotomy as ortho
from .ui import draw_text


def draw_cut(layout, context, kind, label):
    scene = context.scene
    cut = ortho.find_cut(scene, kind)
    row = layout.row(align=True)
    row.operator("omfs.ortho_draw", text=label if cut is None else f"Retracer {label}",
                 icon='GREASEPENCIL' if cut is None else 'FILE_REFRESH').kind = kind
    if cut is not None:
        row.operator("omfs.ortho_remove", text="", icon='X').kind = kind
        col = layout.column(align=True)
        col.prop(cut.omfs_ortho, "depth_ui")
        col.label(text=f"{len(ortho.front_points(cut))} points (orange) · points arrière (bleus) déplaçables", icon='BLANK1')


class OMFS_PT_ortho_osteotomy(bpy.types.Panel):
    bl_label = "Ostéotomies orthognathiques"
    bl_idname = "OMFS_PT_ortho_osteotomy"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return not scene.omfs_volume.is_volume_scene and not scene.omfs_print.is_print_scene

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        draw_text(layout, context, "Cliquez des points sur l'os où passe la coupe, dans la vue qui regarde dans le sens "
                                   "de la scie (Le Fort: de face; BSSO: de côté), puis Entrée. Ajustez les points arrière "
                                   "(bleus, G pour déplacer) pour placer la coupe à distance.")
        for which, label, kinds in (('MAXILLA', "Maxillaire", (('LEFORT', "Le Fort I"),)),
                                    ('MANDIBLE', "Mandibule", (('BSSO_R', "BSSO droite"), ('BSSO_L', "BSSO gauche")))):
            box = layout.box()
            source = ortho.bone(scene, which)
            row = box.row()
            row.prop(scene.omfs_ortho_scene, "maxilla" if which == 'MAXILLA' else "mandible", text=label)
            if source is not None and getattr(scene.omfs_ortho_scene, "maxilla" if which == 'MAXILLA' else "mandible") is None:
                box.label(text=f"→ {source.name}", icon='BLANK1')
            for kind, text in kinds:
                draw_cut(box, context, kind, text)
            pieces = ortho.pieces_of(scene, source) if source is not None else []
            row = box.row(align=True)
            row.enabled = source is not None and bool(ortho.cuts(scene, source))
            row.operator("omfs.ortho_separate", icon='MOD_BOOLEAN').which = which
            if pieces:
                row.operator("omfs.ortho_rejoin", icon='LOOP_BACK').which = which
                for piece in pieces:
                    box.label(text=piece.name, icon='CHECKMARK' if piece.omfs.role == 'SEGMENT' else 'LOCKED')


classes = (OMFS_PT_ortho_osteotomy,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
