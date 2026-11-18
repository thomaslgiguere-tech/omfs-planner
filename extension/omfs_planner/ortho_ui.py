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
        if ortho.is_split(cut):
            col.label(text=f"{len(ortho.vertical_points(cut))} points bleus (trait vertical) · "
                           f"{len(ortho.front_points(cut))} orange (G pour déplacer)", icon='BLANK1')
            p = cut.omfs_ortho
            row = col.row()
            row.alert = not p.nerve_found
            row.label(text=f"Arrêt au nerf : {p.nerve_depth:.1f} mm" if p.nerve_found else
                      f"Canal du nerf introuvable : {p.nerve_depth:.1f} mm par défaut", icon='CHECKMARK' if p.nerve_found else 'ERROR')
        else:
            col.label(text=f"{len(ortho.front_points(cut))} points (orange, G pour déplacer)", icon='BLANK1')
            col.label(text="Stops postérieurs (jonction ptérygo-maxillaire) :")
            row = col.row(align=True)
            for side, name in ortho.STOPS:
                placed = ortho.stop_point(cut, side) is not None
                row.operator("omfs.ortho_stop", text=f"{'Replacer' if placed else 'Stop'} {name}",
                             icon='CHECKMARK' if placed else 'ADD').side = side


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
        draw_text(layout, context, "Le Fort : cliquez des points sur l'os où passe la coupe, de face, puis Entrée ; "
                                   "puis un clic de chaque côté à la jonction ptérygo-maxillaire (stops bleus) : la "
                                   "coupe s'y arrête et descend à 90°. BSSO : vue de côté, 2 points bleus sur la "
                                   "corticale vestibulaire (trait vertical, jusqu'au nerf), puis des points orange "
                                   "vers l'arrière jusqu'au cortex lingual.")
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
            row.enabled = source is not None and bool(ortho.cuts(scene, source)) and all(
                ortho.is_split(c) or ortho.stops_ready(c) for c in ortho.cuts(scene, source))
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
