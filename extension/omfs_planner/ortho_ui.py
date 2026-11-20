"""Panels (OMFS tab): "Ostéotomies orthognathiques" (Le Fort I and BSSO cuts, then the separation of the segments)
and "Mouvement du Le Fort" (the maxilla and the upper teeth moved as one block, live numbers)."""

import bpy

from . import landmarks, movement, ortho_movement, reference_frame
from . import ortho_osteotomy as ortho
from .ui import draw_segment_header, draw_text


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


def signed(value):
    return f"{value:+.1f}"


class OMFS_PT_lefort_movement(bpy.types.Panel):
    bl_label = "Mouvement du Le Fort"
    bl_idname = "OMFS_PT_lefort_movement"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return OMFS_PT_ortho_osteotomy.poll(context)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if reference_frame.find_frame(scene) is None:
            layout.label(text="Construisez d'abord le référentiel", icon='ERROR')
            return
        seg = ortho_movement.segment(scene)
        if seg is None:
            draw_text(layout, context, "Séparez d'abord le Le Fort I (Ostéotomies orthognathiques).")
            return
        props = seg.omfs_move
        if not props.has_preop:
            draw_text(layout, context, "Le Le Fort et les dents du haut bougeront ensemble, sur le référentiel "
                                       "(Frankfort, médiosagittal, coronal).")
            layout.operator("omfs.lefort_prepare", icon='PLAY')
            return
        draw_segment_header(layout, scene, seg)

        box = layout.box()   # what moves as one block
        box.label(text="Bougent ensemble", icon='LINKED')
        col = box.column(align=True)
        col.label(text=seg.name, icon='MESH_DATA')
        for i, item in enumerate(props.followers):
            row = col.row(align=True)
            row.label(text=item.obj.name if item.obj else "(supprimé)", icon='MESH_DATA')
            op = row.operator("omfs.remove_follower", text="", icon='X')
            op.segment_name, op.index = seg.name, i
        box.operator("omfs.add_follower", icon='ADD').segment_name = seg.name

        col = layout.column(align=True)
        col.label(text="Translation du pivot (mm), référentiel")
        col.prop(props, "anterior")
        col.prop(props, "up")
        col.prop(props, "left")
        col = layout.column(align=True)
        col.label(text="Rotation autour du pivot")
        col.prop(props, "pitch", text="Pitch (+ avant vers le haut)")
        col.prop(props, "roll", text="Roll (+ gauche vers le haut)")
        col.prop(props, "yaw", text="Yaw (+ avant vers la gauche)")
        layout.prop(props, "pivot_kind")

        rows = movement.tracked_rows(scene, seg)
        box = layout.box()
        box.label(text="Déplacement des points (mm)", icon='EMPTY_SINGLE_ARROW')
        if rows:
            grid = box.grid_flow(columns=5, even_columns=False, align=True)
            for text in ("", "Total", "Avance", "Haut", "Gauche"):
                grid.label(text=text)
            for name, _point, (left, anterior, up) in rows:
                grid.label(text=name)
                grid.label(text=f"{(left * left + anterior * anterior + up * up) ** 0.5:.1f}")
                for value in (anterior, up, left):
                    grid.label(text=signed(value))
        missing = [k for k in movement.preset(seg)['tracked'] if landmarks.find(scene, k) is None]
        if missing:
            draw_text(box, context, "À placer (Landmarks) : " + ", ".join(landmarks.BY_ID[k][1] for k in missing))

        box = layout.box()
        row = box.row(align=True)
        row.label(text="Plan occlusal / Frankfort", icon='MESH_PLANE')
        row.prop(scene.omfs_lefort, "show_plane", text="", icon='HIDE_OFF' if scene.omfs_lefort.show_plane else 'HIDE_ON')
        report = ortho_movement.occlusal_report(scene, seg)
        if report is None:
            draw_text(box, context, "Placez le milieu incisif du haut et les deux cuspides prémolaires (Landmarks).")
        else:
            (sag0, cant0), (sag1, cant1) = report
            col = box.column(align=True)
            col.label(text=f"Sagittal : {sag0:.1f}° → {sag1:.1f}°  ({sag1 - sag0:+.1f}°)")
            col.label(text=f"Cant : {cant0:.1f}° → {cant1:.1f}°  ({cant1 - cant0:+.1f}°)")
            col.label(text="Sagittal + : descend vers l'avant · cant + : gauche plus haut", icon='INFO')
        layout.prop(props, "show_labels", text="Étiquettes dans la vue")

        box = layout.box()   # placeholder, to be designed with the surgeon
        row = box.row()
        row.enabled = False
        row.prop(scene.omfs_lefort, "mmf_from_occlusion")
        box.label(text="À venir : la mandibule suivra l'occlusion", icon='INFO')


classes = (OMFS_PT_ortho_osteotomy, OMFS_PT_lefort_movement)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
