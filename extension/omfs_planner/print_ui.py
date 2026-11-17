"""Panels of the Impression workspace: "Imprimer" in the OMFS tab (choose the parts), and the "Impression"
tab shown on the print scene (plate, active part, supports and base, resin, checks and export)."""

import json
import math

import bpy

from . import params, print_chain, print_profiles, print_scene
from .ui import draw_text

TAB = "Impression"


class OMFS_PT_print_pick(bpy.types.Panel):
    bl_label = "Imprimer"
    bl_idname = "OMFS_PT_print_pick"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return not context.scene.omfs_print.is_print_scene

    def draw(self, context):
        layout = self.layout
        found = print_scene.printable(context.scene)
        draw_text(layout, context, "Cochez les pièces à imprimer. Elles sont copiées sur un plateau virtuel "
                                   "(scène « Impression »): la planification n'est jamais modifiée.")
        col = layout.column(align=True)
        for obj in found:
            col.prop(obj.omfs_print, "pick", text=obj.name)
        if not found:
            layout.label(text="Aucune pièce imprimable dans la scène", icon='INFO')
        row = layout.row()
        row.scale_y = 1.4
        row.operator("omfs.print_prepare", icon='MOD_BUILD')
        if print_scene.print_scene() is not None:
            layout.operator("omfs.print_show", icon='RESTRICT_VIEW_OFF')
        layout.operator("omfs.print_test_piece", icon='MESH_CUBE')


class PrintPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = TAB

    @classmethod
    def poll(cls, context):
        return context.scene.omfs_print.is_print_scene


class OMFS_PT_print_plate(PrintPanel):
    bl_label = "Plateau"
    bl_idname = "OMFS_PT_print_plate"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.omfs_print
        layout.prop(props, "printer", text="")
        printer = print_scene.printer_of(scene)
        if printer is not None:
            layout.label(text=f"{printer['width']:g} x {printer['height']:g} x {printer['z']:g} mm · pixel "
                              f"{1000 * printer['width'] / printer['pixels_x']:.0f} µm", icon='MESH_GRID')
        active = print_scene.active_part(context)
        box = layout.box()
        for part in print_scene.parts(scene):
            row = box.row(align=True)
            icon = 'ERROR' if part.omfs_print.outside else print_scene.STATES.get(part.omfs_print.state, ('BLANK1', ""))[0]
            op = row.operator("omfs.print_select", text=part.name, icon=icon, depress=part == active, emboss=part == active)
            op.name = part.name
            if part.omfs_print.source:
                row.operator("omfs.print_update_part", text="", icon='FILE_REFRESH').name = part.name
            row.operator("omfs.print_remove", text="", icon='X').name = part.name
            if part.omfs_print.outside:
                box.label(text="Hors du volume d'impression", icon='BLANK1')
        if not print_scene.parts(scene):
            box.label(text="Plateau vide", icon='INFO')
        row = layout.row(align=True)
        row.operator("omfs.print_arrange", icon='MOD_ARRAY')
        row.operator("omfs.print_test_piece", text="Pièce test", icon='MESH_CUBE')
        layout.operator("omfs.print_back", icon='LOOP_BACK')


class OMFS_PT_print_part(PrintPanel):
    bl_label = "Pièce active: position et orientation"
    bl_idname = "OMFS_PT_print_part"

    def draw(self, context):
        layout = self.layout
        part = print_scene.active_part(context)
        if part is None:
            draw_text(layout, context, "Cliquez une pièce dans la vue ou dans la liste. G: déplacer sur le plateau. "
                                       "R: tourner. Les poignées de couleur font la même chose à la souris.")
            return
        layout.label(text=part.name, icon='OBJECT_DATA')
        layout.label(text="Appui vers le haut, incliné de:")
        row = layout.row(align=True)
        for angle in params.PRINT_TILTS:
            op = row.operator("omfs.print_tilt", text=f"{angle}°")
            op.angle, op.axis, op.flip = math.radians(angle), 'X', False
        row = layout.row(align=True)
        op = row.operator("omfs.print_tilt", text="Incliné de côté 30°")
        op.angle, op.axis, op.flip = math.radians(30), 'Y', False
        op = row.operator("omfs.print_tilt", text="Retourner")
        op.angle, op.axis, op.flip = math.radians(30), 'X', True
        col = layout.column(align=True)
        col.prop(part, "location", index=0, text="Position X")
        col.prop(part, "location", index=1, text="Y")
        col = layout.column(align=True)
        col.prop(part, "rotation_euler", index=0, text="Rotation X")
        col.prop(part, "rotation_euler", index=1, text="Y")
        col.prop(part, "rotation_euler", index=2, text="Z")
        draw_text(layout, context, "La hauteur se règle seule (supports ou plateau). Dans la vue: G puis X ou Y, "
                                   "R puis X, Y ou Z, ou les poignées.")


class OMFS_PT_print_supports(PrintPanel):
    bl_label = "Supports et base"
    bl_idname = "OMFS_PT_print_supports"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.omfs_print
        row = layout.row()
        row.prop(props, "supports")
        sub = row.row()
        sub.enabled = props.supports
        sub.prop(props, "live", text="Auto")
        col = layout.column(align=True)
        col.enabled = props.supports
        col.prop(props, "density")
        col.prop(props, "tip")
        col.prop(props, "lift")
        col.separator()
        col.prop(props, "base")
        if props.base == 'RAFT':
            col.prop(props, "raft_thickness")
            col.prop(props, "raft_margin")
            col.prop(props, "raft_lip")
        for part in print_scene.parts(scene):
            icon, text = print_scene.STATES.get(part.omfs_print.state, ('BLANK1', ""))
            if text:
                layout.label(text=f"{part.name}: {text}", icon=icon)
            if part.omfs_print.message:
                draw_text(layout, context, part.omfs_print.message)
        if props.supports and not props.live:
            layout.operator("omfs.print_refresh", icon='FILE_REFRESH')
        box = layout.box()
        box.label(text="Surface d'appui: sans support", icon='STRIP_COLOR_04')
        box.label(text="Supports (aperçu simplifié)", icon='STRIP_COLOR_05')
        box.label(text="Pièce hors du volume", icon='STRIP_COLOR_01')


def minutes(seconds):
    hours, rest = divmod(int(round(seconds / 60)), 60)
    return f"{hours} h {rest:02d}" if hours else f"{rest} min"


class OMFS_PT_print_resin(PrintPanel):
    bl_label = "Résine"
    bl_idname = "OMFS_PT_print_resin"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene.omfs_print, "resin", text="")
        resin = print_scene.resin_of(scene)
        if resin is None:
            layout.label(text="Aucun profil de résine", icon='ERROR')
            return
        e = resin["elegoo"]
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.85

        def line(label, value):
            split = col.split(factor=0.62)
            split.label(text=label)
            split.label(text=value)
        line("Hauteur de couche", f"{resin['layer_height']:g} mm")
        line("Exposition", f"{resin['exposure_time']:g} s")
        line("Couches de base", f"{resin['bottom_layers']} à {resin['bottom_exposure_time']:g} s")
        line("Couches de transition", f"{e.get('TransitionLayerCount', 0):g}")
        line("Soulèvement", f"{e.get('LiftHeight', 0):g} + {e.get('LiftHeight2', 0):g} mm")
        line("Montée (mm/min)", f"{e.get('LiftSpeed', 0):g} puis {e.get('LiftSpeed2', 0):g}")
        line("Descente (mm/min)", f"{e.get('RetractSpeed', 0):g} puis {e.get('RetractSpeed2', 0):g}")
        line("Délai lumière éteinte", f"{e.get('LightOffDelay', 0):g} s")
        line("Attente avant expo.", f"{e.get('WaitTimeBeforeCure', 0):g} s (base {e.get('BottomWaitTimeBeforeCure', 0):g} s)")
        if resin.get("source"):
            draw_text(box, context, resin["source"])
        layout.operator("omfs.print_edit_resin", text="Modifier / nouvelle résine", icon='GREASEPENCIL')


class OMFS_PT_print_export(PrintPanel):
    bl_label = "Vérifications et export"
    bl_idname = "OMFS_PT_print_export"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.omfs_print
        found = print_scene.estimates(scene)
        if found is not None:
            col = layout.column(align=True)
            col.label(text=f"{found['layers']} couches · {minutes(found['time'])} · {found['resin_ml']:.1f} ml", icon='TIME')
            col.label(text="Encombrement: %.0f x %.0f x %.0f mm" % tuple(found["size"]), icon='BLANK1')
        results = print_scene.checks(scene)
        col = layout.column(align=True)
        for ok, text in results:
            col.label(text=text, icon='CHECKMARK' if ok else 'CANCEL')
        layout.prop(props, "case_id")
        row = layout.row()
        row.scale_y = 1.5
        row.enabled = all(ok for ok, _text in results) and not props.busy
        row.operator("omfs.print_export", icon='EXPORT')
        if props.busy:
            layout.label(text=props.busy, icon='SORTTIME')
        last = json.loads(props.result) if props.result else None
        if last:
            box = layout.box()
            if last.get("problems"):
                box.label(text="Dernier export: fichier non gardé", icon='CANCEL')
                for problem in last["problems"]:
                    draw_text(box, context, problem)
            else:
                box.label(text="Dernier export: vérifié", icon='CHECKMARK')
                box.label(text=f"{last['layers']} couches · {minutes(last['print_time'])} · {last['resin_ml']:.1f} ml")
                draw_text(box, context, bpy.path.basename(last["path"]))


classes = (OMFS_PT_print_pick, OMFS_PT_print_plate, OMFS_PT_print_part, OMFS_PT_print_supports, OMFS_PT_print_resin,
           OMFS_PT_print_export)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
