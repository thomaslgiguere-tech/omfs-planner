"""Segmental seating, step 2: each segment of the upper set-up seated on the fixed lower arch (surgeon, 2026-11-19).

The posterior segments first: the surgeon brings each one close with G / R, then "Asseoir": the seating solver of
the whole arch (occlusion_seating: magnet between the arches, overlap limit), with the segment as its own block.
- The angle the surgeon gave the segment is kept by a rotational spring ("Tenue de l'angulation", per segment):
  100 % = translations only, the angle exactly as given; less = some rotation allowed when it improves the
  intercuspation a lot (the degrees turned are shown); 0 % = free. 70 % by default, 100 % for the anterior
  segment (the incisors' angulation is the surgeon's).
- "Selon la courbe": each cut split a contact point in two copies, one on each neighbouring segment; springs
  pull this segment's copies onto its neighbours' (where they are now), next to the magnet, so the contact
  curve stays continuous (no fragment much higher or lower than the rest) while the teeth stay in occlusion.
  Usable on any segment, meant for the last one (often the anterior), before or after G / R.
Overlap limit: the anterior one for the anterior segment, the posterior one for the others. Only the segment's
transform changes; its pre-op pose (where the arch was) is kept.
"""

from math import degrees
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import occlusion_seating as seating
from . import occlusion_segments, params

TRANSLATIONS = np.vstack((np.zeros((3, 3)), np.eye(3)))   # 6 x 3: the motions without rotation


class ContactCopy:
    """A contact point's copy on a segment, where the segment is now (a spring end for the seating solver)."""

    def __init__(self, obj, co):
        self.obj, self.co = obj, Vector(co)
        self.omfs = SimpleNamespace(landmark="")

    @property
    def matrix_world(self):
        return Matrix.Translation(self.obj.matrix_world @ self.co)


def hold(segment):
    """(rotational spring or None, motion basis or None) for the segment's angulation hold."""
    h = segment.omfs_setup.hold / 100.0
    if h >= 0.999:
        return None, TRANSLATIONS
    if h <= 0.0:
        return None, None
    stiffness = params.SEG_HOLD_STIFFNESS * h / (1.0 - h)
    return (stiffness, np.array(segment.matrix_world.to_3x3().normalized())), None


def curve_springs(scene, segment):
    """[(neighbour's copy, this segment's copy)] of the contact points this segment shares."""
    return [(ContactCopy(c.other, c.other_co), ContactCopy(segment, c.co)) for c in segment.omfs_setup.contacts
            if c.other is not None and c.other.name in scene.objects]


def deepest_overlap(job):
    contacts = seating.measure(job, seating.test_points(job, seating.contact_vertices(job)))
    d = np.concatenate((contacts.d1, contacts.d2))
    d = d[~np.isnan(d)]
    return float(max(0.0, -d.min())) if len(d) else 0.0


def seat_segment(scene, segment, curve=False):
    """Seat one segment on the lower arch. Returns (iterations, settled, degrees turned, deepest overlap mm)."""
    props = scene.omfs_occlusion
    reference = np.array(segment.matrix_world.to_3x3().normalized())
    spring, free = hold(segment)
    pairs = curve_springs(scene, segment) if curve else []
    job = seating.Seating(props.lower, segment, segment, pairs, None, None, spring)
    limit = props.tolerance_anterior if segment.omfs_setup.anterior else props.tolerance_posterior
    args = (scene, job, (limit, limit), params.SEG_CURVE_SHARE if pairs else 0.0, 1.0)
    first, _ok = seating.solve(*args, seating.test_points(job), free, params.OCC_MAX_ITERATIONS)
    last, ok = seating.solve(*args, seating.test_points(job, seating.contact_vertices(job)), free,
                             params.OCC_FINISH_ITERATIONS)
    turned = degrees(np.linalg.norm(seating.rotation_error(segment.matrix_world.to_3x3().normalized(), reference)))
    segment.omfs_setup.last_turn = turned
    return first + last, ok, turned, deepest_overlap(job)


def segment_named(context, name):
    obj = context.scene.objects.get(name)
    return obj if obj is not None and obj.omfs.made_by == occlusion_segments.SEGMENT_MADE_BY else None


class OMFS_OT_split_segments(bpy.types.Operator):
    """Séparer la copie de l'arcade du haut en segments le long des traits choisis et du trait médian, sans
    épaisseur (les triangles traversés sont coupés en deux, rien n'est retiré). Un objet par segment"""
    bl_idname = "omfs.split_segments"
    bl_label = "Séparer les segments"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        upper = scene.omfs_occlusion.upper
        copy = occlusion_segments.setup_copy(scene, upper) if upper is not None else None
        if copy is None or not occlusion_segments.chosen(copy):
            self.report({'ERROR'}, "Choisissez d'abord au moins un trait (Segmentaire : choisir les traits)")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        objects = occlusion_segments.split(scene, copy)
        self.report({'INFO'}, f"{len(objects)} segments : " + ", ".join(o.name for o in objects))
        return {'FINISHED'}


class OMFS_OT_unsplit_segments(bpy.types.Operator):
    """Supprimer les segments et revenir à la copie entière (les traits choisis restent)"""
    bl_idname = "omfs.unsplit_segments"
    bl_label = "Recoller (refaire les traits)"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scene = context.scene
        copy = occlusion_segments.setup_copy(scene, scene.omfs_occlusion.upper)
        if copy is None:
            return {'CANCELLED'}
        occlusion_segments.unsplit(scene, copy)
        return {'FINISHED'}


class OMFS_OT_seat_segment(bpy.types.Operator):
    """Asseoir ce segment sur l'arcade du bas : aimant entre les arcades, limite de chevauchement, et la tenue de
    son angulation (ressort de rotation vers l'angle que vous lui avez donné). Selon la courbe : en plus, ses
    points de contact sont tirés vers ceux de ses voisins (continuité de la courbe des contacts)"""
    bl_idname = "omfs.seat_segment"
    bl_label = "Asseoir"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()
    curve: bpy.props.BoolProperty(name="Selon la courbe")

    def execute(self, context):
        scene = context.scene
        segment = segment_named(context, self.segment_name)
        if segment is None or scene.omfs_occlusion.lower is None:
            self.report({'ERROR'}, "Il faut le segment et l'arcade du bas")
            return {'CANCELLED'}
        if self.curve and not segment.omfs_setup.contacts:
            self.report({'ERROR'}, "Ce segment n'a pas de voisin (aucun point de contact partagé)")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        iterations, ok, turned, overlap = seat_segment(scene, segment, self.curve)
        how = "selon la courbe, " if self.curve else ""
        self.report({'INFO'} if ok else {'WARNING'},
                    f"{segment.name} assis ({how}{iterations} itérations) : rotation {turned:.1f}°, "
                    f"chevauchement max {overlap:.2f} mm" + ("" if ok else " ; pas tout à fait stable"))
        return {'FINISHED'}


class OMFS_OT_segment_back(bpy.types.Operator):
    """Remettre ce segment où il était dans l'arcade (avant G / R et l'assise)"""
    bl_idname = "omfs.segment_back"
    bl_label = "Position de départ"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()

    def execute(self, context):
        segment = segment_named(context, self.segment_name)
        if segment is None:
            return {'CANCELLED'}
        segment.matrix_world = Matrix(segment.omfs_setup.preop)
        segment.omfs_setup.last_turn = -1.0
        return {'FINISHED'}


class OMFS_OT_select_segment(bpy.types.Operator):
    """Sélectionner ce segment (puis G / R dans la vue pour l'approcher à la main)"""
    bl_idname = "omfs.select_segment"
    bl_label = "Sélectionner"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()

    def execute(self, context):
        segment = segment_named(context, self.segment_name)
        if segment is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        segment.hide_set(False)
        segment.select_set(True)
        context.view_layer.objects.active = segment
        return {'FINISHED'}


classes = (OMFS_OT_split_segments, OMFS_OT_unsplit_segments, OMFS_OT_seat_segment, OMFS_OT_segment_back,
           OMFS_OT_select_segment)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
