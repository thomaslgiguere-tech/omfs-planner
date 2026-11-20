"""Le Fort I movement: the maxilla (distal segment) and the upper teeth moved as one block, with live numbers.

The maths are those of every segment (movement.py, preset LEFORT): six numbers on the reference frame
axes (advance / up / left, pitch / roll / yaw) about a pivot (upper incisor midline by default). What is
specific here:
- what moves with it (followers): the upper teeth (CBCT, crowns from the scan or not) and the scans
  registered on them; found automatically (teeth whose roots sit in the segment), editable;
- the points followed: ANS, point A, upper incisor midline, PNS (the premolar cusps only serve the plane);
- the occlusal plane (upper incisor midline + both premolar cusps): its angle to Frankfort horizontal
  in profile (sagittal) and from the front (cant), before and after the movement, drawn in the view.

The mandible is not moved here: it will follow the occlusion (maxillomandibular block, to come).
"""

from math import atan2, degrees

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, movement, params, reference_frame

OCCLUSAL = ('UI', 'CUSP_R', 'CUSP_L')


# --- Segment and followers ------------------------------------------------------------------

def segment(scene):
    return movement.segment_of(scene, 'LEFORT')


def root_share(teeth, bvh, to_segment, samples=2000):
    """Share of the teeth's vertices lying within LEFORT_TEETH_REACH of the segment (roots in the maxilla)."""
    co = np.empty(len(teeth.data.vertices) * 3)
    teeth.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)[:: max(1, len(teeth.data.vertices) // samples)]
    m = np.array(to_segment)
    co = co @ m[:3, :3].T + m[:3, 3]
    return float(np.mean([bvh.find_nearest(p, params.LEFORT_TEETH_REACH)[0] is not None for p in co]))


def default_followers(scene, seg):
    """The upper teeth (TEETH objects whose roots sit in the Le Fort segment) and the scans registered on them."""
    co = [v.co for v in seg.data.vertices]
    bvh = BVHTree.FromPolygons(co, [p.vertices for p in seg.data.polygons])
    to_segment = seg.matrix_world.inverted()
    teeth = [o for o in scene.objects if o.type == 'MESH' and o.omfs.role == 'TEETH' and o.parent is None
             and root_share(o, bvh, to_segment @ o.matrix_world) >= params.LEFORT_TEETH_SHARE]
    scans = [o for o in scene.objects if o.type == 'MESH' and o.omfs.role == 'SCAN' and o.parent is None
             and o.omfs_reg.target in teeth]
    return [o for o in teeth + scans if movement.follower_problem(scene, seg, o) is None]


def prepare(scene, seg):
    """Store the pre-op pose of the Le Fort segment and its followers (first use)."""
    movement.init_segment(scene, seg, kind='LEFORT', followers=default_followers(scene, seg))


# --- Occlusal plane -------------------------------------------------------------------------

def occlusal_points(scene):
    """Pre-op world points (upper incisor midline, right and left premolar cusps), or None if one is missing."""
    found = [landmarks.find(scene, k) for k in OCCLUSAL]
    if any(lm is None for lm in found):
        return None
    return [lm.matrix_world.translation.copy() for lm in found]


def plane_angles(frame, incisor, right, left):
    """(sagittal, cant) of the occlusal plane against Frankfort horizontal, degrees.
    Sagittal: + = the plane goes down towards the front. Cant: + = the patient's left side is higher."""
    _l, anterior, up = reference_frame.to_clinical(frame, incisor - (right + left) / 2)
    side_left, _a, side_up = reference_frame.to_clinical(frame, left - right)
    return degrees(atan2(-up, anterior)), degrees(atan2(side_up, side_left))


def occlusal_report(scene, seg):
    """((sagittal, cant) before, (sagittal, cant) now) or None."""
    frame, points = reference_frame.find_frame(scene), occlusal_points(scene)
    if frame is None or points is None:
        return None
    moved = [movement.world_movement(scene, seg) @ p for p in points] if seg.omfs_move.has_preop else points
    return plane_angles(frame, *points), plane_angles(frame, *moved)


# --- Drawing --------------------------------------------------------------------------------

def _draw_plane():
    """The occlusal plane where the movement takes it: a translucent triangle and its outline."""
    context = bpy.context
    scene = context.scene
    seg = segment(scene)
    if seg is None or reference_frame.find_frame(scene) is None:
        return
    if not scene.omfs_lefort.show_plane or not seg.omfs_move.show_labels:   # "Étiquettes dans la vue" hides it too
        return
    points = occlusal_points(scene)
    if points is None:
        return
    if seg.omfs_move.has_preop:
        w = movement.world_movement(scene, seg)
        points = [w @ p for p in points]
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    fill = gpu.shader.from_builtin('UNIFORM_COLOR')
    fill.uniform_float("color", (*params.OCCLUSAL_PLANE_COLOUR[:3], 0.18))
    batch_for_shader(fill, 'TRIS', {"pos": points}).draw(fill)
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    line.uniform_float("viewportSize", (context.region.width, context.region.height))
    line.uniform_float("lineWidth", 2.0)
    line.uniform_float("color", params.OCCLUSAL_PLANE_COLOUR)
    batch_for_shader(line, 'LINES', {"pos": [points[0], points[1], points[1], points[2], points[2], points[0]]}).draw(line)
    gpu.state.blend_set('NONE')


# --- Operators and properties ---------------------------------------------------------------

class OMFS_OT_lefort_prepare(bpy.types.Operator):
    """Prépare le mouvement du Le Fort : position préop gardée, les dents du haut (et leurs scans) bougent avec
    lui. Les nombres sont sur le référentiel (Frankfort, médiosagittal, coronal)"""
    bl_idname = "omfs.lefort_prepare"
    bl_label = "Préparer le mouvement du Le Fort"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        seg = segment(scene)
        if seg is None:
            self.report({'ERROR'}, "Séparez d'abord le Le Fort I (Ostéotomies orthognathiques)")
            return {'CANCELLED'}
        if reference_frame.find_frame(scene) is None:
            self.report({'ERROR'}, "Construisez d'abord le référentiel")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        prepare(scene, seg)
        scene.omfs_movement.segment = seg
        names = ", ".join(f.obj.name for f in seg.omfs_move.followers) or "aucun objet"
        self.report({'INFO'}, f"{seg.name} prêt ; bougent avec lui : {names}")
        return {'FINISHED'}


class OMFS_LeFortSceneProperties(bpy.types.PropertyGroup):
    show_plane: bpy.props.BoolProperty(name="Plan occlusal dans la vue", default=True,
                                       description="Dessiner le plan occlusal (milieu incisif + cuspides prémolaires) "
                                                   "là où le mouvement l'amène")
    mmf_from_occlusion: bpy.props.BoolProperty(
        name="Générer le bloc maxillomandibulaire à partir de l'occlusion",
        description="À venir : la mandibule (segment distal et dents du bas) suit le maxillaire dans l'occlusion "
                    "planifiée")


classes = (OMFS_OT_lefort_prepare, OMFS_LeFortSceneProperties)
_handles = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_lefort = bpy.props.PointerProperty(type=OMFS_LeFortSceneProperties)
    if not bpy.app.background:
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_plane, (), 'WINDOW', 'POST_VIEW'))


def unregister():
    for handle in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _handles.clear()
    del bpy.types.Scene.omfs_lefort
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
