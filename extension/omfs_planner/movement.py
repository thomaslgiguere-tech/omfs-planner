"""Segment movement (Step 5): 6 degrees of freedom in the reference frame, with live numbers.

Generic: any SEGMENT object (the genioplasty chin today, Le Fort I / BSSO segments later).
Genioplasty specifics are presets: the tracked points (pogonion, menton) and the default pivot.

The movement is stored as six numbers on the segment (obj.omfs_move), in clinical terms:
translation left / anterior / up (mm) and rotation pitch / roll / yaw (degrees), about a pivot.
The segment's matrix is always computed from them:

    segment.matrix_world = F @ L @ F^-1 @ preop
    L = Translation(t + p) @ R @ Translation(-p)      (frame-local, p = pivot before the move)

F is the reference frame, so the axes are the clinical ones and the translation is exactly the
net displacement of the pivot. The pre-op pose is stored when the segment is created: the
mesh data is never touched and "Back to pre-op" is always exact.

Dragging (movement_gizmos.py) and typing in the panel write the same six numbers, so they
are always in sync. The segment's own transform is locked, so G / R can't bypass them.
"""

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Vector

from . import landmarks, reference_frame

# Genioplasty preset: the points whose net movement is shown, and the pivots offered.
TRACKED = ('POG', 'ME')
PIVOTS = (
    ('POG', "Pogonion", "Rotate about the pogonion (before the move)"),
    ('ME', "Menton", "Rotate about the menton (before the move)"),
    ('CENTRE', "Segment centre", "Rotate about the centre of the segment's bounding box"),
)
VALUES = ('left', 'anterior', 'up', 'pitch', 'roll', 'yaw')

LABEL_BACKGROUND = (0.0, 0.0, 0.0, 0.65)
LABEL_TEXT = (1.0, 1.0, 1.0, 1.0)
TRACK_COLOUR = (1.0, 0.2, 0.6, 1.0)   # landmark magenta

_syncing = False   # True while several values are written at once: apply only at the end


# --- Maths ---------------------------------------------------------------------------

def frame_matrix(scene):
    frame = reference_frame.find_frame(scene)
    return frame.matrix_world.normalized() if frame else None


def rotation(props):
    """Frame-local rotation. + pitch = anterior up (about -X), + roll = left side up (about -Y),
    + yaw = anterior towards the left (about +Z): same signs as the osteotomy angles."""
    return Euler((-props.pitch, -props.roll, props.yaw), 'XYZ').to_matrix().to_4x4()


def translation(props):
    """Frame-local translation (frame Y is posterior)."""
    return Vector((props.left, -props.anterior, props.up))


def local_movement(props):
    p = Vector(props.pivot)
    return Matrix.Translation(translation(props) + p) @ rotation(props) @ Matrix.Translation(-p)


def world_movement(scene, props):
    f = frame_matrix(scene)
    return f @ local_movement(props) @ f.inverted()


def to_clinical(local):
    """Frame-local displacement -> (left, anterior, up)."""
    return local.x, -local.y, local.z


def displacement(scene, obj, point):
    """Net translation (left, anterior, up) of a pre-op world point carried by the segment."""
    f = frame_matrix(scene)
    p = f.inverted() @ Vector(point)
    return to_clinical(local_movement(obj.omfs_move) @ p - p)


# --- Segment -------------------------------------------------------------------------

def is_segment(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'SEGMENT'


def active_segment(scene):
    chosen = scene.omfs_movement.segment
    if is_segment(chosen):
        return chosen
    return next((o for o in scene.objects if is_segment(o)), None)


def preop_centre(obj):
    corners = [Vector(c) for c in obj.bound_box]
    return obj.omfs_move.preop @ (sum(corners, Vector()) / 8)


def pivot_world(scene, obj, kind):
    """Pre-op world position of a pivot choice (None if its landmark isn't placed)."""
    if kind == 'CENTRE':
        return preop_centre(obj)
    lm = landmarks.find(scene, kind)
    return lm.matrix_world.translation.copy() if lm else None


def init_segment(scene, obj, previous=None):
    """Store the pre-op pose of a new segment, lock its transform, carry over a previous movement."""
    global _syncing
    props = obj.omfs_move
    props.preop = obj.matrix_world.copy()
    props.has_preop = True
    obj.lock_location = obj.lock_rotation = obj.lock_scale = (True,) * 3
    _syncing = True
    try:
        if previous is not None:
            for name in (*VALUES, 'pivot_kind', 'pivot', 'locked', 'show_labels'):
                setattr(props, name, previous[name])
        else:
            kind = next((k for k in ('POG', 'ME') if landmarks.find(scene, k)), 'CENTRE')
            props.pivot_kind = kind
            f = frame_matrix(scene)
            if f is not None:
                props.pivot = f.inverted() @ pivot_world(scene, obj, kind)
    finally:
        _syncing = False
    apply(scene, obj)


def remember(obj):
    """The movement of a segment about to be replaced (re-applied cut), for init_segment."""
    props = obj.omfs_move
    if not props.has_preop:
        return None
    return {name: (tuple(getattr(props, name)) if name == 'pivot' else getattr(props, name))
            for name in (*VALUES, 'pivot_kind', 'pivot', 'locked', 'show_labels')}


def apply(scene, obj):
    """Place the segment from its six numbers."""
    props = obj.omfs_move
    if _syncing or not props.has_preop or frame_matrix(scene) is None:
        return
    obj.matrix_world = world_movement(scene, props) @ props.preop


def set_pivot(scene, obj, kind):
    """Rotate about another point from now on. The segment doesn't move: the translation is
    rewritten as the displacement of the new pivot."""
    global _syncing
    props = obj.omfs_move
    f = frame_matrix(scene)
    world = pivot_world(scene, obj, kind)
    if f is None or world is None:
        return False
    movement = local_movement(props)
    p = f.inverted() @ world
    left, anterior, up = to_clinical(movement @ p - p)
    _syncing = True
    try:
        props.pivot = p
        props.left, props.anterior, props.up = left, anterior, up
    finally:
        _syncing = False
    apply(scene, obj)
    return True


def back_to_preop(scene, obj):
    global _syncing
    _syncing = True
    try:
        for name in VALUES:
            setattr(obj.omfs_move, name, 0.0)
    finally:
        _syncing = False
    apply(scene, obj)


def is_moved(obj):
    return any(abs(getattr(obj.omfs_move, name)) > 1e-6 for name in VALUES)


def tracked_rows(scene, obj):
    """[(landmark name, pre-op world point, (left, anterior, up))] for the tracked points placed."""
    rows = []
    for lm_id in TRACKED:
        lm = landmarks.find(scene, lm_id)
        if lm is not None:
            point = lm.matrix_world.translation.copy()
            rows.append((landmarks.BY_ID[lm_id][1], point, displacement(scene, obj, point)))
    return rows


def describe(left, anterior, up):
    """Clinical words for a displacement, e.g. ["6.0 mm avance", "2.0 mm bas", "0.3 mm gauche"]."""
    def word(value, plus, minus):
        if abs(value) < 0.05:
            return f"0.0 mm {plus}/{minus}"
        return f"{abs(value):.1f} mm {plus if value > 0 else minus}"
    return [word(anterior, "avance", "recul"), word(up, "haut", "bas"), word(left, "gauche", "droite")]


# --- Properties and operators ---------------------------------------------------------

def _on_value(props, context):
    apply(context.scene, props.id_data)


def _on_pivot_kind(props, context):
    if not _syncing:
        set_pivot(context.scene, props.id_data, props.pivot_kind)


class OMFS_MoveProperties(bpy.types.PropertyGroup):
    """Stored on each segment: its pre-op pose and its movement in the reference frame."""
    left: bpy.props.FloatProperty(
        name="Gauche (+) / droite", subtype='DISTANCE', step=10, update=_on_value,
        description="Translation of the pivot to the patient's left (+) or right (-)")
    anterior: bpy.props.FloatProperty(
        name="Avance (+) / recul", subtype='DISTANCE', step=10, update=_on_value,
        description="Translation of the pivot forward (+) or back (-)")
    up: bpy.props.FloatProperty(
        name="Haut (+) / bas", subtype='DISTANCE', step=10, update=_on_value,
        description="Translation of the pivot up (+) or down (-)")
    pitch: bpy.props.FloatProperty(
        name="Pitch", subtype='ANGLE', step=10, update=_on_value,
        description="Rotation about the left-right axis; + = the front of the segment goes up")
    roll: bpy.props.FloatProperty(
        name="Roll", subtype='ANGLE', step=10, update=_on_value,
        description="Rotation about the front-back axis; + = the patient's left side goes up")
    yaw: bpy.props.FloatProperty(
        name="Yaw", subtype='ANGLE', step=10, update=_on_value,
        description="Rotation about the vertical axis; + = the front of the segment turns to the patient's left")
    pivot_kind: bpy.props.EnumProperty(name="Pivot", items=PIVOTS, update=_on_pivot_kind,
                                       description="Point the rotations turn about (the segment doesn't move when it changes)")
    pivot: bpy.props.FloatVectorProperty(size=3, options={'HIDDEN'})   # frame-local, before the move
    preop: bpy.props.FloatVectorProperty(size=(4, 4), subtype='MATRIX', options={'HIDDEN'})
    has_preop: bpy.props.BoolProperty(options={'HIDDEN'})
    locked: bpy.props.BoolProperty(name="Lock segment",
                                   description="Hide the drag handles: the segment moves only by typed values")
    show_labels: bpy.props.BoolProperty(name="Labels in the view", default=True,
                                        description="Show the net movement of the tracked points next to them")


class OMFS_MovementSceneProperties(bpy.types.PropertyGroup):
    segment: bpy.props.PointerProperty(
        name="Segment", type=bpy.types.Object, poll=lambda _self, obj: is_segment(obj),
        description="Segment to move (the first segment, if empty)")


class OMFS_OT_segment_preop(bpy.types.Operator):
    """Put the segment back at its pre-op position (all movements to 0)"""
    bl_idname = "omfs.segment_preop"
    bl_label = "Retour préop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = active_segment(context.scene)
        if obj is None:
            return {'CANCELLED'}
        back_to_preop(context.scene, obj)
        return {'FINISHED'}


# --- Drawing: tracked points and their labels ----------------------------------------------

def _shown(context):
    scene = context.scene
    obj = active_segment(scene)
    if obj is None or not obj.omfs_move.has_preop or not obj.omfs_move.show_labels:
        return None
    if not obj.visible_get() or frame_matrix(scene) is None:
        return None
    return obj


def _draw_tracks():
    """A line from each tracked point before the move to where the segment takes it."""
    context = bpy.context
    obj = _shown(context)
    if obj is None or not is_moved(obj):
        return
    movement = world_movement(context.scene, obj.omfs_move)
    points = [(p, movement @ p) for _name, p, _d in tracked_rows(context.scene, obj)]
    if not points:
        return
    gpu.state.depth_test_set('NONE')
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    line.uniform_float("viewportSize", (context.region.width, context.region.height))
    line.uniform_float("lineWidth", 2.0)
    line.uniform_float("color", TRACK_COLOUR)
    batch_for_shader(line, 'LINES', {"pos": [p for pair in points for p in pair]}).draw(line)
    dots = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    gpu.state.point_size_set(9.0)
    dots.uniform_float("color", TRACK_COLOUR)
    batch_for_shader(dots, 'POINTS', {"pos": [q for _p, q in points]}).draw(dots)
    gpu.state.point_size_set(1.0)


def _draw_labels():
    """A small box next to each tracked point (moved): its net translation in clinical words."""
    context = bpy.context
    obj = _shown(context)
    if obj is None:
        return
    region, rv3d = context.region, context.region_data
    movement = world_movement(context.scene, obj.omfs_move)
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, 13 * scale)
    line_h, pad = 16 * scale, 6 * scale
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    for i, (name, point, d) in enumerate(tracked_rows(context.scene, obj)):
        xy = location_3d_to_region_2d(region, rv3d, movement @ point)
        if xy is None:
            continue
        total = Vector(d).length
        lines = [f"{name}   {total:.1f} mm", *describe(*d)]
        width = max(blf.dimensions(font, text)[0] for text in lines) + 2 * pad
        height = len(lines) * line_h + pad
        # First point's box above-right, the next below-right: pogonion and menton are close.
        x = xy.x + 14 * scale
        y = xy.y + 10 * scale if i % 2 == 0 else xy.y - 10 * scale - height
        corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        shader.uniform_float("color", LABEL_BACKGROUND)
        batch_for_shader(shader, 'TRI_FAN', {"pos": corners}).draw(shader)
        for row, text in enumerate(lines):
            blf.color(font, *(TRACK_COLOUR if row == 0 else LABEL_TEXT))
            blf.position(font, x + pad, y + height - (row + 1) * line_h + 2 * scale, 0)
            blf.draw(font, text)
    gpu.state.blend_set('NONE')


_draw_handles = []

classes = (OMFS_MoveProperties, OMFS_MovementSceneProperties, OMFS_OT_segment_preop)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_move = bpy.props.PointerProperty(type=OMFS_MoveProperties)
    bpy.types.Scene.omfs_movement = bpy.props.PointerProperty(type=OMFS_MovementSceneProperties)
    if not bpy.app.background:
        _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_tracks, (), 'WINDOW', 'POST_VIEW'))
        _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_labels, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    for handle in _draw_handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _draw_handles.clear()
    del bpy.types.Scene.omfs_movement
    del bpy.types.Object.omfs_move
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
