"""Segment movement (Step 5): 6 degrees of freedom on clinical axes, with live numbers.

Generic: any SEGMENT object (the genioplasty chin today, Le Fort I / BSSO segments later).
Genioplasty specifics are presets: the tracked points (pogonion, menton) and the pivots offered.

The movement is stored as six numbers on the segment (obj.omfs_move): translation left /
anterior / up (mm) and rotation pitch / roll / yaw (degrees), about a pivot, on one of two sets
of axes:
- Osteotomy plane (default for a cut segment): "up" is the cut's normal, "anterior" runs along
  the cut in the facial midsagittal plane (forward as in the facial frame), "left" completes them. Advancing slides
  the segment along the cut, as in surgery; "up" opens (+) or closes the cut. The midline is
  the plane perpendicular to the cut through the pivot (pogonion or menton).
- Reference frame: Frankfort / midsagittal / coronal.

The segment's matrix is always computed from the numbers (A = axes rotation, P = pivot before
the move, both in world space):

    segment.matrix_world = W @ preop,  W = Translation(A t + P) @ (A R A^-1) @ Translation(-P)

so the translation is exactly the net displacement of the pivot. The pre-op pose is stored when
the segment is created: the mesh data is never touched and "Back to pre-op" is always exact.
Changing the pivot or the axes rewrites the numbers for the same pose, so the segment never jumps.

Dragging (movement_gizmos.py) and typing in the panel write the same six numbers, so they
are always in sync. The segment's own transform is locked, so G / R can't bypass them.

"Apply the same movement to the cutting guide" moves a guide by the same W, rigidly, with the
tools of its Boolean modifiers (sleeve cylinders, text, cut-edge box) so its holes go with it.
Each keeps its own pre-op pose (custom property), so the guide always goes back exactly.
"""

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Vector

from . import landmarks, locking, reference_frame

# Genioplasty preset: the points whose net movement is shown, and the pivots offered.
TRACKED = ('POG', 'ME')
PIVOTS = (
    ('POG', "Pogonion", "Rotate about the pogonion (before the move); the midline goes through it"),
    ('ME', "Menton", "Rotate about the menton (before the move); the midline goes through it"),
    ('CENTRE', "Segment centre", "Rotate about the centre of the segment's bounding box"),
)
AXES = (
    ('CUT', "Osteotomy plane", "Advance slides along the cut (in the midsagittal plane), "
                               "up is perpendicular to the cut"),
    ('FRAME', "Reference frame", "Frankfort horizontal, midsagittal and coronal planes"),
)
VALUES = ('left', 'anterior', 'up', 'pitch', 'roll', 'yaw')
CARRIED = (*VALUES, 'axes', 'pivot_kind', 'pivot', 'locked', 'show_labels')

LABEL_BACKGROUND = (0.0, 0.0, 0.0, 0.65)
LABEL_TEXT = (1.0, 1.0, 1.0, 1.0)
TRACK_COLOUR = (1.0, 0.2, 0.6, 1.0)   # landmark magenta

_syncing = False   # True while several values are written at once: apply only at the end


# --- Axes ----------------------------------------------------------------------------

def frame_axes(scene):
    """Rotation of the reference frame (columns: left, posterior, up), or None."""
    frame = reference_frame.find_frame(scene)
    return frame.matrix_world.to_3x3().normalized() if frame else None


def cut_of(scene, obj):
    """The osteotomy plane this segment was cut with, if it still exists."""
    cut = scene.objects.get(obj.omfs.made_by)
    return cut if cut is not None and cut.omfs.role == 'CUT' else None


def cut_axes(scene, obj):
    """Columns: left and posterior on the cut, up = the cut's normal. None without a cut."""
    from . import osteotomy   # osteotomy imports this module
    frame, cut = frame_axes(scene), cut_of(scene, obj)
    if frame is None or cut is None:
        return None
    up = (cut.matrix_world.to_3x3().normalized() @ osteotomy.plane_normal(cut.omfs_cut)).normalized()
    # Anterior runs where the cut meets the midsagittal plane: advancing never shifts sideways.
    posterior = up.cross(frame.col[0]).normalized()
    left = posterior.cross(up)
    return Matrix((left, posterior, up)).transposed()


def axes(scene, obj):
    """Rotation whose columns are the movement axes (left, posterior, up), or None."""
    if obj.omfs_move.axes == 'CUT':
        return cut_axes(scene, obj) or frame_axes(scene)
    return frame_axes(scene)


def axes_label(scene, obj):
    if obj.omfs_move.axes == 'CUT' and cut_axes(scene, obj) is None:
        return "Reference frame (no osteotomy plane)"
    return dict((k, n) for k, n, _d in AXES)[obj.omfs_move.axes]


# --- Maths ---------------------------------------------------------------------------

def rotation(props):
    """Rotation on the movement axes. + pitch = anterior up (about -X), + roll = left side up
    (about -Y), + yaw = anterior towards the left (about +Z): same signs as the osteotomy angles."""
    return Euler((-props.pitch, -props.roll, props.yaw), 'XYZ').to_matrix()


def translation(props):
    """Translation on the movement axes (Y is posterior)."""
    return Vector((props.left, -props.anterior, props.up))


def world_movement(scene, obj):
    """World matrix taking the pre-op segment to its planned position."""
    props = obj.omfs_move
    a = axes(scene, obj)
    p = Vector(props.pivot)
    turn = (a @ rotation(props) @ a.transposed()).to_4x4()
    return Matrix.Translation(a @ translation(props) + p) @ turn @ Matrix.Translation(-p)


def current_movement(obj):
    """The movement the segment shows now (whatever the numbers or axes say)."""
    return obj.matrix_world @ obj.omfs_move.preop.inverted()


def to_clinical(local):
    """Displacement on the axes (left, posterior, up) -> (left, anterior, up)."""
    return local.x, -local.y, local.z


def displacement(scene, obj, point, on_axes=None):
    """Net translation (left, anterior, up) of a pre-op world point carried by the segment,
    on the movement axes (or on the axes given)."""
    a = on_axes if on_axes is not None else axes(scene, obj)
    point = Vector(point)
    return to_clinical(a.transposed() @ (world_movement(scene, obj) @ point - point))


def set_movement(scene, obj, movement):
    """Write the six numbers that give this world movement, on the current axes and pivot."""
    global _syncing
    props = obj.omfs_move
    a = axes(scene, obj)
    p = Vector(props.pivot)
    turn = (a.transposed() @ movement.to_3x3().normalized() @ a).to_euler('XYZ')
    _syncing = True
    try:
        props.left, props.anterior, props.up = to_clinical(a.transposed() @ (movement @ p - p))
        props.pitch, props.roll, props.yaw = -turn.x, -turn.y, turn.z
    finally:
        _syncing = False
    apply(scene, obj)


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
            for name in CARRIED:
                setattr(props, name, previous[name])
        else:
            props.axes = 'CUT' if cut_of(scene, obj) else 'FRAME'
            props.pivot_kind = next((k for k in ('POG', 'ME') if landmarks.find(scene, k)), 'CENTRE')
            props.pivot = pivot_world(scene, obj, props.pivot_kind)
    finally:
        _syncing = False
    apply(scene, obj)


def remember(obj):
    """The movement of a segment about to be replaced (re-applied cut), for init_segment."""
    props = obj.omfs_move
    if not props.has_preop:
        return None
    return {name: (tuple(getattr(props, name)) if name == 'pivot' else getattr(props, name))
            for name in CARRIED}


def apply(scene, obj):
    """Place the segment from its six numbers."""
    props = obj.omfs_move
    if _syncing or not props.has_preop or frame_axes(scene) is None:
        return
    obj.matrix_world = world_movement(scene, obj) @ props.preop


def set_pivot(scene, obj, kind):
    """Rotate about another point from now on. The segment doesn't move: the translation is
    rewritten as the displacement of the new pivot."""
    world = pivot_world(scene, obj, kind)
    if world is None or frame_axes(scene) is None:
        return False
    movement = current_movement(obj)
    obj.omfs_move.pivot = world
    set_movement(scene, obj, movement)
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


GUIDE_PREOP = "omfs_preop"      # custom property: an object's pose before following a segment


def companions(guide):
    """The guide and the tools of its Boolean modifiers: they move as one, or the holes stay behind."""
    objects = [guide]
    for mod in guide.modifiers:
        tool = mod.object if mod.type == 'BOOLEAN' else None
        if tool is not None and tool not in objects and not locking.is_locked(tool) and not is_segment(tool):
            objects.append(tool)
    return objects


def preop_of(obj):
    return Matrix([obj[GUIDE_PREOP][i:i + 4] for i in range(0, 16, 4)])


def guide_follows(scene, segment, guide):
    """Move the guide (and its tools) by the segment's movement, from their pre-op poses."""
    w = world_movement(scene, segment)
    for obj in companions(guide):
        if GUIDE_PREOP not in obj:
            obj[GUIDE_PREOP] = [v for row in obj.matrix_world for v in row]
        obj.matrix_world = w @ preop_of(obj)


def guide_back(guide):
    """Guide and tools back where they were before following a segment."""
    for obj in companions(guide):
        if GUIDE_PREOP in obj:
            obj.matrix_world = preop_of(obj)
            del obj[GUIDE_PREOP]


def guide_moved(guide):
    return GUIDE_PREOP in guide


def is_moved(obj):
    return any(abs(getattr(obj.omfs_move, name)) > 1e-6 for name in VALUES)


def tracked_rows(scene, obj, on_axes=None):
    """[(landmark name, pre-op world point, (left, anterior, up))] for the tracked points placed."""
    rows = []
    for lm_id in TRACKED:
        lm = landmarks.find(scene, lm_id)
        if lm is not None:
            point = lm.matrix_world.translation.copy()
            rows.append((landmarks.BY_ID[lm_id][1], point, displacement(scene, obj, point, on_axes)))
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


def _on_axes(props, context):
    """Same pose, numbers rewritten on the new axes."""
    obj = props.id_data
    if not _syncing and props.has_preop and frame_axes(context.scene) is not None:
        set_movement(context.scene, obj, current_movement(obj))


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
    axes: bpy.props.EnumProperty(name="Axes", items=AXES, update=_on_axes,
                                 description="Axes the six values are measured on (the segment doesn't move when it changes)")
    pivot: bpy.props.FloatVectorProperty(size=3, options={'HIDDEN'})   # world, before the move
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


def cutting_guide(context):
    from . import guide_workshop      # here: guide_workshop imports osteotomy, which imports this module
    return guide_workshop.target_guide(context)


class OMFS_OT_guide_follow(bpy.types.Operator):
    """Move the cutting guide by the segment's movement, as one rigid block with its sleeve cylinders,
    text and cut-edge box. Its pre-op position is kept: "Guide en préop" puts it back"""
    bl_idname = "omfs.guide_follow_segment"
    bl_label = "Appliquer le même mouvement au guide de coupe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        segment, guide = active_segment(context.scene), cutting_guide(context)
        if segment is None or guide is None:
            self.report({'ERROR'}, "Il faut un segment et un guide (Confection de guide)")
            return {'CANCELLED'}
        guide_follows(context.scene, segment, guide)
        self.report({'INFO'}, f"{guide.name} déplacé comme {segment.name}")
        return {'FINISHED'}


class OMFS_OT_guide_preop(bpy.types.Operator):
    """Put the cutting guide (and its tools) back where it was before following the segment"""
    bl_idname = "omfs.guide_preop"
    bl_label = "Guide en préop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        guide = cutting_guide(context)
        if guide is not None:
            guide_back(guide)
        return {'FINISHED'}


# --- Drawing: tracked points and their labels ----------------------------------------------

def _shown(context):
    scene = context.scene
    obj = active_segment(scene)
    if obj is None or not obj.omfs_move.has_preop or not obj.omfs_move.show_labels:
        return None
    if not obj.visible_get() or frame_axes(scene) is None:
        return None
    return obj


def _draw_tracks():
    """A line from each tracked point before the move to where the segment takes it."""
    context = bpy.context
    obj = _shown(context)
    if obj is None or not is_moved(obj):
        return
    movement = world_movement(context.scene, obj)
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
    movement = world_movement(context.scene, obj)
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

classes = (OMFS_MoveProperties, OMFS_MovementSceneProperties, OMFS_OT_segment_preop, OMFS_OT_guide_follow,
           OMFS_OT_guide_preop)


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
