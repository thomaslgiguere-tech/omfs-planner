"""Segment movement (Step 5): 6 degrees of freedom on clinical axes, with live numbers.

Generic: any SEGMENT object. What differs between operations is a preset (obj.omfs_move.preset): the
tracked points, the pivots offered and the axes allowed.
- GENIO (genioplasty chin): pogonion and menton; osteotomy plane or reference frame axes.
- LEFORT (Le Fort I maxilla): ANS, point A, upper incisor midline, PNS; reference frame.
- BSSO (mandibular body): moved by the occlusion later; no panel yet.

Followers: other objects carried rigidly by the same movement (e.g. the upper teeth and their scan
with the Le Fort I). Each keeps its own pre-op pose, so it goes back exactly; removing one puts it back.

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

# Every pivot ever offered (the enum's numbers stay fixed: saved files keep their choice).
ALL_PIVOTS = ('POG', 'ME', 'CENTRE', 'UI', 'ANS', 'A', 'PNS')
PRESETS = {   # tracked points (net movement shown), pivots offered, axes allowed
    'GENIO': {'tracked': ('POG', 'ME'), 'pivots': ('POG', 'ME', 'CENTRE'), 'axes': ('CUT', 'FRAME')},
    'LEFORT': {'tracked': ('ANS', 'A', 'UI', 'PNS'),   # premolar cusps: only for the occlusal plane
               'pivots': ('UI', 'ANS', 'A', 'PNS', 'CENTRE'), 'axes': ('FRAME',)},
    'BSSO': {'tracked': ('LI', 'B', 'POG', 'ME'), 'pivots': ('LI', 'POG', 'CENTRE'), 'axes': ('FRAME',)},
}
ORTHO_MADE_BY = "Ortho osteotomy"   # ortho_osteotomy.MADE_BY (that module imports this one)
AXES = (
    ('CUT', "Osteotomy plane", "Advance slides along the cut (in the midsagittal plane), "
                               "up is perpendicular to the cut"),
    ('FRAME', "Reference frame", "Frankfort horizontal, midsagittal and coronal planes"),
)
VALUES = ('left', 'anterior', 'up', 'pitch', 'roll', 'yaw')
CARRIED = ('preset', *VALUES, 'axes', 'pivot_kind', 'pivot', 'locked', 'show_labels')

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


def preset_of(obj):
    """The segment's preset; segments separated before presets existed are recognised by their maker."""
    if obj.omfs_move.preset:
        return obj.omfs_move.preset
    if obj.omfs.made_by.startswith(ORTHO_MADE_BY):
        return 'LEFORT' if "(Le Fort I)" in obj.name else 'BSSO'
    return 'GENIO'


def preset(obj):
    return PRESETS[preset_of(obj)]


def segments(scene, kind=None):
    return [o for o in scene.objects if is_segment(o) and (kind is None or preset_of(o) == kind)]


def segment_of(scene, kind):
    """The segment of this kind shown in its panel: the one the handles drive if it is of this kind, else the first."""
    chosen = scene.omfs_movement.segment
    if is_segment(chosen) and chosen.name in scene.objects and preset_of(chosen) == kind:
        return chosen
    return next(iter(segments(scene, kind)), None)


def active_segment(scene):
    """The segment the drag handles drive: the one chosen, else the genioplasty segment, else the first."""
    chosen = scene.omfs_movement.segment
    if is_segment(chosen) and chosen.name in scene.objects:
        return chosen
    return segment_of(scene, 'GENIO') or next(iter(segments(scene)), None)


def preop_centre(obj):
    corners = [Vector(c) for c in obj.bound_box]
    return obj.omfs_move.preop @ (sum(corners, Vector()) / 8)


def pivot_world(scene, obj, kind):
    """Pre-op world position of a pivot choice (None if its landmark isn't placed)."""
    if kind == 'CENTRE':
        return preop_centre(obj)
    lm = landmarks.find(scene, kind)
    return lm.matrix_world.translation.copy() if lm else None


def init_segment(scene, obj, previous=None, kind=None, followers=()):
    """Store the pre-op pose of a new segment, lock its transform, carry over a previous movement
    (and its followers), or start with these followers."""
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
            props.preset = kind or preset_of(obj)
            props.axes = 'CUT' if cut_of(scene, obj) and 'CUT' in preset(obj)['axes'] else 'FRAME'
            props.pivot_kind = next((k for k in preset(obj)['pivots'] if k == 'CENTRE' or landmarks.find(scene, k)),
                                    'CENTRE')
            props.pivot = pivot_world(scene, obj, props.pivot_kind)
    finally:
        _syncing = False
    props.followers.clear()
    for follower, preop in (previous or {}).get('followers', [(f, None) for f in followers]):
        if follower is not None and follower.name in scene.objects:
            item = props.followers.add()
            item.obj = follower
            item.preop = preop if preop is not None else follower.matrix_world.copy()
    apply(scene, obj)


def remember(obj):
    """The movement of a segment about to be replaced (re-applied cut), for init_segment."""
    props = obj.omfs_move
    if not props.has_preop:
        return None
    kept = {name: (tuple(getattr(props, name)) if name == 'pivot' else getattr(props, name)) for name in CARRIED}
    kept['followers'] = [(f.obj, Matrix(f.preop)) for f in props.followers if f.obj is not None]
    return kept


def apply(scene, obj):
    """Place the segment from its six numbers."""
    props = obj.omfs_move
    if _syncing or not props.has_preop or frame_axes(scene) is None:
        return
    w = world_movement(scene, obj)
    obj.matrix_world = w @ props.preop
    for item in props.followers:
        if item.obj is not None and item.obj.name in scene.objects:
            item.obj.matrix_world = w @ Matrix(item.preop)


# --- Followers -------------------------------------------------------------------------

def follower_problem(scene, segment, obj):
    if obj is None or obj == segment or obj.name not in scene.objects:
        return "Objet introuvable"
    if obj.parent is not None:
        return f"{obj.name} est rattaché à {obj.parent.name} (bloc d'occlusion ?) : retirez-le d'abord"
    if any(f.obj == obj for s in segments(scene) for f in s.omfs_move.followers):
        return f"{obj.name} suit déjà un segment"
    return None


def add_follower(scene, segment, obj):
    """obj moves with the segment from now on. It has not moved yet: where it is now is its pre-op pose,
    and it joins the segment's current movement at once."""
    item = segment.omfs_move.followers.add()
    item.obj = obj
    item.preop = obj.matrix_world.copy()
    apply(scene, segment)


def remove_follower(segment, index):
    """Stop following: the object goes back to its pre-op pose."""
    followers = segment.omfs_move.followers
    item = followers[index]
    if item.obj is not None:
        item.obj.matrix_world = Matrix(item.preop)
    followers.remove(index)


def release_followers(segment):
    """Every follower back to its pre-op pose (the segment is about to be deleted)."""
    while segment.omfs_move.followers:
        remove_follower(segment, 0)


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
    """The guide, the tools of its Boolean modifiers and its pins' Empties: they move as one, or the
    holes stay behind. The pins' parts are not moved but rebuilt from their Empties (pins.build_all),
    so the arms from the cap reach the pins where they are now."""
    objects = [guide]
    for obj in bpy.data.objects:
        if obj.omfs.made_by == "pin site" and obj.omfs_pin.guide == guide and obj.name in bpy.context.scene.objects:
            objects.append(obj)
    for mod in guide.modifiers:
        tool = mod.object if mod.type == 'BOOLEAN' else None
        if (tool is not None and tool not in objects and not locking.is_locked(tool) and not is_segment(tool)
                and tool.omfs.made_by != "pin part"):
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
    rebuild_pins(scene)


def guide_back(guide):
    """Guide and tools back where they were before following a segment."""
    for obj in companions(guide):
        if GUIDE_PREOP in obj:
            obj.matrix_world = preop_of(obj)
            del obj[GUIDE_PREOP]
    rebuild_pins(bpy.context.scene)


def rebuild_pins(scene):
    from . import pins                # here: pins imports this module
    bpy.context.view_layer.update()   # the Empties' new matrices
    pins.build_all(scene)


def guide_moved(guide):
    return GUIDE_PREOP in guide


def is_moved(obj):
    return any(abs(getattr(obj.omfs_move, name)) > 1e-6 for name in VALUES)


def tracked_rows(scene, obj, on_axes=None):
    """[(landmark name, pre-op world point, (left, anterior, up))] for the tracked points placed."""
    rows = []
    for lm_id in preset(obj)['tracked']:
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

_pivot_items = {}


def _pivots(props, _context):
    """Pivots offered for this segment's preset (kept in a cache: Blender needs the strings to stay alive)."""
    kind = props.preset or 'GENIO'
    if kind not in _pivot_items:
        names = {'CENTRE': ("Centre du segment", "Rotate about the centre of the segment's bounding box")}
        _pivot_items[kind] = [(k, *(names.get(k) or (landmarks.BY_ID[k][1],
                                                     f"Rotate about {landmarks.BY_ID[k][1]} (before the move)")),
                               ALL_PIVOTS.index(k)) for k in PRESETS[kind]['pivots']]
    return _pivot_items[kind]


class OMFS_Follower(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(type=bpy.types.Object)
    preop: bpy.props.FloatVectorProperty(size=(4, 4), subtype='MATRIX', options={'HIDDEN'})


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
    pivot_kind: bpy.props.EnumProperty(name="Pivot", items=_pivots, update=_on_pivot_kind,
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
    preset: bpy.props.StringProperty(options={'HIDDEN'})   # GENIO, LEFORT, BSSO (see PRESETS)
    followers: bpy.props.CollectionProperty(type=OMFS_Follower)


class OMFS_MovementSceneProperties(bpy.types.PropertyGroup):
    segment: bpy.props.PointerProperty(
        name="Segment", type=bpy.types.Object, poll=lambda _self, obj: is_segment(obj),
        description="Segment to move (the first segment, if empty)")


class OMFS_OT_segment_preop(bpy.types.Operator):
    """Put the segment back at its pre-op position (all movements to 0)"""
    bl_idname = "omfs.segment_preop"
    bl_label = "Retour préop"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()   # empty: the segment the handles drive

    def execute(self, context):
        obj = context.scene.objects.get(self.segment_name) if self.segment_name else active_segment(context.scene)
        if not is_segment(obj):
            return {'CANCELLED'}
        back_to_preop(context.scene, obj)
        return {'FINISHED'}


class OMFS_OT_segment_handles(bpy.types.Operator):
    """Put the drag handles (arrows and rings) on this segment"""
    bl_idname = "omfs.segment_handles"
    bl_label = "Poignées sur ce segment"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()

    def execute(self, context):
        obj = context.scene.objects.get(self.segment_name)
        if not is_segment(obj):
            return {'CANCELLED'}
        context.scene.omfs_movement.segment = obj
        return {'FINISHED'}


class OMFS_OT_add_follower(bpy.types.Operator):
    """Make the selected objects move with this segment (same rigid movement, their shape untouched)"""
    bl_idname = "omfs.add_follower"
    bl_label = "Ajouter la sélection"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()

    def execute(self, context):
        segment = context.scene.objects.get(self.segment_name)
        if not is_segment(segment):
            return {'CANCELLED'}
        added = []
        for obj in context.selected_objects:
            problem = follower_problem(context.scene, segment, obj)
            if problem:
                self.report({'WARNING'}, problem)
                continue
            add_follower(context.scene, segment, obj)
            added.append(obj.name)
        if not added:
            self.report({'ERROR'}, "Sélectionnez les objets à faire suivre (clic dans la vue ou l'Outliner)")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Suivent {segment.name} : " + ", ".join(added))
        return {'FINISHED'}


class OMFS_OT_remove_follower(bpy.types.Operator):
    """This object stops following the segment and goes back to its pre-op position"""
    bl_idname = "omfs.remove_follower"
    bl_label = "Ne plus suivre"
    bl_options = {'REGISTER', 'UNDO'}

    segment_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        segment = context.scene.objects.get(self.segment_name)
        if not is_segment(segment) or not 0 <= self.index < len(segment.omfs_move.followers):
            return {'CANCELLED'}
        remove_follower(segment, self.index)
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

def shown(context):
    """Segments whose tracked points are drawn: pre-op stored, labels on, visible, a reference frame."""
    scene = context.scene
    if frame_axes(scene) is None:
        return []
    return [o for o in segments(scene) if o.omfs_move.has_preop and o.omfs_move.show_labels and o.visible_get()]


def _draw_tracks():
    """A line from each tracked point before the move to where the segment takes it."""
    context = bpy.context
    points = []
    for obj in shown(context):
        if is_moved(obj):
            movement = world_movement(context.scene, obj)
            points += [(p, movement @ p) for _name, p, _d in tracked_rows(context.scene, obj)]
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


def short(left, anterior, up):
    """One line for a crowded view, e.g. "av 6.0 · bas 2.0 · g 0.3"."""
    def word(value, plus, minus):
        return f"{plus if value >= 0 else minus} {abs(value):.1f}"
    return " · ".join((word(anterior, "av", "rec"), word(up, "haut", "bas"), word(left, "g", "d")))


def _draw_labels():
    """A small box next to each tracked point: its net translation in clinical words (one line per value
    for the genioplasty's two points, one compact line each when there are more)."""
    context = bpy.context
    region, rv3d = context.region, context.region_data
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, 13 * scale)
    line_h, pad = 16 * scale, 6 * scale
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    for obj in shown(context):
        movement = world_movement(context.scene, obj)
        compact = len(preset(obj)['tracked']) > 2
        for i, (name, point, d) in enumerate(tracked_rows(context.scene, obj)):
            xy = location_3d_to_region_2d(region, rv3d, movement @ point)
            if xy is None:
                continue
            total = Vector(d).length
            lines = [f"{name}   {total:.1f} mm", short(*d)] if compact else [f"{name}   {total:.1f} mm", *describe(*d)]
            width = max(blf.dimensions(font, text)[0] for text in lines) + 2 * pad
            height = len(lines) * line_h + pad
            # Boxes alternately above-right and below-right: neighbouring points are close.
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

classes = (OMFS_Follower, OMFS_MoveProperties, OMFS_MovementSceneProperties, OMFS_OT_segment_preop,
           OMFS_OT_segment_handles, OMFS_OT_add_follower, OMFS_OT_remove_follower, OMFS_OT_guide_follow,
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
