"""Osteotomy: a planar cut placed relative to the mental foramina, and the split of the bone.

Version 1 is the standard horizontal sliding genioplasty: one plane. The cut is a CUT object
whose local axes are the reference frame's (X left, Y posterior, Z up) with its origin at the
midpoint of the mental foramina. Its parameters live on the object (obj.omfs_cut), so other
techniques can later add more cuts of the same kind.

Applying the cut never changes the source bone: the segment and the proximal part are new
objects made with an exact Boolean, and the source is only hidden.
"""

from math import cos, radians, sin

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from . import landmarks, locking, params, reference_frame

COLLECTION = "Osteotomy"
CUT_NAME = "Osteotomy plane"
REQUIRED = ('MF_R', 'MF_L')

LATERAL_REACH = 80.0    # mm each side of the midline: wider than any mandible
ANTERIOR_REACH = 60.0   # mm in front of the foramina: past the chin
DEPTH = 100.0           # mm, how far below the plane the cutting volume reaches
CUT_COLOR = (1.0, 0.85, 0.1, 0.35)

STAR_COLOR = (1.0, 0.9, 0.0, 1.0)
STAR_RADIUS = 2.0       # mm
STAR_NAMES = {'MF_R': "Star R", 'MF_L': "Star L"}


# --- Geometry -----------------------------------------------------------------------

def anchor_matrix(scene):
    """Frame rotation with origin at the midpoint of the mental foramina, or None if missing."""
    frame = reference_frame.find_frame(scene)
    foramina = [landmarks.find(scene, lm) for lm in REQUIRED]
    if frame is None or None in foramina:
        return None
    midpoint = (foramina[0].matrix_world.translation + foramina[1].matrix_world.translation) / 2
    matrix = frame.matrix_world.normalized().copy()
    matrix.translation = midpoint
    return matrix


def plane_normal(props):
    """Unit normal of the cut in local axes (X left, Y posterior, Z up)."""
    # + profile angle = rises anteriorly: tip the normal towards posterior (+Y), i.e. about -X.
    # + frontal angle = left side higher: tip the normal towards the right (-X), i.e. about -Y.
    return Euler((-props.profile_angle, -props.frontal_angle, 0)).to_matrix() @ Vector((0, 0, 1))


def height_on_plane(props, x, y):
    """Local Z of the cut plane above the point (x, y)."""
    n = plane_normal(props)
    return -props.below_foramina - (n.x * x + n.y * y) / n.z


def outline(props):
    """Cut outline seen from above (local x, y), counter-clockwise: anterior edge, then the
    posterior limits (right side is -X, left side is +X)."""
    front = -ANTERIOR_REACH
    pts = [(-LATERAL_REACH, front), (LATERAL_REACH, front), (LATERAL_REACH, props.extent_l)]
    if props.extent_l != props.extent_r:
        pts += [(0.0, props.extent_l), (0.0, props.extent_r)]
    pts.append((-LATERAL_REACH, props.extent_r))
    return pts


def plane_mesh_data(props):
    """Vertices and the single face of the visible cut plane, in local coordinates."""
    verts = [(x, y, height_on_plane(props, x, y)) for x, y in outline(props)]
    return verts, [tuple(range(len(verts)))]


def cutter_mesh_data(props):
    """Closed prism: the cut plane on top, vertical walls, everything below is the segment."""
    top, _faces = plane_mesh_data(props)
    n = len(top)
    bottom = [(x, y, z - DEPTH) for x, y, z in top]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    faces += [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    return top + bottom, faces


# --- Safety distances ----------------------------------------------------------------

def world_vertices(obj):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


def heights_above_cut(cut, points):
    """Signed distance (mm) of world points above the cut plane, and their local coordinates."""
    props = cut.omfs_cut
    inv = np.array(cut.matrix_world.inverted())
    local = points @ inv[:3, :3].T + inv[:3, 3]
    n = np.array(plane_normal(props))
    heights = (local - np.array((0.0, 0.0, -props.below_foramina))) @ n
    return heights, local


def min_height_per_side(cut, obj):
    """Closest approach of a mesh to the cut, right and left, over the cut's extent only."""
    props = cut.omfs_cut
    heights, local = heights_above_cut(cut, world_vertices(obj))
    result = {}
    for side, mask_side, extent in (('R', local[:, 0] < 0, props.extent_r),
                                    ('L', local[:, 0] >= 0, props.extent_l)):
        inside = mask_side & (local[:, 1] <= extent) & (np.abs(local[:, 0]) <= LATERAL_REACH)
        result[side] = float(heights[inside].min()) if inside.any() else None
    return result


def safety_rows(scene, cut):
    """[(label, distance in mm or None, margin)] for the panel. Negative = the cut crosses it."""
    rows = []
    for lm, label in (('MF_R', "Mental foramen R"), ('MF_L', "Mental foramen L"),
                      ('CA_R', "Canine apex R"), ('CA_L', "Canine apex L")):
        obj = landmarks.find(scene, lm)
        if obj is not None:
            margin = params.NERVE_SAFETY_MARGIN if lm.startswith('MF') else params.ROOT_SAFETY_MARGIN
            heights, _local = heights_above_cut(cut, np.array([obj.matrix_world.translation]))
            rows.append((label, float(heights[0]), margin))
    for role, label, margin in (('NERVE', "Nerve canal", params.NERVE_SAFETY_MARGIN),
                                ('TEETH', "Tooth roots", params.ROOT_SAFETY_MARGIN)):
        for obj in scene.objects:
            if obj.type == 'MESH' and obj.omfs.role == role and obj.visible_get():
                for side, distance in min_height_per_side(cut, obj).items():
                    rows.append((f"{label} {side}", distance, margin))
    return rows


_safety_cache = {}


def cached_safety_rows(scene, cut):
    """safety_rows, recomputed only when the cut or the landmarks change (panels redraw often)."""
    landmark_positions = tuple(tuple(o.matrix_world.translation) for o in scene.objects
                               if o.omfs.role == 'LANDMARK')
    key = (cut.name, tuple(cut.matrix_world.col[3]), state(cut.omfs_cut), landmark_positions)
    if _safety_cache.get("key") != key:
        _safety_cache["key"] = key
        _safety_cache["rows"] = safety_rows(scene, cut)
    return _safety_cache["rows"]


# --- Objects -------------------------------------------------------------------------

def osteotomy_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def find_cut(scene):
    for obj in scene.objects:
        if obj.omfs.role == 'CUT':
            return obj
    return None


def state(props):
    """Everything the cut's shape depends on, to tell when an applied cut is out of date."""
    return (round(props.below_foramina, 3), round(props.profile_angle, 5), round(props.frontal_angle, 5),
            round(props.extent_r, 3), round(props.extent_l, 3), props.source.name if props.source else "")


def anchor_is_current(scene, cut):
    anchor = anchor_matrix(scene)
    return anchor is not None and all(
        (Vector(a) - Vector(b)).length < 1e-4 for a, b in zip(anchor.col, cut.matrix_world.col))


def update_cut(scene, cut):
    """Place the cut on the current frame and foramina, and rebuild its plane from the parameters."""
    anchor = anchor_matrix(scene)
    if anchor is None:
        return
    locking.set_locked(cut, False)
    cut.matrix_world = anchor
    locking.set_locked(cut, True)
    verts, faces = plane_mesh_data(cut.omfs_cut)
    mesh = cut.data
    mesh.clear_geometry()
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    update_stars(scene, cut.omfs_cut.source)


def default_source(scene):
    """The bone the pogonion (or menton) was placed on: that is the bone to cut."""
    for lm in ('POG', 'ME'):
        obj = landmarks.find(scene, lm)
        if obj is not None:
            source = scene.objects.get(obj.omfs.surface)
            if source is not None and source.omfs.role == 'BONE':
                return source
    return None


def add_cut(scene):
    mesh = bpy.data.meshes.new(CUT_NAME)
    cut = bpy.data.objects.new(CUT_NAME, mesh)
    osteotomy_collection(scene).objects.link(cut)
    cut.omfs.role = 'CUT'
    cut.color = CUT_COLOR
    cut.show_in_front = True
    cut.omfs_cut.source = default_source(scene)
    update_cut(scene, cut)
    return cut


def star_mesh(name):
    verts = []
    for i in range(10):
        r = STAR_RADIUS if i % 2 == 0 else STAR_RADIUS * 0.4
        a = radians(90 + 36 * i)
        verts.append((r * cos(a), r * sin(a), 0.0))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [tuple(range(10))])
    return mesh


def update_stars(scene, bone):
    """Yellow stars below each mental foramen, snapped onto the bone surface (visual only)."""
    frame = reference_frame.find_frame(scene)
    if frame is None or bone is None:
        return
    down = -frame.matrix_world.col[2].xyz.normalized()
    to_bone = bone.matrix_world.inverted()
    for lm, name in STAR_NAMES.items():
        foramen = landmarks.find(scene, lm)
        if foramen is None:
            continue
        target = foramen.matrix_world.translation + down * params.STAR_BELOW_FORAMEN
        found, location, normal, _index = bone.closest_point_on_mesh(to_bone @ target)
        if not found:
            continue
        star = scene.objects.get(name)
        if star is None:
            star = bpy.data.objects.new(name, star_mesh(name))
            osteotomy_collection(scene).objects.link(star)
            star.omfs.role = 'MARKER'
            star.color = STAR_COLOR
            star.show_in_front = True
        locking.set_locked(star, False)
        world_normal = (bone.matrix_world.to_3x3() @ normal).normalized()
        star.matrix_world = Matrix.LocRotScale(bone.matrix_world @ location,
                                               world_normal.to_track_quat('Z', 'Y'), None)
        star.location += world_normal * 0.2   # just above the surface, so it isn't hidden
        locking.set_locked(star, True)


def boolean_copy(context, source, cutter, operation, name):
    """A new object: the source mesh after an exact Boolean with the cutter. Source untouched."""
    temp = bpy.data.objects.new(name, source.data)
    context.scene.collection.objects.link(temp)
    temp.matrix_world = source.matrix_world
    modifier = temp.modifiers.new("Osteotomy", 'BOOLEAN')
    modifier.object = cutter
    modifier.operation = operation
    modifier.solver = 'EXACT'
    mesh = bpy.data.meshes.new_from_object(temp.evaluated_get(context.evaluated_depsgraph_get()))
    mesh.name = name
    bpy.data.objects.remove(temp)
    result = bpy.data.objects.new(name, mesh)
    result.matrix_world = source.matrix_world
    osteotomy_collection(context.scene).objects.link(result)
    return result


def apply_cut(context, cut):
    """Split the source bone into segment + proximal part. Re-applying replaces the old result."""
    props = cut.omfs_cut
    source = props.source
    for obj in [o for o in context.scene.objects if o.omfs.made_by == cut.name]:
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    verts, faces = cutter_mesh_data(props)
    cutter_mesh = bpy.data.meshes.new("Osteotomy cutter")
    cutter_mesh.from_pydata(verts, [], faces)
    cutter = bpy.data.objects.new("Osteotomy cutter", cutter_mesh)
    context.scene.collection.objects.link(cutter)
    cutter.matrix_world = cut.matrix_world

    segment = boolean_copy(context, source, cutter, 'INTERSECT', f"{source.name} segment")
    proximal = boolean_copy(context, source, cutter, 'DIFFERENCE', f"{source.name} proximal")
    bpy.data.objects.remove(cutter)
    bpy.data.meshes.remove(cutter_mesh)

    for obj, role in ((segment, 'SEGMENT'), (proximal, 'BONE')):
        obj.omfs.role = role
        obj.omfs.made_by = cut.name
        locking.set_locked(obj, role == 'BONE')
    source.hide_set(True)
    props.applied_state = repr(state(props))
    return segment, proximal


# --- Properties and operators ---------------------------------------------------------

def _on_cut_changed(props, context):
    update_cut(context.scene, props.id_data)


def _is_bone(_props, obj):
    return obj.type == 'MESH' and obj.omfs.role == 'BONE' and not obj.omfs.made_by


class OMFS_CutProperties(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(
        name="Bone", type=bpy.types.Object, poll=_is_bone, update=_on_cut_changed,
        description="Bone to cut (the bone the pogonion was placed on, by default)")
    below_foramina: bpy.props.FloatProperty(
        name="Below foramina", subtype='DISTANCE', default=params.CUT_BELOW_FORAMINA,
        min=0.0, soft_max=20.0, step=10, update=_on_cut_changed,
        description="Height of the cut below the midpoint of the mental foramina")
    profile_angle: bpy.props.FloatProperty(
        name="Profile angle", subtype='ANGLE', default=radians(params.CUT_PROFILE_ANGLE),
        min=radians(-45), max=radians(45), update=_on_cut_changed,
        description="Angle to Frankfort in profile view; + = the cut rises anteriorly")
    frontal_angle: bpy.props.FloatProperty(
        name="Frontal angle", subtype='ANGLE', default=radians(params.CUT_FRONTAL_ANGLE),
        min=radians(-30), max=radians(30), update=_on_cut_changed,
        description="Tilt in frontal view; + = patient's left side higher")
    extent_r: bpy.props.FloatProperty(
        name="Posterior extent R", subtype='DISTANCE', default=params.CUT_POSTERIOR_EXTENT,
        min=0.0, soft_max=40.0, step=10, update=_on_cut_changed,
        description="How far the cut goes behind the right mental foramen")
    extent_l: bpy.props.FloatProperty(
        name="Posterior extent L", subtype='DISTANCE', default=params.CUT_POSTERIOR_EXTENT,
        min=0.0, soft_max=40.0, step=10, update=_on_cut_changed,
        description="How far the cut goes behind the left mental foramen")
    applied_state: bpy.props.StringProperty(options={'HIDDEN'})


class OMFS_OT_add_cut(bpy.types.Operator):
    """Add the osteotomy plane below the mental foramina"""
    bl_idname = "omfs.add_cut"
    bl_label = "Add Osteotomy Plane"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if anchor_matrix(context.scene) is None:
            self.report({'ERROR'}, "Build the reference frame and place both mental foramina first")
            return {'CANCELLED'}
        add_cut(context.scene)
        return {'FINISHED'}


class OMFS_OT_update_cut(bpy.types.Operator):
    """Move the cut to the current reference frame and mental foramina"""
    bl_idname = "omfs.update_cut"
    bl_label = "Update Cut"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cut = find_cut(context.scene)
        if cut is None:
            return {'CANCELLED'}
        update_cut(context.scene, cut)
        return {'FINISHED'}


class OMFS_OT_apply_cut(bpy.types.Operator):
    """Split the bone along the cut into a mobile segment and the proximal part (source kept, hidden)"""
    bl_idname = "omfs.apply_cut"
    bl_label = "Apply Cut"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cut = find_cut(context.scene)
        if cut is None or cut.omfs_cut.source is None:
            self.report({'ERROR'}, "Choose the bone to cut")
            return {'CANCELLED'}
        segment, proximal = apply_cut(context, cut)
        if not segment.data.polygons:
            self.report({'WARNING'}, "The cut doesn't cross the bone: check its height")
        else:
            self.report({'INFO'}, f"Created {segment.name} and {proximal.name}")
        return {'FINISHED'}


classes = (OMFS_CutProperties, OMFS_OT_add_cut, OMFS_OT_update_cut, OMFS_OT_apply_cut)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_cut = bpy.props.PointerProperty(type=OMFS_CutProperties)


def unregister():
    del bpy.types.Object.omfs_cut
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
