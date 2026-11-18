"""Orthognathic osteotomies: Le Fort I and mandibular (BSSO) cuts, separate from the genioplasty plane.

Le Fort I, like Dolphin 3D's osteotomy tool: the surgeon clicks a few points on the bone where the cut should pass,
from the front (the view looks through the bone the way the saw goes). The line through the points is pushed along
that direction into a cutting sheet ("nappe de coupe"). Then (surgeon, 2026-11-17) one click each side at the
pterygomaxillary junction sets a reliable posterior stop: the first plane is kept up to the plane through the two
stops, and the cut goes down there at 90 degrees, so the segment comes without the pterygoid plates. Points can
be dragged (G); the sheet follows. The line also runs 15 mm past its end points laterally.

"Séparer" cuts a copy of the bone with every sheet of that bone (thickened to a saw kerf, Boolean difference) and
splits the result into its loose pieces: the distal segment (teeth-bearing: the Le Fort I maxilla, the
mandibular body) and the proximal parts (skull base, ramus + condyle each side). The bone's own mesh is never
touched (CBCT data): it is only hidden, and "Réunir" brings it back.

BSSO (surgeon, 2026-11-17, second version): two blue clicks on the buccal cortex, in the view that looks into the
bone, set the anterior vertical cut. A first plane goes from 10 mm outside the bone into it, along the view, through
the whole height of the bone, and stops at the depth of the nerve canal (the NERVE object, found automatically: the
buccal cortex only). From there the cut turns (an acute angle to start with) and follows a curve through orange
points clicked posteriorly, keeping the same vertical direction, until it exits the lingual cortex. Its exact course
is not critical to the outcome (surgeon): it must separate the segments, quickly.

Generic on purpose: a cut is a line + a direction on any bone, or a vertical cut + a curve. Le Fort / BSSO are only the defaults
(depth, names, which piece is distal). Later, the interdental lines of a segmental Le Fort (occlusion_segments.py)
can become cuts of the same kind.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, params, reference_frame, scene_setup

MADE_BY = "Ortho osteotomy"
COLLECTION = "Ostéotomies orthognathiques"
KINDS = (
    ('LEFORT', "Le Fort I", "Trait au-dessus des apex, de part et d'autre, poussé vers l'arrière jusqu'aux ptérygoïdes"),
    ('BSSO_R', "BSSO droite", "Ostéotomie sagittale droite : trait vertical vestibulaire (2 points bleus) jusqu'au "
                              "nerf, puis courbe vers l'arrière jusqu'au cortex lingual (points orange)"),
    ('BSSO_L', "BSSO gauche", "Ostéotomie sagittale gauche : trait vertical vestibulaire, puis courbe"),
)
NAMES = {'LEFORT': "Trait Le Fort I", 'BSSO_R': "Trait BSSO droite", 'BSSO_L': "Trait BSSO gauche"}
SHEET_COLOUR = (1.0, 0.85, 0.1, 0.45)
FRONT_COLOUR, STOP_COLOUR = (1.0, 0.75, 0.0, 1.0), (0.2, 0.8, 1.0, 1.0)
STOPS = (('R', "droit"), ('L', "gauche"))   # Le Fort I posterior stops, at the pterygomaxillary junctions
SPLIT_KINDS = {'BSSO_R', 'BSSO_L'}   # vertical buccal cut + curve; the others: a line pushed along the view


def is_split(cut):
    return cut.omfs_ortho.kind in SPLIT_KINDS


# --- Axes ---------------------------------------------------------------------------------------

def clinical_axes(scene):
    """(left, posterior, up) unit vectors: the reference frame's if built, else the patient's (LPS) axes."""
    frame = reference_frame.find_frame(scene)
    if frame is None:
        return Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    m = frame.matrix_world.to_3x3().normalized()
    return m.col[0].normalized(), m.col[1].normalized(), m.col[2].normalized()


# --- Cut objects --------------------------------------------------------------------------------

def cuts(scene, source=None):
    found = [o for o in scene.objects if o.omfs.made_by == MADE_BY and o.type == 'MESH' and o.omfs.role == 'CUT']
    return [o for o in found if source is None or o.omfs_ortho.source == source]


def find_cut(scene, kind):
    return next((o for o in cuts(scene) if o.omfs_ortho.kind == kind), None)


def vertical_points(cut):
    """BSSO: the two blue points of the buccal vertical cut, in order."""
    return sorted((o for o in cut.children if o.omfs.landmark.startswith("ORTHO_V")), key=lambda o: int(o.omfs.landmark[7:]))


def front_points(cut):
    """The clicked points (Empties), in order: the Le Fort line, or the BSSO's orange curve."""
    return sorted((o for o in cut.children if o.omfs.landmark.startswith("ORTHO_F")), key=lambda o: int(o.omfs.landmark[7:]))


def stop_point(cut, side):
    """The Le Fort I's posterior stop on that side ('R' / 'L'), or None."""
    return next((o for o in cut.children if o.omfs.landmark == f"ORTHO_PTM_{side}"), None)


def stops_ready(cut):
    return all(stop_point(cut, side) is not None for side, _name in STOPS)


def set_stop(scene, cut, side, location):
    """Place (or move) the posterior stop of that side: a click at the pterygomaxillary junction."""
    point = stop_point(cut, side)
    if point is None:
        name = dict(STOPS)[side]
        point = new_point(scene, cut, f"Stop ptérygo-maxillaire {name}", f"ORTHO_PTM_{side}", Vector(location),
                          STOP_COLOUR, 1.2)
    point.location = Vector(location)
    bpy.context.view_layer.update()
    rebuild_sheet(cut)
    return point


def collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def new_point(scene, cut, name, tag, location, colour, size, parent=None):
    obj = bpy.data.objects.new(name, None)
    collection(scene).objects.link(obj)
    obj.empty_display_type, obj.empty_display_size = 'SPHERE', size
    obj.show_in_front = True
    obj.omfs.made_by, obj.omfs.landmark, obj.omfs.role = MADE_BY, tag, 'MARKER'
    obj.color = colour          # after the role: setting the role paints the role's colour
    obj.parent = parent or cut
    obj.matrix_parent_inverse = Matrix()
    if parent is None:
        obj.location = location            # the cut object stays at the world origin
    else:
        obj.location = location            # local: the offset from its front point
    for lock in ("lock_rotation", "lock_scale"):
        setattr(obj, lock, (True, True, True))
    return obj


def make_cut(scene, kind, source, points, direction):
    """A line cut on a bone (Le Fort I): clicked points (world) and the direction the saw goes (world). Replaces a
    cut of the same kind, keeping its posterior stops. The cut is complete once both stops are placed (set_stop).
    Returns the cut object (its mesh is the sheet)."""
    old = find_cut(scene, kind)
    kept = {}
    if old is not None:
        kept = {side: stop_point(old, side).matrix_world.translation.copy() for side, _n in STOPS if stop_point(old, side)}
        remove_cut(old)
    mesh = bpy.data.meshes.new(NAMES[kind])
    cut = bpy.data.objects.new(NAMES[kind], mesh)
    collection(scene).objects.link(cut)
    cut.omfs.role, cut.omfs.made_by = 'CUT', MADE_BY
    cut.color = SHEET_COLOUR
    cut.show_transparent = True
    cut.hide_select = True
    props = cut.omfs_ortho
    props.kind, props.source = kind, source
    props.direction = Vector(direction).normalized()
    for i, p in enumerate(points, 1):
        new_point(scene, cut, f"{NAMES[kind]} {i}", f"ORTHO_F{i}", Vector(p), FRONT_COLOUR, 1.0)
    for side, location in kept.items():
        set_stop(scene, cut, side, location)
    bpy.context.view_layer.update()
    rebuild_sheet(cut)
    return cut


def make_split(scene, kind, source):
    """An empty BSSO cut on a bone: points are added by add_split_point. Replaces a cut of the same kind."""
    old = find_cut(scene, kind)
    if old is not None:
        remove_cut(old)
    mesh = bpy.data.meshes.new(NAMES[kind])
    cut = bpy.data.objects.new(NAMES[kind], mesh)
    collection(scene).objects.link(cut)
    cut.omfs.role, cut.omfs.made_by = 'CUT', MADE_BY
    cut.color = SHEET_COLOUR
    cut.show_transparent = True
    cut.hide_select = True
    cut.omfs_ortho.kind, cut.omfs_ortho.source = kind, source
    return cut


def add_split_point(scene, cut, location, direction=None):
    """The two first clicks are the blue points of the vertical cut (direction: the view's, into the bone, taken
    with the second one); the next ones the orange curve."""
    vertical = vertical_points(cut)
    name = NAMES[cut.omfs_ortho.kind]
    if len(vertical) < 2:
        i = len(vertical) + 1
        point = new_point(scene, cut, f"{name} vertical {i}", f"ORTHO_V{i}", Vector(location), STOP_COLOUR, 1.2)
        if i == 2 and direction is not None:
            cut.omfs_ortho.direction = Vector(direction).normalized()
    else:
        i = len(front_points(cut)) + 1
        point = new_point(scene, cut, f"{name} {i}", f"ORTHO_F{i}", Vector(location), FRONT_COLOUR, 1.0)
    bpy.context.view_layer.update()
    rebuild_sheet(cut)
    return point


def remove_last_point(cut):
    """The last orange point, else the last blue one."""
    points = front_points(cut) or vertical_points(cut)
    if points:
        bpy.data.objects.remove(points[-1])
        bpy.context.view_layer.update()
        rebuild_sheet(cut)


def remove_cut(cut):
    for obj in list(cut.children_recursive):
        bpy.data.objects.remove(obj)
    mesh = cut.data
    bpy.data.objects.remove(cut)
    bpy.data.meshes.remove(mesh)


# --- The sheet ----------------------------------------------------------------------------------

def smooth_line(points, steps):
    """Catmull-Rom curve through the points (n, 3), steps samples per segment, ends included."""
    p = np.asarray(points, float)
    if len(p) < 3:
        t = np.linspace(0, 1, steps + 1)[:, None]
        return p[0] + t * (p[-1] - p[0])
    ext = np.vstack((2 * p[0] - p[1], p, 2 * p[-1] - p[-2]))
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, steps, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1])
    return np.array(out)


def extend(line, length):
    """The line with its ends pushed out along their tangents."""
    start = line[0] - (line[1] - line[0]) / max(np.linalg.norm(line[1] - line[0]), 1e-9) * length
    end = line[-1] + (line[-1] - line[-2]) / max(np.linalg.norm(line[-1] - line[-2]), 1e-9) * length
    return np.vstack((start, line, end))


def sheet_rows(cut):
    """The rows of the Le Fort I cut surface (each (k, 3), world, front to back), or None until the line and both
    stops are placed. In the frame (across: right stop to left stop; back: the saw's direction; normal): the first
    plane is a surface through the clicked line (its height interpolated between the line's points, inverse
    distance), covering the whole width (+ ORTHO_END_EXTENSION_LEFORT each side) from ORTHO_MARGIN in front of the
    line back to the line of the stops; then a last row straight down from there, at 90 degrees, ORTHO_PTM_DROP mm.
    However the line is clicked (from the front, or along the lateral walls past the junction), every bone at that
    level in front of the stops is cut, and the pterygoid plates behind them stay with the skull."""
    fronts = front_points(cut)
    if len(fronts) < 2 or not stops_ready(cut):
        return None
    right, left = (np.array(stop_point(cut, side).matrix_world.translation) for side, _n in STOPS)
    across = left - right
    across /= max(np.linalg.norm(across), 1e-9)
    back = np.array(cut.omfs_ortho.direction) - (np.array(cut.omfs_ortho.direction) @ across) * across
    back /= max(np.linalg.norm(back), 1e-9)
    normal = np.cross(across, back)
    up = np.array(clinical_axes(bpy.context.scene)[2])
    if normal @ up < 0:
        normal = -normal
    line = smooth_line(np.array([f.matrix_world.translation for f in fronts]), params.ORTHO_SMOOTH_STEPS)
    u, v, h = line @ across, line @ back, line @ normal
    reach = params.ORTHO_END_EXTENSION_LEFORT
    spacing = params.ORTHO_LEFORT_SPACING
    columns = np.linspace(u.min() - reach, u.max() + reach, max(2, int(np.ceil((np.ptp(u) + 2 * reach) / spacing)) + 1))
    (ur, vr), (ul, vl) = (right @ across, right @ back), (left @ across, left @ back)
    stop_v = vr + (columns - ur) * (vl - vr) / (ul - ur if abs(ul - ur) > 1e-9 else 1e-9)   # the stops' line, extended
    start_v = v.min() - params.ORTHO_MARGIN
    depth = np.maximum(stop_v - start_v, spacing)
    n_rows = max(2, int(np.ceil(depth.max() / spacing)) + 1)
    rows = []
    for r in range(n_rows):
        rv = start_v + depth * r / (n_rows - 1)                       # the last row exactly on the stops' line
        d2 = (columns[:, None] - u[None]) ** 2 + (rv[:, None] - v[None]) ** 2
        w = 1.0 / np.maximum(d2, 1e-6)
        height = (w * h[None]).sum(axis=1) / w.sum(axis=1)            # through the line, smooth between
        rows.append(columns[:, None] * across + rv[:, None] * back + height[:, None] * normal)
    rows.append(rows[-1] - normal * params.ORTHO_PTM_DROP)            # the posterior stop, straight down
    return rows


def nerve_depth(scene, origin, down, into, along, height):
    """How deep (mm, along `into` from origin) the nerve canal lies at the buccal cut: the centre of the first canal
    met going in (NERVE objects' vertices within ORTHO_BSSO_NERVE_BAND of the cut's plane, within its height + 10 mm,
    and within ORTHO_BSSO_CANAL mm of the nearest one: not the other side's canal further in). None if no canal."""
    found = []
    for obj in scene.objects:
        if obj.type == 'MESH' and obj.omfs.role == 'NERVE':
            co = np.empty(len(obj.data.vertices) * 3)
            obj.data.vertices.foreach_get("co", co)
            m = np.array(obj.matrix_world)
            co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3] - origin
            h = co @ down
            near = (np.abs(co @ along) < params.ORTHO_BSSO_NERVE_BAND) & (co @ into > 0) & (h > -10) & (h < height + 10)
            found.append(co[near] @ into)
    depth = np.concatenate(found) if found else np.zeros(0)
    if not len(depth):
        return None
    first = depth[depth < depth.min() + params.ORTHO_BSSO_CANAL]
    return float(first.mean())


def split_rows(cut):
    """(top row, bottom row) of the BSSO cut (world), or None before the two blue points. Frame: down (blue point 1
    to 2), into (the view's direction, square to it), along (the third). The path, seen from above: from
    ORTHO_BSSO_OUTSIDE outside the bone straight in to the nerve canal's depth (the buccal vertical cut), then the
    smooth curve through the orange points, ORTHO_BSSO_EXIT past the last one. At each point of the path the cut
    spans the bone there, + ORTHO_BSSO_HEIGHT_MARGIN above and below, always along `down`."""
    vertical = vertical_points(cut)
    if len(vertical) < 2:
        return None
    scene = bpy.context.scene
    top, bottom = (np.array(p.matrix_world.translation) for p in vertical[:2])
    down = bottom - top
    down /= max(np.linalg.norm(down), 1e-9)
    into = np.array(cut.omfs_ortho.direction)
    into = into - (into @ down) * down
    if np.linalg.norm(into) < 1e-6:
        return None
    into /= np.linalg.norm(into)
    along = np.cross(down, into)
    depth = nerve_depth(scene, top, down, into, along, float(np.linalg.norm(bottom - top)))
    props = cut.omfs_ortho
    props.nerve_found = depth is not None
    props.nerve_depth = depth if depth is not None else params.ORTHO_BSSO_DEFAULT_DEPTH
    corner = top + into * props.nerve_depth
    flat = [corner]
    for p in front_points(cut):
        q = np.array(p.matrix_world.translation)
        flat.append(q - ((q - top) @ down) * down)                   # same vertical direction: seen from above
    step = params.ORTHO_BSSO_SPACING
    outside = top - into * params.ORTHO_BSSO_OUTSIDE
    n = max(2, int(np.ceil((params.ORTHO_BSSO_OUTSIDE + props.nerve_depth) / step)) + 1)
    path = list(outside + np.linspace(0, 1, n)[:, None] * (corner - outside))
    if len(flat) > 1:
        curve = extend(smooth_line(np.array(flat), params.ORTHO_SMOOTH_STEPS * 2), params.ORTHO_BSSO_EXIT)[1:]
        path += list(curve[1:])
    path = np.array(path)
    # the bone's height around each point of the path, along `down`
    source = props.source
    co = np.empty(len(source.data.vertices) * 3)
    source.data.vertices.foreach_get("co", co)
    m = np.array(source.matrix_world)
    co = co.reshape(-1, 3)[::2] @ m[:3, :3].T + m[:3, 3] - top
    h, x, y = co @ down, co @ into, co @ along
    span = np.full((len(path), 2), np.nan)
    for k, p in enumerate(path - top):
        near = (x - p @ into) ** 2 + (y - p @ along) ** 2 < params.ORTHO_BSSO_HEIGHT_RADIUS ** 2
        if near.any():
            span[k] = h[near].min(), h[near].max()
    if np.isnan(span).all():
        span[:] = (-10.0, np.linalg.norm(bottom - top) + 10.0)
    valid = np.flatnonzero(~np.isnan(span[:, 0]))
    for k in np.flatnonzero(np.isnan(span[:, 0])):                    # outside the bone: the nearest point's span
        span[k] = span[valid[np.argmin(np.abs(valid - k))]]
    lo = np.array([span[max(0, k - 2):k + 3, 0].min() for k in range(len(path))]) - params.ORTHO_BSSO_HEIGHT_MARGIN
    hi = np.array([span[max(0, k - 2):k + 3, 1].max() for k in range(len(path))]) + params.ORTHO_BSSO_HEIGHT_MARGIN
    lo = np.minimum(lo, -params.ORTHO_BSSO_HEIGHT_MARGIN)
    hi = np.maximum(hi, np.linalg.norm(bottom - top) + params.ORTHO_BSSO_HEIGHT_MARGIN)
    return path + lo[:, None] * down, path + hi[:, None] * down


def sheet_geometry(cut):
    """(vertices, faces) of the cutting surface, world; None when not enough points."""
    rows = split_rows(cut) if is_split(cut) else sheet_rows(cut)
    if rows is None:
        return None
    k = len(rows[0])
    faces = [(r * k + i, r * k + i + 1, (r + 1) * k + i + 1, (r + 1) * k + i)
             for r in range(len(rows) - 1) for i in range(k - 1)]
    return np.vstack(rows), faces


def rebuild_sheet(cut):
    """The cut object's mesh: the cutting surface (drawn translucent)."""
    mesh = cut.data
    mesh.clear_geometry()
    geometry = sheet_geometry(cut)
    if geometry is None:
        points = [f.matrix_world.translation for f in front_points(cut)]
        if not is_split(cut) and len(points) >= 2:   # Le Fort before its stops: the line alone
            line = smooth_line(np.array(points), params.ORTHO_SMOOTH_STEPS)
            mesh.from_pydata(line.tolist(), [(i, i + 1) for i in range(len(line) - 1)], [])
            mesh.update()
        return
    verts, faces = geometry
    mesh.from_pydata(np.asarray(verts).tolist(), [], faces)
    mesh.update()


def cutter_mesh(cut):
    """The cutting surface thickened to the saw kerf: a closed slab, for the Boolean."""
    verts, faces = sheet_geometry(cut)
    bm = bmesh.new()
    bverts = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([bverts[i] for i in f])
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * params.ORTHO_KERF / 2      # solidify grows backwards: the slab ends centred on the sheet
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=params.ORTHO_KERF)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new("cutter")
    bm.to_mesh(mesh)
    bm.free()
    return mesh


_dirty = set()


@bpy.app.handlers.persistent
def _on_depsgraph(scene, depsgraph):
    """A point was moved: redraw its sheet just after (never inside the handler: that would loop)."""
    for update in depsgraph.updates:
        obj = update.id
        if isinstance(obj, bpy.types.Object) and update.is_updated_transform and obj.omfs.made_by == MADE_BY and obj.type == 'EMPTY':
            cut = obj.parent if obj.parent is not None and obj.parent.type == 'MESH' else (obj.parent.parent if obj.parent else None)
            if cut is not None:
                _dirty.add(cut.name)
    if _dirty and not bpy.app.timers.is_registered(_redraw):
        bpy.app.timers.register(_redraw, first_interval=0.05)


def _redraw():
    for name in list(_dirty):
        cut = bpy.data.objects.get(name)
        if cut is not None:
            rebuild_sheet(cut)
    _dirty.clear()
    return None


# --- Separation ---------------------------------------------------------------------------------

def mesh_volume(mesh):
    """Signed volume (mm3) and centroid of a closed mesh, from its triangles."""
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    a, b, c = (co[tris.reshape(-1, 3)[:, k]] for k in range(3))
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    total = v.sum()
    centre = ((a + b + c) / 4 * v[:, None]).sum(axis=0) / total if abs(total) > 1e-9 else co.mean(axis=0)
    return float(total), centre


def boolean_cut(context, source, cutters):
    """A new mesh (world coordinates): the source minus the cutter slabs. Fast solver for closed meshes (bones from
    the Segmentation workspace), exact solver otherwise."""
    scene = context.scene
    temp = bpy.data.objects.new("ortho temp", source.data)
    scene.collection.objects.link(temp)
    temp.matrix_world = source.matrix_world
    tool = bpy.data.objects.new("ortho cutter", cutters)
    scene.collection.objects.link(tool)
    mod = temp.modifiers.new("Cut", 'BOOLEAN')
    mod.operation, mod.object = 'DIFFERENCE', tool
    result = None
    for solver in ('MANIFOLD', 'EXACT'):
        try:
            mod.solver = solver
        except TypeError:
            continue
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()
        mesh = bpy.data.meshes.new_from_object(temp.evaluated_get(depsgraph))
        if len(mesh.polygons):
            result = mesh
            break
        bpy.data.meshes.remove(mesh)
    bpy.data.objects.remove(temp)
    bpy.data.objects.remove(tool)
    if result is not None:
        result.transform(source.matrix_world)
    return result


def loose_pieces(context, mesh, name):
    """Split a mesh into its loose parts (Edit Mode > Separate > By Loose Parts, in C): new objects, world coords."""
    scene = context.scene
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    with context.temp_override(active_object=obj, object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
        for other in context.view_layer.objects.selected:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
    found = [o for o in scene.objects if o.type == 'MESH' and (o == obj or o.name.startswith(name + "."))]
    for o in found:
        o.select_set(False)
    return found


def merge(context, target, others):
    """Join chips into a piece (Object > Join)."""
    if not others:
        return
    with context.temp_override(active_object=target, selected_editable_objects=[target] + others, selected_objects=[target] + others):
        bpy.ops.object.join()


def cutter_slabs(found):
    """Every cut of a bone as one closed mesh (the slabs side by side)."""
    slabs = bmesh.new()
    for cut in found:
        part = cutter_mesh(cut)
        slabs.from_mesh(part)
        bpy.data.meshes.remove(part)
    mesh = bpy.data.meshes.new("ortho cutters")
    slabs.to_mesh(mesh)
    slabs.free()
    return mesh


def touches(obj, tree, box):
    """Some vertex of the piece lies on a cut face (within the kerf): the cut went through it. Only the vertices
    inside the cutters' box are tested (the cut faces are a few hundred among a million vertices)."""
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co = co[np.all((co >= box[0]) & (co <= box[1]), axis=1)]
    reach = params.ORTHO_KERF + 0.3
    return any(tree.find_nearest(Vector(p), reach)[0] is not None for p in co)


def pieces_of(scene, source):
    return [o for o in scene.objects if o.omfs.made_by == f"{MADE_BY}: {source.name}"]


def set_cuts_hidden(scene, source, hide):
    """Hide (or show) the bone's cutting surfaces and their points: hidden once the segments are separated, shown
    again by "Réunir" to adjust them."""
    for cut in cuts(scene, source):
        for obj in [cut] + list(cut.children_recursive):
            if obj.name in bpy.context.view_layer.objects:
                obj.hide_set(hide)


def rejoin(scene, source):
    for obj in pieces_of(scene, source):
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    source.hide_set(False)
    set_cuts_hidden(scene, source, False)


def separate(context, source):
    """Cut the bone with all its cuts and name the pieces. Returns (pieces, warning or None)."""
    scene = context.scene
    if any(not is_split(c) and not stops_ready(c) for c in cuts(scene, source)):
        return [], "Placez d'abord les deux stops ptérygo-maxillaires (droit et gauche)"
    found = [c for c in cuts(scene, source) if sheet_geometry(c) is not None]
    if not found:
        return [], "Tracez d'abord un trait sur cet os"
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    rejoin(scene, source)
    cutters = cutter_slabs(found)
    mesh = boolean_cut(context, source, cutters)
    bpy.data.meshes.remove(cutters)
    if mesh is None:
        return [], "La coupe a échoué (maillage de l'os non fermé ?)"
    base = f"{source.name} coupe"
    objects = loose_pieces(context, mesh, base)
    slab = cutter_slabs(found)
    tree = BVHTree.FromPolygons([v.co for v in slab.vertices], [p.vertices for p in slab.polygons])
    corners = np.array([v.co for v in slab.vertices])
    box = (corners.min(axis=0) - 1.0, corners.max(axis=0) + 1.0)
    bpy.data.meshes.remove(slab)
    sizes = {o.name: mesh_volume(o.data) for o in objects}
    touched = {o.name for o in objects if touches(o, tree, box)}
    # pieces the cut went through: what was cut; the others (bones already apart, e.g. vertebrae) stay fixed
    cut_big = [o for o in objects if o.name in touched and sizes[o.name][0] >= params.ORTHO_MIN_PIECE]
    if not cut_big:
        cut_big = [max(objects, key=lambda o: sizes[o.name][0])]
    kinds = {c.omfs_ortho.kind for c in found}
    left, posterior, up = (np.array(v) for v in clinical_axes(scene))
    centre = sum(sizes[o.name][1] * sizes[o.name][0] for o in cut_big) / sum(sizes[o.name][0] for o in cut_big)
    if 'LEFORT' in kinds:
        distal = min(cut_big, key=lambda o: sizes[o.name][1] @ up)                      # the maxilla, below the cut
    else:
        distal = min(cut_big, key=lambda o: abs((sizes[o.name][1] - centre) @ left))    # the body, across the midline
    keep = cut_big
    for other in [o for o in objects if o not in keep]:                     # chips and other bones join a fixed piece
        c = sizes[other.name][1]
        fixed = [o for o in keep if o != distal] or [distal]
        target = distal if other.name in touched and sizes[other.name][0] < params.ORTHO_MIN_PIECE and \
            np.linalg.norm(sizes[distal.name][1] - c) < min(np.linalg.norm(sizes[o.name][1] - c) for o in fixed) else \
            min(fixed, key=lambda o: np.linalg.norm(sizes[o.name][1] - c))
        merge(context, target, [other])
    big = keep
    centres = {o: mesh_volume(o.data)[1] for o in big}       # by object: the names change below
    expected = 2 if 'LEFORT' in kinds else 1 + len(kinds & {'BSSO_R', 'BSSO_L'})
    for o in big:
        o.omfs.made_by = f"{MADE_BY}: {source.name}"
        if o == distal:
            o.name = f"{source.name} distal (Le Fort I)" if 'LEFORT' in kinds else f"{source.name} distale"
            o.omfs.role = 'SEGMENT'
            locking.set_locked(o, False)
        else:
            side = "gauche" if (centres[o] - centres[distal]) @ left > 0 else "droit"
            o.name = f"{source.name} proximal" if 'LEFORT' in kinds else f"{source.name} proximal {side}"
            o.omfs.role = 'BONE'
            locking.set_locked(o, True)
        o.data.name = o.name
        o.color = (0.55, 0.85, 0.55, 1.0) if o == distal else (0.90, 0.84, 0.70, 1.0)
        if o.name not in collection(scene).objects:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            collection(scene).objects.link(o)
    source.hide_set(True)
    set_cuts_hidden(scene, source, True)      # the segments are seen without the cuts and their points
    warning = None
    if len(big) < expected:
        how = ("BSSO : prolongez la courbe orange jusqu'à sortir par le cortex lingual" if kinds & SPLIT_KINDS else
               "Prolongez le trait au-delà de l'os (points d'extrémité) ou vérifiez les stops ptérygo-maxillaires")
        warning = f"Coupe incomplète : {len(big)} morceau(x) au lieu de {expected}. {how}"
    return big, warning


# --- Drawing ------------------------------------------------------------------------------------

class OMFS_OT_ortho_draw(bpy.types.Operator):
    """Le Fort I : cliquez des points sur l'os où l'ostéotomie doit passer, de face ; Entrée : le trait est poussé dans
    l'os selon la direction de la vue. BSSO : vue de côté, 2 clics bleus sur la corticale vestibulaire (le trait
    vertical antérieur, qui s'enfonce jusqu'au nerf), puis des clics orange vers l'arrière jusqu'au cortex lingual
    (tournez la vue au besoin). Retour arrière : dernier point. Échap : annuler"""
    bl_idname = "omfs.ortho_draw"
    bl_label = "Tracer le trait"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(items=KINDS)

    def invoke(self, context, event):
        source = bone_for(context.scene, self.kind)
        if source is None:
            self.report({'ERROR'}, "Choisissez d'abord l'os à couper")
            return {'CANCELLED'}
        self.source = source
        self.points = []
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        self.markers = []
        self.split = self.kind in SPLIT_KINDS
        self.cut = None
        if self.split:                 # live: the cut follows each click
            old = find_cut(context.scene, self.kind)
            self.previous = None if old is None else (
                [p.matrix_world.translation.copy() for p in vertical_points(old) + front_points(old)],
                Vector(old.omfs_ortho.direction))
            self.cut = make_split(context.scene, self.kind, source)
        self.prompt()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def count(self):
        return len(vertical_points(self.cut)) + len(front_points(self.cut)) if self.split else len(self.points)

    def view_direction(self):
        view = next(r.data for r in self.area.regions if r.type == 'WINDOW')
        return view.view_rotation @ Vector((0.0, 0.0, -1.0))

    def prompt(self):
        if self.split:
            blue = len(vertical_points(self.cut))
            what = (f"point bleu {blue + 1}/2 sur la corticale vestibulaire (trait vertical, vue de côté)" if blue < 2
                    else f"points orange vers l'arrière jusqu'au cortex lingual ({len(front_points(self.cut))})")
        else:
            what = f"({self.count()} point(s))"
        self.area.header_text_set(f"{NAMES[self.kind]} : cliquez sur {self.source.name} : {what}   |   "
                                  "Entrée / clic droit : terminer   |   Retour arrière : dernier point   |   Échap : annuler")

    def clear(self, context):
        for marker in self.markers:
            bpy.data.objects.remove(marker)
        self.markers = []
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def cancel_split(self, context):
        remove_cut(self.cut)
        if self.previous is not None:   # the cut there before comes back
            cut = make_split(context.scene, self.kind, self.source)
            for p in self.previous[0]:
                add_split_point(context.scene, cut, p, self.previous[1])

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self.clear(context)
            if self.split:
                self.cancel_split(context)
            return {'CANCELLED'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS' and self.count():
            if self.split:
                remove_last_point(self.cut)
            else:
                self.points.pop()
                bpy.data.objects.remove(self.markers.pop())
            self.prompt()
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self.split and (len(vertical_points(self.cut)) < 2 or not front_points(self.cut)):
                self.report({'WARNING'}, "2 points bleus, puis au moins 1 point orange")
                return {'RUNNING_MODAL'}
            if not self.split and len(self.points) < 2:
                self.report({'WARNING'}, "Au moins 2 points")
                return {'RUNNING_MODAL'}
            self.clear(context)
            if not self.split:
                make_cut(context.scene, self.kind, self.source, list(self.points), self.view_direction())
            return {'FINISHED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under = landmarks.viewport_region_under_mouse(self.area, event)
            if under is None:
                return {'PASS_THROUGH'}
            hit = landmarks.object_hit(context, under[0], under[1], self.source)
            if hit is None:
                return {'RUNNING_MODAL'}
            if self.split:
                add_split_point(context.scene, self.cut, hit, self.view_direction())
            else:
                self.points.append(hit.copy())
                marker = bpy.data.objects.new("point", None)
                context.scene.collection.objects.link(marker)
                marker.empty_display_type, marker.empty_display_size, marker.location = 'SPHERE', 1.0, hit
                marker.show_in_front = True
                self.markers.append(marker)
            self.prompt()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


class OMFS_OT_ortho_stop(bpy.types.Operator):
    """Cliquez sur l'os à la jonction ptérygo-maxillaire de ce côté : la coupe s'arrête là et descend à 90°,
    sans prendre les plaques ptérygoïdes. Échap : annuler"""
    bl_idname = "omfs.ortho_stop"
    bl_label = "Stop ptérygo-maxillaire"
    bl_options = {'REGISTER', 'UNDO'}

    side: bpy.props.EnumProperty(items=(('R', "Droit", ""), ('L', "Gauche", "")))

    def invoke(self, context, event):
        self.cut = find_cut(context.scene, 'LEFORT')
        if self.cut is None:
            self.report({'ERROR'}, "Tracez d'abord le trait du Le Fort I")
            return {'CANCELLED'}
        self.source = self.cut.omfs_ortho.source
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        self.area.header_text_set(f"Stop ptérygo-maxillaire {dict(STOPS)[self.side]} : cliquez sur l'os à la jonction "
                                  "(vue de côté)   |   Échap : annuler")
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def end(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self.end(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under = landmarks.viewport_region_under_mouse(self.area, event)
            if under is None:
                return {'PASS_THROUGH'}
            hit = landmarks.object_hit(context, under[0], under[1], self.source)
            if hit is None:
                return {'RUNNING_MODAL'}
            set_stop(context.scene, self.cut, self.side, hit)
            self.end(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}


def bone_for(scene, kind):
    """The bone a cut of this kind goes through: the one chosen, else "Maxillaire" / "Mandibule" (the names the
    Segmentation workspace gives)."""
    return bone(scene, 'MAXILLA' if kind == 'LEFORT' else 'MANDIBLE')


def bone(scene, which):
    props = scene.omfs_ortho_scene
    chosen = props.maxilla if which == 'MAXILLA' else props.mandible
    if chosen is not None and chosen.name in scene.objects:
        return chosen
    return _default(scene, "Maxillaire" if which == 'MAXILLA' else "Mandibule")


class OMFS_OT_ortho_remove(bpy.types.Operator):
    """Effacer ce trait (ses points et sa nappe)"""
    bl_idname = "omfs.ortho_remove"
    bl_label = "Effacer le trait"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(items=KINDS)

    def execute(self, context):
        cut = find_cut(context.scene, self.kind)
        if cut is not None:
            remove_cut(cut)
        return {'FINISHED'}


class OMFS_OT_ortho_separate(bpy.types.Operator):
    """Couper une copie de l'os avec ses traits et séparer les segments (distal, proximal). L'os d'origine n'est
    jamais modifié: il est seulement caché"""
    bl_idname = "omfs.ortho_separate"
    bl_label = "Séparer les segments"
    bl_options = {'REGISTER', 'UNDO'}

    which: bpy.props.EnumProperty(items=(('MAXILLA', "Maxillaire", ""), ('MANDIBLE', "Mandibule", "")))

    def execute(self, context):
        source = bone(context.scene, self.which)
        if source is None:
            self.report({'ERROR'}, "Choisissez l'os")
            return {'CANCELLED'}
        pieces, warning = separate(context, source)
        if not pieces:
            self.report({'ERROR'}, warning)
            return {'CANCELLED'}
        if warning:
            self.report({'WARNING'}, warning)
        else:
            self.report({'INFO'}, f"{len(pieces)} segments: " + ", ".join(p.name for p in pieces))
        return {'FINISHED'}


class OMFS_OT_ortho_rejoin(bpy.types.Operator):
    """Effacer les segments et revenir à l'os entier (les traits restent, pour ajuster et séparer de nouveau)"""
    bl_idname = "omfs.ortho_rejoin"
    bl_label = "Réunir"
    bl_options = {'REGISTER', 'UNDO'}

    which: bpy.props.EnumProperty(items=(('MAXILLA', "Maxillaire", ""), ('MANDIBLE', "Mandibule", "")))

    def execute(self, context):
        source = bone(context.scene, self.which)
        if source is not None:
            rejoin(context.scene, source)
        return {'FINISHED'}


# --- Properties ---------------------------------------------------------------------------------

def _is_bone(_self, obj):
    return obj.type == 'MESH' and obj.omfs.role == 'BONE' and not obj.omfs.made_by.startswith(MADE_BY)


class OMFS_OrthoCut(bpy.types.PropertyGroup):
    kind: bpy.props.StringProperty()
    source: bpy.props.PointerProperty(type=bpy.types.Object)
    direction: bpy.props.FloatVectorProperty(size=3)
    nerve_depth: bpy.props.FloatProperty()     # BSSO: mm the buccal cut goes in (to the nerve canal)
    nerve_found: bpy.props.BoolProperty()


def _default(scene, name):
    obj = scene.objects.get(name)
    return obj if obj is not None and obj.type == 'MESH' else None


class OMFS_OrthoScene(bpy.types.PropertyGroup):
    maxilla: bpy.props.PointerProperty(name="Maxillaire", type=bpy.types.Object, poll=_is_bone,
                                       description="L'os coupé par le Le Fort I (ex. « Maxillaire » de la segmentation)")
    mandible: bpy.props.PointerProperty(name="Mandibule", type=bpy.types.Object, poll=_is_bone,
                                        description="L'os coupé par les BSSO")


classes = (OMFS_OrthoCut, OMFS_OrthoScene, OMFS_OT_ortho_draw, OMFS_OT_ortho_stop, OMFS_OT_ortho_remove, OMFS_OT_ortho_separate,
           OMFS_OT_ortho_rejoin)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_ortho = bpy.props.PointerProperty(type=OMFS_OrthoCut)
    bpy.types.Scene.omfs_ortho_scene = bpy.props.PointerProperty(type=OMFS_OrthoScene)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    del bpy.types.Scene.omfs_ortho_scene
    del bpy.types.Object.omfs_ortho
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
