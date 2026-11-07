"""Occlusion seating (part 2): the lower arch is pulled onto the fixed upper arch by springs.

Like IPS CaseDesigner's virtual occlusion: the surgeon clicks corresponding points on both arches
(e.g. a cusp tip on the upper arch and the fossa it goes into on the lower arch). Each pair is a
spring. The solver moves the lower block, step by step (shown as an animation), so the springs
shorten, while the teeth stop at an overlap limit: one limit canine to canine (the two canine
springs mark that zone), another on the premolars and molars. Only the block's transform changes:
rigid, nothing touches the mesh data.

Contact is tested both ways (lower vertices against the upper surface and upper vertices against
the lower surface), with signed distances from the face normals: negative = overlap. Faces on the
open border of a scan are ignored, since the sign of a distance there means nothing.

The overlap map colours both arches by that signed distance (bands in params.py) and reports the
deepest overlap and the contact area right / anterior / left.
"""

import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, occlusion, params, registration

MAP_ATTRIBUTE = "Occlusion overlap"
MADE_BY = "Occlusion"
POINT_SIZE = 0.6        # mm, radius of the point spheres
SPRING_COILS = 8
SPRING_RADIUS = 0.4     # mm, zig-zag amplitude of the drawn springs
MAX_STEP = 1.0          # mm and degrees per iteration, so the animation stays smooth
CONVERGED = 1e-3        # mm per iteration
SIGN_MIN_COS = 0.3      # inside only if the point is this clearly behind the closest face
FIT_FRAMES = 40         # animation frames of the first phase (pure point fit)
MIN_SPREAD = 3.0        # mm: lower points closer than this to a single line are refused

# Colour bands, deepest overlap first (see the legend in the panel and params.py).
MAGENTA, RED, ORANGE, YELLOW = (1.0, 0.0, 1.0, 1.0), (0.9, 0.1, 0.1, 1.0), (1.0, 0.5, 0.0, 1.0), (1.0, 0.9, 0.1, 1.0)
GREEN, BLUE, GREY = (0.1, 0.8, 0.2, 1.0), (0.4, 0.7, 1.0, 1.0), (0.6, 0.6, 0.6, 1.0)


def legend():
    """[(icon, text)] for the panel, built from the parameters."""
    a, b, c = params.OCC_OVERLAP_STEPS
    contact, far = params.OCC_CONTACT, params.OCC_GAP_FAR
    return [
        ('STRIP_COLOR_07', f"Chevauchement > {c:g} mm"),
        ('STRIP_COLOR_01', f"Chevauchement {b:g} – {c:g} mm"),
        ('STRIP_COLOR_02', f"Chevauchement {a:g} – {b:g} mm"),
        ('STRIP_COLOR_03', f"Chevauchement 0 – {a:g} mm"),
        ('STRIP_COLOR_04', f"Contact (écart < {contact:g} mm)"),
        ('STRIP_COLOR_05', f"Écart {contact:g} – {far:g} mm"),
        ('STRIP_COLOR_09', f"Écart > {far:g} mm"),
    ]


def overlap_colours(d):
    """RGBA per signed distance (negative = overlap, NaN = nothing within reach)."""
    a, b, c = params.OCC_OVERLAP_STEPS
    overlap = -d
    conditions = [np.isnan(d), overlap > c, overlap > b, overlap > a, overlap > 0,
                  d < params.OCC_CONTACT, d < params.OCC_GAP_FAR]
    choices = [GREY, MAGENTA, RED, ORANGE, YELLOW, GREEN, BLUE]
    with np.errstate(invalid='ignore'):
        index = np.select(conditions, range(len(choices)), default=len(choices))
    return np.array(choices + [GREY])[index]


# --- Surfaces and signed distances -----------------------------------------------------

def border_polygons(mesh):
    """True for faces touching the open border of the mesh (an edge used by one face only)."""
    edge_of_loop = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", edge_of_loop)
    uses = np.bincount(edge_of_loop, minlength=len(mesh.edges))
    totals = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    poly_of_loop = np.repeat(np.arange(len(mesh.polygons)), totals)
    border = np.zeros(len(mesh.polygons), dtype=bool)
    border[poly_of_loop[uses[edge_of_loop] == 1]] = True
    return border


_surfaces = {}


def surface(obj):
    """(BVH in the object's local coordinates, border flag per triangle), built once per mesh."""
    mesh = obj.data
    key = (mesh.name, len(mesh.vertices), len(mesh.polygons))
    if key not in _surfaces:
        tris = registration.triangles(mesh)
        poly = np.empty(len(mesh.loop_triangles), dtype=np.int64)
        mesh.loop_triangles.foreach_get("polygon_index", poly)
        bvh = BVHTree.FromPolygons(registration.vertex_coords(obj, world=False).tolist(), tris.tolist())
        _surfaces[key] = (bvh, border_polygons(mesh)[poly])
    return _surfaces[key]


def signed_distances(points, target, max_distance):
    """For world points: signed distance to the target surface (negative = inside, i.e. overlap),
    the closest world point and the face normal there. NaN where nothing is within reach or the
    closest face is on the open border."""
    bvh, border = surface(target)
    m = target.matrix_world
    inv, rot = m.inverted(), m.to_3x3()
    n = len(points)
    d = np.full(n, np.nan)
    closest = np.full((n, 3), np.nan)
    normals = np.full((n, 3), np.nan)
    for i, p in enumerate(points):
        p = Vector(p)
        loc, nor, index, dist = bvh.find_nearest(inv @ p, max_distance)
        if loc is None or border[index]:
            continue
        loc, nor = m @ loc, (rot @ nor).normalized()
        # Inside only if clearly behind the face. When the closest point is on an edge, the face
        # normal says nothing about the sign (the point looks along the face): count it as outside.
        d[i] = -dist if dist > 1e-9 and (p - loc).dot(nor) < -SIGN_MIN_COS * dist else dist
        closest[i], normals[i] = loc, nor
    return d, closest, normals


def world_points(obj, local):
    m = np.array(obj.matrix_world)
    return local @ m[:3, :3].T + m[:3, 3]


_samples = {}


def samples(obj):
    """A fixed random subset of the vertices (local coordinates), the same at every iteration."""
    key = (obj.data.name, len(obj.data.vertices))
    if key not in _samples:
        co = registration.vertex_coords(obj, world=False)
        if len(co) > params.OCC_SAMPLES:
            co = co[np.random.default_rng(0).choice(len(co), params.OCC_SAMPLES, replace=False)]
        _samples[key] = co
    return _samples[key]


def contact_vertices(scene):
    """(lower, upper) local coordinates of every vertex near the other arch: random samples can miss
    the tip of a cusp, which is exactly what sets the overlap."""
    props = scene.omfs_occlusion
    found = []
    for obj, other in ((props.lower, props.upper), (props.upper, props.lower)):
        d = signed_distances(registration.vertex_coords(obj), other, params.OCC_FINISH_BAND)[0]
        found.append(registration.vertex_coords(obj, world=False)[~np.isnan(d)])
    return found


# --- Point pairs (springs) -------------------------------------------------------------

def occlusion_points(scene):
    return [o for o in scene.objects if o.omfs.made_by == MADE_BY and o.omfs.landmark.startswith("OCC_")]


CANINES = (   # (key, label): the two canine springs, weighted more than the others
    ('CR', "Canine droite"),
    ('CL', "Canine gauche"),
)
MIDLINE = ('ML', "Midline")   # upper to lower dental midline, an ordinary weight
NAMED_SPRINGS = CANINES + (MIDLINE,)


def pair_key(point):
    """'1', '2', ... for ordinary springs, 'CR' / 'CL' for the canine springs, 'ML' for the midline."""
    return point.omfs.landmark[5:]


def is_canine(point):
    return pair_key(point) in dict(CANINES)


def pair_label(point):
    key = pair_key(point)
    return dict(NAMED_SPRINGS).get(key, f"Ressort {key}")


def point_pairs(scene):
    """[(upper point, lower point)]: canines first, the midline, then the numbered springs in order.
    Each point is a child of its arch scan."""
    points = {o.omfs.landmark: o for o in occlusion_points(scene)}
    keys = {lm[5:] for lm in points}
    ordered = [k for k, _label in NAMED_SPRINGS if k in keys] + sorted((k for k in keys if k.isdigit()), key=int)
    return [(points[f"OCC_U{k}"], points[f"OCC_L{k}"]) for k in ordered
            if f"OCC_U{k}" in points and f"OCC_L{k}" in points]


def pair_weights(scene, canine_weight):
    return np.array([canine_weight if is_canine(up) else 1.0 for up, _low in point_pairs(scene)])


def next_pair_number(scene):
    return max((int(pair_key(o)) for o in occlusion_points(scene) if pair_key(o).isdigit()), default=0) + 1


def remove_pair(scene, key):
    for obj in [o for o in occlusion_points(scene) if o.omfs.landmark in (f"OCC_U{key}", f"OCC_L{key}")]:
        bpy.data.objects.remove(obj)


def add_point(scene, point_id, name, location, arch):
    obj = bpy.data.objects.new(name, None)
    occlusion.occlusion_collection(scene).objects.link(obj)
    obj.empty_display_type = 'SPHERE'
    obj.empty_display_size = POINT_SIZE
    obj.show_name = True
    obj.show_in_front = True
    obj.omfs.role = 'MARKER'
    obj.omfs.landmark = point_id
    obj.omfs.made_by = MADE_BY
    obj.parent = arch
    obj.matrix_parent_inverse = arch.matrix_world.inverted()
    obj.matrix_world = Matrix.Translation(location)
    locking.set_locked(obj, True)
    return obj


def spring_lengths(scene):
    return [(up.matrix_world.translation - low.matrix_world.translation).length
            for up, low in point_pairs(scene)]


def points_visible(scene):
    return any(not o.hide_get() for o in occlusion_points(scene))


def set_points_hidden(scene, hide):
    for obj in occlusion_points(scene):
        obj.hide_set(hide)


# --- Anterior zone (canine to canine) ---------------------------------------------------

def occlusal_axis(arch):
    """(centre, unit axis), world: the arch is flat along this axis (its direction is arbitrary)."""
    co = world_points(arch, samples(arch))
    centre = co.mean(axis=0)
    return centre, np.linalg.svd(co - centre, full_matrices=False)[2][2]


def zone_plane(scene, arch, tag, margin):
    """(point, normal towards the incisors), world, of the plane closing the anterior zone of one
    arch: through the two canine points of that arch and its occlusal axis, moved margin mm distally.
    None without both canine springs."""
    points = {o.omfs.landmark: o for o in occlusion_points(scene)}
    right, left = points.get(f"OCC_{tag}CR"), points.get(f"OCC_{tag}CL")
    if arch is None or right is None or left is None:
        return None
    right, left = np.array(right.matrix_world.translation), np.array(left.matrix_world.translation)
    centre, occlusal = occlusal_axis(arch)
    normal = np.cross(left - right, occlusal)
    normal /= np.linalg.norm(normal)
    middle = (left + right) / 2
    if (middle - centre) @ normal < 0:   # the canines are in front of the arch's centre
        normal = -normal
    return middle - normal * margin, normal


def zone_planes(scene):
    """(upper plane, lower plane). The upper point is the canine tip, so the distal slope of the canine
    needs a margin; the lower point is the canine / premolar embrasure, already the limit."""
    props = scene.omfs_occlusion
    return (zone_plane(scene, props.upper, 'U', params.OCC_CANINE_ZONE_MARGIN),
            zone_plane(scene, props.lower, 'L', 0.0))


def in_zone(plane, points):
    if plane is None:
        return np.zeros(len(points), dtype=bool)
    with np.errstate(invalid='ignore'):
        return (points - plane[0]) @ plane[1] > 0


def anterior_contacts(own_plane, other_plane, points, closest):
    """A contact is anterior as soon as one of its two sides is: grinding either side of it touches
    an anterior tooth."""
    return in_zone(own_plane, points) | in_zone(other_plane, closest)


def zone_ready(scene):
    return all(plane is not None for plane in zone_planes(scene))


def limits_of(props):
    """(anterior, posterior) overlap limits, mm."""
    return props.tolerance_anterior, props.tolerance_posterior


# --- Solver --------------------------------------------------------------------------

def skew(v):
    """Matrix S with S @ w = v x w."""
    x, y, z = v
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rigid_rows(points, normals, centre):
    """Rows of the linearised rigid motion (rotation, translation) along a normal at each point."""
    v = points - centre
    return np.hstack((np.cross(v, normals), normals))


def closing_motions(scene):
    """6 x 3 basis of the motions (rotation, translation) that only close the bite: no translation
    in the occlusal plane (across the arch or antero-posterior), no rotation about the occlusal axis
    (it would swing the incisors sideways). What is left: along the occlusal axis, and tilts.
    Across the arch = the line of the lower canine points, else the block's X axis."""
    props = scene.omfs_occlusion
    occlusal = occlusal_axis(props.lower)[1]
    points = {o.omfs.landmark: o.matrix_world.translation for o in occlusion_points(scene)}
    right, left = points.get("OCC_LCR"), points.get("OCC_LCL")
    if right is not None and left is not None:
        across = np.array(left - right)
    else:
        across = np.array(occlusion.find_block(scene, 'LOWER').matrix_world.to_3x3().col[0])
    across -= (across @ occlusal) * occlusal
    across /= np.linalg.norm(across)
    locked = np.zeros((3, 6))
    locked[0, :3] = occlusal
    locked[1, 3:] = across
    locked[2, 3:] = np.cross(occlusal, across)
    return np.linalg.svd(locked)[2][3:].T


def held_points(rows, d, limit):
    """The points near their overlap limit, as (rows, rhs): rows @ x >= rhs keeps each one within it."""
    with np.errstate(invalid='ignore'):
        held = d < -limit + params.OCC_LIMIT_BAND
    return rows[held], -limit[held] - d[held]


def solve_step(a, b, held_rows, held_rhs):
    """Least squares a x = b, with held_rows @ x >= held_rhs. Active set: the points that would pass
    their limit are held in turn, the worst first, stiff enough to win over the guidance; a held point
    the motion leaves behind is released, or it would glue the arches and block a tilt."""
    h, g = a.T @ a, a.T @ b
    held = np.zeros(len(held_rows), dtype=bool)
    x = np.linalg.solve(h, g)
    for _round in range(params.OCC_LIMIT_ROUNDS):
        excess = held_rhs - held_rows @ x
        released = held & (excess < 0)
        waiting = np.where(held, 0.0, excess)
        worst = np.argsort(waiting)[-6:]
        worst = worst[waiting[worst] > 1e-3]
        if not len(worst) and not released.any():
            break
        held[released] = False
        held[worst] = True
        rows, rhs = held_rows[held], held_rhs[held]
        k = params.OCC_LIMIT_STIFFNESS
        x = np.linalg.solve(h + k * rows.T @ rows, g + k * rows.T @ rhs)
    return x


def seat_step(scene, limits, spring_share=params.OCC_SPRING_SHARE / 100, canine_weight=params.OCC_CANINE_WEIGHT,
              vertices=None, hold_position=False):
    """One iteration: a small rigid motion of the lower block. Two guides, solved together:
    - springs pull each lower point a fraction of the way towards its upper point;
    - the magnet closes the gaps between the arches (within range), like pushing the models together.
    Springs and magnet share the guidance (spring_share vs 1 - spring_share); each group is weighted
    as a whole, so a thousand magnet points don't drown three springs. Within the springs, each
    canine spring counts canine_weight times an ordinary one.
    The teeth are not a third guide but a limit: no point goes deeper than the overlap limit of its
    zone, limits = (anterior, posterior), whatever the guides want.
    vertices = (lower, upper) local points to test, random samples by default.
    hold_position: the block cannot slide in the occlusal plane (see closing_motions), exactly: the
    solver only searches the other motions.
    Returns (translation mm, rotation deg) applied."""
    props = scene.omfs_occlusion
    upper, lower = props.upper, props.lower
    block = occlusion.find_block(scene, 'LOWER')
    centre = np.array(block.matrix_world.translation)
    groups = []   # (rows, rhs, total weight of the group)

    pairs = point_pairs(scene)
    if pairs:
        rows, rhs = [], []
        weights = pair_weights(scene, canine_weight)
        for (up, low), w in zip(pairs, np.sqrt(weights / weights.mean())):
            p, q = np.array(low.matrix_world.translation), np.array(up.matrix_world.translation)
            rows.append(w * np.hstack((-skew(p - centre), np.eye(3))))
            rhs.append(w * (q - p) * params.OCC_SPRING_STEP)
        groups.append((np.vstack(rows), np.concatenate(rhs), spring_share))

    # Lower vertices against the upper surface (move along the upper normal), and upper vertices
    # against the lower surface (move the lower surface against its own normal: signs flipped).
    upper_plane, lower_plane = zone_planes(scene)
    lower_local, upper_local = vertices or (samples(lower), samples(upper))
    lp = world_points(lower, lower_local)
    d1, c1, n1 = signed_distances(lp, upper, params.OCC_SEARCH_DISTANCE)
    up_ = world_points(upper, upper_local)
    d2, c2, n2 = signed_distances(up_, lower, params.OCC_SEARCH_DISTANCE)
    rows1, rows2 = rigid_rows(lp, n1, centre), rigid_rows(c2, n2, centre)
    with np.errstate(invalid='ignore'):
        near1 = (d1 > 0) & (d1 < params.OCC_MAGNET_RANGE)
        near2 = (d2 > 0) & (d2 < params.OCC_MAGNET_RANGE)

    magnet_rows = np.vstack((rows1[near1], rows2[near2]))
    magnet_rhs = np.concatenate((-d1[near1], d2[near2])) * params.OCC_MAGNET_STEP   # close the gap
    groups.append((magnet_rows, magnet_rhs, 1.0 - spring_share))

    a, b = [1e-3 * np.eye(6)], [np.zeros(6)]
    for rows, rhs, weight in groups:
        if len(rows) and weight > 0:
            w = np.sqrt(weight / len(rows))
            a.append(w * rows)
            b.append(w * rhs)

    anterior, posterior = limits
    limit1 = np.where(anterior_contacts(lower_plane, upper_plane, lp, c1), anterior, posterior)
    limit2 = np.where(anterior_contacts(upper_plane, lower_plane, up_, c2), anterior, posterior)
    held1, rhs1 = held_points(rows1, d1, limit1)
    held2, rhs2 = held_points(-rows2, d2, limit2)
    free = closing_motions(scene) if hold_position else np.eye(6)
    x = free @ solve_step(np.vstack(a) @ free, np.concatenate(b), np.vstack((held1, held2)) @ free,
                          np.concatenate((rhs1, rhs2)))

    omega, t = Vector(x[:3]), Vector(x[3:])
    scale = min(1.0, MAX_STEP / max(t.length, 1e-12), np.radians(MAX_STEP) / max(omega.length, 1e-12))
    omega, t = omega * scale, t * scale   # together, so the motion keeps its direction
    angle = omega.length
    rotation = Matrix.Rotation(angle, 4, omega.normalized()) if angle > 1e-12 else Matrix()
    c = Vector(centre)
    motion = Matrix.Translation(c + t) @ rotation @ Matrix.Translation(-c)
    block.matrix_world = registration.as_rigid(motion @ block.matrix_world)
    bpy.context.view_layer.update()   # so the arch and its points report their new position
    return t.length, np.degrees(angle)


def converged(step):
    shift, angle = step
    return shift < CONVERGED and angle < CONVERGED


def settled(step):
    """The last step is too small to matter clinically (it may still vibrate between two contacts)."""
    shift, angle = step
    return shift < params.OCC_SETTLED and angle < params.OCC_SETTLED


def pair_arrays(scene):
    pairs = point_pairs(scene)
    lower = np.array([low.matrix_world.translation for _up, low in pairs])
    upper = np.array([up.matrix_world.translation for up, _low in pairs])
    return lower, upper


def points_spread(scene):
    """Second largest spread of the lower points (mm): near 0 means they are on one line, and the
    rotation about that line is undetermined."""
    lower, _upper = pair_arrays(scene)
    return float(np.linalg.svd(lower - lower.mean(axis=0), compute_uv=False)[1]) if len(lower) >= 3 else 0.0


def point_fit(scene, canine_weight=params.OCC_CANINE_WEIGHT):
    """World matrix of the lower block after a pure least-squares fit of the springs (Kabsch,
    canines weighted): gets there from any starting pose, even a scan exported upside down."""
    block = occlusion.find_block(scene, 'LOWER')
    lower, upper = pair_arrays(scene)
    fit = Matrix(registration.rigid_fit(lower, upper, pair_weights(scene, canine_weight)).tolist())
    return registration.as_rigid(fit @ block.matrix_world)


def seat(scene, limits, spring_share=params.OCC_SPRING_SHARE / 100, canine_weight=params.OCC_CANINE_WEIGHT,
         max_iterations=params.OCC_MAX_ITERATIONS, from_here=False):
    """Run the solver to the end (no animation): point fit, then springs and magnet up to the overlap
    limits. from_here skips the point fit and holds the block against any slide in the occlusal
    plane, to keep a position adjusted by hand.
    Returns (contact iterations, True if it settled)."""
    if not from_here:
        block = occlusion.find_block(scene, 'LOWER')
        block.matrix_world = point_fit(scene, canine_weight)
        bpy.context.view_layer.update()
    for i in range(1, max_iterations + 1):
        if converged(seat_step(scene, limits, spring_share, canine_weight, hold_position=from_here)):
            break
    vertices = contact_vertices(scene)
    for _i in range(params.OCC_FINISH_ITERATIONS):
        step = seat_step(scene, limits, spring_share, canine_weight, vertices, from_here)
        if converged(step):
            break
    return i, settled(step)


# --- Overlap map ---------------------------------------------------------------------

def vertex_areas(obj):
    """Area around each vertex (a third of its triangles), mm²."""
    co = registration.vertex_coords(obj, world=False)
    tris = registration.triangles(obj.data)
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    out = np.zeros(len(co))
    for k in range(3):
        np.add.at(out, tris[:, k], area / 3)
    return out


def robust_overlap(d, area):
    """Deepest overlap reached over at least OCC_OVERLAP_MIN_AREA mm², so a single spiky vertex
    (a scan artefact) doesn't set the maximum."""
    with np.errstate(invalid='ignore'):
        inside = d < 0
    if not inside.any():
        return 0.0
    order = np.argsort(d[inside])                      # deepest first
    cumulative = np.cumsum(area[inside][order])
    k = min(np.searchsorted(cumulative, params.OCC_OVERLAP_MIN_AREA), len(order) - 1)
    return float(-d[inside][order][k])


def write_overlap_map(scene):
    """Colour both arches by their signed distance to the other one; store the summary."""
    props = scene.omfs_occlusion
    upper, lower = props.upper, props.lower
    planes = dict(zip((upper, lower), zone_planes(scene)))
    deepest = {True: 0.0, False: 0.0}   # anterior, posterior
    for obj, other in ((lower, upper), (upper, lower)):
        points = registration.vertex_coords(obj)
        d, closest, _normals = signed_distances(points, other, params.OCC_SEARCH_DISTANCE)
        mesh = obj.data
        attr = mesh.color_attributes.get(MAP_ATTRIBUTE) or mesh.color_attributes.new(
            MAP_ATTRIBUTE, 'FLOAT_COLOR', 'POINT')
        attr.data.foreach_set("color", overlap_colours(d).ravel())
        mesh.color_attributes.active_color = attr
        area = vertex_areas(obj)
        anterior = anterior_contacts(planes[obj], planes[other], points, closest)
        for zone, part in ((True, anterior), (False, ~anterior)):
            deepest[zone] = max(deepest[zone], robust_overlap(d[part], area[part]))
        if obj is lower:
            contact_areas(scene, lower, points, d, anterior)
    props.max_overlap_anterior, props.max_overlap_posterior = deepest[True], deepest[False]
    props.max_overlap = max(deepest.values())
    props.map_state = repr(map_key(scene))


def midline_offset(scene, points):
    """Signed distance (mm, + = patient's left) of world points from the midline: from the lower canine
    points if placed, else from the block centre along its X axis."""
    canines = {o.omfs.landmark: o.matrix_world.translation for o in occlusion_points(scene)}
    right, left = canines.get("OCC_LCR"), canines.get("OCC_LCL")
    if right is not None and left is not None:
        return (points - np.array((right + left) / 2)) @ np.array((left - right).normalized())
    m = occlusion.find_block(scene, 'LOWER').matrix_world
    return (points - np.array(m.translation)) @ np.array(m.to_3x3().col[0])   # block X = patient left


def contact_areas(scene, lower, points, d, anterior):
    """Contact area (mm²) on the lower arch: patient's right, anterior, left. Anterior = the canine to
    canine zone, or a band around the midline without the canine springs."""
    props = scene.omfs_occlusion
    x = midline_offset(scene, points)
    if not zone_ready(scene):
        anterior = np.abs(x) <= params.OCC_ANTERIOR_HALF_WIDTH
    with np.errstate(invalid='ignore'):
        touching = d < params.OCC_CONTACT
    area = vertex_areas(lower)
    props.area_right = float(area[touching & ~anterior & (x < 0)].sum())
    props.area_anterior = float(area[touching & anterior].sum())
    props.area_left = float(area[touching & ~anterior & (x > 0)].sum())


def tripod(props):
    """Stable occlusion: contacts on the right, anterior and left at once."""
    return min(props.area_right, props.area_anterior, props.area_left) >= params.OCC_TRIPOD_MIN_AREA


def map_key(scene):
    """What the map depends on: both arch transforms."""
    props = scene.omfs_occlusion
    return tuple(tuple(round(v, 4) for row in obj.matrix_world for v in row) for obj in (props.upper, props.lower))


def map_is_current(scene):
    props = scene.omfs_occlusion
    return props.map_state == repr(map_key(scene))


def set_map_shading(context, on):
    for area in (context.screen.areas if context.screen else ()):
        if area.type == 'VIEW_3D':
            area.spaces.active.shading.color_type = 'VERTEX' if on else 'OBJECT'


# Refresh the map shortly after the lower block stops moving (e.g. after a G / R by hand).
_seating = False
_last_change = 0.0


def _refresh_map():
    scene = bpy.context.scene
    props = scene.omfs_occlusion
    if _seating or time.monotonic() - _last_change < 0.4:
        return 0.2   # still moving: check again soon
    if props.show_map and props.upper and props.lower and not map_is_current(scene):
        write_overlap_map(scene)
    return None


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    global _last_change
    props = scene.omfs_occlusion
    if _seating or not (props.show_map and props.upper and props.lower):
        return
    if any(u.is_updated_transform and isinstance(u.id, bpy.types.Object) for u in depsgraph.updates):
        if not map_is_current(scene):
            _last_change = time.monotonic()
            if not bpy.app.timers.is_registered(_refresh_map):
                bpy.app.timers.register(_refresh_map, first_interval=0.4)


# --- Springs in the viewport ---------------------------------------------------------

def spring_coords(p, q):
    """Zig-zag line from p to q."""
    axis = q - p
    if axis.length < 1e-6:
        return [p, q]
    side = axis.orthogonal().normalized() * SPRING_RADIUS
    n = 2 * SPRING_COILS
    return [p] + [p + axis * (k / n) + side * (1 if k % 2 else -1) for k in range(1, n)] + [q]


def spring_colour(length):
    if length < params.OCC_CONTACT * 2:
        return GREEN
    return YELLOW if length < 2.0 else ORANGE


def _draw_springs():
    scene = bpy.context.scene
    if not points_visible(scene):
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    gpu.state.depth_test_set('NONE')   # in front of the teeth, like the points
    for up, low in point_pairs(scene):
        p, q = low.matrix_world.translation, up.matrix_world.translation
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": spring_coords(p, q)})
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 4.0 if is_canine(up) else 2.0)
        shader.uniform_float("color", spring_colour((q - p).length))
        batch.draw(shader)
    _draw_zone_limits(scene, shader, region)


def _draw_zone_limits(scene, shader, region):
    """A line across each arch, where the anterior zone ends."""
    points = {o.omfs.landmark: o.matrix_world.translation for o in occlusion_points(scene)}
    for tag, plane in zip("UL", zone_planes(scene)):
        if plane is None:
            continue
        shift = Vector(plane[0]) - (points[f"OCC_{tag}CR"] + points[f"OCC_{tag}CL"]) / 2
        ends = [points[f"OCC_{tag}CR"] + shift, points[f"OCC_{tag}CL"] + shift]
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": ends})
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 2.0)
        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch.draw(shader)


_draw_handle = None


# --- Operators -----------------------------------------------------------------------

def guided(scene):
    """Something can move the lower block: an active spring, or the magnet with the arches in range."""
    props = scene.omfs_occlusion
    if point_pairs(scene) and props.spring_share > 0:
        return True
    if props.spring_share >= 100:
        return False
    d = signed_distances(world_points(props.lower, samples(props.lower)), props.upper, params.OCC_MAGNET_RANGE)[0]
    return not np.isnan(d).all()


def arches_ready(scene):
    props = scene.omfs_occlusion
    return props.upper is not None and props.lower is not None and occlusion.find_block(scene, 'LOWER')


class OMFS_OT_add_occlusion_pair(bpy.types.Operator):
    """Cliquez un point sur l'arcade du haut (ex. pointe de cuspide), puis le point correspondant sur
    l'arcade du bas (ex. fosse où elle doit s'asseoir). Échap / clic droit : annuler"""
    bl_idname = "omfs.add_occlusion_pair"
    bl_label = "Ajouter une paire"
    bl_options = {'REGISTER', 'UNDO'}

    key: bpy.props.StringProperty(options={'SKIP_SAVE'},
                                  description="Vide : nouveau ressort numéroté. CR / CL : ressort canin droit / gauche. ML : Midline")

    @classmethod
    def description(cls, context, properties):
        if properties.key in dict(CANINES):
            return (f"{dict(CANINES)[properties.key]} : cliquez la pointe de la canine du haut, puis l'embrasure "
                    "canine / 1re prémolaire du bas où elle doit s'accoter (classe I canine). Ce ressort pèse plus "
                    "que les autres (Poids des canines). Refaire remplace l'ancien")
        if properties.key == MIDLINE[0]:
            return ("Midline : cliquez le milieu interincisif du haut, puis le milieu interincisif du bas. "
                    "Pèse comme un ressort ordinaire. Refaire remplace l'ancien")
        return cls.__doc__

    def invoke(self, context, event):
        if not arches_ready(context.scene):
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return {'CANCELLED'}
        props = context.scene.omfs_occlusion
        self.arches = (props.upper, props.lower)
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.key:
            self.label = dict(NAMED_SPRINGS)[self.key]
        else:
            self.key = str(next_pair_number(context.scene))
            self.label = f"Ressort {self.key}"
        self.upper_point = None
        set_points_hidden(context.scene, False)
        self.prompt()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self):
        if self.key in dict(CANINES):
            which = ("la pointe de la canine du HAUT" if self.upper_point is None
                     else "l'embrasure canine / 1re prémolaire du BAS")
        elif self.key == MIDLINE[0]:
            which = "le milieu interincisif du " + ("HAUT" if self.upper_point is None else "BAS")
        else:
            which = "l'arcade du HAUT" if self.upper_point is None else "l'arcade du BAS (point correspondant)"
        self.area.header_text_set(f"{self.label} : cliquez sur {which}   |   Échap / clic droit : annuler")

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self.upper_point is not None:
                bpy.data.objects.remove(self.upper_point)
            self.finish(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
            if under_mouse is None:
                return {'PASS_THROUGH'}
            arch = self.arches[0] if self.upper_point is None else self.arches[1]
            location = landmarks.object_hit(context, *under_mouse, arch)
            if location is None:
                self.report({'WARNING'}, f"Cliquez sur {arch.name}")
                return {'RUNNING_MODAL'}
            short = {'CR': "Can. D", 'CL': "Can. G", 'ML': "Midline"}.get(self.key, self.key)
            if self.upper_point is None:
                remove_pair(context.scene, self.key)   # redoing a canine spring replaces it
                self.upper_point = add_point(context.scene, f"OCC_U{self.key}", f"Haut {short}", location, arch)
                self.prompt()
                return {'RUNNING_MODAL'}
            add_point(context.scene, f"OCC_L{self.key}", f"Bas {short}", location, arch)
            self.finish(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class OMFS_OT_delete_occlusion_pair(bpy.types.Operator):
    """Supprimer ce ressort (ses deux points)"""
    bl_idname = "omfs.delete_occlusion_pair"
    bl_label = "Supprimer le ressort"
    bl_options = {'REGISTER', 'UNDO'}

    key: bpy.props.StringProperty()

    def execute(self, context):
        remove_pair(context.scene, self.key)
        return {'FINISHED'}


class OMFS_OT_toggle_occlusion_points(bpy.types.Operator):
    """Afficher ou cacher les points et les ressorts"""
    bl_idname = "omfs.toggle_occlusion_points"
    bl_label = "Afficher / cacher les ressorts"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        set_points_hidden(context.scene, points_visible(context.scene))
        return {'FINISHED'}


class OMFS_OT_seat_occlusion(bpy.types.Operator):
    """L'arcade du bas vient s'asseoir sur l'arcade du haut fixe : les ressorts guident (part réglable),
    l'aimant rapproche les arcades jusqu'aux contacts stables, et les dents s'arrêtent au chevauchement
    permis de chaque zone. Animé dans la vue ; Échap : annuler et revenir au départ"""
    bl_idname = "omfs.seat_occlusion"
    bl_label = "Asseoir l'occlusion"
    bl_options = {'REGISTER', 'UNDO'}

    from_here: bpy.props.BoolProperty(options={'SKIP_SAVE'},
                                      description="Partir de la position actuelle, sans ajustement aux points et sans glissement")

    @classmethod
    def description(cls, context, properties):
        if properties.from_here:
            return ("Rasseoir depuis la position actuelle (ex. après un ajustement à la main) : seulement les "
                    "derniers millimètres, avec les ressorts présents, l'aimant et les limites. Le glissement "
                    "latéral et antéro-postérieur est bloqué : l'arcade ne fait que se fermer")
        return cls.__doc__

    def ready(self, context):
        if not arches_ready(context.scene):
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return False
        if self.from_here:
            if not guided(context.scene):
                self.report({'ERROR'}, "Rien ne guide l'assise : ajoutez un ressort, ou rapprochez les arcades à moins "
                                       f"de {params.OCC_MAGNET_RANGE:g} mm avec une part des ressorts sous 100 %")
                return False
            return True
        if len(point_pairs(context.scene)) < 3:
            self.report({'ERROR'}, "Placez au moins 3 paires de points")
            return False
        if points_spread(context.scene) < MIN_SPREAD:
            self.report({'ERROR'}, "Points presque alignés : étalez-les (ex. incisives + molaires gauche et droite)")
            return False
        return True

    def report_end(self, settled):
        if not settled:
            self.report({'WARNING'}, "Assise non stabilisée : relancez « Reprendre d'ici » ou vérifiez les ressorts")

    def execute(self, context):   # scripts and tests: no animation
        if not self.ready(context):
            return {'CANCELLED'}
        scene = context.scene
        props = scene.omfs_occlusion
        iterations, settled = seat(scene, limits_of(props), props.spring_share / 100, props.canine_weight,
                                   from_here=self.from_here)
        write_overlap_map(scene)
        self.report({'INFO'}, f"Assise : {iterations} itérations")
        self.report_end(settled)
        return {'FINISHED'}

    def invoke(self, context, event):
        global _seating
        if not self.ready(context):
            return {'CANCELLED'}
        self.block = occlusion.find_block(context.scene, 'LOWER')
        self.start = self.block.matrix_world.copy()
        # Phase 1: glide to the point fit (none from here). Phase 2: guides up to the limits.
        self.frame = FIT_FRAMES if self.from_here else 0
        self.fit = self.start if self.from_here else point_fit(context.scene, context.scene.omfs_occlusion.canine_weight)
        self.iteration = 0
        self.vertices = None
        _seating = True
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        self.timer = context.window_manager.event_timer_add(0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        global _seating
        _seating = False
        context.window_manager.event_timer_remove(self.timer)
        if self.area:
            self.area.header_text_set(None)

    def modal(self, context, event):
        scene = context.scene
        if event.type == 'ESC' and event.value == 'PRESS':
            self.block.matrix_world = self.start
            self.finish(context)
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self.frame < FIT_FRAMES:
            self.frame += 1
            f = self.frame / FIT_FRAMES
            self.block.matrix_world = self.start.lerp(self.fit, f * f * (3 - 2 * f))   # ease in / out
            if self.area:
                self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        self.iteration += 1
        props = scene.omfs_occlusion
        step = seat_step(scene, limits_of(props), props.spring_share / 100, props.canine_weight, self.vertices,
                         self.from_here)
        longest = max(spring_lengths(scene), default=0.0)
        if self.area:
            self.area.header_text_set(f"Assise : itération {self.iteration}, ressort le plus long "
                                      f"{longest:.2f} mm   |   Échap : annuler")
            self.area.tag_redraw()
        if self.vertices is None and (converged(step) or self.iteration >= params.OCC_MAX_ITERATIONS):
            self.vertices = contact_vertices(scene)   # last iterations: every vertex near the contacts
            self.last_iteration = self.iteration + params.OCC_FINISH_ITERATIONS
            return {'RUNNING_MODAL'}
        if self.vertices is not None and (converged(step) or self.iteration >= self.last_iteration):
            self.finish(context)
            self.report_end(settled(step))
            write_overlap_map(scene)
            scene.omfs_occlusion.show_map = True
            set_map_shading(context, True)
            return {'FINISHED'}
        return {'RUNNING_MODAL'}


class OMFS_OT_toggle_overlap_map(bpy.types.Operator):
    """Colorer les deux arcades selon le chevauchement / l'écart en mm (légende dans le panneau)"""
    bl_idname = "omfs.toggle_overlap_map"
    bl_label = "Carte de chevauchement"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if not arches_ready(scene):
            return {'CANCELLED'}
        props = scene.omfs_occlusion
        props.show_map = not props.show_map
        if props.show_map:
            write_overlap_map(scene)
        set_map_shading(context, props.show_map)   # off: back to the role colours
        return {'FINISHED'}


class OMFS_OT_refresh_overlap_map(bpy.types.Operator):
    """Recalculer la carte et les mesures pour la position actuelle"""
    bl_idname = "omfs.refresh_overlap_map"
    bl_label = "Actualiser la carte"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not arches_ready(context.scene):
            return {'CANCELLED'}
        write_overlap_map(context.scene)
        return {'FINISHED'}


classes = (OMFS_OT_add_occlusion_pair, OMFS_OT_delete_occlusion_pair, OMFS_OT_toggle_occlusion_points,
           OMFS_OT_seat_occlusion, OMFS_OT_toggle_overlap_map, OMFS_OT_refresh_overlap_map)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_springs, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    if bpy.app.timers.is_registered(_refresh_map):
        bpy.app.timers.unregister(_refresh_map)
    bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
