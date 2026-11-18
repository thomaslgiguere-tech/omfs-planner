"""Occlusion seating (part 2): the upper arch is pulled onto the fixed lower arch by springs.

Like IPS CaseDesigner's virtual occlusion: the surgeon clicks corresponding points on both arches
(e.g. a cusp tip on the upper arch and the fossa it goes into on the lower arch). Each pair is a
spring. The lower arch never moves (surgeon, 2026-11-09: always the upper onto the lower, it is
more visual). The solver moves the upper block, or each upper segment on its own (segmental
surgery: its angles are the surgeon's, the seating only translates it, occlusion_segments.py) so the
springs shorten, while the teeth stop at an overlap limit: one limit canine to canine (the two canine
springs mark that zone), another on the premolars and molars. Only the block's transform changes:
rigid, nothing touches the mesh data.

Speed (2026-11-14): each step aims straight at the equilibrium (Gauss-Newton) and is shortened only if
it does not lower the energy (line search), so a seating takes a handful of steps; distances are read
from a grid built once per arch (exact again near the contacts); the operator computes the whole
seating first, then plays it back at an even speed.

Contact is tested both ways (lower vertices against the upper surface and upper vertices against
the lower surface), with signed distances from the face normals: negative = overlap. Faces on the
open border of a scan are ignored, since the sign of a distance there means nothing.

The overlap map colours both arches by that signed distance (bands in params.py) and reports the
deepest overlap and the contact area right / anterior / left.
"""

import time
from collections import namedtuple

import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, occlusion, params, registration

MAP_ATTRIBUTE = "Occlusion overlap"
MADE_BY = "Occlusion"
POINT_SIZE = 0.6        # mm, radius of the point spheres
SPRING_COILS = 8
SPRING_RADIUS = 0.4     # mm, zig-zag amplitude of the drawn springs
CONVERGED = 1e-3        # mm and degrees: a step this small ends the seating
SIGN_MIN_COS = 0.3      # inside only if the point is this clearly behind the closest face
FIT_FRAMES = 40         # playback: at most this many frames per stretch (the point fit glide)
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


def nearest_signed(target, local, max_distance):
    """Exact, point by point (BVH): for points in the target's local coordinates, (signed distance,
    closest point, face normal), local. NaN where nothing is within reach or the closest face is on
    the open border."""
    bvh, border = surface(target)
    n = len(local)
    d = np.full(n, np.nan)
    closest = np.full((n, 3), np.nan)
    normals = np.full((n, 3), np.nan)
    find = bvh.find_nearest
    for i, p in enumerate(local):
        p = Vector(p)
        loc, nor, index, dist = find(p, max_distance)
        if loc is None or border[index]:
            continue
        # Inside only if clearly behind the face. When the closest point is on an edge, the face
        # normal says nothing about the sign (the point looks along the face): count it as outside.
        d[i] = -dist if dist > 1e-9 and (p - loc).dot(nor) < -SIGN_MIN_COS * dist else dist
        closest[i], normals[i] = loc, nor
    return d, closest, normals


def dilate(mask, steps):
    """Grow a 3D boolean grid by steps cells (6-neighbourhood)."""
    for _ in range(steps):
        grown = mask.copy()
        for axis in range(3):
            lead = (slice(None),) * axis + (slice(1, None),)
            lag = (slice(None),) * axis + (slice(None, -1),)
            grown[lead] |= mask[lag]
            grown[lag] |= mask[lead]
        mask = grown
    return mask


def surface_points(co, tris, spacing):
    """The vertices plus points spread over the triangles longer than spacing, so no part of the
    surface is missed (large faces of a coarse mesh)."""
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], axis=0)
    count = np.ceil(longest / spacing).astype(int)
    found = [co]
    for k in np.unique(count[count > 1]):
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')
        keep = i + j <= k
        u, v = i[keep] / k, j[keep] / k
        sel = count == k
        found.append((a[sel, None] + u[:, None] * (b - a)[sel, None] + v[:, None] * (c - a)[sel, None]).reshape(-1, 3))
    return np.concatenate(found)


_grids = {}


def distance_grid(obj):
    """(distance, normal, closest, origin, cell): at the nodes of a regular grid (the object's local
    coordinates, float32), what nearest_signed gives there (signed distance, face normal, closest point),
    built once per mesh within OCC_SEARCH_DISTANCE of the surface (NaN farther, and where the closest face
    is on the open border). Then thousands of points are read in one go, by interpolation, instead of being
    searched point by point. In Blender terms: a signed distance field, what Mesh to Volume / Volume to
    Mesh use internally; Meshmixer has no equivalent."""
    mesh = obj.data
    key = (mesh.name, len(mesh.vertices), len(mesh.polygons))
    if key not in _grids:
        cell, reach = params.OCC_GRID_CELL, params.OCC_SEARCH_DISTANCE
        co = registration.vertex_coords(obj, world=False)
        origin = co.min(axis=0) - reach - 2 * cell
        shape = tuple(np.ceil((co.max(axis=0) + reach + 2 * cell - origin) / cell).astype(int) + 1)
        near = np.zeros(shape, dtype=bool)
        points = surface_points(co, registration.triangles(mesh), cell)
        near[tuple(np.rint((points - origin) / cell).astype(int).T)] = True
        nodes = tuple(np.argwhere(dilate(near, int(np.ceil(reach / cell)) + 1)).T)
        d, closest, normals = nearest_signed(obj, np.stack(nodes, axis=1) * cell + origin, reach + cell)
        grids = (np.full(shape, np.nan, dtype=np.float32), np.full(shape + (3,), np.nan, dtype=np.float32),
                 np.full(shape + (3,), np.nan, dtype=np.float32))
        for grid, values in zip(grids, (d, normals, closest)):
            grid[nodes] = values
        _grids[key] = grids + (origin, cell)
    return _grids[key]


def grid_lookup(obj, local):
    """(signed distance, face normal, closest point) at local points, by trilinear interpolation of the
    distance grid. NaN outside the grid or next to a NaN node."""
    distance, normal, closest, origin, cell = distance_grid(obj)
    f = (local - origin) / cell
    i = np.floor(f).astype(np.int64)
    outside = np.any((i < 0) | (i >= np.array(distance.shape) - 1), axis=1)
    i = np.clip(i, 0, np.array(distance.shape) - 2)
    t = f - i
    d, n, c = np.zeros(len(local)), np.zeros((len(local), 3)), np.zeros((len(local), 3))
    for corner in np.ndindex(2, 2, 2):
        at = (i[:, 0] + corner[0], i[:, 1] + corner[1], i[:, 2] + corner[2])
        w = np.prod([t[:, k] if corner[k] else 1 - t[:, k] for k in range(3)], axis=0)
        d += w * distance[at]
        n += w[:, None] * normal[at]
        c += w[:, None] * closest[at]
    d[outside] = np.nan
    return d, n, c


def signed_distances(points, target, max_distance, matrix=None):
    """For world points: signed distance to the target surface (negative = inside, i.e. overlap),
    the closest world point and the face normal there. NaN where nothing is within reach or the
    closest face is on the open border. matrix: the target's world matrix if not where it is now
    (a trial position of the solver).
    Read from the distance grid, and recomputed exactly (point by point) closer than OCC_EXACT_BAND,
    where the contacts and the overlap limits are decided; beyond the grid's depth, all exact."""
    m = np.array(target.matrix_world) if matrix is None else np.asarray(matrix)
    inv = np.linalg.inv(m)
    local = points @ inv[:3, :3].T + inv[:3, 3]
    if max_distance > params.OCC_SEARCH_DISTANCE:
        d, closest, normals = nearest_signed(target, local, max_distance)
    else:
        d, normals, closest = grid_lookup(target, local)
        with np.errstate(invalid='ignore'):
            exact = np.abs(d) < params.OCC_EXACT_BAND
            far = ~(np.abs(d) <= max_distance)
        if exact.any():
            d[exact], closest[exact], normals[exact] = nearest_signed(target, local[exact], max_distance)
        d[far & ~exact], closest[far & ~exact], normals[far & ~exact] = np.nan, np.nan, np.nan
    closest = closest @ m[:3, :3].T + m[:3, 3]
    normals = normals @ m[:3, :3].T
    with np.errstate(invalid='ignore'):
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]   # interpolated: unit again
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


def contact_vertices(job):
    """(mobile, fixed) local coordinates of every vertex near the other arch (read from the distance grid,
    which is precise enough to choose them): random samples can miss the tip of a cusp, which is exactly
    what sets the overlap."""
    found = []
    for obj, other in ((job.mobile, job.fixed), (job.fixed, job.mobile)):
        local = registration.vertex_coords(obj, world=False)
        m = np.linalg.inv(np.array(other.matrix_world)) @ np.array(obj.matrix_world)
        d = grid_lookup(other, local @ m[:3, :3].T + m[:3, 3])[0]
        with np.errstate(invalid='ignore'):
            near = local[np.abs(d) < params.OCC_FINISH_BAND]
        # one vertex per OCC_FINISH_SPACING cube: a scan has several per 0.1 mm, and a cusp tip missed by
        # 0.15 mm sits about 0.01 mm lower
        cells = np.floor(near / params.OCC_FINISH_SPACING).astype(np.int64)
        found.append(near[np.unique(cells, axis=0, return_index=True)[1]])
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


def pair_weights(pairs, canine_weight):
    return np.array([canine_weight if is_canine(point) else 1.0 for point, _other in pairs])


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


# --- What moves onto what ---------------------------------------------------------------

# One seating: the mobile object (and its block) is pulled onto the fixed one. pairs = [(point on the
# fixed side, point on the mobile side)]; the planes close the anterior zone of each side.
Seating = namedtuple("Seating", "fixed mobile block pairs fixed_plane mobile_plane")


def upper_parts(scene):
    """What stands for the upper arch: its segments once separated (segmental surgery), else the scan."""
    from . import occlusion_segments
    return occlusion_segments.segments(scene) or [scene.omfs_occlusion.upper]


def seatings(scene):
    """The seatings to run: the lower arch is always fixed. Whole arch: the upper block onto the lower.
    Segmented upper arch: each segment onto the lower, independently (the order does not matter)."""
    from . import occlusion_segments
    props = scene.omfs_occlusion
    upper_plane, lower_plane = zone_planes(scene)
    pairs = point_pairs(scene)
    parts = occlusion_segments.segments(scene)
    if not parts:
        return [Seating(props.lower, props.upper, occlusion.find_block(scene, 'UPPER'),
                        [(low, up) for up, low in pairs], lower_plane, upper_plane)]
    return [Seating(props.lower, part, occlusion_segments.block_of(part),
                    [(low, up) for up, low in pairs if up.parent is part], lower_plane, upper_plane)
            for part in parts]


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


def closing_motions(scene, job):
    """6 x 3 basis of the motions (rotation, translation) that only close the bite: no translation
    in the occlusal plane (across the arch or antero-posterior), no rotation about the occlusal axis
    (it would swing the incisors sideways). What is left: along the occlusal axis, and tilts.
    Across the arch = the line of the lower canine points, else the block's X axis."""
    occlusal = occlusal_axis(job.mobile)[1]
    points = {o.omfs.landmark: o.matrix_world.translation for o in occlusion_points(scene)}
    right, left = points.get("OCC_LCR"), points.get("OCC_LCL")
    if right is not None and left is not None:
        across = np.array(left - right)
    else:
        across = np.array(job.block.matrix_world.to_3x3().col[0])
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


def spring_fade(scene, pairs, spring_share):
    """Per spring, the share of its pull it keeps: 1 when longer than the release distance, falling
    linearly to OCC_SPRING_RESIDUAL at length 0. All 1 with springs only (100 %) or no release."""
    release = scene.omfs_occlusion.spring_release
    if release <= 0 or spring_share >= 1.0:
        return np.ones(len(pairs))
    lengths = np.array([(f.matrix_world.translation - m.matrix_world.translation).length for f, m in pairs])
    r = params.OCC_SPRING_RESIDUAL
    return r + (1 - r) * np.clip(lengths / release, 0.0, 1.0)


# Where both arches are, for the test points: (mobile points, d1, c1, n1, fixed points, d2, c2, n2), world.
# d1: mobile points against the fixed surface; d2: fixed points against the mobile surface. magnet1 / magnet2:
# the points the magnet pulls (the samples); the others (every vertex near the contacts) only check the limits.
Contacts = namedtuple("Contacts", "mobile_points d1 c1 n1 fixed_points d2 c2 n2 magnet1 magnet2")


def test_points(job, extra=None):
    """(mobile local points, fixed local points, magnet mask mobile, magnet mask fixed): the random samples,
    plus extra = (mobile, fixed) points that only check the overlap limits."""
    mobile, fixed = samples(job.mobile), samples(job.fixed)
    extra = extra or (np.zeros((0, 3)), np.zeros((0, 3)))
    return (np.vstack((mobile, extra[0])), np.vstack((fixed, extra[1])),
            np.arange(len(mobile) + len(extra[0])) < len(mobile), np.arange(len(fixed) + len(extra[1])) < len(fixed))


def measure(job, vertices, motion=None):
    """Contacts with the mobile block moved by motion (4 x 4, world) from where it is now (None: here)."""
    mobile_local, fixed_local, magnet1, magnet2 = vertices
    mobile = np.array(job.mobile.matrix_world)
    if motion is not None:
        mobile = motion @ mobile
    mp = mobile_local @ mobile[:3, :3].T + mobile[:3, 3]
    d1, c1, n1 = signed_distances(mp, job.fixed, params.OCC_SEARCH_DISTANCE)
    fp = world_points(job.fixed, fixed_local)
    d2, c2, n2 = signed_distances(fp, job.mobile, params.OCC_SEARCH_DISTANCE, mobile)
    return Contacts(mp, d1, c1, n1, fp, d2, c2, n2, magnet1, magnet2)


def rigid_motion(x, centre):
    """4 x 4 world matrix of the motion x = (rotation vector, translation) about centre."""
    omega, t = Vector(x[:3]), Vector(x[3:])
    angle = omega.length
    rotation = Matrix.Rotation(angle, 4, omega.normalized()) if angle > 1e-12 else Matrix()
    c = Vector(centre)
    return np.array(Matrix.Translation(c + t) @ rotation @ Matrix.Translation(-c))


# The guides at one position, with their weights frozen there, so a trial position is judged by the same rules.
Guides = namedtuple("Guides", "spring_weights magnet_weight limit1 limit2")


def guides_here(scene, job, limits, spring_share, canine_weight, contacts):
    """Weights of the two guides, which share the guidance (spring_share vs 1 - spring_share); each group is
    weighted as a whole, so a thousand magnet points don't drown three springs. Within the springs, each
    canine spring counts canine_weight times an ordinary one. A spring shorter than the release distance
    (spring_release) pulls less and less, down to OCC_SPRING_RESIDUAL at 0, and what the springs lose goes
    to the magnet, so the intercuspation sets the final position (none of that with springs only, 100 %).
    The limits: each point's overlap limit, anterior or posterior by its zone."""
    pairs = job.pairs
    pull, spring_weights = 0.0, np.zeros(0)
    if pairs:
        fade = spring_fade(scene, pairs, spring_share)
        weights = pair_weights(pairs, canine_weight) * fade
        pull = spring_share * float(fade.mean())
        spring_weights = weights / weights.mean() * pull / len(pairs)
    c = contacts
    with np.errstate(invalid='ignore'):
        near = int((c.magnet1 & (c.d1 > 0) & (c.d1 < params.OCC_MAGNET_RANGE)).sum()
                   + (c.magnet2 & (c.d2 > 0) & (c.d2 < params.OCC_MAGNET_RANGE)).sum())
    magnet = 1.0 - (pull if pairs else spring_share)
    anterior, posterior = limits
    limit1 = np.where(anterior_contacts(job.mobile_plane, job.fixed_plane, c.mobile_points, c.c1), anterior, posterior)
    limit2 = np.where(anterior_contacts(job.fixed_plane, job.mobile_plane, c.fixed_points, c.c2), anterior, posterior)
    return Guides(spring_weights, magnet / near if near else 0.0, limit1, limit2)


def energy(job, guides, contacts, motion=None):
    """What the seating minimises: springs (weighted squared lengths) + magnet (squared gaps, each up to the
    magnet range) + a stiff penalty beyond each point's overlap limit. motion: a trial position of the
    mobile block (its spring points moved with it)."""
    total = 0.0
    if job.pairs:
        moving, fixed = pair_arrays(job)
        if motion is not None:
            moving = moving @ motion[:3, :3].T + motion[:3, 3]
        total += float(guides.spring_weights @ ((fixed - moving) ** 2).sum(axis=1))
    reach = params.OCC_MAGNET_RANGE
    for d, limit, magnet in ((contacts.d1, guides.limit1, contacts.magnet1), (contacts.d2, guides.limit2, contacts.magnet2)):
        gap = np.clip(np.nan_to_num(d[magnet], nan=reach), 0.0, reach)
        excess = np.clip(np.nan_to_num(-limit - d, nan=0.0), 0.0, None)
        total += guides.magnet_weight * float(gap @ gap) + params.OCC_LIMIT_STIFFNESS * float(excess @ excess)
    return total


def full_step(job, guides, contacts, centre, free):
    """The linearised motion (rotation vector, translation) towards the equilibrium, in one step (Gauss-Newton):
    the springs close their whole length, the magnet its whole gaps (within range), and the points near their
    overlap limit are held at it (solve_step), whatever the guides want."""
    a, b = [1e-3 * np.eye(6)], [np.zeros(6)]
    if job.pairs:
        moving, fixed = pair_arrays(job)
        for (p, q), w in zip(zip(moving, fixed), np.sqrt(guides.spring_weights)):
            a.append(w * np.hstack((-skew(p - centre), np.eye(3))))
            b.append(w * (q - p))
    c = contacts
    rows1, rows2 = rigid_rows(c.mobile_points, c.n1, centre), rigid_rows(c.c2, c.n2, centre)
    with np.errstate(invalid='ignore'):
        near1 = c.magnet1 & (c.d1 > 0) & (c.d1 < params.OCC_MAGNET_RANGE)
        near2 = c.magnet2 & (c.d2 > 0) & (c.d2 < params.OCC_MAGNET_RANGE)
    w = np.sqrt(guides.magnet_weight)
    a += [w * rows1[near1], w * rows2[near2]]
    b += [-w * c.d1[near1], w * c.d2[near2]]   # close the gaps
    held1, rhs1 = held_points(rows1, c.d1, guides.limit1)
    held2, rhs2 = held_points(-rows2, c.d2, guides.limit2)
    free = np.eye(6) if free is None else free
    x = free @ solve_step(np.vstack(a) @ free, np.concatenate(b), np.vstack((held1, held2)) @ free,
                          np.concatenate((rhs1, rhs2)))
    scale = min(1.0, params.OCC_MAX_STEP / max(np.linalg.norm(x[3:]), 1e-12),
                np.radians(params.OCC_MAX_STEP) / max(np.linalg.norm(x[:3]), 1e-12))
    return x * scale   # rotation and translation together, so the motion keeps its direction


def solve(scene, job, limits, spring_share, canine_weight, vertices, free, max_iterations, path=None):
    """Seat the mobile block with the given test points: full steps (full_step), each tried whole, then halved
    while it does not lower the energy (line search, up to OCC_LINE_SEARCH times), so the solver can't swing
    between two contacts. Stops when no step helps any more, or below CONVERGED. Only the block's transform
    changes. path: the block's world matrix is added after each step (to play the seating back).
    Returns (iterations, True if it settled)."""
    block = job.block
    contacts = measure(job, vertices)
    for i in range(1, max_iterations + 1):
        guides = guides_here(scene, job, limits, spring_share, canine_weight, contacts)
        now = energy(job, guides, contacts)
        x = full_step(job, guides, contacts, np.array(block.matrix_world.translation), free)
        for _halving in range(params.OCC_LINE_SEARCH + 1):
            motion = rigid_motion(x, np.array(block.matrix_world.translation))
            trial = measure(job, vertices, motion)
            if energy(job, guides, trial, motion) < now:
                break
            x = x / 2
        else:
            return i, True          # no step lowers the energy: this is the equilibrium
        block.matrix_world = registration.as_rigid(Matrix(motion.tolist()) @ block.matrix_world)
        bpy.context.view_layer.update()   # so the arch and its points report their new position
        contacts = trial
        if path is not None:
            path.append(block.matrix_world.copy())
        shift, angle = np.linalg.norm(x[3:]), np.degrees(np.linalg.norm(x[:3]))
        if shift < CONVERGED and angle < CONVERGED:
            return i, True
    return max_iterations, shift < params.OCC_SETTLED and angle < params.OCC_SETTLED


def pair_arrays(job):
    """(mobile points, fixed points), world."""
    moving = np.array([m.matrix_world.translation for _f, m in job.pairs]).reshape(-1, 3)
    fixed = np.array([f.matrix_world.translation for f, _m in job.pairs]).reshape(-1, 3)
    return moving, fixed


def points_spread(job):
    """Second largest spread of the mobile points (mm): near 0 means they are on one line, and the
    rotation about that line is undetermined."""
    moving, _fixed = pair_arrays(job)
    return float(np.linalg.svd(moving - moving.mean(axis=0), compute_uv=False)[1]) if len(moving) >= 3 else 0.0


def can_fit(job):
    """Enough springs, spread out, for the point fit (3 pairs or more, not on one line)."""
    return len(job.pairs) >= 3 and points_spread(job) >= MIN_SPREAD


def point_fit(job, canine_weight=params.OCC_CANINE_WEIGHT):
    """World matrix of the mobile block after a pure least-squares fit of the springs (Kabsch,
    canines weighted): gets there from any starting pose, even a scan exported upside down."""
    moving, fixed = pair_arrays(job)
    fit = Matrix(registration.rigid_fit(moving, fixed, pair_weights(job.pairs, canine_weight)).tolist())
    return registration.as_rigid(fit @ job.block.matrix_world)


def is_segment_job(job):
    from . import occlusion_segments
    return occlusion_segments.is_segment(job.mobile)


def motions(scene, job, from_here):
    """6 x k basis of the motions the seating may use, or None for all of them. A segment: only
    translations (its angles are the surgeon's, it never turns). The whole arch "from here": no slide in
    the occlusal plane, to keep a position adjusted by hand."""
    from . import occlusion_segments
    if is_segment_job(job):
        return occlusion_segments.free_motions(job.block)
    return closing_motions(scene, job) if from_here else None


def translation_fit(job, free, canine_weight=params.OCC_CANINE_WEIGHT):
    """World matrix of the mobile block moved by the springs' weighted mean, along the free translations
    only: one spring is enough, the angles are kept."""
    moving, fixed = pair_arrays(job)
    weights = pair_weights(job.pairs, canine_weight)
    shift = ((fixed - moving) * weights[:, None]).sum(axis=0) / weights.sum()
    along = [free[3:, j] for j in range(free.shape[1]) if np.linalg.norm(free[3:, j]) > 0.5]
    shift = sum((shift @ e) * e for e in along) if along else np.zeros(3)
    return Matrix.Translation(Vector(shift)) @ job.block.matrix_world


def fit_target(scene, job, from_here, canine_weight=params.OCC_CANINE_WEIGHT):
    """Where the first phase takes the block: the pure point fit (whole arch, springs spread out), the
    springs' mean translation (a segment), or None (from here, or too few springs)."""
    if from_here:
        return None
    if is_segment_job(job):
        return translation_fit(job, motions(scene, job, False), canine_weight) if job.pairs else None
    return point_fit(job, canine_weight) if can_fit(job) else None


def seat_one(scene, job, limits, spring_share=params.OCC_SPRING_SHARE / 100, canine_weight=params.OCC_CANINE_WEIGHT,
             max_iterations=params.OCC_MAX_ITERATIONS, from_here=False, path=None):
    """Run one seating to the end: first fit (see fit_target), then springs and magnet up to the overlap
    limits, with the motions allowed (see motions): on random samples, then again with every vertex near
    the contacts also held at its limit (samples can miss the tip of a cusp, which is exactly what sets the
    overlap; the magnet stays on the samples, so both phases aim at the same equilibrium).
    path: the block's world matrices along the way (start, fit, each step), to play it back.
    Returns (iterations, True if it settled)."""
    if path is not None:
        path.append(job.block.matrix_world.copy())
    target = fit_target(scene, job, from_here, canine_weight)
    if target is not None:
        job.block.matrix_world = target
        bpy.context.view_layer.update()
        if path is not None:
            path.append(target.copy())
    free = motions(scene, job, from_here)
    args = (scene, job, limits, spring_share, canine_weight)
    first, _ok = solve(*args, test_points(job), free, max_iterations, path)
    last, ok = solve(*args, test_points(job, contact_vertices(job)), free, params.OCC_FINISH_ITERATIONS, path)
    return first + last, ok


def is_anterior_segment(job):
    from . import occlusion_segments
    return is_segment_job(job) and occlusion_segments.block_of(job.mobile).omfs_seg.side == 0


def seat(scene, limits, spring_share=params.OCC_SPRING_SHARE / 100, canine_weight=params.OCC_CANINE_WEIGHT,
         max_iterations=params.OCC_MAX_ITERATIONS, from_here=False, paths=None, messages=None):
    """Every seating in order (see seatings). Segmented upper arch (separated once the whole arch is seated):
    the optimisation, from where each segment is: the posterior segments only translate (springs, magnet,
    limits), the anterior one glides along the incisal guidance to the target overbite (occlusion_incisors).
    paths: a list that gets (block, its world matrices along the way) per seating. messages: a list that gets
    (level, text) for the surgeon. Returns (total iterations, True if all settled)."""
    from . import occlusion_incisors
    total, all_settled = 0, True
    props = scene.omfs_occlusion
    for job in seatings(scene):
        path = [] if paths is not None else None
        if is_anterior_segment(job):
            start = job.block.matrix_world.copy()
            result = occlusion_incisors.glide(scene, job.mobile, job.fixed, props.overbite_target, limits[0])
            bpy.context.view_layer.update()
            if isinstance(result, str):
                all_settled = False
                if messages is not None:
                    messages.append(('WARNING', result))
            elif messages is not None:
                messages.append(('WARNING' if result.status == 'free' or result.short else 'INFO',
                                 f"{job.mobile.name} : {occlusion_incisors.describe(result)}"))
            if paths is not None:
                paths.append((job.block, [start, job.block.matrix_world.copy()]))
            continue
        iterations, ok = seat_one(scene, job, limits, spring_share, canine_weight, max_iterations,
                                  from_here or is_segment_job(job), path)
        if paths is not None:
            paths.append((job.block, path))
        total += iterations
        all_settled &= ok
    return total, all_settled


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


def signed_distances_to_any(points, targets, max_distance):
    """signed_distances against several surfaces (the upper segments): for each point, the deepest
    overlap, else the smallest gap, and the closest point on that surface."""
    best_d = np.full(len(points), np.nan)
    best_c = np.full((len(points), 3), np.nan)
    for target in targets:
        d, closest, _normals = signed_distances(points, target, max_distance)
        with np.errstate(invalid='ignore'):
            better = np.isnan(best_d) | (d < best_d)
        better &= ~np.isnan(d)
        best_d[better], best_c[better] = d[better], closest[better]
    return best_d, best_c


def neighbour_overlap(part, others):
    """How much deeper (mm, negative; NaN = no new overlap) each vertex of a segment now goes into the other
    segments than it did pre-op: two neighbours whose angles were changed can run into each other at the
    osteotomy line. Compared with pre-op because real scans already "overlap" there (interproximal contacts,
    the ragged edge of the cut), which is not a planning problem."""
    from . import occlusion_segments
    points = registration.vertex_coords(part)
    block = occlusion_segments.block_of(part)
    to_preop = occlusion.preop_matrix(block) @ block.matrix_world.inverted()
    best = np.full(len(points), np.nan)
    for other in others:
        d_now = signed_distances(points, other, params.SEG_NEIGHBOUR_RANGE)[0]
        other_block = occlusion_segments.block_of(other)
        # the part's points where they were pre-op, seen from the other segment where it is now
        relative = np.array(other_block.matrix_world @ occlusion.preop_matrix(other_block).inverted() @ to_preop)
        before = points @ relative[:3, :3].T + relative[:3, 3]
        d_pre = signed_distances(before, other, params.SEG_NEIGHBOUR_RANGE)[0]
        with np.errstate(invalid='ignore'):
            excess = d_now - np.minimum(np.nan_to_num(d_pre, nan=0.0), 0.0)
            better = excess < np.nan_to_num(best, nan=0.0)   # deeper than 0 and than the others so far
        best[better] = excess[better]
    return best


def segment_overlaps(scene):
    """[(segment, other segment, deepest overlap mm)] for every pair that overlaps, deepest first."""
    from . import occlusion_segments
    parts = occlusion_segments.segments(scene)
    bpy.context.view_layer.update()   # a block just turned: its segment's world matrix follows on update
    found = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            depth = max(robust_overlap(neighbour_overlap(a, [b]), vertex_areas(a)),
                        robust_overlap(neighbour_overlap(b, [a]), vertex_areas(b)))
            if depth > 0:
                found.append((a, b, depth))
    return sorted(found, key=lambda f: -f[2])


def write_overlap_map(scene):
    """Colour both arches (every upper segment) by their signed distance to the other one; store the summary.
    Segments are also coloured where they overlap a neighbouring segment (the deeper of the two wins)."""
    props = scene.omfs_occlusion
    lower, parts = props.lower, upper_parts(scene)
    upper_plane, lower_plane = zone_planes(scene)
    deepest = {True: 0.0, False: 0.0}   # anterior, posterior
    sides = [(lower, parts, lower_plane, upper_plane)] + [(part, [lower], upper_plane, lower_plane) for part in parts]
    for obj, others, own_plane, other_plane in sides:
        points = registration.vertex_coords(obj)
        d, closest = signed_distances_to_any(points, others, params.OCC_SEARCH_DISTANCE)
        shown = d
        if obj is not lower and len(parts) > 1:
            between = neighbour_overlap(obj, [p for p in parts if p is not obj])
            with np.errstate(invalid='ignore'):
                shown = np.where(between < np.nan_to_num(d, nan=np.inf), between, d)
        mesh = obj.data
        attr = mesh.color_attributes.get(MAP_ATTRIBUTE) or mesh.color_attributes.new(
            MAP_ATTRIBUTE, 'FLOAT_COLOR', 'POINT')
        attr.data.foreach_set("color", overlap_colours(shown).ravel())
        mesh.color_attributes.active_color = attr
        area = vertex_areas(obj)
        anterior = anterior_contacts(own_plane, other_plane, points, closest)
        for zone, part in ((True, anterior), (False, ~anterior)):
            deepest[zone] = max(deepest[zone], robust_overlap(d[part], area[part]))
        if obj is lower:
            contact_areas(scene, lower, points, d, anterior)
    props.max_overlap_anterior, props.max_overlap_posterior = deepest[True], deepest[False]
    props.max_overlap = max(deepest.values())
    between = segment_overlaps(scene) if len(parts) > 1 else []
    props.segment_overlap = between[0][2] if between else 0.0
    from . import occlusion_incisors
    front = next((p for p in parts if p is props.upper or p.name.startswith("Segment antérieur")), None)
    found = occlusion_incisors.measure(scene, front, lower) if front is not None else None
    props.incisors_found = found is not None
    if found is not None:
        props.overbite, props.overjet = found.overbite, found.overjet
    props.segment_overlap_pair = f"{between[0][0].name[8:]} ↔ {between[0][1].name[8:]}" if between else ""
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
    """What the map depends on: the transforms of both arches (every upper segment)."""
    props = scene.omfs_occlusion
    return tuple(tuple(round(v, 4) for row in obj.matrix_world for v in row)
                 for obj in [props.lower] + upper_parts(scene))


def map_is_current(scene):
    props = scene.omfs_occlusion
    return props.map_state == repr(map_key(scene))


def set_map_shading(context, on):
    for area in (context.screen.areas if context.screen else ()):
        if area.type == 'VIEW_3D':
            area.spaces.active.shading.color_type = 'VERTEX' if on else 'OBJECT'


def show_map(context, on):
    """Map on: the points, springs and zone lines are hidden so the colours show; map off: shown again."""
    context.scene.omfs_occlusion.show_map = on
    if on:
        write_overlap_map(context.scene)
    set_points_hidden(context.scene, on)
    set_map_shading(context, on)   # off: back to the role colours


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

def playback_frames(path, eased=True):
    """One world matrix per frame along the recorded poses, at an even speed (OCC_PLAYBACK_MM mm or
    OCC_PLAYBACK_DEG degrees per frame, whichever needs more frames); the first glide (the point fit) eased
    in and out, and at most FIT_FRAMES frames per stretch."""
    frames = [path[0]]
    for k, (a, b) in enumerate(zip(path, path[1:])):
        shift = (b.translation - a.translation).length
        angle = np.degrees(a.to_quaternion().rotation_difference(b.to_quaternion()).angle)
        n = int(min(FIT_FRAMES, max(1, np.ceil(max(shift / params.OCC_PLAYBACK_MM, angle / params.OCC_PLAYBACK_DEG)))))
        for f in range(1, n + 1):
            t = f / n
            if eased and k == 0:
                t = t * t * (3 - 2 * t)
            frames.append(a.lerp(b, t))
    return frames


def guided(scene, job):
    """Something can move the mobile block: an active spring, or the magnet with the arches in range."""
    props = scene.omfs_occlusion
    if job.pairs and props.spring_share > 0:
        return True
    if props.spring_share >= 100:
        return False
    d = signed_distances(world_points(job.mobile, samples(job.mobile)), job.fixed, params.OCC_MAGNET_RANGE)[0]
    return not np.isnan(d).all()


def arches_ready(scene):
    props = scene.omfs_occlusion
    return props.upper is not None and props.lower is not None and occlusion.find_block(scene, 'LOWER')


def closest_hit(context, under_mouse, objects):
    """(object, world location) of the nearest hit along the mouse ray among objects, or (None, None)."""
    region, mouse = under_mouse
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
    hits = [(obj, landmarks.object_hit(context, region, mouse, obj)) for obj in objects]
    hits = [(obj, loc) for obj, loc in hits if loc is not None]
    return min(hits, key=lambda h: (h[1] - origin).length, default=(None, None))


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
            arches = upper_parts(context.scene) if self.upper_point is None else [self.arches[1]]
            arch, location = closest_hit(context, under_mouse, arches)
            if location is None:
                self.report({'WARNING'}, f"Cliquez sur {' / '.join(a.name for a in arches)}")
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
    """Afficher ou cacher d'un clic les points, les ressorts et les traits blancs qui délimitent la zone
    antérieure (canine à canine). Rien n'est supprimé : les ressorts restent actifs pour l'assise"""
    bl_idname = "omfs.toggle_occlusion_points"
    bl_label = "Afficher / cacher les ressorts et les limites"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        set_points_hidden(context.scene, points_visible(context.scene))
        return {'FINISHED'}


class OMFS_OT_seat_occlusion(bpy.types.Operator):
    """L'arcade du haut (ou chaque segment) vient s'asseoir sur l'arcade du bas fixe : les ressorts guident (part réglable),
    l'aimant rapproche les arcades jusqu'aux contacts stables, et les dents s'arrêtent au chevauchement
    permis de chaque zone. Calculé d'abord, puis animé dans la vue ; Échap : annuler et revenir au départ"""
    bl_idname = "omfs.seat_occlusion"
    bl_label = "Asseoir l'occlusion"
    bl_options = {'REGISTER', 'UNDO'}

    from_here: bpy.props.BoolProperty(options={'SKIP_SAVE'},
                                      description="Partir de la position actuelle, sans ajustement aux points et sans glissement")

    @classmethod
    def description(cls, context, properties):
        from . import occlusion_segments
        if occlusion_segments.segments(context.scene):
            return ("Optimiser l'occlusion segmentée, depuis la position de chaque segment : les segments "
                    "postérieurs ne font que se translater (aucune rotation) pour mieux s'intercuspider ; le "
                    "segment antérieur glisse le long de la pente incisive jusqu'au surplomb vertical visé, "
                    "en gardant le contact. Échap : annuler")
        if properties.from_here:
            return ("Rasseoir depuis la position actuelle (ex. après un ajustement à la main) : seulement les "
                    "derniers millimètres, avec les ressorts présents, l'aimant et les limites. Arcade entière : "
                    "le glissement latéral et antéro-postérieur est bloqué, elle ne fait que se fermer")
        return cls.__doc__

    def ready(self, context):
        scene = context.scene
        if not arches_ready(scene):
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return False
        jobs = seatings(scene)
        if is_segment_job(jobs[0]) or self.from_here:   # translations (or a hand position): one spring will do
            for job in jobs:
                if not is_anterior_segment(job) and not guided(scene, job):
                    where = f"{job.mobile.name} : " if is_segment_job(job) else ""
                    self.report({'ERROR'}, f"{where}Rien ne guide l'assise : ajoutez un ressort, ou rapprochez les "
                                           f"arcades à moins de {params.OCC_MAGNET_RANGE:g} mm avec une part des "
                                           "ressorts sous 100 %")
                    return False
            return True
        if len(jobs[0].pairs) < 3:
            self.report({'ERROR'}, "Placez au moins 3 paires de points")
            return False
        if points_spread(jobs[0]) < MIN_SPREAD:
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
        messages = []
        iterations, settled = seat(scene, limits_of(props), props.spring_share / 100, props.canine_weight,
                                   from_here=self.from_here, messages=messages)
        write_overlap_map(scene)
        self.report({'INFO'}, f"Assise : {iterations} itérations")
        for level, text in messages:
            self.report({level}, text)
        self.report_end(settled)
        return {'FINISHED'}

    def invoke(self, context, event):
        """Computed first (a few seconds, the cursor waits), then played back at OCC_PLAYBACK_FPS: the animation
        stays fluid whatever the time of one iteration. Every segment moves at the same time."""
        global _seating
        if not self.ready(context):
            return {'CANCELLED'}
        self.starts = [(job.block, job.block.matrix_world.copy()) for job in seatings(context.scene)]   # Esc: back to these
        self.frames = None
        _seating = True
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area:
            self.area.header_text_set("Assise : calcul…")
        context.window.cursor_set('WAIT')
        self.timer = context.window_manager.event_timer_add(1 / params.OCC_PLAYBACK_FPS, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def compute(self, context):
        scene = context.scene
        props = scene.omfs_occlusion
        start = time.monotonic()
        paths, messages = [], []
        iterations, self.all_settled = seat(scene, limits_of(props), props.spring_share / 100, props.canine_weight,
                                            from_here=self.from_here, paths=paths, messages=messages)
        self.report({'INFO'}, f"Assise : {iterations} itérations, {time.monotonic() - start:.1f} s de calcul")
        for level, text in messages:
            self.report({level}, text)
        for block, matrix in self.starts:
            block.matrix_world = matrix
        self.frames = [(block, playback_frames(path, eased=not self.from_here or len(path) == 2))
                       for block, path in paths]
        self.frame = 0
        context.window.cursor_set('DEFAULT')
        if self.area:
            self.area.header_text_set("Assise   |   Échap : annuler")

    def finish(self, context):
        global _seating
        _seating = False
        context.window_manager.event_timer_remove(self.timer)
        context.window.cursor_set('DEFAULT')
        if self.area:
            self.area.header_text_set(None)

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            for block, matrix in self.starts:
                block.matrix_world = matrix
            self.finish(context)
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self.frames is None:
            self.compute(context)
            return {'RUNNING_MODAL'}
        self.frame += 1
        for block, frames in self.frames:
            block.matrix_world = frames[min(self.frame, len(frames) - 1)]
        if self.area:
            self.area.tag_redraw()
        if self.frame < max(len(frames) for _block, frames in self.frames):
            return {'RUNNING_MODAL'}
        self.finish(context)
        self.report_end(self.all_settled)
        show_map(context, True)
        return {'FINISHED'}


class OMFS_OT_toggle_overlap_map(bpy.types.Operator):
    """Colorer les deux arcades selon le chevauchement / l'écart en mm (légende dans le panneau). Cache les
    ressorts et les limites de zone pendant la carte, les réaffiche ensuite"""
    bl_idname = "omfs.toggle_overlap_map"
    bl_label = "Carte de chevauchement"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if not arches_ready(scene):
            return {'CANCELLED'}
        show_map(context, not scene.omfs_occlusion.show_map)
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
