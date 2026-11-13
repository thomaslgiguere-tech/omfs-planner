"""Object creator: Splint / Chape. Intermediate or final occlusal splint, or a single-arch cap.

Splint (surgeon, 2026-11-13, to look like the companies' splints): no curve to click. One point per spot
on the buccal surface of the upper teeth marks how far up the splint covers them there, the same on
the lower teeth; through those points run two smooth surfaces (top, bottom), interpolated along the arch,
across it perpendicular to the insertion axis. The outline is computed: an automatic, smooth centre line
of the crowns, and a buccal / lingual envelope of the crowns plus the clearances (0.5 / 2.0 mm), smoothed
over a long stretch so the arch is round. The block between the surfaces then has its cusps hollowed out
(the imprint, as before).

Cap (chape), unchanged. Built in steps (docs/workflow.md, "Object creator"):
1. Arch curve: points clicked on the arches, joined by a smooth curve (done).
2. Parametric blank swept along the curve, as wide as the crowns plus margins, ending with the curve (done).
3. Tooth imprints by projection onto the arches along the insertion axis, 0.10 mm clearance (done).
4. Solidify: the blank frozen into its own object, real holes at the contacts (done).
5. Engraved text and print checks, STL export: the universal text tool (text_labels.py) (done).

The arch curve is a Bezier curve object (role ARCH) whose points are the clicks, with automatic
handles so the curve passes through every point. For a splint between two arches, each click is
moved halfway between the upper and lower occlusal surfaces at that spot, so the curve runs
through the middle of the bite. The scans are only read, never modified.
"""

import json

import blf
import bmesh
import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, params, reference_frame, text_labels

COLLECTION = "Splint"
CURVE_NAME = "Arch curve"
KINDS = (
    ('INTERMEDIATE', "Intermédiaire", "Gouttière entre l'arcade du haut déplacée et l'arcade du bas préop"),
    ('FINAL', "Finale", "Gouttière entre les deux arcades en occlusion finale"),
    ('CHAPE', "Chape", "Coiffe sur une seule arcade"),
)
CURVE_COLOUR = (1.0, 0.55, 0.1, 1.0)   # orange
CENTRE_REACH = 15.0                    # mm: how far up / down to look for the other arch's surface


# --- Arches and axes --------------------------------------------------------------------

def is_scan(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'SCAN'


def upper_arch(scene):
    """The upper scan: chosen here, else the one chosen for the occlusion."""
    chosen = scene.omfs_splint.upper
    return chosen if is_scan(chosen) else (scene.omfs_occlusion.upper if is_scan(scene.omfs_occlusion.upper) else None)


def lower_arch(scene):
    chosen = scene.omfs_splint.lower
    return chosen if is_scan(chosen) else (scene.omfs_occlusion.lower if is_scan(scene.omfs_occlusion.lower) else None)


def arches(scene):
    """The arch meshes the curve is drawn on: both arches for a splint, one for a cap."""
    props = scene.omfs_splint
    if props.kind == 'CHAPE':
        return [o for o in (props.cap_arch,) if is_scan(o)]
    return [o for o in (upper_arch(scene), lower_arch(scene)) if o is not None]


def up_axis(scene):
    """Occlusal "up": the reference frame's, else found from the arches (scanners do not agree on
    which file axis is up: case1's scans have it along Y)."""
    frame = reference_frame.find_frame(scene)
    if frame:
        return frame.matrix_world.col[2].xyz.normalized()
    return arch_up(scene) or Vector((0, 0, 1))


_axes = {}


def thinnest_axis(obj):
    """World direction across which the arch is thinnest (the occlusal axis, either way), and the
    arch's centre. Cached until the mesh or its placement changes."""
    key = (obj.name, len(obj.data.vertices), tuple(round(x, 6) for row in obj.matrix_world for x in row))
    if key not in _axes:
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        m = np.array(obj.matrix_world)
        points = co.reshape(-1, 3)[::max(1, len(obj.data.vertices) // 20000)] @ m[:3, :3].T + m[:3, 3]
        centre = points.mean(0)
        _axes[key] = (np.linalg.svd(points - centre, full_matrices=False)[2][2], centre)
    return _axes[key]


def towards_teeth(obj, axis):
    """The occlusal axis of one arch pointed at its teeth: away from the open border of a scan, or
    away from the flat base of a closed model (its biggest flat area facing out along the axis)."""
    mesh = obj.data
    to_world = np.array(obj.matrix_world)[:3, :3]
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    heights = co.reshape(-1, 3) @ to_world.T @ axis
    edges = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("edge_index", edges)
    open_edges = np.flatnonzero(np.bincount(edges, minlength=len(mesh.edges)) == 1)
    if len(open_edges):
        pairs = np.empty(len(mesh.edges) * 2, np.int32)
        mesh.edges.foreach_get("vertices", pairs)
        border = pairs.reshape(-1, 2)[open_edges].ravel()
        return axis if heights[border].mean() < heights.mean() else -axis
    normals = np.empty(len(mesh.polygons) * 3)
    mesh.polygons.foreach_get("normal", normals)
    areas = np.empty(len(mesh.polygons))
    mesh.polygons.foreach_get("area", areas)
    facing = normals.reshape(-1, 3) @ to_world.T @ axis
    return axis if areas[facing < -0.999].sum() > areas[facing > 0.999].sum() else -axis


def arch_up(scene):
    """Up from the arches: with both, the mean of their occlusal axes, pointing from the lower arch to
    the upper; with one (a cap), its occlusal axis pointing at its teeth (the cap side)."""
    props = scene.omfs_splint
    if props.kind == 'CHAPE':
        arch = props.cap_arch
        if not is_scan(arch):
            return None
        return Vector(towards_teeth(arch, thinnest_axis(arch)[0])).normalized()
    upper, lower = upper_arch(scene), lower_arch(scene)
    if upper is None or lower is None:
        return None
    (a, upper_centre), (b, lower_centre) = thinnest_axis(upper), thinnest_axis(lower)
    axis = a + (b if a @ b > 0 else -b)
    if axis @ (upper_centre - lower_centre) < 0:
        axis = -axis
    return Vector(axis).normalized()


def ray_hit(obj, origin, direction, depsgraph):
    """First hit of a world-space ray on one object (world location), or None."""
    to_local = obj.matrix_world.inverted()
    hit, location, _normal, _index = obj.ray_cast(to_local @ origin, to_local.to_3x3() @ direction,
                                                  depsgraph=depsgraph)
    return obj.matrix_world @ location if hit else None


def centre_between_arches(scene, point, depsgraph):
    """The point halfway between the lower arch's occlusal surface (seen from above) and the upper
    arch's (seen from below), on the vertical through the point. The point itself if either is missed."""
    up = up_axis(scene)
    lower_scan, upper_scan = lower_arch(scene), upper_arch(scene)
    lower = ray_hit(lower_scan, point + up * CENTRE_REACH, -up, depsgraph) if lower_scan else None
    upper = ray_hit(upper_scan, point - up * CENTRE_REACH, up, depsgraph) if upper_scan else None
    if lower is None or upper is None or (upper - lower).dot(up) < 0:
        return point
    return (lower + upper) / 2


# --- Arch curve ---------------------------------------------------------------------------

def splint_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def find_curve(scene):
    return next((o for o in scene.objects if o.type == 'CURVE' and o.omfs.role == 'ARCH'), None)


def ensure_curve(scene):
    obj = find_curve(scene)
    if obj is not None:
        return obj
    curve = bpy.data.curves.new(CURVE_NAME, 'CURVE')
    curve.dimensions = '3D'
    obj = bpy.data.objects.new(CURVE_NAME, curve)
    splint_collection(scene).objects.link(obj)
    obj.omfs.role = 'ARCH'
    obj.show_in_front = True
    return obj


def curve_points(obj):
    """World positions of the clicked points, in order."""
    if obj is None or not obj.data.splines:
        return []
    return [obj.matrix_world @ p.co for p in obj.data.splines[0].bezier_points]


def set_points(obj, points):
    """Rebuild the curve through these world points (automatic handles: passes through each)."""
    curve = obj.data
    curve.splines.clear()
    if not points:
        return
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    to_local = obj.matrix_world.inverted()
    for bp, p in zip(spline.bezier_points, points):
        bp.co = to_local @ p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    spline.resolution_u = 24


def add_point(scene, location, depsgraph):
    """Append a point (centred between the arches for a splint). Returns the point placed."""
    props = scene.omfs_splint
    if props.kind != 'CHAPE' and props.centre:
        location = centre_between_arches(scene, location, depsgraph)
    obj = ensure_curve(scene)
    set_points(obj, curve_points(obj) + [location])
    return location


def remove_last_point(scene):
    obj = find_curve(scene)
    if obj is not None:
        set_points(obj, curve_points(obj)[:-1])


def sampled_curve(obj, depsgraph):
    """World points along the evaluated curve (the smooth line, not only the clicks)."""
    if obj is None or not obj.data.splines:
        return []
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = [evaluated.matrix_world @ v.co for v in mesh.vertices]
    evaluated.to_mesh_clear()
    return points


def curve_length(obj):
    if obj is None or not obj.data.splines or len(obj.data.splines[0].bezier_points) < 2:
        return 0.0
    return obj.data.splines[0].calc_length()


# --- Splint: coverage points and the automatic arch --------------------------------------------

POINT_TAG = "splint point"      # made_by of a coverage point (an Empty on its arch's buccal surface)
POINT_SIZE = 0.6


def coverage_points(scene, side=None):
    """Coverage points, 'U' (upper teeth) and 'L' (lower teeth), in the order placed."""
    found = [o for o in scene.objects if o.omfs.made_by == POINT_TAG and (side is None or o.omfs.landmark[4] == side)]
    return sorted(found, key=lambda o: int(o.omfs.landmark[5:]))


def add_coverage_point(scene, arch, location, side):
    """A point on the buccal surface of an arch's teeth: the splint covers them up to there. It moves
    with its arch (child of the scan), never on its own."""
    number = max((int(o.omfs.landmark[5:]) for o in coverage_points(scene)), default=0) + 1
    obj = bpy.data.objects.new(f"Couverture {'haut' if side == 'U' else 'bas'} {number}", None)
    splint_collection(scene).objects.link(obj)
    obj.empty_display_type = 'SPHERE'
    obj.empty_display_size = POINT_SIZE
    obj.show_in_front = True
    obj.omfs.role = 'MARKER'
    obj.omfs.made_by = POINT_TAG
    obj.omfs.landmark = f"SPL_{side}{number}"
    obj.parent = arch
    obj.matrix_parent_inverse = arch.matrix_world.inverted()
    obj.matrix_world = Matrix.Translation(location)
    locking.set_locked(obj, True)
    return obj


def plane_axes(up):
    """Two unit vectors across the insertion axis."""
    u = Vector(up).orthogonal().normalized()
    return np.array(u), np.array(Vector(up).cross(u).normalized())


def gaussian(values, sigma):
    """Gaussian smoothing of a 1D array (edges held), sigma in samples."""
    if sigma < 0.5 or len(values) < 3:
        return values
    half = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, half, mode='edge'), kernel, mode='valid')


def resample(line, step):
    """2D polyline resampled every `step` mm."""
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    at = np.linspace(0.0, s[-1], max(3, int(round(s[-1] / step)) + 1))
    return np.stack([np.interp(at, s, line[:, i]) for i in range(2)], axis=1)


def centre_on_crowns(line, flat, tips, step, rounds=3):
    """2D centre line moved, station by station, to the middle of the crowns across it (seen from above),
    smoothed, then cut or carried along its end tangents to span the crown points near the bite (`tips`)."""
    line = resample(line, step)
    for _ in range(rounds):
        tangents = np.gradient(line, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        normals = np.stack((-tangents[:, 1], tangents[:, 0]), 1)
        nearest = np.empty(len(flat), int)
        sq = (line ** 2).sum(1)
        for chunk in np.array_split(np.arange(len(flat)), max(1, len(flat) // 20000)):
            nearest[chunk] = np.argmin((flat[chunk] ** 2).sum(1)[:, None] - 2 * flat[chunk] @ line.T + sq[None, :], axis=1)
        rel = flat - line[nearest]
        lateral = np.einsum('ij,ij->i', rel, normals[nearest])
        low, high = np.full(len(line), np.inf), np.full(len(line), -np.inf)
        np.minimum.at(low, nearest, lateral)
        np.maximum.at(high, nearest, lateral)
        seen = np.isfinite(low)
        shift = np.where(seen, (low + high) / 2, np.nan)
        shift = np.interp(np.arange(len(line)), np.flatnonzero(seen), shift[seen]) if seen.any() else np.zeros(len(line))
        line = line + normals * gaussian(shift, 3.0)[:, None]
        line = resample(line, step)
    # The ends: from the first to the last crown point near the bite (the teeth, not the gingiva behind
    # the last molar, lower down), plus a small margin; the line is carried along its end tangents.
    t0, t1 = line[0] - line[1], line[-1] - line[-2]
    t0, t1 = t0 / np.linalg.norm(t0), t1 / np.linalg.norm(t1)
    reach = 30.0
    longer = resample(np.vstack((line[0] + t0 * reach, line, line[-1] + t1 * reach)), step / 2)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(longer, axis=0), axis=1))))
    tangents = np.gradient(longer, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    sq = (longer ** 2).sum(1)
    near = np.argmin((tips ** 2).sum(1)[:, None] - 2 * tips @ longer.T + sq[None, :], axis=1)
    where = arc[near] + np.einsum('ij,ij->i', tips - longer[near], tangents[near])
    start, stop = where.min() - params.SPLINT_END_MARGIN, where.max() + params.SPLINT_END_MARGIN
    at = np.linspace(start, stop, max(3, int(round((stop - start) / step)) + 1))
    line = np.stack([np.interp(at, arc, longer[:, i]) for i in range(2)], axis=1)
    return resample(line, step)


_auto = {}


def auto_centreline(scene, step):
    """The splint's centre line and its two surfaces, from the coverage points: (points at mid-height,
    tangents, buccal normals, length, depth above, depth below per station), or an error message.
    Seen along the insertion axis from the middle of the crowns, the crowns between the two surfaces are
    measured in sectors (median distance), smoothed: a round arch, from one end of the teeth to the other."""
    upper, lower = upper_arch(scene), lower_arch(scene)
    tops, bottoms = coverage_points(scene, 'U'), coverage_points(scene, 'L')
    if upper is None or lower is None:
        return "Choisissez les deux arcades"
    if not tops or not bottoms:
        return "Placez au moins un point de couverture sur les dents du haut et un sur celles du bas"
    up = np.array(up_axis(scene))
    top_at = np.array([tuple(o.matrix_world.translation) for o in tops])
    bottom_at = np.array([tuple(o.matrix_world.translation) for o in bottoms])
    key = (repr(np.round(top_at, 4).tolist()), repr(np.round(bottom_at, 4).tolist()), upper.name, lower.name,
           repr([tuple(map(tuple, o.matrix_world)) for o in (upper, lower)]), step, tuple(np.round(up, 6)))
    if key in _auto:
        return _auto[key]
    hi, lo = (top_at @ up).max(), (bottom_at @ up).min()
    if hi <= lo:
        return "Les points du haut doivent être plus hauts que ceux du bas"
    verts = np.concatenate([world_vertices(upper), world_vertices(lower)])
    h = verts @ up
    crowns = verts[(h > lo) & (h < hi)]
    if len(crowns) < 50:
        return "Aucune dent entre les points du haut et du bas"
    u, v = plane_axes(up)
    flat = np.stack((crowns @ u, crowns @ v), 1)
    middle = flat.mean(0)
    rel = flat - middle
    theta, r = np.arctan2(rel[:, 1], rel[:, 0]), np.linalg.norm(rel, axis=1)
    order = np.sort(theta)
    gaps = np.diff(np.concatenate((order, [order[0] + 2 * np.pi])))
    start = order[(np.argmax(gaps) + 1) % len(order)]        # the arch's opening is the widest empty sector
    t = (theta - start) % (2 * np.pi)
    span = t.max()
    width = np.radians(params.SPLINT_ARCH_BIN)
    bins = max(3, int(np.ceil(span / width)) + 1)
    index = np.minimum((t / span * (bins - 1)).round().astype(int), bins - 1)
    radius = np.full(bins, np.nan)
    for b in np.unique(index):
        radius[b] = np.median(r[index == b])
    known = ~np.isnan(radius)
    radius = np.interp(np.arange(bins), np.flatnonzero(known), radius[known])
    radius = gaussian(radius, params.SPLINT_ARCH_SMOOTH / params.SPLINT_ARCH_BIN)
    angles = start + np.linspace(0.0, span, bins)
    line2 = middle + radius[:, None] * np.stack((np.cos(angles), np.sin(angles)), 1)
    bite = crowns @ up
    near_bite = np.abs(bite - (hi + lo) / 2) < params.SPLINT_END_BAND     # crowns at the occlusal level
    line2 = centre_on_crowns(line2, flat, flat[near_bite] if near_bite.sum() > 10 else flat, step)
    line = np.outer(line2[:, 0], u) + np.outer(line2[:, 1], v)          # on the plane through the origin
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    length = s[-1]
    at = np.linspace(0.0, length, max(2, int(round(length / step)) + 1))
    points = np.stack([np.interp(at, s, line[:, i]) for i in range(3)], axis=1)
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    normals = np.cross(tangents, up)
    if np.mean(np.einsum('ij,ij->i', normals, points - points.mean(axis=0))) < 0:
        normals = -normals                                  # buccal = away from the inside of the arch

    def surface(at_points):
        """Height of a surface through the points, at each station (smooth along the arch, flat past the ends)."""
        flat_points = at_points - np.outer(at_points @ up, up)
        where = np.array([at[np.argmin(np.linalg.norm(points - p, axis=1))] for p in flat_points])
        heights = at_points @ up
        order = np.argsort(where)
        line_h = np.interp(at, where[order], heights[order])
        return gaussian(line_h, params.SPLINT_SURFACE_SMOOTH / step / 2)

    top, bottom = surface(top_at), surface(bottom_at)
    if (top <= bottom).any():
        return "Les surfaces se croisent: les points du haut doivent rester au-dessus de ceux du bas"
    mid = (top + bottom) / 2
    result = (points + np.outer(mid, up), tangents, normals, length, top - mid, mid - bottom)
    _auto.clear()
    _auto[key] = result
    return result


def centreline(scene, depsgraph, step):
    """(points, tangents, buccal normals, length, depth above, depth below per station) of the blank, or an
    error message: the splint's automatic arch, or the cap's clicked curve with its constant depths."""
    if scene.omfs_splint.kind != 'CHAPE':
        return auto_centreline(scene, step)
    curve = find_curve(scene)
    if curve is None or len(curve_points(curve)) < 2:
        return "Cliquez d'abord la courbe d'arcade (2 points au moins)"
    up = up_axis(scene)
    points, tangents, normals, length = stations(curve, depsgraph, step, up)
    depth_up, depth_low = depths(scene)
    return points, tangents, normals, length, np.full(len(points), depth_up), np.full(len(points), depth_low)


# --- Blank ----------------------------------------------------------------------------------
#
# The blank is a closed slab following the arch curve: as wide as the crowns at each station
# (measured on the scans, both arches together) plus a lingual and a buccal margin, reaching
# "depth" above and below the curve (the middle of the bite). It stops at the ends of the curve.
# Its top and bottom faces are fine grids (vertex groups "Top" / "Bottom"): step 3 projects them
# onto the teeth.

BLANK_NAME = "Splint"
BLANK_TAG = "splint blank"      # made_by of the blank
BLANK_COLOUR = (0.95, 0.70, 0.30, 0.55)   # guide orange, see-through
TOP_GROUP, BOTTOM_GROUP = "Top", "Bottom"

_measures = {}                  # measurement key -> raw extents (lingual, buccal)


def stations(obj, depsgraph, step, up):
    """Points every `step` mm along the curve, with tangent and buccal normal (horizontal, unit).
    Returns (points, tangents, normals, length) as numpy arrays."""
    line = np.array([tuple(p) for p in sampled_curve(obj, depsgraph)])
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    line = line[keep]
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    length = s[-1]
    at = np.linspace(0.0, length, max(2, int(round(length / step)) + 1))
    points = np.stack([np.interp(at, s, line[:, i]) for i in range(3)], axis=1)
    up = np.array(up)
    tangents = np.gradient(points, axis=0)
    tangents -= np.outer(tangents @ up, up)                       # horizontal
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    normals = np.cross(tangents, up)
    # Buccal = away from the inside of the arch (the centre of a horseshoe is lingual).
    if np.mean(np.einsum('ij,ij->i', normals, points - points.mean(axis=0))) < 0:
        normals = -normals
    return points, tangents, normals, length


def world_vertices(obj):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def fill_gaps(values):
    """Stations where nothing was measured (a missing tooth) take their neighbours' values."""
    known = ~np.isnan(values)
    if not known.any():
        return None
    idx = np.arange(len(values))
    return np.interp(idx, idx[known], values[known])


def smooth(values, window):
    """Running maximum then running mean: covers every tooth, without corners."""
    half = max(1, window // 2)
    padded = np.pad(values, half, mode='edge')
    maxed = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).max(axis=1)
    padded = np.pad(maxed, half, mode='edge')
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).mean(axis=1)


def measure(scene, points, tangents, normals, up, depth_up, depth_low, step):
    """Crown extents at each station: (lingual, buccal) in mm from the curve. Only the crown band the
    splint will cover is measured (no palate, no gingiva below it)."""
    up = np.array(up)
    depth_up, depth_low = np.broadcast_to(depth_up, len(points)), np.broadcast_to(depth_low, len(points))
    verts = np.concatenate([world_vertices(o) for o in arches(scene)])
    heights = verts @ up
    band = (heights > (points @ up - depth_low).min()) & (heights < (points @ up + depth_up).max())
    verts = verts[band]
    lingual = np.full(len(points), np.nan)
    buccal = np.full(len(points), np.nan)
    sq = (points ** 2).sum(axis=1)
    for chunk in np.array_split(verts, max(1, len(verts) // 20000)):
        if not len(chunk):
            continue
        nearest = np.argmin((chunk ** 2).sum(axis=1)[:, None] - 2 * chunk @ points.T + sq[None, :], axis=1)
        rel = chunk - points[nearest]
        along = np.einsum('ij,ij->i', rel, tangents[nearest])
        h = rel @ up
        lat = np.einsum('ij,ij->i', rel, normals[nearest])
        ok = (np.abs(along) <= 0.75 * step) & (h <= depth_up[nearest]) & (h >= -depth_low[nearest])
        idx, l = nearest[ok], lat[ok]
        b_now = np.full(len(points), -np.inf)
        l_now = np.full(len(points), -np.inf)
        np.maximum.at(b_now, idx, l)
        np.maximum.at(l_now, idx, -l)
        seen = np.isfinite(b_now)
        buccal[seen] = np.fmax(buccal[seen], b_now[seen])
        lingual[seen] = np.fmax(lingual[seen], l_now[seen])
    lingual, buccal = fill_gaps(lingual), fill_gaps(buccal)
    if lingual is None:
        return None
    return lingual, buccal


def cap_on_upper(scene):
    """A cap on the upper arch hangs under it: the arch (crowns, gingiva, palate) is above the curve."""
    arch, curve = scene.omfs_splint.cap_arch, find_curve(scene)
    points = curve_points(curve)
    if not is_scan(arch) or not points:
        return False
    up = up_axis(scene)
    centre = arch.matrix_world @ (sum((Vector(c) for c in arch.bound_box), Vector()) / 8)
    return centre.dot(up) > sum(p.dot(up) for p in points) / len(points)


def depths(scene):
    """(above, below) the curve: a cap sits on the occlusal curve and reaches only a little onto the teeth."""
    props = scene.omfs_splint
    if props.kind != 'CHAPE':
        return props.depth_up, props.depth_low
    return (props.cap_depth, props.depth_up) if cap_on_upper(scene) else (props.depth_up, props.cap_depth)


# --- Imprint ------------------------------------------------------------------------------
#
# Each top and bottom vertex is pushed along the insertion axis (frame up) onto the first tooth
# surface met coming from the other arch: the lower arch is seen from above, the upper arch from
# below. That is a height field: where a crown bulges, the splint stops at its widest point and
# never goes under it, so the undercuts are blocked out and the splint goes on and off. Where no
# tooth is met within the blank (margins, embrasures deeper than the depth) the face stays flat.
# The clearance is a ball rolled over that height field (a dilation): the gap holds on the walls too,
# and a face stepping off a wall does so outside the tooth, never cutting its corner.

_trees = {}                     # (object, mesh) name -> BVH tree in the object's own coordinates


def imprint_faces(scene):
    """(arch, 'TOP' or 'BOTTOM'): which arch each face of the blank is pressed onto."""
    props = scene.omfs_splint
    if props.kind == 'CHAPE':
        return [(props.cap_arch, 'TOP' if cap_on_upper(scene) else 'BOTTOM')] if is_scan(props.cap_arch) else []
    return [(a, face) for a, face in ((upper_arch(scene), 'TOP'), (lower_arch(scene), 'BOTTOM')) if a is not None]


def tree_of(obj, depsgraph):
    key = (obj.name, obj.data.name, len(obj.data.vertices))
    if key not in _trees:
        _trees[key] = BVHTree.FromObject(obj, depsgraph)
    return _trees[key]


def first_hits(obj, origins, direction, depsgraph):
    """Parallel world rays on one object: distance to the first hit (inf where missed).
    The arches move rigidly (no scale)."""
    tree = tree_of(obj, depsgraph)
    to_local = obj.matrix_world.inverted()
    d = (to_local.to_3x3() @ Vector(direction)).normalized()
    dist = np.full(len(origins), np.inf)
    for i, o in enumerate(origins):
        location, _normal, _index, t = tree.ray_cast(to_local @ Vector(o), d)
        if location is not None:
            dist[i] = t
    return dist


def dilate(heights, gap, radius, row_step, col_steps):
    """Tooth height field (rows x cols, -inf where no tooth) pushed out by a rolled ball: `gap` above
    the surface, `radius` sideways off the walls (an ellipsoid when radius > gap). Rows are `row_step`
    apart, columns `col_steps[row]` apart (the blank is wider at the molars, same number of columns)."""
    rows, cols = heights.shape
    j_max, k_max = int(radius / row_step), int(radius / col_steps.min())
    padded = np.pad(heights, ((j_max, j_max), (k_max, k_max)), constant_values=-np.inf)
    out = np.full_like(heights, -np.inf)
    for j in range(-j_max, j_max + 1):
        for k in range(-k_max, k_max + 1):
            d = np.hypot(j * row_step, k * col_steps) / radius        # per row
            bump = np.where(d <= 1, gap * np.sqrt(np.clip(1 - d * d, 0, None)), -np.inf)
            shifted = padded[j_max + j:j_max + j + rows, k_max + k:k_max + k + cols]
            out = np.maximum(out, shifted + bump[:, None])
    return out


def imprint(scene, base, up, depth_up, depth_low, depsgraph, row_step, col_steps):
    """Heights of the top and bottom faces above each base point (rows x cols grid on the curve's
    surface), pressed onto the teeth with the clearance, and the mask of the points where the two
    arches leave no room (holes). depth_up / depth_low: per base point (the splint's surfaces vary)."""
    props = scene.omfs_splint
    rows, cols = len(col_steps), len(base) // len(col_steps)
    radius = max(props.clearance, props.resolution)     # a step off a wall lands at least one column out
    top = np.array(depth_up, float)
    bottom = -np.array(depth_low, float)
    for arch, face in imprint_faces(scene):
        side = 1.0 if face == 'TOP' else -1.0          # the upper arch is met going up
        dist = first_hits(arch, base - side * CENTRE_REACH * up, side * up, depsgraph)
        # How far the teeth reach into the blank, from their side (-inf where none): the lower
        # arch's height, minus the upper arch's height.
        into = (CENTRE_REACH - dist).reshape(rows, cols)
        reach = dilate(into, props.clearance, radius, row_step, col_steps).ravel()
        if face == 'TOP':
            top = np.minimum(top, -reach)
        else:
            bottom = np.maximum(bottom, reach)
    contact = top < bottom
    middle = (top + bottom) / 2
    return np.where(contact, middle, top), np.where(contact, middle, bottom), contact


def round_edges(top, bottom, contact, depth_up, depth_low, radius, rows_s, f, widths):
    """Rounded rim: near the outline (lingual and buccal sides, both ends) the flat top comes down and
    the flat bottom comes up along a quarter circle of this radius, meeting the vertical wall
    tangentially. Imprints deeper than that are kept. The radius is capped below half the thickness."""
    radius = np.minimum(radius, 0.45 * (depth_up + depth_low))     # per point (the splint's thickness varies)
    across = np.minimum(f[None, :], 1.0 - f[None, :]) * widths[:, None]        # mm to the nearer side wall
    along = np.minimum(rows_s, rows_s[-1] - rows_s)[:, None]                    # mm to the nearer end
    u = np.minimum(across, along).ravel()
    drop = np.where(u < radius, radius - np.sqrt(np.clip(radius ** 2 - (radius - u) ** 2, 0.0, None)), 0.0)
    top = np.where(contact, top, np.minimum(top, depth_up - drop))
    bottom = np.where(contact, bottom, np.maximum(bottom, -depth_low + drop))
    meet = top < bottom                                                         # thin there: holes, like contacts
    middle = (top + bottom) / 2
    return np.where(meet, middle, top), np.where(meet, middle, bottom), contact | meet


def curve_key(obj):
    return repr([tuple(round(c, 4) for c in p) for p in curve_points(obj)])


def blank_key(scene):
    if scene.omfs_splint.kind != 'CHAPE':   # the splint: its coverage points (they move with their arch)
        where = [tuple(round(c, 4) for c in o.matrix_world.translation) for o in coverage_points(scene)]
    else:
        where = [curve_key(find_curve(scene)), [round(d, 3) for d in depths(scene)]]
    return repr((scene.omfs_splint.kind == 'CHAPE', where, [o.name for o in arches(scene)],
                 [tuple(map(tuple, o.matrix_world)) for o in arches(scene)]))


def blank_data(scene, depsgraph):
    """Vertices, faces, and the top / bottom vertex indices of the blank, or an error message."""
    props = scene.omfs_splint
    if not arches(scene):
        return "Choisissez les arcades"
    up = up_axis(scene)
    step = params.SPLINT_MEASURE_STEP
    line = centreline(scene, depsgraph, step)
    if isinstance(line, str):
        return line
    points, tangents, normals, length, depth_up, depth_low = line
    key = blank_key(scene)
    if key not in _measures:
        _measures.clear()
        _measures[key] = measure(scene, points, tangents, normals, up, depth_up, depth_low, step)
    measured = _measures[key]
    if measured is None:
        return "Aucune dent trouvée le long de l'arcade: vérifiez les arcades et les points"
    lingual, buccal = measured
    if props.kind == 'CHAPE':
        window = int(round(params.SPLINT_SMOOTH / step))
        lingual = smooth(lingual, window) + props.lingual_margin
        buccal = smooth(buccal, window) + props.buccal_margin
    else:   # a round envelope: never inside the clearance, smoothed over a long stretch
        window = int(round(props.outline_smooth / step))
        lingual = gaussian(smooth(lingual, window), window / 4) + props.splint_lingual_margin
        buccal = gaussian(smooth(buccal, window), window / 4) + props.splint_buccal_margin

    # Rows every `resolution` mm, from one end of the curve to the other.
    res = props.resolution
    station_s = np.linspace(0.0, length, len(points))
    rows_s = np.linspace(0.0, length, max(2, int(np.ceil(length / res)) + 1))
    c = np.stack([np.interp(rows_s, station_s, points[:, i]) for i in range(3)], axis=1)
    n = np.stack([np.interp(rows_s, station_s, normals[:, i]) for i in range(3)], axis=1)
    n /= np.linalg.norm(n, axis=1)[:, None]
    lin = np.interp(rows_s, station_s, lingual)
    buc = np.interp(rows_s, station_s, buccal)
    above = np.interp(rows_s, station_s, depth_up)
    below = np.interp(rows_s, station_s, depth_low)

    cols = int(np.ceil((lin + buc).max() / res))
    f = np.linspace(0.0, 1.0, cols + 1)
    left = c - n * lin[:, None]
    right = c + n * buc[:, None]
    grid = left[:, None, :] + (right - left)[:, None, :] * f[None, :, None]   # rows x cols x 3
    up = np.array(up)
    base = grid.reshape(-1, 3)
    above, below = np.repeat(above, cols + 1), np.repeat(below, cols + 1)   # per base point
    top_h, bottom_h = above.copy(), -below
    contact = np.zeros(len(base), bool)
    if props.imprint:
        top_h, bottom_h, contact = imprint(scene, base, up, above, below, depsgraph,
                                           length / (len(rows_s) - 1), (lin + buc) / cols)
    if props.edge_round > 0:
        top_h, bottom_h, contact = round_edges(top_h, bottom_h, contact, above, below, props.edge_round,
                                               rows_s, f, lin + buc)
    rows, w = len(rows_s), cols + 1
    rim = np.zeros((rows, w), bool)                              # the outline: walls and end caps
    rim[[0, -1], :] = rim[:, [0, -1]] = True
    rim = rim.ravel() & contact                                  # teeth meeting on the outline: a thin rim
    middle = (top_h + bottom_h) / 2                              # stays, so the hole never opens the wall
    top_h = np.where(rim, middle + params.SPLINT_RIM_MIN / 2, top_h)
    bottom_h = np.where(rim, middle - params.SPLINT_RIM_MIN / 2, bottom_h)
    contact = contact & ~rim
    verts = np.concatenate([base + up * top_h[:, None], base + up * bottom_h[:, None]])

    idx = np.arange(rows * w).reshape(rows, w)
    bot = idx + rows * w
    a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = [np.stack([a, b, cc, d], axis=-1).reshape(-1, 4)]
    a, b, cc, d = bot[:-1, :-1], bot[:-1, 1:], bot[1:, 1:], bot[1:, :-1]
    faces.append(np.stack([d, cc, b, a], axis=-1).reshape(-1, 4))
    for col in (0, -1):                                          # lingual and buccal walls
        faces.append(np.stack([idx[:-1, col], idx[1:, col], bot[1:, col], bot[:-1, col]], axis=-1))
    for row in (0, -1):                                          # end caps
        faces.append(np.stack([idx[row, :-1], bot[row, :-1], bot[row, 1:], idx[row, 1:]], axis=-1))
    faces = np.concatenate(faces)
    cell = (length / (rows - 1)) * ((lin + buc) / cols)            # mm2 around a grid point, per row
    info = {"length": float(rows_s[-1] - rows_s[0]), "width_min": float((lin + buc).min()),
            "width_max": float((lin + buc).max()), "imprint": bool(props.imprint),
            "contact_area": float((contact.reshape(rows, w) * cell[:, None]).sum()),
            "thinnest": float((top_h - bottom_h)[~contact].min()) if (~contact).any() else 0.0}
    return verts, faces, np.arange(rows * w), np.arange(rows * w) + rows * w, info


def find_blank(scene):
    return next((o for o in scene.objects if o.type == 'MESH' and o.omfs.made_by == BLANK_TAG), None)


def build_blank(scene, depsgraph):
    """Create or rebuild the blank. Returns the object, or an error message."""
    data = blank_data(scene, depsgraph)
    if isinstance(data, str):
        return data
    verts, faces, top, bottom, info = data
    obj = find_blank(scene)
    if obj is None:
        obj = bpy.data.objects.new(BLANK_NAME, bpy.data.meshes.new(BLANK_NAME))
        splint_collection(scene).objects.link(obj)
        obj.omfs.role = 'GUIDE'
        obj.omfs.made_by = BLANK_TAG
        obj.color = BLANK_COLOUR
    mesh = obj.data
    mesh.clear_geometry()
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 4, dtype=np.int32))
    mesh.update(calc_edges=True)
    bm = bmesh.new()                                             # outward normals, whatever the click order
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    obj.matrix_world.identity()
    for name, indices in ((TOP_GROUP, top), (BOTTOM_GROUP, bottom)):
        group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        group.add(indices.tolist(), 1.0, 'REPLACE')
    obj["omfs_blank_key"] = blank_key(scene)
    obj["omfs_blank_info"] = json.dumps(info)
    text_labels.rebuild_on(obj)                                 # its text follows the new surface
    return obj


def blank_info(obj):
    return json.loads(obj.get("omfs_blank_info", "{}"))


def blank_stale(scene, obj):
    return obj.get("omfs_blank_key") != blank_key(scene)


def _on_blank_param(props, context):
    if find_blank(context.scene) is not None:
        build_blank(context.scene, context.evaluated_depsgraph_get())


# --- Solid ----------------------------------------------------------------------------------
#
# "Solidifier" freezes the blank into its own object: the parameters no longer touch it, so it can
# be sculpted, polished or exported. Where the arches leave no room (top and bottom meet), real
# holes are opened. Text already put on the blank comes along (its Boolean modifiers are copied).
# The blank stays, hidden, to start again from the parameters; each solidify makes a new object.

SOLID_TAG = "splint solid"      # made_by of a solidified splint
SOLID_COLOUR = (0.95, 0.70, 0.30, 1.0)


def contact_holes(mesh):
    """Open the patches where the blank's top and bottom meet (zero thickness): their faces are removed
    and the meeting top / bottom vertices welded, so the rim of each hole closes the slab. The mesh
    must still be the blank's (top grid, then bottom grid). Returns the number of vertices welded."""
    n = len(mesh.vertices) // 2
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    touch = np.linalg.norm(co[:n] - co[n:], axis=1) < 1e-4
    if not touch.any():
        return 0
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    inside = [f for f in bm.faces if all(touch[v.index % n] for v in f.verts)
              and len({v.index < n for v in f.verts}) == 1]          # top or bottom grid, not a wall
    # Only the rims of real holes are welded: a contact one vertex wide (no face of it fully touching) is
    # left as two coincident vertices, or welding would join four faces on one edge.
    rims = {v.index % n for f in inside for v in f.verts}
    bmesh.ops.delete(bm, geom=inside, context='FACES_ONLY')
    weld = {bm.verts[i + n]: bm.verts[i] for i in sorted(rims)}
    bmesh.ops.weld_verts(bm, targetmap=weld)
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context='EDGES')
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return len(weld)


def open_edges(mesh):
    """Edges not shared by exactly two faces (0 for a closed, printable mesh)."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    count = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return count


def find_solids(scene):
    return [o for o in scene.objects if o.type == 'MESH' and o.omfs.made_by == SOLID_TAG]


def solidify(scene):
    """A new object from the blank, with holes at the contacts. Returns it, or an error message."""
    blank = find_blank(scene)
    if blank is None:
        return "Construisez d'abord le bloc"
    if blank_stale(scene, blank):
        return "Les points, la courbe ou les arcades ont changé: mettez le bloc à jour d'abord"
    label = dict((k, name) for k, name, _d in KINDS)[scene.omfs_splint.kind]
    obj = blank.copy()                  # copies the modifiers too (text put on the blank)
    obj.data = blank.data.copy()
    obj.name = obj.data.name = f"{label} (solide)"
    splint_collection(scene).objects.link(obj)
    obj.omfs.made_by = SOLID_TAG
    obj.color = SOLID_COLOUR
    del obj["omfs_blank_key"]
    welded = contact_holes(obj.data)
    info = dict(blank_info(blank), welded=welded, open_edges=open_edges(obj.data))
    obj["omfs_blank_info"] = json.dumps(info)
    blank.hide_set(True)
    return obj


# --- Text and print checks -------------------------------------------------------------------

def is_part(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.made_by in (BLANK_TAG, SOLID_TAG)


def current_part(context):
    """What text and checks apply to: the active splint (solid or blank, or the part of the active
    text), else the latest visible solid, else the blank."""
    active = context.view_layer.objects.active
    if text_labels.is_label(active) and is_part(active.omfs_text.part):
        return active.omfs_text.part
    if is_part(active) and active.visible_get():
        return active
    solids = [o for o in find_solids(context.scene) if o.visible_get()]
    return solids[-1] if solids else find_blank(context.scene)


def check(part, depsgraph):
    """The generic print check (text_labels.print_check), plus the splint's thinnest spot."""
    result = text_labels.print_check(part, depsgraph)
    result["thinnest"] = blank_info(part).get("thinnest")
    part["omfs_check"] = json.dumps(result)
    return result


def last_check(part):
    return text_labels.last_check(part)


# --- Properties and operators ---------------------------------------------------------

def _on_wire_hole_diameter(props, context):
    from . import splint_holes
    splint_holes.set_diameter(context.scene, props.wire_hole_diameter)


class OMFS_SplintProperties(bpy.types.PropertyGroup):
    kind: bpy.props.EnumProperty(name="Type", items=KINDS, default='FINAL', update=_on_blank_param)
    upper: bpy.props.PointerProperty(name="Arcade du haut", type=bpy.types.Object,
                                     poll=lambda _p, o: is_scan(o), description="Scan de l'arcade du haut (par défaut: celle de l'assise de l'occlusion)")
    lower: bpy.props.PointerProperty(name="Arcade du bas", type=bpy.types.Object,
                                     poll=lambda _p, o: is_scan(o), description="Scan de l'arcade du bas (par défaut: celle de l'assise de l'occlusion)")
    cap_arch: bpy.props.PointerProperty(name="Arcade", type=bpy.types.Object, poll=lambda _p, o: is_scan(o),
                                        description="Scan de l'arcade que la chape recouvre")
    buccal_margin: bpy.props.FloatProperty(
        name="Vestibulaire", subtype='DISTANCE', default=params.SPLINT_BUCCAL_MARGIN, min=0.0, soft_max=10.0,
        step=10, update=_on_blank_param,
        description="Débord au-delà de la face vestibulaire des couronnes (plus pour une gouttière segmentaire)")
    lingual_margin: bpy.props.FloatProperty(
        name="Lingual", subtype='DISTANCE', default=params.SPLINT_LINGUAL_MARGIN, min=0.0, soft_max=10.0,
        step=10, update=_on_blank_param, description="Débord au-delà de la face linguale / palatine des couronnes")
    splint_buccal_margin: bpy.props.FloatProperty(
        name="Vestibulaire", subtype='DISTANCE', default=params.SPLINT_BUCCAL_CLEARANCE, min=0.0, soft_max=10.0,
        step=10, update=_on_blank_param,
        description="Gouttière: débord au-delà de la face vestibulaire des couronnes (enveloppe lissée)")
    splint_lingual_margin: bpy.props.FloatProperty(
        name="Lingual", subtype='DISTANCE', default=params.SPLINT_LINGUAL_CLEARANCE, min=0.0, soft_max=10.0,
        step=10, update=_on_blank_param,
        description="Gouttière: débord au-delà de la face linguale / palatine des couronnes (enveloppe lissée)")
    outline_smooth: bpy.props.FloatProperty(
        name="Rondeur du contour", subtype='DISTANCE', default=params.SPLINT_OUTLINE_SMOOTH, min=1.0, soft_max=30.0,
        step=100, precision=0, update=_on_blank_param,
        description="Le contour enveloppe les couronnes, lissé sur cette longueur d'arcade: plus long = plus rond "
                    "(jamais à l'intérieur des débords)")
    depth_up: bpy.props.FloatProperty(
        name="Haut", subtype='DISTANCE', default=params.SPLINT_DEPTH, min=0.5, soft_max=8.0, step=10,
        update=_on_blank_param,
        description="Jusqu'où la gouttière monte sur les dents du haut, depuis le milieu de l'occlusion")
    depth_low: bpy.props.FloatProperty(
        name="Bas", subtype='DISTANCE', default=params.SPLINT_DEPTH, min=0.5, soft_max=8.0, step=10,
        update=_on_blank_param,
        description="Jusqu'où la gouttière descend sur les dents du bas, depuis le milieu de l'occlusion")
    cap_depth: bpy.props.FloatProperty(
        name="Sur les dents", subtype='DISTANCE', default=params.SPLINT_CAP_DEPTH, min=0.1, soft_max=5.0, step=10,
        update=_on_blank_param, description="Jusqu'où la chape descend sur les dents, depuis la courbe occlusale")
    resolution: bpy.props.FloatProperty(
        name="Résolution", subtype='DISTANCE', default=params.SPLINT_RESOLUTION, min=0.05, soft_max=1.0,
        step=1, precision=2, update=_on_blank_param,
        description="Distance entre les sommets du bloc: l'empreinte des dents ne sera pas plus fine que ça. "
                    "Les scans intraoraux ont environ 0,15 mm entre leurs sommets: sous 0,05 mm, rien de plus à copier")
    edge_round: bpy.props.FloatProperty(
        name="Arrondi des bords", subtype='DISTANCE', default=params.SPLINT_EDGE_ROUND, min=0.0, soft_max=3.0,
        step=10, precision=2, update=_on_blank_param,
        description="Rayon des rebords arrondis (arêtes du dessus et du dessous, tout le pourtour). 0 = bords vifs")
    wire_hole_diameter: bpy.props.FloatProperty(
        name="Diamètre", subtype='DISTANCE', default=params.SPLINT_WIRE_HOLE_DIAMETER, min=0.3, soft_max=3.0,
        step=10, precision=2, update=lambda props, context: _on_wire_hole_diameter(props, context),
        description="Diamètre des trous pour le fil d'ostéosynthèse (tous les trous)")
    imprint: bpy.props.BoolProperty(
        name="Empreinte des dents", default=False, update=_on_blank_param,
        description="Presse le dessus et le dessous du bloc sur les dents le long de l'axe d'insertion "
                    "(le haut du référentiel): pas de contre-dépouille, la gouttière s'insère et s'enlève")
    clearance: bpy.props.FloatProperty(
        name="Jeu", subtype='DISTANCE', default=params.SPLINT_CLEARANCE, min=0.0, soft_max=0.5, step=1,
        precision=2, update=_on_blank_param,
        description="Espace entre l'empreinte et les dents, mesuré perpendiculairement à la dent")
    centre: bpy.props.BoolProperty(
        name="Centrer entre les arcades", default=True,
        description="Chaque point cliqué est placé à mi-hauteur entre les faces occlusales du haut et du bas")


def arch_under_mouse(context, region, mouse):
    """(arch, side 'U' / 'L', world point) of the nearest of the two arches under the mouse, or None."""
    scene = context.scene
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
    depsgraph = context.evaluated_depsgraph_get()
    hits = [(arch, side, ray_hit(arch, origin, direction, depsgraph))
            for arch, side in ((upper_arch(scene), 'U'), (lower_arch(scene), 'L')) if arch is not None]
    hits = [h for h in hits if h[2] is not None]
    return min(hits, key=lambda h: (h[2] - origin).length) if hits else None


class OMFS_OT_splint_points(bpy.types.Operator):
    """Click on the buccal surface of the teeth where the splint must stop covering them, upper and lower
    (a few per side and one in front). Click and drag an existing point to slide it on its tooth.
    Backspace: remove the last one. Right-click or Enter: done"""
    bl_idname = "omfs.splint_points"
    bl_label = "Placer / glisser les points"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        scene = context.scene
        if upper_arch(scene) is None or lower_arch(scene) is None:
            self.report({'ERROR'}, "Choisissez d'abord les deux arcades")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.dragging = None
        self.header(context)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def header(self, context):
        up, low = len(coverage_points(context.scene, 'U')), len(coverage_points(context.scene, 'L'))
        self.area.header_text_set(f"Couverture: cliquez sur la face vestibulaire des dents, jusqu'où la gouttière "
                                  f"les recouvre (haut {up}, bas {low}); cliquez-glissez un point pour l'ajuster   |   "
                                  "Retour arrière: retirer le dernier   |   Clic droit / Entrée: terminer")

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        return {'FINISHED'}

    def rebuild(self, context):
        if find_blank(context.scene) is not None:
            build_blank(context.scene, context.evaluated_depsgraph_get())

    def nearest_point(self, context, region, mouse):
        best, best_d = None, 20.0                           # pixels
        for obj in coverage_points(context.scene):
            p = location_3d_to_region_2d(region, region.data, obj.matrix_world.translation)
            if p is not None and (p - Vector(mouse)).length < best_d:
                best, best_d = obj, (p - Vector(mouse)).length
        return best

    def modal(self, context, event):
        scene = context.scene
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER', 'ESC'}:
            return self.finish(context)
        if event.value == 'PRESS' and event.type == 'BACK_SPACE':
            placed = coverage_points(scene)
            if placed:
                bpy.data.objects.remove(max(placed, key=lambda o: int(o.omfs.landmark[5:])))
                self.rebuild(context)
            self.header(context)
            return {'RUNNING_MODAL'}
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if event.type == 'MOUSEMOVE' and self.dragging is not None and under_mouse is not None:
            hit = arch_under_mouse(context, *under_mouse)
            if hit is not None and hit[0] is self.dragging.parent:   # slides on its own arch
                self.dragging.matrix_world = Matrix.Translation(hit[2])
                self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.dragging is not None:
            self.dragging = None
            self.rebuild(context)
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS' or under_mouse is None:
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        self.dragging = self.nearest_point(context, *under_mouse)
        if self.dragging is not None:
            return {'RUNNING_MODAL'}
        hit = arch_under_mouse(context, *under_mouse)
        if hit is None:
            self.report({'WARNING'}, "Cliquez sur une arcade")
            return {'RUNNING_MODAL'}
        arch, side, location = hit
        add_coverage_point(scene, arch, location, side)
        self.rebuild(context)
        self.header(context)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}


class OMFS_OT_remove_splint_point(bpy.types.Operator):
    """Remove this coverage point (empty name: all of them)"""
    bl_idname = "omfs.remove_splint_point"
    bl_label = "Retirer le point"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty()

    def execute(self, context):
        for obj in coverage_points(context.scene):
            if not self.name or obj.name == self.name:
                bpy.data.objects.remove(obj)
        if find_blank(context.scene) is not None and coverage_points(context.scene):
            build_blank(context.scene, context.evaluated_depsgraph_get())
        return {'FINISHED'}


class OMFS_OT_arch_curve_click(bpy.types.Operator):
    """Click points along the arch, from the last molar on one side to the last molar on the other.
    Right-click, Enter or Esc to finish"""
    bl_idname = "omfs.arch_curve_click"
    bl_label = "Cliquer les points"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if not arches(context.scene):
            self.report({'ERROR'}, "Choisissez d'abord la ou les arcades (scans)")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.header(context)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def header(self, context):
        n = len(curve_points(find_curve(context.scene)))
        self.area.header_text_set(f"Cliquez sur les faces occlusales, d'une dernière molaire à l'autre ({n} points)"
                                  "   |   Retour arrière: retirer le dernier   |   Clic droit / Entrée / Échap: terminer")

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER', 'ESC'}:
            self.finish(context)
            return {'FINISHED'}
        if event.value == 'PRESS' and event.type == 'BACK_SPACE':
            remove_last_point(context.scene)
            self.header(context)
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        region, mouse = under_mouse
        origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
        direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
        depsgraph = context.evaluated_depsgraph_get()
        hits = [h for h in (ray_hit(o, origin, direction, depsgraph) for o in arches(context.scene)) if h]
        if not hits:
            self.report({'WARNING'}, "Cliquez sur une arcade")
            return {'RUNNING_MODAL'}
        add_point(context.scene, min(hits, key=lambda h: (h - origin).length), depsgraph)
        self.header(context)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}


class OMFS_OT_build_blank(bpy.types.Operator):
    """Build (or update) the blank: a splint between the surfaces through its coverage points, a cap along
    its arch curve; crown width plus the margins"""
    bl_idname = "omfs.build_splint_blank"
    bl_label = "Construire le bloc"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        result = build_blank(context.scene, context.evaluated_depsgraph_get())
        if isinstance(result, str):
            self.report({'ERROR'}, result)
            return {'CANCELLED'}
        return {'FINISHED'}


class OMFS_OT_solidify_splint(bpy.types.Operator):
    """Freeze the splint into its own object (the settings no longer change it): holes where the
    teeth touch, text already placed comes along. The parametric blank is kept, hidden"""
    bl_idname = "omfs.solidify_splint"
    bl_label = "Solidifier en un objet"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        result = solidify(context.scene)
        if isinstance(result, str):
            self.report({'ERROR'}, result)
            return {'CANCELLED'}
        for other in context.selected_objects:
            other.select_set(False)
        context.view_layer.objects.active = result
        result.select_set(True)
        info = blank_info(result)
        self.report({'INFO'} if info["open_edges"] == 0 else {'WARNING'},
                    f"{result.name}: fermé" if info["open_edges"] == 0
                    else f"{result.name}: {info['open_edges']} arêtes non fermées (voir Vérification)")
        return {'FINISHED'}


class OMFS_OT_check_splint(bpy.types.Operator):
    """Check the splint as it will be printed: closed mesh, text Booleans applied, material left
    under the engravings, letter size"""
    bl_idname = "omfs.check_splint"
    bl_label = "Vérifier"

    part_name: bpy.props.StringProperty()

    def execute(self, context):
        part = bpy.data.objects.get(self.part_name)
        if part is None:
            return {'CANCELLED'}
        result = check(part, context.evaluated_depsgraph_get())
        self.report({'INFO'} if result["ok"] else {'WARNING'},
                    "Prêt à imprimer" if result["ok"] else "À corriger: voir le panneau")
        return {'FINISHED'}


class OMFS_OT_show_blank(bpy.types.Operator):
    """Show the parametric blank again, to change the settings and solidify anew"""
    bl_idname = "omfs.show_splint_blank"
    bl_label = "Montrer le bloc paramétrique"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        blank = find_blank(context.scene)
        if blank is not None:
            blank.hide_set(False)
        return {'FINISHED'}


class OMFS_OT_arch_curve_undo(bpy.types.Operator):
    """Remove the last point of the arch curve"""
    bl_idname = "omfs.arch_curve_undo"
    bl_label = "Retirer le dernier point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        remove_last_point(context.scene)
        return {'FINISHED'}


class OMFS_OT_arch_curve_clear(bpy.types.Operator):
    """Delete the arch curve to start again"""
    bl_idname = "omfs.arch_curve_clear"
    bl_label = "Effacer la courbe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = find_curve(context.scene)
        if obj is not None:
            curve = obj.data
            bpy.data.objects.remove(obj)
            if curve.users == 0:
                bpy.data.curves.remove(curve)
        return {'FINISHED'}


class OMFS_OT_arch_curve_edit(bpy.types.Operator):
    """Edit the curve's points: G to move (snapped onto the surfaces), X to delete,
    Ctrl + right-click to add one. Tab to finish"""
    bl_idname = "omfs.arch_curve_edit"
    bl_label = "Modifier les points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = find_curve(context.scene)
        if obj is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        locking.set_locked(obj, False)
        for other in context.selected_objects:
            other.select_set(False)
        context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        ts = context.scene.tool_settings
        ts.use_snap = True
        ts.snap_elements = {'FACE_PROJECT'}
        return {'FINISHED'}


# --- Drawing --------------------------------------------------------------------------------

def _draw_curve():
    """The arch curve, thick and on top, so it reads over the teeth."""
    context = bpy.context
    obj = find_curve(context.scene)
    if obj is None or not obj.visible_get():
        return
    line = sampled_curve(obj, context.evaluated_depsgraph_get())
    points = curve_points(obj)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    if len(line) > 1:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("viewportSize", (context.region.width, context.region.height))
        shader.uniform_float("lineWidth", 4.0)
        shader.uniform_float("color", CURVE_COLOUR)
        batch_for_shader(shader, 'LINE_STRIP', {"pos": line}).draw(shader)
    if points:
        dots = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
        gpu.state.point_size_set(11.0)
        dots.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_for_shader(dots, 'POINTS', {"pos": points}).draw(dots)
        gpu.state.point_size_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_numbers():
    context = bpy.context
    obj = find_curve(context.scene)
    if obj is None or not obj.visible_get():
        return
    font = 0
    blf.size(font, 13 * context.preferences.system.ui_scale)
    blf.color(font, *CURVE_COLOUR)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
    for i, p in enumerate(curve_points(obj), start=1):
        xy = location_3d_to_region_2d(context.region, context.region_data, p)
        if xy is not None:
            blf.position(font, xy.x + 8, xy.y + 8, 0)
            blf.draw(font, str(i))
    blf.disable(font, blf.SHADOW)


_draw_handles = []

classes = (OMFS_SplintProperties, OMFS_OT_splint_points, OMFS_OT_remove_splint_point, OMFS_OT_arch_curve_click, OMFS_OT_arch_curve_undo,
           OMFS_OT_arch_curve_clear, OMFS_OT_arch_curve_edit, OMFS_OT_build_blank, OMFS_OT_solidify_splint,
           OMFS_OT_show_blank, OMFS_OT_check_splint)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_splint = bpy.props.PointerProperty(type=OMFS_SplintProperties)
    if not bpy.app.background:
        _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_curve, (), 'WINDOW', 'POST_VIEW'))
        _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_numbers, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    for handle in _draw_handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _draw_handles.clear()
    del bpy.types.Scene.omfs_splint
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
