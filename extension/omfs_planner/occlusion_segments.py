"""Segmental seating, first step: where the upper arch is cut into segments (surgeon, 2026-11-19).

Works on a copy of the upper arch scan (the occlusal set-up: the scan itself is never cut).
- A cut along the palatal midline is there from the start: from behind the palate up to just behind the
  two central incisors (never between them). Every cut between two teeth ends on it, so a segment is
  always closed on the palatal side.
- Between every two teeth, a candidate cut (black, dashed): a plane through the contact point, across the
  arch, tilted to run midway between the two roots (CBCT), then a palatal piece straight to the midline cut.
  Drawn as its exact section of the scan. Hovering highlights one (with the narrowest gap between the
  roots); clicking chooses it (click again: removed).
- The chosen cuts and the midline split the arch into segments, numbered from the patient's right to left,
  tinted in the view. Zero thickness (surgeon, 2026-11-19): the arch is cut exactly at the junction, no saw
  kerf, so the dental dimensions stay; a kerf only comes when the cut is carried onto the bone.

How the arch is read (no landmark needed; the upper incisor midline landmark refines the midline if placed):
- the crowns (crown_fusion.crown_vertex_mask) give the arch; the occlusal plane is their least-spread axis
  (pointing from the palate to the cusp tips), the midline their axis of mirror symmetry, the front the
  side where the U bends;
- the arch curve runs through the middle of the crowns (polar bins around the crowns' centre);
- contact points: where the proximal walls are. In a thin slab across the arch, the surface points facing
  along the arch (towards the next tooth) are many at a contact (the slab runs along both teeth's proximal
  walls) and few mid-tooth (only the outline). Read on the scan's crowns and on the CBCT teeth (crown and root
  trunks, at their pre-op place): on the patient's scans the peaks fall every 7-10.5 mm, the widths of the
  teeth. Every clear peak is a candidate: no assumption on the teeth's widths; a rare extra peak is only a
  line the surgeon doesn't click.
"""

import json

import blf
import bpy
import gpu
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import crown_fusion, landmarks, locking, movement, occlusion, occlusion_prepare, params, registration


# --- Arch frame ---------------------------------------------------------------------------------

def _symmetry_error(p2, axis):
    """Mean distance from the points mirrored about this axis (through the origin) to the nearest point."""
    from mathutils.kdtree import KDTree
    sample = p2[:: max(1, len(p2) // 1500)]
    tree = KDTree(len(sample))
    for i, (x, y) in enumerate(sample):
        tree.insert((x, y, 0.0), i)
    tree.balance()
    normal = np.array((-axis[1], axis[0]))
    mirrored = sample - 2 * np.outer(sample @ normal, normal)
    return float(np.mean([tree.find((x, y, 0.0))[2] for x, y in mirrored]))


def arch_frame(co, crown, palate, incisor=None):
    """(origin, anterior, left, occlusal) of an upper arch, world vectors. co: world vertices; crown, palate:
    masks (crown vertices, the others). occlusal points from the palate towards the cusp tips."""
    pts = co[crown]
    origin = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - origin, full_matrices=False)
    occlusal = vt[2]
    if (co[palate] - origin).mean(axis=0) @ occlusal > 0:   # the palate is on the other side of the cusps
        occlusal = -occlusal
    e1, e2 = vt[0], vt[1]
    p2 = np.column_stack(((pts - origin) @ e1, (pts - origin) @ e2))
    if incisor is not None:   # the midline through the upper incisor landmark
        d = np.array(incisor) - origin
        d -= (d @ occlusal) * occlusal
        anterior = d / np.linalg.norm(d)
    else:
        axes = [np.array((1.0, 0.0)), np.array((0.0, 1.0))]
        sym = min(axes, key=lambda a: _symmetry_error(p2, a))
        near_axis = np.abs(p2 @ np.array((-sym[1], sym[0]))) < params.SEG_MIDLINE_BAND
        sign = 1.0 if (p2[near_axis] @ sym).mean() > 0 else -1.0   # the bend of the U: the front
        anterior = sign * (sym[0] * e1 + sym[1] * e2)
    left = np.cross(anterior, occlusal)   # upper arch: occlusal points down, so anterior x down = left
    return origin, anterior, left, occlusal


# --- Arch curve and contacts --------------------------------------------------------------------

def arch_curve(p2):
    """Curve through the middle of the crowns (2D points (left, anterior) relative to the crowns' centre):
    (stations (n, 2), arch length s (n,)), from the patient's right end to the left end."""
    angle = np.degrees(np.arctan2(p2[:, 0], p2[:, 1]))   # 0 at the front, + to the left
    bins = np.arange(-179.0, 180.0, 2.0)
    stations = []
    for lo in bins:
        inside = (angle >= lo) & (angle < lo + 2.0)
        if inside.sum() >= params.SEG_MIN_BIN_POINTS:
            stations.append(np.median(p2[inside], axis=0))
    stations = np.array(stations)
    for _ in range(3):   # smooth (ends kept)
        stations[1:-1] = (stations[:-2] + 2 * stations[1:-1] + stations[2:]) / 4
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(stations, axis=0), axis=1))))
    fine = np.arange(0.0, s[-1], params.SEG_STEP)
    curve = np.column_stack([np.interp(fine, s, stations[:, k]) for k in range(2)])
    return curve, fine


def tangents(curve):
    """Direction of the curve, averaged over +/- SEG_TANGENT_SPAN mm (steady at the ends of the arch)."""
    k = max(1, int(params.SEG_TANGENT_SPAN / params.SEG_STEP))
    ahead = curve[np.minimum(np.arange(len(curve)) + k, len(curve) - 1)]
    behind = curve[np.maximum(np.arange(len(curve)) - k, 0)]
    t = ahead - behind
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def proximal_walls(curve, p2, height, normals2, crown_p2, crown_height):
    """Per station: number of surface points facing along the arch (normal within SEG_WALL_ANGLE of the arch's
    direction) in a thin slab across it, from the gum line's level (minus SEG_ROOT_DEPTH, the root trunks) to
    the cusps. Those are the proximal walls: the slab cuts them at a contact, only the outline mid-tooth.
    p2, height, normals2: the points (2D), their height towards the cusps, their normals (2D, unnormalised)."""
    tan = tangents(curve)
    walls = np.zeros(len(curve))
    sharp = np.cos(np.radians(params.SEG_WALL_ANGLE))
    for i, (c, t) in enumerate(zip(curve, tan)):
        normal = np.array((-t[1], t[0]))
        near = (np.abs((crown_p2 - c) @ t) < 2.0) & (np.abs((crown_p2 - c) @ normal) < params.SEG_CROSS)
        if near.sum() < 3:
            continue
        gum = np.percentile(crown_height[near], 2)
        d = p2 - c
        inside = ((np.abs(d @ t) < params.SEG_SLAB) & (np.abs(d @ normal) < params.SEG_CROSS)
                  & (height > gum - params.SEG_ROOT_DEPTH))
        n = normals2[inside]
        facing = np.abs(n @ t) / np.maximum(np.linalg.norm(n, axis=1), 1e-9)
        walls[i] = (facing > sharp).sum()
    return walls


def contact_score(*walls):
    """Higher at a contact: the proximal walls of each source (scan, CBCT), each scaled to its own spread."""
    score = np.zeros(len(walls[0]))
    for w in walls:
        if w is not None and w.any():
            score += (w - w.mean()) / (w.std() or 1.0)
    k = max(1, int(params.SEG_SMOOTH / params.SEG_STEP))
    return np.convolve(score, np.ones(2 * k + 1) / (2 * k + 1), mode='same')


def contacts(s, score, s_mid):
    """Arch positions (s) of the contact points: every clear peak of the proximal-wall signal (higher than
    SEG_PEAK above the signal's mean, at least SEG_MIN_GAP from a higher peak), away from the arch's ends.
    No assumption on the teeth's widths; the peak nearest the midline is the contact between the centrals."""
    k = max(1, int(params.SEG_MIN_GAP / params.SEG_STEP))
    order = np.argsort(-score)
    taken = []
    for i in order:
        if score[i] < params.SEG_PEAK:
            break
        if not (s[0] + params.SEG_END_MARGIN < s[i] < s[-1] - params.SEG_END_MARGIN):
            continue
        if all(abs(i - j) > k for j in taken):
            taken.append(i)
    found = sorted(float(s[i]) for i in taken)
    if found:   # the contact between the centrals: the peak nearest the midline, if close to it
        nearest = min(found, key=lambda x: abs(x - s_mid))
        if abs(nearest - s_mid) > params.SEG_MIDLINE_SEARCH:
            found = sorted(found + [s_mid])
    return found


def station_of(curve, p2):
    """Index of the nearest curve station for each 2D point."""
    from mathutils.kdtree import KDTree
    tree = KDTree(len(curve))
    for i, (x, y) in enumerate(curve):
        tree.insert((x, y, 0.0), i)
    tree.balance()
    return np.array([tree.find((x, y, 0.0))[1] for x, y in p2])


def long_axes(arch, t_co, cuts):
    """Long axis (unit 3D vector, towards the cusps) of each tooth between two consecutive contacts, from the
    CBCT teeth (crown and root) at their pre-op place: the main direction of the tooth's points. t_co: world
    points of the CBCT teeth; cuts: contact positions (s). Returns a list, one per gap between contacts."""
    origin, anterior, left, occlusal = arch['frame']
    rel = t_co - origin
    p2 = np.column_stack((rel @ left, rel @ anterior))
    near = station_of(arch['curve'], p2)
    s_pt = arch['s'][near]
    across = np.linalg.norm(p2 - arch['curve'][near], axis=1)
    axes = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        margin = 0.2 * (hi - lo)   # the middle of the tooth: no neighbour's proximal wall
        pts = rel[(s_pt > lo + margin) & (s_pt < hi - margin) & (across < params.SEG_CROSS)]
        if len(pts) < 50:
            axes.append(occlusal.copy())
            continue
        _u, _sv, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        axis = vt[0] if vt[0] @ occlusal > 0 else -vt[0]
        axes.append(axis if axis @ occlusal > params.SEG_AXIS_MIN_COS else occlusal.copy())
    return axes


def analyse(obj, scene=None):
    """Everything the cut chooser needs about this upper arch copy, in world space:
    {'frame', 'curve' (2D), 's', 'contacts' (s), 'p2' (all vertices 2D), 'height', 'stop' (2D),
     'mid_s'}."""
    zero = np.array(movement.preop_matrix(bpy.context.scene, obj))   # position zero (it may be moved in its block)
    co = registration.vertex_coords(obj, world=False) @ zero[:3, :3].T + zero[:3, 3]
    crown = crown_fusion.crown_vertex_mask(obj)
    incisor = None
    if scene is not None:
        lm = landmarks.find(scene, 'UI')
        incisor = tuple(lm.matrix_world.translation) if lm is not None else None
    origin, anterior, left, occlusal = arch_frame(co, crown, ~crown, incisor)
    rel = co - origin
    p2 = np.column_stack((rel @ left, rel @ anterior))
    height = rel @ occlusal
    curve, s = arch_curve(p2[crown])
    normals = np.empty(len(co) * 3)
    obj.data.vertex_normals.foreach_get("vector", normals)
    normals = normals.reshape(-1, 3) @ zero[:3, :3].T
    scan_walls = proximal_walls(curve, p2[crown], height[crown], np.column_stack((normals @ left, normals @ anterior))[crown],
                                p2[crown], height[crown])
    cbct_walls = t_world = None
    teeth = obj.omfs_reg.target
    if teeth is not None and teeth.type == 'MESH':   # the CBCT teeth at their pre-op place (they may follow a segment)
        m = np.array(movement.preop_matrix(bpy.context.scene, teeth))
        t_co = registration.vertex_coords(teeth, world=False) @ m[:3, :3].T + m[:3, 3] - origin
        t_n = np.empty(len(teeth.data.vertices) * 3)
        teeth.data.vertex_normals.foreach_get("vector", t_n)
        t_n = t_n.reshape(-1, 3) @ m[:3, :3].T
        cbct_walls = proximal_walls(curve, np.column_stack((t_co @ left, t_co @ anterior)), t_co @ occlusal,
                                    np.column_stack((t_n @ left, t_n @ anterior)), p2[crown], height[crown])
        t_world = t_co + origin
    score = contact_score(scan_walls, cbct_walls)
    s_mid = float(s[np.argmin(np.abs(curve[:, 0]) + 1000 * (curve[:, 1] < 0))])   # front crossing of the midline
    found = contacts(s, score, s_mid)
    crown_mid = np.zeros(len(curve))   # height of the crowns' middle along the arch (where a cut's plane turns)
    near = station_of(curve, p2[crown])
    for i in range(len(curve)):
        h = height[crown][near == i]
        crown_mid[i] = np.median(h) if len(h) else (crown_mid[i - 1] if i else 0.0)
    # midline cut: from behind everything to just behind the central incisors
    near = crown & (np.abs(p2[:, 0]) < params.SEG_MIDLINE_BAND) & (p2[:, 1] > 0)
    stop = (np.min(p2[near, 1]) if near.any() else np.interp(s_mid, s, curve[:, 1]) - 8.0) - params.SEG_STOP_BEHIND
    arch = {'frame': (origin, anterior, left, occlusal), 'curve': curve, 's': s, 'score': score, 't_co': t_world,
            'contacts': found, 'mid_s': s_mid, 'p2': p2, 'height': height, 'crown': crown,
            'stop': np.array((0.0, stop)), 'back': float(p2[:, 1].min()) - 5.0, 'crown_mid': crown_mid}
    arch['cuts'] = candidates(arch)
    return arch


def candidates(arch):
    """The candidate cuts (cut_geometry) at every contact, each tilted between its two teeth."""
    found = arch['contacts']
    ends = [arch['s'][0]] + list(found) + [arch['s'][-1]]
    axes = long_axes(arch, arch['t_co'], ends) if arch['t_co'] is not None else [arch['frame'][3]] * (len(ends) - 1)
    return [cut_geometry(arch, at, axes[k] + axes[k + 1]) for k, at in enumerate(found)]


# --- Cut geometry -------------------------------------------------------------------------------

def to3(arch, q, h=0.0):
    origin, anterior, left, occlusal = arch['frame']
    return origin + q[0] * left + q[1] * anterior + h * occlusal


def cut_geometry(arch, s_at, bisector):
    """One candidate cut, as two pieces of plane (zero thickness):
    - interdental: through the contact point (crown middle) and across the arch, tilted to run midway between
      the two roots (centre_between_roots, CBCT); without readable roots, along the bisector of the two teeth's
      long axes; from SEG_REACH_OUT outside to SEG_REACH_IN inside the crowns' middle;
    - palatal: from there, upright, straight to the midline cut (or to its front end behind the centrals).
    Between the central incisors: the midline cut carried through to the front (flag 'central')."""
    origin, anterior, left, occlusal = arch['frame']
    curve, s = arch['curve'], arch['s']
    i = int(np.clip(np.searchsorted(s, s_at), 0, len(s) - 1))
    p2 = curve[i]
    t2 = tangents(curve)[i]
    away = t2 if s_at > arch['mid_s'] else -t2   # along the arch, away from the midline
    u2 = np.array((-t2[1], t2[0]))
    if u2 @ (np.array((0.0, np.mean(curve[:, 1]))) - p2) < 0:
        u2 = -u2
    cut = {'s': s_at, 'side': 1 if s_at > arch['mid_s'] else -1,
           'central': abs(p2[0]) < params.SEG_MIDLINE_BAND and abs(s_at - arch['mid_s']) < params.SEG_MIDLINE_SEARCH}
    point = to3(arch, p2, arch['crown_mid'][i])
    b = bisector / np.linalg.norm(bisector)
    u3 = to3(arch, u2) - origin
    u3 -= (u3 @ b) * b
    u3 /= np.linalg.norm(u3)
    n = np.cross(u3, b)
    away3 = to3(arch, away) - origin
    if n @ away3 < 0:
        n = -n
    cut['root_gap'] = None
    if arch.get('t_co') is not None:   # centred between the two roots (CBCT), still through the contact
        centred, gap = centre_between_roots(point, n, u3, occlusal, arch['t_co'])
        if centred is not None:
            n = centred if centred @ away3 > 0 else -centred
            cut['root_gap'] = gap
            u3 = u3 - (u3 @ n) * n
            u3 /= np.linalg.norm(u3)
    q2 = p2 + u2 * params.SEG_REACH_IN
    stop = arch['stop']
    e2 = np.array((0.0, q2[1])) if q2[1] <= stop[1] else stop
    w2 = e2 - q2
    m2 = np.array((-w2[1], w2[0]))
    if m2 @ away < 0:
        m2 = -m2
    cut.update(point=point, normal=n, across=u3, q2=q2, e2=e2, m2=m2 / max(np.linalg.norm(m2), 1e-9))
    return cut


def centre_between_roots(point, normal, across, cusps, t_co):
    """Tilt of the interdental plane, kept through the crown contact `point`, so that it runs midway between the
    two facing root surfaces (CBCT): at each level from SEG_ROOT_LEVELS[0] to [1] mm towards the roots, the middle
    of the gap between the last point of one root and the first of the next (near the middle of the roots,
    across +/- SEG_ROOT_ACROSS); the plane's direction is the line from the contact that follows those middles
    best. Returns (normal, narrowest gap in mm) or (None, None) when the roots can't be read."""
    for _ in range(2):   # once more with the plane found: the roots are told apart by it
        up = np.cross(normal, across)
        up = up if up @ cusps > 0 else -up
        rel = t_co - point
        along, height, acr = rel @ normal, rel @ up, rel @ across
        middles, gaps = [], []
        for level in np.arange(params.SEG_ROOT_LEVELS[0], params.SEG_ROOT_LEVELS[1], params.SEG_ROOT_STEP):
            at = (np.abs(height + level) < params.SEG_ROOT_STEP / 2) & (np.abs(acr) < params.SEG_ROOT_ACROSS) \
                & (np.abs(along) < params.SEG_ROOT_REACH)
            before, after = along[at & (along < 0)], along[at & (along > 0)]
            if len(before) < 3 or len(after) < 3:
                continue
            a, b = before.max(), after.min()
            middles.append(-level * up + (a + b) / 2 * normal)
            gaps.append(b - a)
        if len(middles) < 2:
            return None, None
        v = np.array(middles)
        _w, vecs = np.linalg.eigh(v.T @ v)   # line through the contact that follows the middles best
        d = vecs[:, -1]
        d = d if d @ cusps > 0 else -d
        across_d = across - (across @ d) * d
        across_d /= np.linalg.norm(across_d)
        new = np.cross(across_d, d)
        normal = new if new @ normal > 0 else -new
    return normal, float(min(gaps))


def posterior_side(arch, cut, co, p2, across_in):
    """For each vertex: on the far side of this cut (away from the midline along the arch)?
    co: world vertices; p2: their 2D (left, anterior); across_in: how far inside the arch's middle each is."""
    tilted = (co - cut['point']) @ cut['normal'] > 0
    palatal = (p2 - cut['q2']) @ cut['m2'] > 0
    far = np.where(across_in <= params.SEG_REACH_IN, tilted, palatal)
    behind_stop = p2[:, 1] <= arch['stop'][1]
    other_half = behind_stop & (np.sign(p2[:, 0]) == -cut['side'])   # the other half of the palate: never
    return far & ~other_half


def labels(arch, cuts, co, p2, across_in):
    """Segment of each vertex: 0 = anterior, then the posterior segments; numbers from the patient's right to
    left. Returns (labels, names)."""
    right = sorted([c for c in cuts if c['side'] < 0 and not c['central']], key=lambda c: -c['s'])   # from the midline out
    left = sorted([c for c in cuts if c['side'] > 0 and not c['central']], key=lambda c: c['s'])
    r_idx = np.zeros(len(co), dtype=int)
    for j, c in enumerate(right, 1):
        r_idx[posterior_side(arch, c, co, p2, across_in)] = j
    l_idx = np.zeros(len(co), dtype=int)
    for j, c in enumerate(left, 1):
        l_idx[posterior_side(arch, c, co, p2, across_in)] = j
    central = any(c['central'] for c in cuts)
    # order from the right: R_n .. R_1, anterior (or anterior right, anterior left), L_1 .. L_n
    n_r, n_l = len(right), len(left)
    lab = np.where(r_idx > 0, n_r - r_idx, -1)
    anterior_first = n_r
    if central:
        ant = np.where(p2[:, 0] < 0, anterior_first, anterior_first + 1)
        after = anterior_first + 2
    else:
        ant = np.full(len(co), anterior_first)
        after = anterior_first + 1
    lab = np.where(lab >= 0, lab, np.where(l_idx > 0, after + l_idx - 1, ant))
    return lab, after + n_l


def plane_section(co, tris, point, normal):
    """Segments where the surface crosses a plane: (k, 2, 3) world points."""
    d = (co - point) @ normal
    dt = d[tris]
    crossing = (dt.min(axis=1) < 0) & (dt.max(axis=1) > 0)
    tri = tris[crossing]
    dd = dt[crossing]
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = dd[:, a], dd[:, b]
        f = np.where(np.sign(da) != np.sign(db), da / np.where(da - db == 0, 1, da - db), np.nan)
        pts.append(co[tri[:, a]] + f[:, None] * (co[tri[:, b]] - co[tri[:, a]]))
    pts = np.stack(pts, axis=1)   # (k, 3 edges, 3)
    ok = ~np.isnan(pts[:, :, 0])
    seg = np.array([p[o][:2] for p, o in zip(pts, ok) if o.sum() >= 2])
    return seg.reshape(-1, 2, 3)


def cut_drawing(arch, cut, co, tris):
    """The cut's line on the surface: the exact sections of the scan by its pieces, kept where each piece is."""
    origin, anterior, left, occlusal = arch['frame']
    out = []
    if cut['central']:
        return midline_drawing(arch, co, tris, through=True)
    seg = plane_section(co, tris, cut['point'], cut['normal'])
    if len(seg):
        mid = seg.mean(axis=1)
        u = (mid - cut['point']) @ cut['across']
        out.append(seg[(u > -params.SEG_REACH_OUT) & (u < params.SEG_REACH_IN)])
    w2 = cut['e2'] - cut['q2']
    normal = to3(arch, np.array((-w2[1], w2[0]))) - origin
    if np.linalg.norm(normal) > 1e-6:
        normal /= np.linalg.norm(normal)
        seg = plane_section(co, tris, to3(arch, cut['q2']), normal)
        if len(seg):
            rel = seg.mean(axis=1) - origin
            q = np.column_stack((rel @ left, rel @ anterior))
            f = (q - cut['q2']) @ w2 / max(w2 @ w2, 1e-9)
            out.append(seg[(f > 0) & (f < 1)])
    return np.concatenate(out) if out else np.zeros((0, 2, 3))


def midline_drawing(arch, co, tris, through=False):
    origin, anterior, left, occlusal = arch['frame']
    seg = plane_section(co, tris, origin, left)
    if not len(seg):
        return seg
    a = (seg.mean(axis=1) - origin) @ anterior
    return seg[a > arch['stop'][1]] if through else seg[a <= arch['stop'][1]]


def surface_bvh(obj):
    co = registration.vertex_coords(obj, world=True)
    return BVHTree.FromPolygons(co.tolist(), registration.triangles(obj.data).tolist())


def across_inside(arch, p2):
    """How far inside the arch's middle (towards the palate) each 2D point is."""
    curve = arch['curve']
    near = station_of(curve, p2)
    t = tangents(curve)[near]
    inward = np.column_stack((-t[:, 1], t[:, 0]))
    centre = np.array((0.0, np.mean(curve[:, 1])))
    flip = np.einsum("ij,ij->i", inward, centre - curve[near]) < 0
    inward[flip] *= -1
    return np.einsum("ij,ij->i", p2 - curve[near], inward)


# --- The set-up copy and its state ----------------------------------------------------------

MADE_BY = "Montage occlusal"
_state = {}   # copy name -> analysis, drawings and segment labels (rebuilt when needed)


def setup_copy(scene, source):
    """The set-up copy of this arch scan, if made."""
    return next((o for o in scene.objects if o.omfs.made_by == MADE_BY and o.omfs_setup.source == source), None)


def make_copy(scene, source):
    """A copy of the arch scan for the set-up, at the scan's position zero, not in any block. The scan is hidden
    (remembered with the isolated objects: "Réafficher le reste" brings it back)."""
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = copy.data.name = f"{source.name} (montage)"
    for coll in source.users_collection:
        coll.objects.link(copy)
    copy.parent = None
    copy.matrix_world = source.matrix_basis.copy()   # occlusion invariant: matrix_basis is the pre-op pose
    copy.omfs.made_by = MADE_BY
    copy.omfs_setup.source = source
    copy.omfs_occl.fingerprint = ""
    locking.set_locked(copy, True)
    hidden = set(occlusion_prepare.hidden_names(scene)) | {source.name}
    source.hide_set(True)
    scene.omfs_isolated = json.dumps(sorted(hidden))
    copy.hide_set(False)
    adopt(scene)
    return copy


def stand_in(scene):
    """What the whole-arch seating moves for the upper arch: the set-up copy once it is in the upper block,
    else the upper scan."""
    upper = scene.omfs_occlusion.upper
    copy = setup_copy(scene, upper) if upper is not None else None
    if copy is not None and copy.parent is not None and copy.parent == occlusion.find_block(scene, 'UPPER'):
        return copy
    return upper


def adopt(scene):
    """The set-up copy takes the upper scan's place in the upper block, exactly where the scan was, with the
    springs' upper points; the scan leaves the block (back at its pre-op pose, hidden): the seating, "Ajuster le
    haut", "Départ" and the ghost then act on the copy, and the scan never moves. Not once split in segments."""
    upper = scene.omfs_occlusion.upper
    copy = setup_copy(scene, upper) if upper is not None else None
    block = occlusion.find_block(scene, 'UPPER')
    if copy is None or block is None or copy.parent == block or segments_of(scene, copy):
        return copy
    if upper.parent == block:
        copy.parent = block
        copy.matrix_parent_inverse = upper.matrix_parent_inverse.copy()
        copy.matrix_basis = upper.matrix_basis.copy()
        for point in [c for c in upper.children if c.omfs.role == 'MARKER']:   # springs' upper points
            inverse, basis = point.matrix_parent_inverse.copy(), point.matrix_basis.copy()
            point.parent = copy
            point.matrix_parent_inverse, point.matrix_basis = inverse, basis
        occlusion.remove_member(upper)
        occlusion.remember_shape(copy)
    else:
        occlusion.add_member(block, copy)
    locking.set_locked(copy, True)
    upper.hide_set(True)
    return copy


def motion(copy):
    """World movement of the copy since its position zero (the whole-arch seating moves it with its block)."""
    return np.array(copy.matrix_world) @ np.linalg.inv(np.array(movement.preop_matrix(bpy.context.scene, copy)))


def chosen(copy):
    return sorted(item.s for item in copy.omfs_setup.cuts)


def state(copy):
    """Analysis of the copy (cached), its candidate cuts' drawings and the segments of the chosen cuts."""
    st = _state.get(copy.name)
    if st is None or st['vertices'] != len(copy.data.vertices):
        arch = analyse(copy, bpy.context.scene)
        zero = np.array(movement.preop_matrix(bpy.context.scene, copy))
        co = registration.vertex_coords(copy, world=False) @ zero[:3, :3].T + zero[:3, 3]   # position zero
        tris = registration.triangles(copy.data)
        rel = co - arch['frame'][0]
        p2 = np.column_stack((rel @ arch['frame'][2], rel @ arch['frame'][1]))
        st = {'vertices': len(copy.data.vertices), 'arch': arch, 'co': co, 'tris': tris, 'p2': p2,
              'inside': across_inside(arch, p2),
              'drawings': [cut_drawing(arch, c, co, tris) for c in arch['cuts']],
              'midline': midline_drawing(arch, co, tris), 'hover': None, 'labels': None, 'key': None}
        _state[copy.name] = st
    key = tuple(chosen(copy))
    if st['key'] != key:
        picked = [c for c in st['arch']['cuts'] if any(abs(c['s'] - x) < 1e-3 for x in key)]
        st['labels'], st['count'] = labels(st['arch'], picked, st['co'], st['p2'], st['inside'])
        st['key'], st['batch'] = key, None
    return st


def picked_indices(copy, st):
    key = chosen(copy)
    return [i for i, c in enumerate(st['arch']['cuts']) if any(abs(c['s'] - x) < 1e-3 for x in key)]


def toggle(copy, index):
    st = state(copy)
    at = st['arch']['cuts'][index]['s']
    items = copy.omfs_setup.cuts
    for k, item in enumerate(items):
        if abs(item.s - at) < 1e-3:
            items.remove(k)
            return False
    items.add().s = at
    return True


def segment_names(st):
    return [str(k + 1) for k in range(st['count'])]


# --- Splitting into segments (zero thickness) ---------------------------------------------------

SEGMENT_MADE_BY = "Montage occlusal: segment"


def picked_cuts(copy, st):
    return [st['arch']['cuts'][i] for i in picked_indices(copy, st)]


def point_labels(st, cuts, points):
    """Segment of any world points (same rules as the copy's vertices)."""
    arch = st['arch']
    rel = points - arch['frame'][0]
    p2 = np.column_stack((rel @ arch['frame'][2], rel @ arch['frame'][1]))
    return labels(arch, cuts, points, p2, across_inside(arch, p2))[0], p2


def sheet_values(arch, cut, points, p2, inside):
    """Signed distance to the cut's sheet (tilted interdental plane near the arch, palatal plane beyond)."""
    if cut is None:   # the midline
        return p2[:, 0]
    tilted = (points - cut['point']) @ cut['normal']
    palatal = (p2 - cut['q2']) @ cut['m2']
    return np.where(inside <= params.SEG_REACH_IN, tilted, palatal)


def split_points(st, cuts, a, b):
    """Where each edge a-b (world, rows) crosses the sheet that separates its two ends: fraction from a."""
    arch = st['arch']
    pa, pb = (point_labels(st, cuts, x)[1] for x in (a, b))
    ia, ib = across_inside(arch, pa), across_inside(arch, pb)
    t = np.full(len(a), 0.5)
    found = np.zeros(len(a), dtype=bool)
    for cut in cuts:
        if cut['central']:
            continue
        sa, sb = (posterior_side(arch, cut, x, p, i) for x, p, i in ((a, pa, ia), (b, pb, ib)))
        here = (sa != sb) & ~found
        if here.any():
            fa = sheet_values(arch, cut, a[here], pa[here], ia[here])
            fb = sheet_values(arch, cut, b[here], pb[here], ib[here])
            t[here] = fa / np.where(fa - fb == 0, 1, fa - fb)
            found |= here
    rest = ~found   # across the midline
    fa, fb = pa[rest, 0], pb[rest, 0]
    t[rest] = fa / np.where(fa - fb == 0, 1, fa - fb)
    return np.clip(t, 0.0, 1.0)


def split(scene, copy):
    """Cut the set-up copy along the chosen cuts and the midline into one object per segment, with zero thickness:
    every edge whose two ends belong to different segments gets a vertex exactly on the sheet between them, the
    faces are split there, nothing is removed. Returns the segment objects, numbered from the patient's right."""
    import bmesh
    st = state(copy)
    cuts = picked_cuts(copy, st)
    w = np.array(movement.preop_matrix(scene, copy))   # the cuts are at position zero
    move = Matrix(motion(copy).tolist())               # the segments start where the copy is now
    bm = bmesh.new()
    bm.from_mesh(copy.data)
    bm.verts.ensure_lookup_table()
    lab = st['labels']
    edges = [e for e in bm.edges if lab[e.verts[0].index] != lab[e.verts[1].index]]
    if edges:
        local = np.array([[tuple(e.verts[0].co), tuple(e.verts[1].co)] for e in edges])
        a, b = (local[:, k] @ w[:3, :3].T + w[:3, 3] for k in (0, 1))
        fractions = split_points(st, cuts, a, b)
        on_cut = []
        for edge, f in zip(edges, fractions):
            _e, vert = bmesh.utils.edge_split(edge, edge.verts[0], float(f))
            on_cut.append(vert)
        bmesh.ops.connect_verts(bm, verts=on_cut)   # each face split between its two new vertices (one call)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])   # every face a triangle
    bm.verts.index_update()
    co = np.array([tuple(v.co) for v in bm.verts]) @ w[:3, :3].T + w[:3, 3]   # world
    faces = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
    bm.free()
    face_labels = point_labels(st, cuts, co[faces].mean(axis=1))[0]
    objects = []
    for k in range(st['count']):
        mine = faces[face_labels == k]
        if not len(mine):
            continue
        used, local_faces = np.unique(mine, return_inverse=True)
        centre = co[used].mean(axis=0)
        name = f"Segment {k + 1}"
        old = scene.objects.get(name)
        if old is not None and old.omfs.made_by == SEGMENT_MADE_BY:
            bpy.data.objects.remove(old)
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(len(used))
        mesh.vertices.foreach_set("co", (co[used] - centre).ravel())
        mesh.loops.add(local_faces.size)
        mesh.loops.foreach_set("vertex_index", local_faces.ravel())
        mesh.polygons.add(len(mine))
        mesh.polygons.foreach_set("loop_start", np.arange(0, local_faces.size, 3))
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new(name, mesh)
        for coll in copy.users_collection:
            coll.objects.link(obj)
        obj.matrix_world = move @ Matrix.Translation(Vector(centre))
        obj.omfs.role, obj.omfs.made_by = 'SCAN', SEGMENT_MADE_BY
        obj.color = params.SEG_COLOURS[k % len(params.SEG_COLOURS)][:3] + (1.0,)
        seg = obj.omfs_setup
        seg.source, seg.number = copy, k + 1
        seg.preop = obj.matrix_world.copy()
        objects.append((k, obj))
    by_label = dict(objects)
    # anterior segment(s): those holding the front of the arch (the incisors): the angulation kept at 100 %
    origin, anterior_axis, left_axis, _occlusal = st['arch']['frame']
    front = origin + anterior_axis * (st['p2'][:, 1].max() - 0.5)
    front_labels = set(point_labels(st, cuts, np.array([front - left_axis, front + left_axis]))[0].tolist())
    for k, obj in objects:
        anterior = k in front_labels   # both halves when the centrals are split
        obj.omfs_setup.anterior = anterior
        obj.omfs_setup.hold = 100.0 if anterior else params.SEG_HOLD_DEFAULT
    # each chosen cut's contact point, copied on its two segments (the contact curve's continuity)
    for cut in cuts:
        if cut['central']:
            pts = np.array([cut['point'] - 0.3 * st['arch']['frame'][2], cut['point'] + 0.3 * st['arch']['frame'][2]])
        else:
            pts = np.array([cut['point'] - 0.3 * cut['normal'], cut['point'] + 0.3 * cut['normal']])
        sides = point_labels(st, cuts, pts)[0]
        if sides[0] == sides[1] or sides[0] not in by_label or sides[1] not in by_label:
            continue
        one, two = by_label[sides[0]], by_label[sides[1]]
        for obj, other in ((one, two), (two, one)):
            item = obj.omfs_setup.contacts.add()
            item.other = other
            item.co = obj.matrix_world.inverted() @ (move @ Vector(cut['point']))
            item.other_co = other.matrix_world.inverted() @ (move @ Vector(cut['point']))
    copy.hide_set(True)
    scene.omfs_segmental.show = False
    return [obj for _k, obj in objects]


def segments_of(scene, copy):
    found = [o for o in scene.objects if o.omfs.made_by == SEGMENT_MADE_BY and o.omfs_setup.source == copy]
    return sorted(found, key=lambda o: o.omfs_setup.number)


def unsplit(scene, copy):
    for obj in segments_of(scene, copy):
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    copy.hide_set(False)
    scene.omfs_segmental.show = True


# --- Drawing --------------------------------------------------------------------------------

def _active_copy(context):
    scene = context.scene
    upper = scene.omfs_occlusion.upper
    copy = setup_copy(scene, upper) if upper is not None else None
    if copy is None or copy.name not in _state or not copy.visible_get() or not scene.omfs_segmental.show:
        return None
    return copy


def _dashed(segs, centre):
    """Every other stretch of SEG_DASH mm (measured from the cut's contact point): a dashed line."""
    if not len(segs):
        return segs
    d = np.linalg.norm(segs.mean(axis=1) - centre, axis=1)
    return segs[(d // params.SEG_DASH).astype(int) % 2 == 0]


def _lines(segs, colour, width):
    if not len(segs):
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", colour)
    batch_for_shader(shader, 'LINES', {"pos": segs.reshape(-1, 3).tolist()}).draw(shader)


def _draw_3d():
    context = bpy.context
    copy = _active_copy(context)
    if copy is None:
        return
    st = state(copy)
    gpu.matrix.push()
    gpu.matrix.multiply_matrix(Matrix(motion(copy).tolist()))   # drawn where the copy is now
    gpu.state.blend_set('ALPHA')
    if st['count'] > 1:   # the segments, tinted
        if st.get('batch') is None:
            colours = np.array(params.SEG_COLOURS)[st['labels'] % len(params.SEG_COLOURS)]
            shader = gpu.shader.from_builtin('SMOOTH_COLOR')
            st['batch'] = (shader, batch_for_shader(shader, 'TRIS', {"pos": st['co'].tolist(), "color": colours.tolist()},
                                                    indices=st['tris'].astype(np.int32).tolist()))
        gpu.state.depth_test_set('LESS_EQUAL')
        shader, batch = st['batch']
        batch.draw(shader)
    gpu.state.depth_test_set('NONE')
    _lines(st['midline'], params.SEG_MIDLINE_COLOUR, 3.0)
    picked = picked_indices(copy, st)
    for i, (cut, segs) in enumerate(zip(st['arch']['cuts'], st['drawings'])):
        if i in picked:
            _lines(segs, params.SEG_CHOSEN_COLOUR, 3.5)
        elif i == st['hover']:
            _lines(segs, params.SEG_HOVER_COLOUR, 4.0)
        else:
            _lines(_dashed(segs, cut['point']), params.SEG_CANDIDATE_COLOUR, 2.0)
    gpu.state.blend_set('NONE')
    gpu.matrix.pop()


def _draw_text():
    context = bpy.context
    copy = _active_copy(context)
    if copy is None:
        return
    st = state(copy)
    move = Matrix(motion(copy).tolist())
    region, rv3d = context.region, context.region_data
    scale = context.preferences.system.ui_scale
    font = 0
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
    if st['count'] > 1:   # the segments' numbers
        blf.size(font, 26 * scale)
        blf.color(font, 1.0, 1.0, 1.0, 1.0)
        for k in range(st['count']):
            mine = st['labels'] == k
            if not mine.any():
                continue
            centre = st['co'][mine & st['arch']['crown']].mean(axis=0) if (mine & st['arch']['crown']).any() \
                else st['co'][mine].mean(axis=0)
            xy = location_3d_to_region_2d(region, rv3d, move @ Vector(centre))
            if xy is not None:
                w, h = blf.dimensions(font, str(k + 1))
                blf.position(font, xy.x - w / 2, xy.y - h / 2, 0)
                blf.draw(font, str(k + 1))
    if st['hover'] is not None:   # the cut under the mouse: the room between the roots
        cut = st['arch']['cuts'][st['hover']]
        xy = location_3d_to_region_2d(region, rv3d, move @ Vector(cut['point']))
        if xy is not None:
            gap = cut['root_gap']
            text = "racines : ?" if gap is None else f"entre les racines : {gap:.1f} mm"
            blf.size(font, 14 * scale)
            blf.color(font, *params.SEG_HOVER_COLOUR)
            blf.position(font, xy.x + 12 * scale, xy.y + 12 * scale, 0)
            blf.draw(font, text)
    blf.disable(font, blf.SHADOW)


# --- Operators ------------------------------------------------------------------------------

def hovered(st, region, rv3d, mouse, move=None):
    """Index of the candidate cut under the mouse (within SEG_HOVER_PX), or None. move: the copy's movement."""
    m = np.array(rv3d.perspective_matrix) @ (move if move is not None else np.eye(4))
    best, best_d = None, params.SEG_HOVER_PX * bpy.context.preferences.system.ui_scale
    for i, segs in enumerate(st['drawings']):
        if not len(segs):
            continue
        pts = segs.reshape(-1, 3)
        h = np.column_stack((pts, np.ones(len(pts)))) @ m.T
        front = h[:, 3] > 1e-6
        xy = (h[front, :2] / h[front, 3:4] + 1) / 2 * (region.width, region.height)
        if not len(xy):
            continue
        d = np.min(np.linalg.norm(xy - mouse, axis=1))
        if d < best_d:
            best, best_d = i, d
    return best


class OMFS_OT_segmental_cuts(bpy.types.Operator):
    """Segmentaire : sur une copie de l'arcade du haut (le scan n'est jamais coupé), un trait médian palatin
    fixe et un trait proposé entre chaque dent (pointillés). Survolez un trait pour le voir, cliquez pour le
    choisir (recliquez pour l'enlever). Les segments se numérotent de droite à gauche. Entrée ou Échap : fin"""
    bl_idname = "omfs.segmental_cuts"
    bl_label = "Segmentaire : choisir les traits"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        scene = context.scene
        upper = scene.omfs_occlusion.upper
        if upper is None:
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.copy = setup_copy(scene, upper) or make_copy(scene, upper)
        self.copy.hide_set(False)
        scene.omfs_segmental.show = True
        try:
            st = state(self.copy)
        except Exception as error:   # not readable as an arch
            self.report({'ERROR'}, f"Arcade illisible : {error}")
            return {'CANCELLED'}
        if not st['arch']['cuts']:
            self.report({'ERROR'}, "Aucun contact entre les dents trouvé")
            return {'CANCELLED'}
        self.prompt(st)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self, st):
        n = len(chosen(self.copy))
        self.area.header_text_set(f"Segmentaire : survolez un trait, cliquez pour le choisir (recliquez : l'enlever)"
                                  f"   |   {n} trait(s), {st['count']} segment(s)   |   Entrée / Échap : terminer")

    def finish(self, context):
        st = _state.get(self.copy.name)
        if st is not None:
            st['hover'] = None
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        self.area.tag_redraw()

    def modal(self, context, event):
        if event.type in {'ESC', 'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.finish(context)
            return {'FINISHED'}
        under = landmarks.viewport_region_under_mouse(self.area, event)
        st = state(self.copy)
        if event.type == 'MOUSEMOVE':
            hover = hovered(st, under[0], under[0].data, np.array(under[1]), motion(self.copy)) if under else None
            if hover != st['hover']:
                st['hover'] = hover
                self.area.tag_redraw()
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and under is not None:
            if st['hover'] is None:
                return {'RUNNING_MODAL'}
            toggle(self.copy, st['hover'])
            st = state(self.copy)
            self.prompt(st)
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}   # navigation and the sidebar keep working


class OMFS_OT_segmental_clear(bpy.types.Operator):
    """Enlever tous les traits choisis (le trait médian reste)"""
    bl_idname = "omfs.segmental_clear"
    bl_label = "Effacer les traits"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        upper = context.scene.omfs_occlusion.upper
        copy = setup_copy(context.scene, upper) if upper is not None else None
        if copy is None:
            return {'CANCELLED'}
        copy.omfs_setup.cuts.clear()
        return {'FINISHED'}


class OMFS_SetupCut(bpy.types.PropertyGroup):
    s: bpy.props.FloatProperty()   # position along the arch curve (mm): which candidate cut


class OMFS_SetupContact(bpy.types.PropertyGroup):
    """A contact point of the arch split by a cut: its copy on this segment and on the neighbour (local points)."""
    other: bpy.props.PointerProperty(type=bpy.types.Object)
    co: bpy.props.FloatVectorProperty(size=3)
    other_co: bpy.props.FloatVectorProperty(size=3)


class OMFS_SetupProperties(bpy.types.PropertyGroup):
    """Stored on a set-up copy of an arch scan (source, cuts), or on one of its segments (the rest)."""
    source: bpy.props.PointerProperty(type=bpy.types.Object, description="The scan (or set-up copy) this was made from")
    cuts: bpy.props.CollectionProperty(type=OMFS_SetupCut)
    number: bpy.props.IntProperty()
    anterior: bpy.props.BoolProperty()
    hold: bpy.props.FloatProperty(
        name="Tenue de l'angulation", subtype='PERCENTAGE', min=0.0, max=100.0, default=70.0,
        description="Ressort de rotation vers l'angle que vous avez donné au segment (G / R) pendant l'assise. "
                    "100 % : translations seulement, l'angle est gardé exactement ; moins : un peu de rotation "
                    "permise si elle améliore beaucoup l'intercuspidation ; 0 % : libre")
    preop: bpy.props.FloatVectorProperty(size=(4, 4), subtype='MATRIX', options={'HIDDEN'})
    last_turn: bpy.props.FloatProperty(default=-1.0, options={'HIDDEN'})   # degrees turned by the last seating
    contacts: bpy.props.CollectionProperty(type=OMFS_SetupContact)


class OMFS_SegmentalSceneProperties(bpy.types.PropertyGroup):
    show: bpy.props.BoolProperty(name="Traits et segments dans la vue", default=True)


classes = (OMFS_SetupCut, OMFS_SetupContact, OMFS_SetupProperties, OMFS_SegmentalSceneProperties, OMFS_OT_segmental_cuts,
           OMFS_OT_segmental_clear)
_handles = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_setup = bpy.props.PointerProperty(type=OMFS_SetupProperties)
    bpy.types.Scene.omfs_segmental = bpy.props.PointerProperty(type=OMFS_SegmentalSceneProperties)
    if not bpy.app.background:
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_3d, (), 'WINDOW', 'POST_VIEW'))
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_text, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    for handle in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _handles.clear()
    _state.clear()
    del bpy.types.Scene.omfs_segmental
    del bpy.types.Object.omfs_setup
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
