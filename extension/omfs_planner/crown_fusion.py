"""Teeth registration, last step: "Modifier la géométrie des dents CBCT".

Once the scan's teeth position is canned (registration.freeze_scan), the crowns of the CBCT teeth
(imprecise: voxel size, metal scatter) are brought onto the scan's crowns (precise), so the scan with
its gingiva can be set aside. The roots of the CBCT teeth never move.

How (a "magnet" that knows which part of each mesh is which):
1. Scan side: only the scan's crowns (the "Crowns" vertex group of the registration, gingiva excluded).
2. CBCT side: a tooth vertex is a crown vertex when the scan crowns are within FUSE_SNAP_MAX of it,
   facing the same way (so the buccal side never snaps onto the lingual side), and not at the edge of
   the scan crown patch (the gum line: there the scan stops, the tooth goes on into the gingiva).
3. Those crown areas are first subdivided down to FUSE_MAX_EDGE, then every crown vertex is moved onto
   the closest point of the scan crown surface: the crown now lies on the scan's own surface.
   Grooves and pits narrower than the CBCT could see: a reverse pull (each scan crown point pulls
   its closest crown vertex, then all go back onto the scan surface), FUSE_GROOVE_PASSES rounds.
4. A transition band (FUSE_BLEND mm) below the crown follows smoothly (displacement diffused along the
   mesh); every vertex beyond it, the roots, keeps its exact CBCT position.
5. Exact crowns (FUSE_EXACT, surgeon 2026-11-19: "littéralement le même STL" on the occlusal and incisal
   anatomy): projection alone can't keep detail finer than the teeth mesh (mamelons, sharp incisal
   edges, fissures are cut into chords). So, on the crown proper, the scan's own triangles replace the
   projected crown: same vertices, same triangles. The zone: scan crown triangles the projected crown
   lies on (FUSE_EXACT_TOL, same side), more than FUSE_EXACT_MARGIN from the gum line (interproximal
   bridges of the scan, with no tooth under them, stay out; holes inside the zone are taken). The teeth
   triangles under it are removed with a small clearance, and a seam (zipper of triangles) joins each
   border of the zone to the border of the hole around it; small leftover holes get a fan. Pieces whose
   border can't be joined are left projected (retried, FUSE_EXACT_RETRIES). The result is closed.
6. Report: distance from the scan's crown points to the new crowns (mean, 95 %, max), on the crown
   proper (points more than FUSE_REPORT_MARGIN from the gum line, where the transition is).
The scan's crown selection is first completed by the scan points lying on the CBCT teeth (FUSE_CROWN_REACH,
same side: gingiva covers the root at a distance), as the registration's "Crowns" selection is often patchy
on the molars, then "closed" (grown then shrunk by FUSE_CROWN_CLOSE rings): small holes are each a border
where nothing would snap.

Nothing is changed in place: the teeth object receives a new mesh data-block; the CBCT mesh data-block
is kept untouched on the object (omfs_fusion.original) and "Dents CBCT d'origine" puts it back.
"""

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import locking, occlusion_prepare, params, registration

CROWN_ATTRIBUTE = "Couronne STL"   # boolean point attribute on the new mesh: True on the crowns now from the scan
EXACT_ATTRIBUTE = "Triangles du scan"   # boolean face attribute: True on the scan's own triangles (exact crowns)
BLEND_ITERATIONS = 80


# --- Geometry -----------------------------------------------------------------------

def close_mask(mask, edges, rings):
    """Morphological closing on the mesh: grow the selection by `rings` rings, then shrink it back.
    Fills the small holes and notches of a painted / suggested selection; its outer edge stays."""
    a, b = edges[:, 0], edges[:, 1]
    n = len(mask)
    for _ in range(rings):   # grow: a vertex joins if a neighbour is in
        hit = np.bincount(a, weights=mask[b], minlength=n) + np.bincount(b, weights=mask[a], minlength=n)
        mask = mask | (hit > 0)
    degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    for _ in range(rings):   # shrink: a vertex stays if all its neighbours are in
        inside = np.bincount(a, weights=mask[b], minlength=n) + np.bincount(b, weights=mask[a], minlength=n)
        mask = mask & (inside >= degree)
    return mask


def near_teeth(co, normals, teeth_co, teeth_tris):
    """Scan vertices lying on the CBCT teeth surface (within FUSE_CROWN_REACH, same side): the crowns.
    Gingiva covers the root at a distance (its thickness): left out, except right at the gum line."""
    bvh = BVHTree.FromPolygons(teeth_co.tolist(), teeth_tris.tolist())
    closest, face_normals = registration.nearest(bvh, co, params.FUSE_CROWN_REACH)
    found = ~np.isnan(closest[:, 0])
    facing = np.einsum("ij,ij->i", normals[found], face_normals[found])
    if len(facing) and np.median(facing) < 0:
        facing = -facing
    near = np.zeros(len(co), dtype=bool)
    near[np.flatnonzero(found)[facing > np.cos(np.radians(params.FUSE_MAX_ANGLE))]] = True
    return near


def crown_vertex_mask(scan):
    """Per scan vertex: crown (the registration's "Crowns" selection, completed by the points lying on its CBCT
    teeth at their pre-op place, then closed), in the scan's own vertex order. For tools that only need to know
    which part is teeth. The scan is taken where it is (a set-up copy: at position zero)."""
    co = registration.vertex_coords(scan, world=False)
    is_crown = np.zeros(len(co), dtype=bool)
    is_crown[registration.crown_indices(scan)] = True
    teeth = scan.omfs_reg.target
    if teeth is not None and teeth.type == 'MESH':
        mesh = teeth.omfs_fusion.original or teeth.data
        from . import movement   # here: movement is loaded after this module
        scene = bpy.context.scene
        to_teeth = np.array(movement.preop_matrix(scene, teeth).inverted() @ movement.preop_matrix(scene, scan))
        normals = np.empty(len(co) * 3)
        scan.data.vertex_normals.foreach_get("vector", normals)
        teeth_co, _n, _e = mesh_arrays(mesh)
        is_crown |= near_teeth(co @ to_teeth[:3, :3].T + to_teeth[:3, 3], normals.reshape(-1, 3) @ to_teeth[:3, :3].T,
                               teeth_co, registration.triangles(mesh))
    edges = np.empty(len(scan.data.edges) * 2, dtype=np.int64)
    scan.data.edges.foreach_get("vertices", edges)
    return close_mask(is_crown, edges.reshape(-1, 2), params.FUSE_CROWN_CLOSE)


def scan_crown_surface(scan, to_teeth, teeth_mesh=None):
    """(BVH of the scan's crown faces, KDTree of the crown patch's border vertices, crown vertex
    coordinates away from that border, (all scan vertices, all scan triangles, crown triangle mask, KDTree
    of the gum line only)),
    all in the teeth's local coordinates."""
    indices = registration.crown_indices(scan)
    co = registration.vertex_coords(scan, world=False)
    co = co @ to_teeth[:3, :3].T + to_teeth[:3, 3]
    is_crown = np.zeros(len(co), dtype=bool)
    is_crown[indices] = True
    if teeth_mesh is not None:   # the registration's "Crowns" selection is often patchy: the CBCT teeth complete it
        normals = np.empty(len(co) * 3)
        scan.data.vertex_normals.foreach_get("vector", normals)
        normals = normals.reshape(-1, 3) @ to_teeth[:3, :3].T
        teeth_co, _n, _e = mesh_arrays(teeth_mesh)
        is_crown |= near_teeth(co, normals, teeth_co, registration.triangles(teeth_mesh))
    edges = np.empty(len(scan.data.edges) * 2, dtype=np.int64)
    scan.data.edges.foreach_get("vertices", edges)
    is_crown = close_mask(is_crown, edges.reshape(-1, 2), params.FUSE_CROWN_CLOSE)
    tris = registration.triangles(scan.data)
    tris = tris[is_crown[tris].all(axis=1)]
    bvh = BVHTree.FromPolygons(co.tolist(), tris.tolist())
    edges = np.sort(np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = np.unique(unique[counts == 1])
    tree = KDTree(max(len(border), 1))
    for i, p in enumerate(co[border]):
        tree.insert(p, i)
    tree.balance()
    points = co[is_crown]
    if len(border):   # the crown proper: the deviation is measured away from the gum line
        points = np.array([p for p in points if tree.find(p)[2] > params.FUSE_REPORT_MARGIN])
    all_tris = registration.triangles(scan.data)
    # the gum line alone: crown border edges that are not also holes of the scan itself (spots the camera
    # missed, often at the bottom of a fissure: the crown goes on around them)
    all_edges = np.sort(np.concatenate((all_tris[:, [0, 1]], all_tris[:, [1, 2]], all_tris[:, [2, 0]])), axis=1)
    scan_unique, scan_counts = np.unique(all_edges, axis=0, return_counts=True)
    holes = {tuple(e) for e in scan_unique[scan_counts == 1]}
    gum = np.unique(np.array([v for e in unique[counts == 1] if tuple(e) not in holes for v in e], dtype=np.int64))
    gum_tree = KDTree(max(len(gum), 1))
    for i, p in enumerate(co[gum]):
        gum_tree.insert(p, i)
    gum_tree.balance()
    patch = (co, all_tris, is_crown[all_tris].all(axis=1), gum_tree if len(gum) else None)   # for the exact crowns
    return bvh, (tree if len(border) else None), points, patch


def snap_targets(co, normals, bvh, border_tree):
    """Closest scan crown point per vertex, and whether the vertex is a crown vertex (see module doc)."""
    closest, face_normals = registration.nearest(bvh, co, params.FUSE_SNAP_MAX)
    found = ~np.isnan(closest[:, 0])
    facing = np.zeros(len(co))
    facing[found] = np.einsum("ij,ij->i", normals[found], face_normals[found])
    if found.any() and np.median(facing[found]) < 0:   # scan normals pointing inwards: same test, flipped
        facing = -facing
    crown = found & (facing > np.cos(np.radians(params.FUSE_MAX_ANGLE)))
    if border_tree is not None:
        for i in np.flatnonzero(crown):
            crown[i] = border_tree.find(closest[i])[2] > params.FUSE_BORDER_GAP
    return closest, crown


def mesh_arrays(mesh):
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    normals = np.empty(len(mesh.vertices) * 3)
    mesh.vertex_normals.foreach_get("vector", normals)
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    return co.reshape(-1, 3), normals.reshape(-1, 3), edges.reshape(-1, 2)


def subdivide_crowns(mesh, crown):
    """Split the crown edges longer than FUSE_MAX_EDGE (twice at most), so the new crowns sample the
    scan surface finely. Returns True if anything was split."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    is_crown = [bool(c) for c in crown]
    changed = False
    for _ in range(params.FUSE_SUBDIVIDE_PASSES):
        is_crown += [True] * (len(bm.verts) - len(is_crown))   # new vertices sit inside the crown area
        long_edges = [e for e in bm.edges if is_crown[e.verts[0].index] and is_crown[e.verts[1].index]
                      and e.calc_length() > params.FUSE_MAX_EDGE]
        if not long_edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        changed = True
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return changed


def blend(co, edges, crown, displacement):
    """Diffuse the crown displacement into the band below it; vertices beyond the band stay put."""
    if not crown.any():
        return displacement
    tree = KDTree(int(crown.sum()))
    for i, p in enumerate(co[crown]):
        tree.insert(p, i)
    tree.balance()
    candidates = np.flatnonzero(~crown)
    band = np.zeros(len(co), dtype=bool)
    for i in candidates:
        band[i] = tree.find(co[i])[2] <= params.FUSE_BLEND
    if not band.any():
        return displacement
    a, b = edges[:, 0], edges[:, 1]
    count = np.bincount(a, minlength=len(co)) + np.bincount(b, minlength=len(co))
    count[count == 0] = 1
    d = displacement.copy()
    for _ in range(BLEND_ITERATIONS):
        mean = np.column_stack([(np.bincount(a, weights=d[b, k], minlength=len(co))
                                 + np.bincount(b, weights=d[a, k], minlength=len(co))) / count for k in range(3)])
        d[band] = mean[band]
    return d


def neighbour_mean(co, edges):
    a, b = edges[:, 0], edges[:, 1]
    n = len(co)
    count = np.maximum(np.bincount(a, minlength=n) + np.bincount(b, minlength=n), 1)
    return np.column_stack([(np.bincount(a, weights=co[b, k], minlength=n)
                             + np.bincount(b, weights=co[a, k], minlength=n)) / count for k in range(3)])


def onto_scan(positions, scan_bvh):
    closest, _normals = registration.nearest(scan_bvh, positions, params.FUSE_SNAP_MAX)
    return np.where(np.isnan(closest[:, :1]), positions, closest)


def relax(co, crown, edges, scan_bvh):
    """Even out the crown triangles without leaving the scan surface: each crown vertex slides halfway
    to the mean of its neighbours along the surface (the part of the move across the surface is
    dropped, so it doesn't climb out of a groove), then goes back onto the scan. Closest-point
    projection bunches vertices in hollows and folds a few triangles: this unfolds them."""
    for _ in range(params.FUSE_RELAX_PASSES):
        p = co[crown]
        _closest, normals = registration.nearest(scan_bvh, p, params.FUSE_SNAP_MAX)
        normals = np.nan_to_num(normals)
        move = neighbour_mean(co, edges)[crown] - p
        move -= np.einsum("ij,ij->i", move, normals)[:, None] * normals
        co[crown] = onto_scan(p + 0.5 * move, scan_bvh)
    return co


def fill_grooves(co, crown, edges, scan_bvh, scan_points):
    """Crown vertices placed on the scan surface by closest point never reach the bottom of a groove
    or a pit narrower than the CBCT could see (they stop on its walls). Reverse pull: each scan crown
    point pulls the crown vertex closest to it, then every vertex goes back onto the scan surface,
    and the triangles are evened out. co: all vertices (crown ones already on the scan)."""
    co = relax(co, crown, edges, scan_bvh)
    if not crown.any() or not len(scan_points):
        return co
    for _ in range(params.FUSE_GROOVE_PASSES):
        positions = co[crown]
        tree = KDTree(len(positions))
        for i, p in enumerate(positions):
            tree.insert(p, i)
        tree.balance()
        owner = np.empty(len(scan_points), dtype=np.int64)
        near = np.zeros(len(scan_points), dtype=bool)
        for k, s in enumerate(scan_points):
            _co, owner[k], dist = tree.find(s)
            near[k] = dist <= params.FUSE_SNAP_MAX
        owner, pulled_by = owner[near], scan_points[near]
        count = np.bincount(owner, minlength=len(positions))
        pull = np.column_stack([np.bincount(owner, weights=pulled_by[:, k], minlength=len(positions))
                                for k in range(3)])
        has = count > 0
        target = positions.copy()
        target[has] = 0.5 * positions[has] + 0.5 * pull[has] / count[has, None]
        co[crown] = onto_scan(target, scan_bvh)
        co = relax(co, crown, edges, scan_bvh)
    return co


def fuse_mesh(mesh, scan_bvh, border_tree, scan_points=()):
    """Bring the crowns of this (teeth) mesh onto the scan crowns, in place. Returns the crown mask."""
    co, normals, _edges = mesh_arrays(mesh)
    _closest, crown = snap_targets(co, normals, scan_bvh, border_tree)
    subdivide_crowns(mesh, crown)
    co, normals, edges = mesh_arrays(mesh)
    closest, crown = snap_targets(co, normals, scan_bvh, border_tree)
    displacement = np.zeros_like(co)
    moved = co.copy()
    moved[crown] = closest[crown]
    moved = fill_grooves(moved, crown, edges, scan_bvh, np.asarray(scan_points, dtype=float).reshape(-1, 3))
    displacement[crown] = moved[crown] - co[crown]
    displacement = blend(co, edges, crown, displacement)
    mesh.vertices.foreach_set("co", (co + displacement).ravel())
    attribute = mesh.attributes.get(CROWN_ATTRIBUTE) or mesh.attributes.new(CROWN_ATTRIBUTE, 'BOOLEAN', 'POINT')
    attribute.data.foreach_set("value", crown.tolist())
    mesh.update()
    return crown


# --- Exact crowns: the scan's own triangles transplanted --------------------------------

def _face_components(tris, keep):
    """Connected components (through shared edges) of the triangles where keep is True: list of index arrays."""
    idx = np.flatnonzero(keep)
    if not len(idx):
        return []
    pos = {f: k for k, f in enumerate(idx)}
    edge_faces = {}
    for f in idx:
        a, b, c = tris[f]
        for e in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(e), max(e)), []).append(f)
    parent = list(range(len(idx)))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for faces in edge_faces.values():
        for f in faces[1:]:
            ra, rb = root(pos[faces[0]]), root(pos[f])
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for k, f in enumerate(idx):
        groups.setdefault(root(k), []).append(f)
    return [np.array(g) for g in groups.values()]


def _boundary(tris):
    """Directed boundary edges of a triangle set, as a dict start -> end, walking the hole side
    (each boundary edge reversed from its triangle's winding). None if a vertex has two exits."""
    count, direction = {}, {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            count[key] = count.get(key, 0) + 1
            direction[key] = (u, v)
    nxt = {}
    for key, n in count.items():
        if n == 1:
            u, v = direction[key]
            if v in nxt:
                return None
            nxt[v] = u
    return nxt


def _loops(nxt):
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v)
            if v is None:
                return None
        if v != start:
            return None
        loops.append(loop)
    return loops


def _clean_faces(tris, keep):
    """Peel the triangle set until its border is made of simple loops: triangles with two or three
    border edges (spikes), the triangles around a vertex where the border pinches, and those left on an
    edge where two teeth touch (four triangles) when only three of them remain, are removed."""
    keep = keep.copy()
    for _ in range(50):
        sub = np.flatnonzero(keep)
        count = {}
        for f in sub:
            a, b, c = tris[f]
            for e in ((a, b), (b, c), (c, a)):
                key = (min(e), max(e))
                count[key] = count.get(key, 0) + 1
        border_degree = {}
        for (u, v), n in count.items():
            if n == 1:
                border_degree[u] = border_degree.get(u, 0) + 1
                border_degree[v] = border_degree.get(v, 0) + 1
        pinched = {v for v, d in border_degree.items() if d > 2}
        drop = []
        for f in sub:
            a, b, c = tris[f]
            n = [count[(min(e), max(e))] for e in ((a, b), (b, c), (c, a))]
            if n.count(1) >= 2 or 3 in n or pinched.intersection((a, b, c)):   # 3: half of a teeth contact edge
                drop.append(f)
        if not drop:
            break
        keep[drop] = False
    return keep


def exact_zone(co, tris, crown_tris, fused_bvh, border_tree):
    """Scan crown triangles to transplant: the crown proper (FUSE_EXACT_MARGIN from the gum line), where
    the fused crown already lies on the scan (FUSE_EXACT_TOL, same side): interproximal bridges of the
    scan, which no tooth lies under, are left out. Holes inside the zone are filled; its border is
    cleaned to simple loops; pieces under FUSE_EXACT_MIN_FACES triangles are dropped."""
    closest, normals = registration.nearest(fused_bvh, co, params.FUSE_EXACT_TOL)
    on = ~np.isnan(closest[:, 0])
    ok = on.copy()
    if border_tree is not None:
        for i in np.flatnonzero(ok):
            ok[i] = border_tree.find(co[i])[2] > params.FUSE_EXACT_MARGIN
    tri_n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-12)
    keep = crown_tris & ok[tris].all(axis=1)
    centre = co[tris].mean(axis=1)
    _c, fused_n = registration.nearest(fused_bvh, centre[keep], params.FUSE_EXACT_TOL * 2)
    facing = np.einsum("ij,ij->i", np.nan_to_num(fused_n), tri_n[keep])
    flip = bool(len(facing)) and np.median(facing) < 0
    sub = np.flatnonzero(keep)
    if flip:
        facing = -facing
    keep[sub[facing < np.cos(np.radians(params.FUSE_MAX_ANGLE))]] = False   # same side as the tooth only
    # holes in the zone (a groove the fused crown missed, a mamelon...): everything enclosed is taken
    outside = _face_components(tris, ~keep)
    if outside:
        largest = max(range(len(outside)), key=lambda k: len(outside[k]))
        for k, group in enumerate(outside):
            if k != largest:
                keep[group] = True
    keep = _clean_faces(tris, keep)
    for group in _face_components(tris, keep):
        if len(group) < params.FUSE_EXACT_MIN_FACES:
            keep[group] = False
    return keep, flip


def zipper(a_loop, b_loop, pos):
    """Triangles joining two parallel loops (a: forward, b: already reversed to run along a)."""
    na, nb = len(a_loop), len(b_loop)
    start = min(range(nb), key=lambda k: np.linalg.norm(pos[b_loop[k]] - pos[a_loop[0]]))
    b_loop = b_loop[start:] + b_loop[:start]
    i = k = 0
    out = []
    while i < na or k < nb:
        a0, a1 = a_loop[i % na], a_loop[(i + 1) % na]
        b0, b1 = b_loop[k % nb], b_loop[(k + 1) % nb]
        advance_a = k >= nb or (i < na and np.linalg.norm(pos[a1] - pos[b0]) <= np.linalg.norm(pos[a0] - pos[b1]))
        if advance_a:
            out.append((a0, a1, b0))
            i += 1
        else:
            out.append((a0, b1, b0))
            k += 1
    return out


skipped = ""   # why the last transplant was not done (shown to the surgeon)


def _skip(reason):
    global skipped
    skipped = reason
    return 0


def _outer_loops(z_tris, z_loops, co):
    """ids of the loops that are the outer border of their zone piece (the longest loop of each piece)."""
    owner = {}
    for k, group in enumerate(_face_components(np.asarray(z_tris), np.ones(len(z_tris), dtype=bool))):
        for v in np.unique(np.asarray(z_tris)[group]):
            owner[int(v)] = k
    best = {}
    for loop in z_loops:
        piece = owner.get(int(loop[0]))
        length = np.linalg.norm(np.diff(co[loop + loop[:1]], axis=0), axis=1).sum()
        if piece not in best or length > best[piece][0]:
            best[piece] = (length, id(loop))
    return {i for _length, i in best.values()}


def _hole_plan(co, normals, tris, scan_co, z_tris, force):
    """Teeth triangles kept around the zone, and the pairing of the zone's border loops with the teeth
    holes. Returns (keep, pairs, fans, drop, grow): drop = zone vertices whose pieces must go, grow = teeth
    vertices around a hole under the zone that no zone border matches (then retry). force: teeth
    triangles removed anyway."""
    z_bvh = BVHTree.FromPolygons(scan_co.tolist(), z_tris.tolist())
    z_nxt = _boundary(z_tris)
    z_loops = _loops(z_nxt) if z_nxt is not None else None
    if not z_loops:
        return _skip("bord de la zone du scan non simple")
    rim = np.unique([v for loop in z_loops for v in loop])
    rim_tree = KDTree(len(rim))
    for i, v in enumerate(rim):
        rim_tree.insert(scan_co[v], i)
    rim_tree.balance()
    # teeth triangles under the zone, and a thin clearance around its border
    closest, z_normals = registration.nearest(z_bvh, co, params.FUSE_EXACT_TOL)
    under = ~np.isnan(closest[:, 0])
    # same side only (zone triangles now wound like the teeth): a neighbouring tooth at a contact point stays
    same_side = np.cos(np.radians(params.FUSE_MAX_ANGLE))
    under[under] = np.einsum("ij,ij->i", normals[under], z_normals[under]) > same_side
    near, n_near = registration.nearest(z_bvh, co, params.FUSE_EXACT_GAP * 2)
    near_rim = ~np.isnan(near[:, 0])
    near_rim[near_rim] = np.einsum("ij,ij->i", normals[near_rim], n_near[near_rim]) > same_side
    for i in np.flatnonzero(near_rim):
        near_rim[i] = rim_tree.find(co[i])[2] <= params.FUSE_EXACT_GAP
    remove = (under | near_rim)[tris].any(axis=1) | force
    keep = _clean_faces(tris, ~remove)
    # little pieces of teeth left inside a hole: removed with it
    for group in _face_components(tris, keep):
        if len(group) < params.FUSE_EXACT_MIN_FACES:
            keep[group] = False
    new_edges = set()
    for f in np.flatnonzero(~keep):
        a, b, c = tris[f]
        new_edges.update((min(e), max(e)) for e in ((a, b), (b, c), (c, a)))
    t_nxt = _boundary(tris[keep])
    if t_nxt is None:
        return _skip("bord du trou des dents non simple")
    t_nxt = {v: u for v, u in t_nxt.items() if (min(u, v), max(u, v)) in new_edges}   # only the new holes
    t_loops = _loops(t_nxt)
    if t_loops is None:
        return _skip("bord du trou des dents non simple")
    # each zone loop -> the teeth hole border closest to it
    claims = {}
    unmatched = []
    for z in z_loops:
        zp = scan_co[z][::max(1, len(z) // 40)]
        scores = [np.mean([np.min(np.linalg.norm(co[t] - p, axis=1)) for p in zp]) for t in t_loops]
        best = int(np.argmin(scores)) if scores else -1
        if best >= 0 and scores[best] <= params.FUSE_EXACT_PAIR:
            claims.setdefault(best, []).append(z)
        else:
            unmatched.append(z)
    pairs, fans, drop, grow = [], [], [], []
    for t, zs in claims.items():
        zs.sort(key=len, reverse=True)
        pairs.append((t_loops[t], zs[0]))
        unmatched += zs[1:]   # two zone pieces over one teeth hole: the smaller ones can't be joined
    outer = _outer_loops(z_tris, z_loops, scan_co)
    for z in unmatched:
        if len(z) <= params.FUSE_EXACT_SMALL_HOLE or id(z) not in outer:
            fans.append(z)   # a hole inside a zone piece (e.g. a spot the camera missed): closed
        else:
            drop.append(z[0])
    paired = set(claims)
    for t, loop in enumerate(t_loops):
        if t not in paired:
            if len(loop) > params.FUSE_EXACT_SMALL_HOLE:   # under the zone, no border: joins the main hole
                grow += list(loop)
            else:
                fans.append(-1 - np.asarray(loop))   # teeth vertices, marked negative
    return keep, pairs, fans, drop, grow


def transplant(mesh, scan_co, scan_tris, crown_tris, border_tree):
    """Replace the crown proper of this (already fused) teeth mesh by the scan's own triangles.
    Zone pieces whose border can't be joined to the teeth are left out (those crowns stay projected).
    Returns the number of scan triangles transplanted (0: nothing done, the fused mesh is kept)."""
    co, normals, _edges = mesh_arrays(mesh)
    tris = registration.triangles(mesh)   # the mesh is rebuilt from its triangles (quads, if any, split)
    fused_bvh = BVHTree.FromPolygons(co.tolist(), tris.tolist())
    zone, flip = exact_zone(scan_co, scan_tris, crown_tris, fused_bvh, border_tree)
    force = np.zeros(len(tris), dtype=bool)
    for _attempt in range(params.FUSE_EXACT_RETRIES):
        if not zone.any():
            return _skip("aucune zone de couronne à transplanter")
        z_tris = scan_tris[zone][:, ::-1] if flip else scan_tris[zone]
        plan = _hole_plan(co, normals, tris, scan_co, z_tris, force)
        if plan == 0:
            return 0
        keep, pairs, fans, drop, grow = plan
        if not drop and not grow:
            break
        if grow:   # widen that hole by two rings of teeth triangles
            ring = np.zeros(len(co), dtype=bool)
            ring[grow] = True
            for _ in range(2):
                touched = ring[tris].any(axis=1)
                ring[tris[touched].ravel()] = True
            force |= ring[tris].any(axis=1)
        for group in _face_components(scan_tris, zone):   # leave out the pieces that could not be joined
            if np.isin(scan_tris[group], drop).any():
                zone[group] = False
    else:
        return _skip("bords impossibles à joindre")
    n = len(co)
    pos = np.vstack((co, scan_co))
    seam = []
    for t_loop, z_loop in pairs:
        seam += zipper(list(t_loop), [v + n for v in reversed(z_loop)], pos)
    extra = []   # fan centres
    for loop in fans:
        loop = [-1 - v if v < 0 else v + n for v in loop]
        centre = len(pos) + len(extra)
        extra.append(pos[loop].mean(axis=0))
        seam += [(centre, loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]
    if extra:
        pos = np.vstack((pos, extra))
    # build the new mesh: kept teeth triangles + scan zone triangles + seam
    all_tris = np.vstack((tris[keep], z_tris + n, np.array(seam, dtype=np.int64).reshape(-1, 3)))
    used_v = np.unique(all_tris)
    remap = np.full(len(pos), -1, dtype=np.int64)
    remap[used_v] = np.arange(len(used_v))
    crown_before = np.zeros(n, dtype=bool)
    attribute = mesh.attributes.get(CROWN_ATTRIBUTE)
    if attribute is not None:
        attribute.data.foreach_get("value", crown_before)
    is_crown = np.concatenate((crown_before, np.ones(len(scan_co) + len(extra), dtype=bool)))[used_v]
    mesh.clear_geometry()
    mesh.vertices.add(len(used_v))
    mesh.vertices.foreach_set("co", pos[used_v].ravel())
    faces = remap[all_tris]
    mesh.loops.add(len(faces) * 3)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 3, 3))
    mesh.update(calc_edges=True)
    attribute = mesh.attributes.get(CROWN_ATTRIBUTE) or mesh.attributes.new(CROWN_ATTRIBUTE, 'BOOLEAN', 'POINT')
    attribute.data.foreach_set("value", is_crown.tolist())
    exact = mesh.attributes.new(EXACT_ATTRIBUTE, 'BOOLEAN', 'FACE')
    exact.data.foreach_set("value", np.concatenate((np.zeros(int(keep.sum()), dtype=bool),
                                                     np.ones(len(z_tris), dtype=bool),
                                                     np.zeros(len(seam), dtype=bool))).tolist())
    return len(z_tris)


def deviation(mesh, scan_points):
    """Distances (mm) from the scan's crown points to the new teeth surface."""
    co, _normals, _edges = mesh_arrays(mesh)
    bvh = BVHTree.FromPolygons(co.tolist(), registration.triangles(mesh).tolist())
    closest, _ = registration.nearest(bvh, scan_points, params.FUSE_SNAP_MAX * 2)
    found = ~np.isnan(closest[:, 0])
    return np.linalg.norm(closest[found] - scan_points[found], axis=1)


# --- Objects ------------------------------------------------------------------------

def is_fused(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'TEETH' and obj.omfs_fusion.original is not None


def crown_mask_of(mesh):
    attribute = mesh.attributes.get(CROWN_ATTRIBUTE)
    if attribute is None:
        return None
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    attribute.data.foreach_get("value", mask)
    return mask


def crown_mask(obj):
    """Per vertex: True on the crowns taken from the scan (fused teeth only)."""
    return crown_mask_of(obj.data)


def fuse(scan):
    """Modify the geometry of the scan's CBCT teeth (registration target). Returns the teeth object."""
    teeth = scan.omfs_reg.target
    original = teeth.omfs_fusion.original or teeth.data
    to_teeth = np.array(teeth.matrix_world.inverted() @ scan.matrix_world)
    scan_bvh, border_tree, scan_points, (scan_co, scan_tris, crown_tris, gum_tree) = scan_crown_surface(scan, to_teeth, original)
    fused = original.copy()
    crown = fuse_mesh(fused, scan_bvh, border_tree, scan_points)
    global skipped
    skipped = ""
    exact = transplant(fused, scan_co, scan_tris, crown_tris, gum_tree) if params.FUSE_EXACT else 0
    if exact:
        crown = crown_mask_of(fused)
    distances = deviation(fused, scan_points)
    if teeth.omfs_fusion.original is None:
        teeth.omfs_fusion.original = original
        name = original.name
        original.name = f"{name} (CBCT d'origine)"
        fused.name = name
    old = teeth.data
    occlusion_prepare.swap_mesh(teeth, fused)   # also its pre-op ghost; re-takes the block's shape proof
    if old is not original and old.users == 0:
        bpy.data.meshes.remove(old)
    props = teeth.omfs_fusion
    props.scan = scan
    props.crown_count = int(crown.sum())
    props.exact_faces = exact
    props.exact_note = skipped
    props.mean = float(distances.mean()) if len(distances) else -1.0
    props.p95 = float(np.percentile(distances, 95)) if len(distances) else -1.0
    props.max = float(distances.max()) if len(distances) else -1.0
    return teeth


def restore(teeth):
    original = teeth.omfs_fusion.original
    if original is None:
        return
    old = teeth.data
    teeth.omfs_fusion.original = None
    name = old.name
    old.name = f"{name} (couronnes STL)"
    original.name = name
    occlusion_prepare.swap_mesh(teeth, original)
    if old.users == 0:
        bpy.data.meshes.remove(old)
    teeth.omfs_fusion.crown_count = 0


# --- Operators ----------------------------------------------------------------------

class OMFS_OT_fuse_crowns(bpy.types.Operator):
    """Modifier la géométrie des dents CBCT : leurs couronnes sont ramenées exactement sur les couronnes
    du scan (gencive exclue) ; les racines ne bougent pas. Le maillage CBCT d'origine est gardé.
    Le scan est ensuite mis de côté (caché)"""
    bl_idname = "omfs.fuse_crowns"
    bl_label = "Modifier la géométrie des dents CBCT"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = registration.scan_for(context, self.scan_name)
        if scan is None or not scan.omfs_reg.frozen:
            self.report({'ERROR'}, "Cannez d'abord la position dentaire initiale")
            return {'CANCELLED'}
        if scan.omfs_reg.target is None or len(registration.crown_indices(scan)) < 100:
            self.report({'ERROR'}, "Il faut les dents CBCT et les couronnes du scan (étape 2 du recalage)")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        teeth = fuse(scan)
        scan.hide_set(True)   # set aside: the teeth now carry its crowns
        props = teeth.omfs_fusion
        self.report({'INFO'}, f"{teeth.name} : {props.crown_count} sommets de couronne sur le scan, "
                              f"écart moyen {props.mean:.3f} mm, 95 % sous {props.p95:.3f} mm")
        return {'FINISHED'}


class OMFS_OT_restore_cbct_teeth(bpy.types.Operator):
    """Remettre la géométrie CBCT d'origine des dents (annule la modification des couronnes)"""
    bl_idname = "omfs.restore_cbct_teeth"
    bl_label = "Dents CBCT d'origine"
    bl_options = {'REGISTER', 'UNDO'}

    teeth_name: bpy.props.StringProperty()

    def execute(self, context):
        teeth = context.scene.objects.get(self.teeth_name)
        if not is_fused(teeth):
            return {'CANCELLED'}
        restore(teeth)
        return {'FINISHED'}


class OMFS_OT_toggle_scan(bpy.types.Operator):
    """Montrer ou cacher le scan mis de côté"""
    bl_idname = "omfs.toggle_scan"
    bl_label = "Scan"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = context.scene.objects.get(self.scan_name)
        if scan is None:
            return {'CANCELLED'}
        scan.hide_set(not scan.hide_get())
        return {'FINISHED'}


class OMFS_FusionProperties(bpy.types.PropertyGroup):
    """Stored on the CBCT teeth object."""
    original: bpy.props.PointerProperty(type=bpy.types.Mesh, description="CBCT teeth mesh, untouched")
    scan: bpy.props.PointerProperty(type=bpy.types.Object, description="Scan whose crowns were taken")
    crown_count: bpy.props.IntProperty()
    exact_faces: bpy.props.IntProperty(description="Scan triangles transplanted as they are (0: crowns only projected)")
    exact_note: bpy.props.StringProperty(description="Why the scan triangles could not be transplanted")
    mean: bpy.props.FloatProperty(default=-1.0)
    p95: bpy.props.FloatProperty(default=-1.0)
    max: bpy.props.FloatProperty(default=-1.0)


classes = (OMFS_FusionProperties, OMFS_OT_fuse_crowns, OMFS_OT_restore_cbct_teeth, OMFS_OT_toggle_scan)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_fusion = bpy.props.PointerProperty(type=OMFS_FusionProperties)


def unregister():
    del bpy.types.Object.omfs_fusion
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
