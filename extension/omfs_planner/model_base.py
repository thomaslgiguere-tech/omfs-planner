"""Model base ("Création de modèle"): turn an open arch scan into a closed, printable dental model.

What the surgeon did by hand on case1, automated:
1. The scan's surface is kept exactly: no vertex of the scan ever moves. Only faces are removed:
   loose scraps, the ragged fringe along the open border, and whatever lies beyond the base plane.
2. The teeth are protected: every vertex less than `teeth_band` mm below the cusp tips around it
   (within `teeth_radius` mm, seen from above) is never removed. The check is reported after each build.
3. Small holes inside the scan are closed by flat patches (faces added, no vertex moved).
4. A wall drops from the open border, along the occlusal axis, to a flat base `height` mm below the
   highest cusp. The base's outline is smoothed (only the new vertices move), so the wall is clean.

Lower arch: the border runs buccal then lingual, so the base is a horseshoe. Upper arch: the border
goes around the palate, so the base is a full plate.

The occlusal axis: the reference frame's up if there is one, else the direction in which the arch
is thinnest, pointing away from the open border (towards the teeth). It can be flipped.
"""

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import params, reference_frame, registration, text_labels

COLLECTION = "Modèles"
TAG = "dental model"          # made_by of a generated model


def is_scan(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'SCAN'


def is_model(obj):
    return isinstance(obj, bpy.types.Object) and obj.type == 'MESH' and obj.omfs.made_by == TAG


def model_of(scene, scan):
    return next((o for o in scene.objects if is_model(o) and o.omfs_model_source == scan.name), None)


def model_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


# --- Geometry helpers ---------------------------------------------------------------------

def arrays(mesh):
    """Vertex positions, edges (pairs) and, per face, its vertex indices (flat) and sizes."""
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    edges = np.empty(len(mesh.edges) * 2, np.int32)
    mesh.edges.foreach_get("vertices", edges)
    sizes = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    corners = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", corners)
    return co.reshape(-1, 3), edges.reshape(-1, 2), corners, sizes


def components(count, a, b):
    """Connected component label of each of `count` items joined by the pairs (a[i], b[i])."""
    labels = np.arange(count)
    while True:   # each item takes the smallest label among its neighbours, until stable
        low = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]                                  # jump along the chain of labels
        if np.array_equal(new, labels):
            return labels
        labels = new


def occlusal_axis(scan, co, edges):
    """Local-space unit "up" of the scan: towards the teeth, away from the open border."""
    frame = reference_frame.find_frame(bpy.context.scene)
    if frame is not None:
        up = np.array((scan.matrix_world.inverted().to_3x3() @ frame.matrix_world.col[2].xyz).normalized())
    else:
        sample = co[::max(1, len(co) // 20000)]
        _u, _s, vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
        up = vt[2]                                      # the arch is thinnest across the occlusal plane
    border = border_vertices(co, edges, scan.data)
    if len(border) and (co[border] @ up).mean() > (co @ up).mean():
        up = -up                                        # the open border is on the base side
    return up


def border_vertices(co, edges, mesh):
    """Vertices on an open border (edges used by one face only)."""
    corners = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("edge_index", corners)
    uses = np.bincount(corners, minlength=len(edges))
    return np.unique(edges[uses == 1])


def plane_basis(up):
    """Two unit vectors across the occlusal axis."""
    u = Vector(up).orthogonal().normalized()
    v = Vector(up).cross(u).normalized()
    return np.array(u), np.array(v)


def teeth_zone(co, up, band, radius):
    """Boolean per vertex: less than `band` below the cusp tips around it (within `radius`, seen from
    above): the crowns and the gingiva right under them, never removed. Only points near the occlusal
    level count as cusp tips, so the palate or the vestibule far from the teeth is not protected."""
    u, v = plane_basis(up)
    h = co @ up
    cell = 1.0
    i = np.floor(co @ u / cell).astype(np.int64)
    j = np.floor(co @ v / cell).astype(np.int64)
    i -= i.min()
    j -= j.min()
    r = int(np.ceil(radius / cell))
    top = np.full((i.max() + 1 + 2 * r, j.max() + 1 + 2 * r), -np.inf)
    np.maximum.at(top, (i + r, j + r), h)
    top[top < h.max() - params.MODEL_OCCLUSAL_SPREAD] = -np.inf   # not a cusp tip
    near = np.full_like(top, -np.inf)
    for di in range(-r, r + 1):                         # highest cusp tip within the radius (a disc)
        for dj in range(-r, r + 1):
            if di * di + dj * dj <= r * r:
                near = np.maximum(near, np.roll(np.roll(top, di, 0), dj, 1))
    tips = near[i + r, j + r]
    return np.isfinite(tips) & (h > tips - band)        # no cusp tip around: never protected


def full_palate(co, up, edges, keep, mesh):
    """Upper arch with its whole palate (True) or horseshoe (False): seen from above, is the middle of
    the teeth zone (lingual to the incisors, between the molars) inside the scan's outline?"""
    u, v = plane_basis(up)
    flat = np.stack((co @ u, co @ v), 1)
    centre = flat[keep].mean(0)
    cell = 1.0                                          # occupancy grid of the scan seen from above
    ij = np.floor((flat - flat.min(0)) / cell).astype(np.int64)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    c = np.floor((centre - flat.min(0)) / cell).astype(np.int64)
    r = 2                                               # a 5 mm square around the middle
    return bool(grid[max(0, c[0] - r):c[0] + r + 1, max(0, c[1] - r):c[1] + r + 1].mean() > 0.5)


# --- Build --------------------------------------------------------------------------------

def reattach(remove, optional, keep, corners, sizes, count):
    """Put back, one ring at a time, the `optional` faces (the fringe) that join a protected part to
    the main piece: trimming the fringe must never cut off a protected part (it would be thrown away
    as a scrap)."""
    face_of_corner = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.cumsum(sizes) - sizes
    following = np.arange(len(corners)) + 1
    last = starts + sizes - 1
    following[last] = starts                            # each corner's next corner in its face
    for _ in range(params.MODEL_REATTACH_RINGS):
        kept = ~remove[face_of_corner]
        a, b = corners[kept], corners[following][kept]
        labels = components(count, a, b)
        used = np.zeros(count, bool)
        used[a] = True
        main = np.bincount(labels[used]).argmax()
        cut_off = used & keep & (labels != main)
        if not cut_off.any():
            break
        stranded = np.zeros(count, bool)
        stranded[np.isin(labels, np.unique(labels[cut_off]))] = True
        touching = np.zeros(len(sizes), bool)
        np.logical_or.at(touching, face_of_corner, stranded[corners])
        back = remove & optional & touching
        if not back.any():
            break
        remove = remove & ~back
    return remove


def next_border_edge(edge, vert):
    """The open edge that follows `edge` at `vert` along the same border: turn around the vertex
    through the faces, starting from the edge's face. Exact even where two runs of border touch at
    one vertex (a bow-tie)."""
    face, current = edge.link_faces[0], edge
    while True:
        current = next(e for e in face.edges if e != current and vert in e.verts)
        if current.is_boundary:
            return current
        face = next(f for f in current.link_faces if f != face)


def border_loops(bm):
    """Open borders as vertex lists, each in the direction its faces run (so a wall face can run the
    other way and keep the normals consistent). Longest first."""
    left = {e for e in bm.edges if e.is_boundary}
    loops = []
    while left:
        first = left.pop()
        loop = first.link_loops[0]
        verts, edge, vert = [loop.vert], first, loop.link_loop_next.vert
        while True:
            edge = next_border_edge(edge, vert)
            if edge == first:
                break
            left.discard(edge)
            verts.append(vert)
            vert = edge.other_vert(vert)
        loops.append(verts)
    return sorted(loops, key=len, reverse=True)


def clean_border(bm, keep):
    """Remove spikes along the border (faces with two open edges and no protected vertex), until
    none is left. A spike whose tip also touches other faces is kept: it may be the only link of a
    part pinched at that vertex. Returns the number of faces removed."""
    removed = 0
    for _ in range(100):
        bad = [f for f in bm.faces if spike_tip(f) is not None and len(spike_tip(f).link_faces) == 1
               and not any(keep[v.index] for v in f.verts)]
        if not bad:
            break
        bmesh.ops.delete(bm, geom=bad, context='FACES')
        removed += len(bad)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        keep_index(bm, keep)
    return removed


def remove_pinched(bm, origin):
    """Remove the small parts hanging on the rest by one vertex only (a bow-tie): printed, such a
    sliver of gingiva is a loose scrap, and Boolean solvers cut it off or fail on it. Parts bigger
    than MODEL_PINCH_MAX vertices stay. Returns the scan indices (`origin` layer) of the vertices removed."""
    removed = []
    for vert in [v for v in bm.verts if v.is_valid and sum(e.is_boundary for e in v.link_edges) > 2]:
        if not vert.is_valid:
            continue
        sides = []
        seen = set()
        for start in vert.link_faces:                    # the faces reached from each fan, never through vert
            if start in seen:
                continue
            side, stack = set(), [start]
            seen.add(start)
            while stack and len(side) <= params.MODEL_PINCH_MAX:
                face = stack.pop()
                side.add(face)
                for edge in face.edges:
                    if vert in edge.verts:
                        continue
                    for other in edge.link_faces:
                        if other not in seen:
                            seen.add(other)
                            stack.append(other)
            sides.append((len(side) > params.MODEL_PINCH_MAX, side))
        small = [side for big, side in sides if not big]
        if len(sides) < 2 or len(small) == len(sides):
            continue
        for side in small:                               # the whole part goes, its edges too
            removed += [v[origin] for v in {v for f in side for v in f.verts} - {vert}]
            bmesh.ops.delete(bm, geom=list(side), context='FACES')
    return removed


def spike_tip(face):
    """The vertex between a face's two open edges, or None."""
    edges = [e for e in face.edges if e.is_boundary]
    if len(edges) < 2:
        return None
    shared = set(edges[0].verts) & set(edges[1].verts)
    return shared.pop() if shared else None


def keep_index(bm, keep):
    """Re-number the protection flags after a deletion (stored on the vertices as a layer)."""
    bm.verts.index_update()
    layer = bm.verts.layers.int.get("keep")
    keep.resize(len(bm.verts), refcheck=False)
    for v in bm.verts:
        keep[v.index] = v[layer]


def smooth_outline(points, iterations):
    """Laplacian smoothing of a closed 2D outline (only the base's new vertices move)."""
    for _ in range(iterations):
        points = 0.5 * points + 0.25 * (np.roll(points, 1, 0) + np.roll(points, -1, 0))
    return points


def build(scan, settings):
    """New closed mesh: the scan's kept faces (vertices untouched), hole patches, wall and base.
    Returns (mesh, report dict)."""
    mesh = scan.data
    co, edges, corners, sizes = arrays(mesh)
    up = occlusal_axis(scan, co, edges) * (-1 if settings.flip else 1)
    h = co @ up
    top = h.max()
    keep = teeth_zone(co, up, settings.teeth_band, settings.teeth_radius)
    palate = full_palate(co, up, edges, keep, mesh)
    height = settings.height
    if palate:   # the whole palate is kept: the base goes under its deepest point
        labels = components(len(co), edges[:, 0], edges[:, 1])
        main = labels == np.bincount(labels).argmax()
        height = max(height, top - h[main].min() + params.MODEL_UNDER_PALATE)
    base = top - height

    # Faces to remove: beyond the base plane (with room for a wall), scraps, the border fringe.
    face_of_corner = np.repeat(np.arange(len(sizes)), sizes)
    low = np.zeros(len(sizes), bool)
    np.logical_or.at(low, face_of_corner, h[corners] < base + params.MODEL_MIN_WALL)
    labels = components(len(co), edges[:, 0], edges[:, 1])   # (again: cheap, and clearer than sharing)
    sizes_of = np.bincount(labels, minlength=len(co))
    scrap_vertex = sizes_of[labels] < settings.scrap_fraction * len(co)
    scrap = np.zeros(len(sizes), bool)
    np.logical_or.at(scrap, face_of_corner, scrap_vertex[corners])
    border = border_vertices(co, edges, mesh)
    fringe_vertex = np.zeros(len(co), bool)
    if settings.fringe > 0 and len(border):
        tree = KDTree(len(border))
        for k, index in enumerate(border):
            tree.insert(co[index], k)
        tree.balance()
        for index in range(len(co)):
            if tree.find(co[index])[2] < settings.fringe:
                fringe_vertex[index] = True
    fringe = np.zeros(len(sizes), bool)
    np.logical_or.at(fringe, face_of_corner, fringe_vertex[corners])
    guarded = np.zeros(len(sizes), bool)
    np.logical_or.at(guarded, face_of_corner, keep[corners])
    remove = (low | scrap | fringe) & ~guarded
    remove = reattach(remove, fringe & ~low & ~scrap, keep, corners, sizes, len(co))
    protected = int(keep.sum())
    protected_at_start = keep.copy()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    layer = bm.verts.layers.int.new("keep")
    origin = bm.verts.layers.int.new("origin")       # the scan's vertex index, to tell what was lost
    for v in bm.verts:
        v[layer] = int(keep[v.index])
        v[origin] = v.index
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.flatnonzero(remove)], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    keep_index(bm, keep)
    cleaned, pinched_origins = 0, []
    for _ in range(10):                                  # removing a sliver can leave a new spike or bow-tie
        cleaned += clean_border(bm, keep)
        removed = remove_pinched(bm, origin)
        keep_index(bm, keep)
        if not removed:
            break
        pinched_origins += removed

    # Scraps cut off by the removal (keep the biggest piece only).
    bm.verts.ensure_lookup_table()
    pieces = bmesh_pieces(bm)
    for piece in pieces[1:]:
        bmesh.ops.delete(bm, geom=piece, context='VERTS')
    kept_vertices = len(bm.verts)
    present = np.zeros(len(co), bool)
    present[[v[origin] for v in bm.verts]] = True
    lost = protected_at_start & ~present
    on_edge = np.zeros(len(co), bool)                   # the scan's open border, or a scrap already loose in it
    on_edge[border] = True
    on_edge |= sizes_of[labels] < sizes_of.max()
    on_edge[pinched_origins] = True                      # slivers hanging by one vertex
    for _ in range(2):                                  # and the next two rings in (slivers at the border)
        on_edge[edges[on_edge[edges[:, 0]], 1]] = True
        on_edge[edges[on_edge[edges[:, 1]], 0]] = True

    loops = border_loops(bm)
    if not loops:
        bm.free()
        return None, {"error": "Le scan n'a pas de bord ouvert: rien à fermer"}
    outer, holes = loops[0], loops[1:]
    patched = 0
    for hole in holes:
        bmesh.ops.contextual_create(bm, geom=hole_edges(hole))
        patched += 1

    # Wall and base.
    tops = np.array([tuple(v.co) for v in outer])
    drop = tops @ up - base
    bottom = tops - np.outer(drop, up)
    u, v = plane_basis(up)
    flat = np.stack((bottom @ u, bottom @ v), 1)
    flat = smooth_outline(flat, settings.outline_smooth)
    extra = 0
    while crossings(flat) and extra < params.MODEL_OUTLINE_SMOOTH_MAX:
        flat = smooth_outline(flat, 10)                  # a wall crossing itself is not a solid
        extra += 10
    bottom = np.outer(flat[:, 0], u) + np.outer(flat[:, 1], v) + base * up
    new = [bm.verts.new(p) for p in bottom]
    count = len(outer)
    for k in range(count):
        n = (k + 1) % count
        bm.faces.new((outer[n], outer[k], new[k], new[n]))
    plate = bm.faces.new(list(reversed(new)))
    bmesh.ops.triangulate(bm, faces=[plate], quad_method='BEAUTY', ngon_method='BEAUTY')

    result = bpy.data.meshes.new(f"Modèle — {scan.name}")
    bm.to_mesh(result)
    bm.free()
    radius = settings.bar_diameter / 2
    ends = None if palate or not settings.use_bar else bar_ends(flat, radius)
    bar = None if ends is None else [[float(x) for x in end[0] * u + end[1] * v + (base + radius) * up]
                                     for end in ends]
    report = {
        "kept": kept_vertices, "removed": len(co) - kept_vertices, "patched": patched, "palate": palate,
        "bar": bar, "up_vector": [float(x) for x in up],
        "pinched": len(pinched_origins), "outline_extra_smooth": extra,
        "protected": protected, "protected_edge": int((lost & on_edge).sum()),
        "protected_removed": int((lost & ~on_edge).sum()), "cleaned": cleaned, "height": float(height),
        "outline_crossings": crossings(flat),
    }
    return result, report


def hole_edges(hole):
    count = len(hole)
    return [next(e for e in hole[k].link_edges if e.other_vert(hole[k]) == hole[(k + 1) % count])
            for k in range(count)]


def bmesh_pieces(bm):
    """Connected pieces of the bmesh as vertex lists, biggest first."""
    seen = set()
    pieces = []
    for start in bm.verts:
        if start in seen:
            continue
        seen.add(start)
        stack, piece = [start], []
        while stack:
            vert = stack.pop()
            piece.append(vert)
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        pieces.append(piece)
    return sorted(pieces, key=len, reverse=True)


def hull(points):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, points))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return np.array(half(pts) + half(reversed(pts)))


def line_spans(outline, start, along):
    """Where the line start + t * along is inside a closed 2D outline: sorted (t_in, t_out) pairs."""
    a, b = outline, np.roll(outline, -1, 0)
    edge = b - a
    denom = along[0] * edge[:, 1] - along[1] * edge[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((a[:, 0] - start[0]) * edge[:, 1] - (a[:, 1] - start[1]) * edge[:, 0]) / denom
        s = ((a[:, 0] - start[0]) * along[1] - (a[:, 1] - start[1]) * along[0]) / denom
    t = np.sort(t[(s >= 0) & (s < 1) & np.isfinite(t)])
    return list(zip(t[0::2], t[1::2]))


def bar_ends(outline, radius):
    """2D ends of the bar across a horseshoe's opening, each in the middle of one branch so it is
    sunk into the solid on both sides. The bar runs parallel to the opening (the longest edge of the
    outline's convex hull), moved towards the arch until its line crosses both branches."""
    ring = hull(outline)
    nxt = np.roll(ring, -1, 0)
    k = int(np.argmax(np.linalg.norm(nxt - ring, axis=1)))
    p, q = ring[k], nxt[k]
    along = (q - p) / np.linalg.norm(q - p)
    towards = np.array((-along[1], along[0]))            # the hull is counter-clockwise: left is inside
    offset = radius + params.MODEL_BAR_INSET
    while offset < params.MODEL_BAR_MAX_INSET:
        start = p + towards * offset
        spans = line_spans(outline, start, along)
        if len(spans) >= 2:                              # first and last branch crossed
            (a0, a1), (b0, b1) = spans[0], spans[-1]
            return start + along * (a0 + a1) / 2, start + along * (b0 + b1) / 2
        offset += 0.5
    return None


def crossings(flat):
    """Number of self-crossings of the base outline (0 = the base is a simple flat plate)."""
    a = flat
    b = np.roll(flat, -1, 0)
    count = 0
    for k in range(len(a)):
        d1 = b - a
        p, q = a[k], b[k]
        r = q - p
        denom = r[0] * d1[:, 1] - r[1] * d1[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((a[:, 0] - p[0]) * d1[:, 1] - (a[:, 1] - p[1]) * d1[:, 0]) / denom
            s = ((a[:, 0] - p[0]) * r[1] - (a[:, 1] - p[1]) * r[0]) / denom
        hit = (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9)
        hit[[k, (k - 1) % len(a), (k + 1) % len(a)]] = False
        count += int(hit.sum())
    return count // 2


def set_bar(scene, obj, ends, diameter):
    """The horseshoe's bar: a cylinder between `ends` (local coordinates of the model), unioned by a
    Boolean modifier (Manifold solver). No ends: the bar and its modifier are removed."""
    name = f"Barre — {obj.name}"
    bar = bpy.data.objects.get(name)
    mod = obj.modifiers.get("Barre de renfort")
    if ends is None:
        if mod is not None:
            obj.modifiers.remove(mod)
        if bar is not None:
            bpy.data.objects.remove(bar)
        return
    a, b = Vector(ends[0]), Vector(ends[1])
    if bar is None:
        bar = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        model_collection(scene).objects.link(bar)
        bar.omfs.made_by = TAG + " bar"
        bar.display_type = 'WIRE'
        bar.hide_render = True
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=diameter / 2, radius2=diameter / 2,
                          depth=(b - a).length)
    bm.to_mesh(bar.data)
    bm.free()
    bar.matrix_world = obj.matrix_world @ Matrix.LocRotScale((a + b) / 2, (b - a).to_track_quat('Z', 'Y'), None)
    bar.hide_set(True)
    if mod is None:
        mod = obj.modifiers.new("Barre de renfort", 'BOOLEAN')
        mod.operation, mod.solver = 'UNION', 'MANIFOLD'   # the model is a clean solid: no part pinched at a vertex
    mod.object = bar


def inside(tree, point):
    """Is a point inside a closed surface (odd number of crossings along a ray)?"""
    origin, direction, hits = Vector(point), Vector((0.0123, 0.0417, 0.999)).normalized(), 0
    while True:
        location = tree.ray_cast(origin, direction)[0]
        if location is None:
            return hits % 2 == 1
        hits += 1
        origin = location + direction * 1e-4


def bar_anchored(obj, ends):
    """Are both ends of the bar inside the model itself (without the bar)?"""
    mesh = obj.data
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    tree = BVHTree.FromPolygons(co.reshape(-1, 3).tolist(), tris.reshape(-1, 3).tolist())
    return all(inside(tree, end) for end in ends)


def pieces_with_bar(obj):
    """Connected pieces of the model with its bar unioned (1 = the bar joins it into one solid)."""
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    count = len(bmesh_pieces(bm))
    bm.free()
    evaluated.to_mesh_clear()
    return count


def check(obj):
    """Closed (no open or non-manifold edge) and scan vertices kept exactly."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    open_edges = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return open_edges


def make_model(scene, scan):
    settings = scene.omfs_model
    mesh, report = build(scan, settings)
    if mesh is None:
        return report["error"]
    obj = model_of(scene, scan)
    if obj is None:
        obj = bpy.data.objects.new(mesh.name, mesh)
        model_collection(scene).objects.link(obj)
        obj.omfs.made_by = TAG
        obj.omfs_model_source = scan.name
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
    obj.matrix_world = scan.matrix_world
    set_bar(scene, obj, report["bar"], settings.bar_diameter)
    text_labels.rebuild_on(obj)                          # its text follows the new surface
    report["open_edges"] = check(obj)
    if report["bar"]:
        report["bar_anchored"] = bar_anchored(obj, report["bar"])
        report["pieces"] = pieces_with_bar(obj)
    obj["omfs_model_report"] = str(report)
    return None


def last_report(obj):
    import ast
    return ast.literal_eval(obj.get("omfs_model_report", "{}")) if obj is not None else {}


# --- Bite check ------------------------------------------------------------------------------

BITE_WORDS = ("bite", "byte", "occlu", "mordu")
UPPER_WORDS = ("upper", "haut", "maxil", "sup")
LOWER_WORDS = ("lower", "bas", "mandib", "inf")


def named(scene, words, exclude=()):
    """The first scan whose name contains one of the words (case ignored), and none of `exclude`."""
    for obj in scene.objects:
        name = obj.name.lower()
        if is_scan(obj) and any(w in name for w in words) and not any(w in name for w in exclude):
            return obj
    return None


def bite_scans(scene):
    """(bite, upper, lower): chosen in the panel, else found by name."""
    settings = scene.omfs_model
    bite = settings.bite if is_scan(settings.bite) else named(scene, BITE_WORDS)
    upper = settings.upper if is_scan(settings.upper) else named(scene, UPPER_WORDS, BITE_WORDS)
    lower = settings.lower if is_scan(settings.lower) else named(scene, LOWER_WORDS, BITE_WORDS)
    return bite, upper, lower


def world_points(obj):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def world_tree(objects):
    """One BVH over several objects' surfaces, in world coordinates."""
    points, tris, offset = [], [], 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        t = np.empty(len(obj.data.loop_triangles) * 3, np.int64)
        obj.data.loop_triangles.foreach_get("vertices", t)
        points.append(world_points(obj))
        tris.append(t.reshape(-1, 3) + offset)
        offset += len(points[-1])
    return BVHTree.FromPolygons(np.vstack(points).tolist(), np.vstack(tris).tolist())


def distances(bvh, points, reach):
    """Distance to the surface and closest points / normals (NaN beyond `reach`)."""
    closest, normals = registration.nearest(bvh, points, reach)
    found = ~np.isnan(closest[:, 0])
    dist = np.full(len(points), np.inf)
    dist[found] = np.linalg.norm(points[found] - closest[found], axis=1)
    return dist, closest, normals


def moved(matrix, local):
    m = np.array(matrix)
    return local @ m[:3, :3].T + m[:3, 3]


def fit(bvh, local, matrix, iterations, reach, trim=None):
    """Point-to-plane ICP of bite points (local coordinates) onto a surface, from `matrix`. With
    `trim`, only that share of the closest pairs pulls (the part of the bite this surface holds);
    else pairs farther than 3 x the median are dropped. Only the matrix changes, never a mesh."""
    for _ in range(iterations):
        points = moved(matrix, local)
        dist, closest, normals = distances(bvh, points, reach)
        found = np.isfinite(dist)
        if found.sum() < 12:
            break
        d = dist[found]
        keep = d <= (np.quantile(d, trim) if trim else max(3 * np.median(d), 0.05))
        step = registration.point_to_plane_step(points[found][keep], closest[found][keep], normals[found][keep])
        matrix = step @ matrix
        if step.translation.length < 1e-5 and step.to_quaternion().angle < 1e-6:
            break
    return matrix


def search_starts(points, local, count):
    """Start poses for the bite on the arches: at points spread over the arches (farthest point
    sampling), the bite's long axis along the arch there and its height or thickness along the
    occlusal axis, both ways: 8 orientations per point."""
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(4000, len(points)), replace=False)]
    _u, _s, axes = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    up = axes[2]                                        # the arches are thinnest across the occlusal plane
    centre = local.mean(0)
    _u, _s, bite_axes = np.linalg.svd(local - centre, full_matrices=False)
    spots = [sample[0]]
    nearest_spot = np.linalg.norm(sample - sample[0], axis=1)
    for _ in range(count - 1):
        spots.append(sample[int(nearest_spot.argmax())])
        nearest_spot = np.minimum(nearest_spot, np.linalg.norm(sample - spots[-1], axis=1))
    for spot in spots:
        around = sample[np.linalg.norm(sample - spot, axis=1) < params.BITE_NEIGHBOURHOOD]
        if len(around) < 10:
            continue
        flat = around - around.mean(0)
        flat -= np.outer(flat @ up, up)
        tangent = np.linalg.svd(flat, full_matrices=False)[2][0]
        for height in (1, 2):
            source = np.stack((bite_axes[0], bite_axes[height], np.cross(bite_axes[0], bite_axes[height])), 1)
            for s1 in (1, -1):
                for s2 in (1, -1):
                    e1, e2 = s1 * tangent, s2 * up
                    rotation = np.stack((e1, e2, np.cross(e1, e2)), 1) @ source.T
                    yield (Matrix.Translation(Vector(spot)) @ Matrix(rotation).to_4x4()
                           @ Matrix.Translation(-Vector(centre)))


def check_bite(bite, upper, lower):
    """Does the bite's shape say the arches are in the right occlusion? The bite is a picture of the
    side of the intercuspation: a strip of upper teeth and a strip of lower teeth, captured together.
    Where the bite object sits does not matter, and nothing moves (the arches never; the bite's own
    position is only computed):
    1. global search: the one pose where the bite fits both arches at once (share of its points
       within 0.3 mm);
    2. its upper part is fitted alone on the upper arch, its lower part alone on the lower arch;
    3. in the right occlusion both give the same pose: their gap (mm at the bite, degrees) is the
       occlusion error."""
    rng = np.random.default_rng(0)
    co = np.empty(len(bite.data.vertices) * 3)
    bite.data.vertices.foreach_get("co", co)
    local = co.reshape(-1, 3)
    coarse = local[rng.choice(len(local), min(300, len(local)), replace=False)]
    fine = local[rng.choice(len(local), min(3000, len(local)), replace=False)]
    both = world_tree((upper, lower))
    arch_points = np.vstack((world_points(upper), world_points(lower)))

    def share(matrix, points):
        dist = distances(both, moved(matrix, points), 2.0)[0]
        return float((dist < params.BITE_INLIER).mean())

    tries = []
    for start in search_starts(arch_points, local, params.BITE_SEARCH_SPOTS):
        matrix = fit(both, coarse, start, 12, 4.0, trim=0.8)
        tries.append((share(matrix, coarse), matrix))
    tries.sort(key=lambda t: -t[0])
    best = max((fit(both, fine, m, 25, 2.0, trim=0.8) for _s, m in tries[:6]), key=lambda m: share(m, fine))
    together = share(best, fine)

    points = moved(best, fine)
    to_upper = distances(world_tree((upper,)), points, 1.0)[0]
    to_lower = distances(world_tree((lower,)), points, 1.0)[0]
    part_upper = fine[(to_upper < params.BITE_INLIER) & (to_upper < to_lower)]
    part_lower = fine[(to_lower < params.BITE_INLIER) & (to_lower <= to_upper)]
    result = {"together": together, "upper_share": len(part_upper) / len(fine),
              "lower_share": len(part_lower) / len(fine), "gap": None, "angle": None,
              "pose": [float(x) for x in np.array(best).ravel()],
              "objects": [o.name for o in (bite, upper, lower)],
              "where": [[float(x) for x in np.array(o.matrix_world).ravel()] for o in (bite, upper, lower)]}
    enough = min(result["upper_share"], result["lower_share"]) >= params.BITE_MIN_SHARE
    if enough:
        on_upper = fit(world_tree((upper,)), part_upper, best, 40, 1.0)
        on_lower = fit(world_tree((lower,)), part_lower, best, 40, 1.0)
        centre = Vector(np.vstack((part_upper, part_lower)).mean(0))
        result["gap"] = (on_upper @ centre - on_lower @ centre).length
        result["angle"] = float(np.degrees((on_upper @ on_lower.inverted()).to_quaternion().angle))
    result["ok"] = bool(enough and together >= params.BITE_MIN_FIT and result["gap"] <= params.BITE_TOLERANCE
                        and result["angle"] <= params.BITE_MAX_ANGLE)
    return result


def show_fitted_bite(scene, bite, result):
    """A ghost of the bite where its shape says it belongs (same mesh data, nothing copied or moved)."""
    name = f"{bite.name} (recalé, contrôle)"
    ghost = bpy.data.objects.get(name)
    if ghost is None:
        ghost = bpy.data.objects.new(name, bite.data)
        model_collection(scene).objects.link(ghost)
        ghost.omfs.role = 'ANALYSIS'
        ghost.display_type = 'WIRE'
    ghost.matrix_world = Matrix(np.array(result["pose"]).reshape(4, 4).tolist())
    return ghost


def bite_stale(scene, result):
    """Has the bite or an arch moved since the check (or been swapped for another object)?"""
    objects = [scene.objects.get(name) for name in result["objects"]]
    if any(o is None for o in objects) or list(bite_scans(scene)) != objects:
        return True
    return any(not np.allclose(np.array(o.matrix_world).ravel(), where, atol=1e-5)
               for o, where in zip(objects, result["where"]))


def pin_models(scene):
    """(lower, upper) models joined by the repositioning pins: chosen in the panel, else the models
    whose name says lower / upper."""
    settings = scene.omfs_model
    models = [o for o in scene.objects if is_model(o)]

    def by_name(words):
        return next((o for o in models if any(w in o.name.lower() for w in words)
                     and not any(w in o.name.lower() for w in BITE_WORDS)), None)
    lower = settings.pin_lower if is_model(settings.pin_lower) else by_name(LOWER_WORDS)
    upper = settings.pin_upper if is_model(settings.pin_upper) else by_name(UPPER_WORDS)
    return lower, upper


def last_bite(scene):
    import ast
    return ast.literal_eval(scene.get("omfs_bite_check", "None"))


# --- Properties and operators ---------------------------------------------------------------

def _rebuild(settings, context):
    scan = settings.scan
    if is_scan(scan) and model_of(context.scene, scan) is not None:
        make_model(context.scene, scan)


class OMFS_ModelSettings(bpy.types.PropertyGroup):
    scan: bpy.props.PointerProperty(name="Scan", type=bpy.types.Object, poll=lambda _s, o: is_scan(o),
                                    description="Arcade (scan intra-oral ouvert) à fermer en modèle")
    height: bpy.props.FloatProperty(
        name="Hauteur du modèle", subtype='DISTANCE', default=params.MODEL_HEIGHT, min=5.0, soft_max=45.0,
        step=50, precision=1, update=_rebuild,
        description="Des pointes cuspidiennes à la base plane, en mm (le scan au-delà est coupé)")
    fringe: bpy.props.FloatProperty(
        name="Frange retirée", subtype='DISTANCE', default=params.MODEL_FRINGE, min=0.0, soft_max=5.0,
        step=10, precision=1, update=_rebuild,
        description="Retire le bord irrégulier du scan sur cette largeur (jamais dans la zone des dents)")
    teeth_band: bpy.props.FloatProperty(
        name="Zone des dents protégée", subtype='DISTANCE', default=params.MODEL_TEETH_BAND, min=0.0,
        soft_max=15.0, step=50, precision=1, update=_rebuild,
        description="Rien n'est retiré à moins de cette hauteur sous les pointes cuspidiennes voisines")
    teeth_radius: bpy.props.FloatProperty(
        name="Rayon de voisinage", subtype='DISTANCE', default=params.MODEL_TEETH_RADIUS, min=1.0,
        soft_max=15.0, step=50, precision=1, update=_rebuild,
        description="Vu d'en haut, distance à laquelle on cherche la pointe cuspidienne la plus haute")
    outline_smooth: bpy.props.IntProperty(
        name="Lissage du contour", default=params.MODEL_OUTLINE_SMOOTH, min=0, soft_max=200, update=_rebuild,
        description="Passes de lissage du contour de la base (ne déplace que la base, jamais le scan)")
    scrap_fraction: bpy.props.FloatProperty(
        name="Débris", default=params.MODEL_SCRAP_FRACTION, min=0.0, max=0.5, update=_rebuild,
        description="Morceaux détachés plus petits que cette fraction du scan: retirés")
    use_bar: bpy.props.BoolProperty(
        name="Barre de renfort", default=True, update=_rebuild,
        description="Arcade en fer à cheval: un cylindre relie les deux bouts, au ras de la base, "
                    "pour que le modèle ne se déforme pas (pas de barre avec un palais complet)")
    bar_diameter: bpy.props.FloatProperty(
        name="Diamètre de la barre", subtype='DISTANCE', default=params.MODEL_BAR_DIAMETER, min=1.0,
        soft_max=10.0, step=50, precision=1, update=_rebuild)
    bite: bpy.props.PointerProperty(name="STL Bite", type=bpy.types.Object, poll=lambda _s, o: is_scan(o),
                                    description="Scan d'occlusion (mot « bite » dans le nom, par défaut): "
                                                "une partie du haut et une partie du bas")
    upper: bpy.props.PointerProperty(name="Haut", type=bpy.types.Object, poll=lambda _s, o: is_scan(o),
                                     description="Arcade du haut (par défaut: le scan dont le nom dit upper / haut)")
    lower: bpy.props.PointerProperty(name="Bas", type=bpy.types.Object, poll=lambda _s, o: is_scan(o),
                                     description="Arcade du bas (par défaut: le scan dont le nom dit lower / bas)")
    pin_lower: bpy.props.PointerProperty(name="Modèle du bas", type=bpy.types.Object, poll=lambda _s, o: is_model(o),
                                         description="Porte les femelles, debout sur sa base (par défaut: son nom dit lower / bas)")
    pin_upper: bpy.props.PointerProperty(name="Modèle du haut", type=bpy.types.Object, poll=lambda _s, o: is_model(o),
                                         description="Porte les bras et les pins mâles (par défaut: son nom dit upper / haut)")
    flip: bpy.props.BoolProperty(name="Inverser l'axe", default=False, update=_rebuild,
                                 description="Si la base apparaît du côté des dents")


class OMFS_OT_make_model(bpy.types.Operator):
    """Close the chosen arch scan into a printable model: base, wall and flat bottom (the scan's vertices
    never move; the teeth zone is never trimmed)"""
    bl_idname = "omfs.make_model"
    bl_label = "Créer le modèle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scan = context.scene.omfs_model.scan
        if not is_scan(scan):
            self.report({'WARNING'}, "Choisissez d'abord un scan d'arcade")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        error = make_model(context.scene, scan)
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        return {'FINISHED'}


class OMFS_OT_check_bite(bpy.types.Operator):
    """Does the bite's shape say the arches are in the right occlusion? Fits the bite (never the arches)
    on both, then each part on its arch, and compares the two poses (about 10 s)"""
    bl_idname = "omfs.check_bite"
    bl_label = "Vérifier l'occlusion (STL Bite)"

    def execute(self, context):
        bite, upper, lower = bite_scans(context.scene)
        missing = [label for label, obj in (("STL Bite", bite), ("haut", upper), ("bas", lower)) if obj is None]
        if missing:
            self.report({'WARNING'}, "Introuvable: " + ", ".join(missing) + " (choisissez-le dans le panneau)")
            return {'CANCELLED'}
        result = check_bite(bite, upper, lower)
        context.scene["omfs_bite_check"] = str(result)
        show_fitted_bite(context.scene, bite, result)
        return {'FINISHED'}


classes = (OMFS_ModelSettings, OMFS_OT_make_model, OMFS_OT_check_bite)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_model = bpy.props.PointerProperty(type=OMFS_ModelSettings)
    bpy.types.Object.omfs_model_source = bpy.props.StringProperty()


def unregister():
    del bpy.types.Object.omfs_model_source
    del bpy.types.Scene.omfs_model
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
