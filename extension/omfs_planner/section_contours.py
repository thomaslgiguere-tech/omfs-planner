"""Section contours: where each visible mesh crosses the section plane, and filled caps.

Read-only on the meshes. Per object, the vertex coordinates and triangles are read once and
cached; moving the plane along its normal then costs two comparisons per triangle (each
triangle's lowest and highest point along the normal are cached too), so the slider stays live
on CT meshes of a million triangles.

A triangle crossing the plane gives one segment. On a closed mesh, the segments chain into
closed loops (each crossed edge is shared by two triangles); the loops are filled for the caps,
row by row (even-odd rule: holes and self-intersecting segmentations are handled).
An open mesh (intraoral scan) gives open lines: contour only, no cap.

Overlap: two caps filled on the same row grid are intersected row by row; what is inside both
objects is drawn in red, and its area is the sum of the row lengths times the row step.
"""

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import params, section

_meshes = {}       # object name -> (vertex coords (N, 3), triangles (T, 3)), local coordinates
_heights = {}      # object name -> (normal key, vertex heights, triangle min, triangle max)
_sections = {}     # object name -> (state key, Section)
_rows = {}         # object name -> (state key, filled rows of its cap)
_overlaps = {}     # 'all' -> (state key, [(name A, name B, rows, area mm²)])
_batches = {}      # (object name, what) -> (state key, GPU batch)


class Section:
    """One object cut by the plane, in world coordinates."""

    def __init__(self, segments, loops, open_lines):
        self.segments = segments     # (K, 2, 3) every crossing segment
        self.loops = loops           # closed polylines [(M, 3)], for caps and overlap
        self.open_lines = open_lines  # True if some segments did not close (open mesh)


def mesh_arrays(obj, depsgraph):
    """Vertex coordinates and triangles of the evaluated mesh (modifiers included), cached."""
    cached = _meshes.get(obj.name)
    if cached is None:
        mesh = obj.evaluated_get(depsgraph).data
        co = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", co)
        tris = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        cached = _meshes[obj.name] = (co.reshape(-1, 3), tris.reshape(-1, 3))
    return cached


def local_plane(obj, plane):
    """The world plane n . p = c in the object's local coordinates: (normal, c), unnormalised."""
    n = plane.col[2].xyz
    a, t = obj.matrix_world.to_3x3(), obj.matrix_world.translation
    return a.transposed() @ n, n.dot(plane.translation) - n.dot(t)


def heights(obj, co, tris, normal):
    """Height of every vertex along the normal, lowest and highest per triangle (cached per normal)."""
    key = tuple(normal)
    cached = _heights.get(obj.name)
    if cached is None or cached[0] != key:
        h = co @ np.array(key, np.float32)
        ht = h[tris]
        cached = _heights[obj.name] = (key, h, ht.min(axis=1), ht.max(axis=1))
    return cached[1:]


def crossing_segments(co, tris, h, tmin, tmax, c):
    """Segments where the plane at height c cuts the triangles, and the crossed edges' ids."""
    hit = tris[(tmin <= c) & (tmax > c)]   # vertices above (h > c) and not above
    if not len(hit):
        return np.empty((0, 2, 3), np.float32), np.empty((0, 2), np.int64)
    d = h[hit] - c                                    # (K, 3)
    i, j = hit, hit[:, [1, 2, 0]]                     # edges 0-1, 1-2, 2-0
    di, dj = d, d[:, [1, 2, 0]]
    crossed = (di > 0) != (dj > 0)                    # exactly two edges per triangle
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(crossed, di / (di - dj), 0.0)[..., None]
    points = co[i] + f * (co[j] - co[i])              # (K, 3, 3), one per edge
    pick = np.argsort(~crossed, axis=1, kind='stable')[:, :2]
    rows = np.arange(len(hit))[:, None]
    n = len(co)
    edge_ids = np.minimum(i, j).astype(np.int64) * n + np.maximum(i, j)
    return points[rows, pick], edge_ids[rows, pick]


def chain_loops(edge_ids):
    """Chain segments sharing a crossed edge: (closed loops as node lists, node ids of each segment end,
    True if some segments stay open)."""
    unique, nodes = np.unique(edge_ids, return_inverse=True)
    nodes = nodes.reshape(-1, 2)
    links = [[] for _ in range(len(unique))]
    for s, (a, b) in enumerate(nodes.tolist()):
        links[a].append(s)
        links[b].append(s)
    used = bytearray(len(nodes))
    loops, open_lines = [], False
    node_list = nodes.tolist()
    for s in range(len(nodes)):
        if used[s]:
            continue
        used[s] = 1
        start, node = node_list[s]
        ring = [start]
        while node != start:
            ring.append(node)
            nxt = [t for t in links[node] if not used[t]] if len(links[node]) == 2 else []
            if not nxt:
                break
            used[nxt[0]] = 1
            a, b = node_list[nxt[0]]
            node = b if a == node else a
        if node == start and len(ring) >= 3:
            loops.append(ring)
        else:
            open_lines = True
    return loops, nodes, open_lines


def cut(obj, plane, depsgraph):
    """The object's Section on the plane (world coordinates), cached until the plane or object moves."""
    key = (tuple(map(tuple, plane)), tuple(map(tuple, obj.matrix_world)))
    cached = _sections.get(obj.name)
    if cached and cached[0] == key:
        return cached[1]
    co, tris = mesh_arrays(obj, depsgraph)
    normal, c = local_plane(obj, plane)
    h, tmin, tmax = heights(obj, co, tris, normal)
    segments, edge_ids = crossing_segments(co, tris, h, tmin, tmax, c)
    loops, open_lines = [], False
    if len(segments):
        rings, nodes, open_lines = chain_loops(edge_ids)
        node_points = np.empty((nodes.max() + 1, 3), np.float32)
        node_points[nodes.ravel()] = segments.reshape(-1, 3)
        loops = [to_world(obj, node_points[ring]) for ring in rings]
    result = Section(to_world(obj, segments.reshape(-1, 3)).reshape(-1, 2, 3), loops, open_lines)
    _sections[obj.name] = (key, result)
    return result


def to_world(obj, points):
    m = np.array(obj.matrix_world, np.float32)
    return points @ m[:3, :3].T + m[:3, 3]


def plane_2d(plane, points):
    """World points on the plane -> (x, y) in the plane's axes."""
    m = np.array(plane, np.float32)
    return (points - m[:3, 3]) @ m[:3, :2]


def fill_rows(loops_2d, step):
    """Even-odd fill of closed 2D polylines, row by row: (row index, x start, x end) arrays.

    Row k is the line y = (k + 0.5) * step, on a grid shared by every object, so fills of two
    objects can be compared row by row. Self-intersecting or nested loops (holes) are fine.
    """
    if not loops_2d:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    a = np.concatenate(loops_2d)
    b = np.concatenate([np.roll(loop, -1, axis=0) for loop in loops_2d])
    lo, hi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    k0 = np.ceil(lo / step - 0.5).astype(np.int64)   # first row with y >= lo
    k1 = np.ceil(hi / step - 0.5).astype(np.int64)   # rows with y < hi end before k1
    count = np.maximum(k1 - k0, 0)
    seg = np.repeat(np.arange(len(a)), count)
    k = k0[seg] + np.arange(len(seg)) - np.repeat(np.cumsum(count) - count, count)
    y = (k + 0.5) * step
    t = (y - a[seg, 1]) / (b[seg, 1] - a[seg, 1])
    x = a[seg, 0] + t * (b[seg, 0] - a[seg, 0])
    order = np.lexsort((x, k))   # each row crosses closed loops an even number of times
    k, x = k[order], x[order]
    return k[0::2], x[0::2], x[1::2]


def cap_rows(plane, loops):
    return fill_rows([plane_2d(plane, loop) for loop in loops], params.SECTION_FILL_STEP)


def rows_area(rows):
    return float((rows[2] - rows[1]).sum() * params.SECTION_FILL_STEP)


def rows_quads(plane, rows):
    """Filled rows as world-space triangles (one thin quad per row interval) for drawing."""
    k, x0, x1 = rows
    step = params.SECTION_FILL_STEP
    y0, y1 = (k - 0.01) * step, (k + 1.01) * step   # rows overlap a hair: no pixel seams
    corners = np.stack([np.stack(c, axis=1) for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], axis=1)
    m = np.array(plane, np.float32)
    points = (corners.reshape(-1, 2) @ m[:3, :2].T + m[:3, 3]).astype(np.float32)
    first = np.arange(len(k), dtype=np.int32)[:, None] * 4
    tris = np.concatenate([first + (0, 1, 2), first + (0, 2, 3)]).astype(np.int32)   # the GPU wants 4-byte ints
    return points, tris


def object_rows(obj, plane, result):
    """Filled rows of the object's cap, cached with its section."""
    state = _sections[obj.name][0]
    cached = _rows.get(obj.name)
    if cached is None or cached[0] != state:
        cached = _rows[obj.name] = (state, cap_rows(plane, result.loops))
    return cached[1]


def rows_intersection(a, b):
    """Where two fills overlap, row by row: same (row, x start, x end) form."""
    k = np.concatenate([a[0], a[0], b[0], b[0]])
    x = np.concatenate([a[1], a[2], b[1], b[2]])
    na, nb = len(a[0]), len(b[0])
    in_a = np.concatenate([np.ones(na), -np.ones(na), np.zeros(2 * nb)])
    in_b = np.concatenate([np.zeros(2 * na), np.ones(nb), -np.ones(nb)])
    order = np.lexsort((x, k))
    k, x = k[order], x[order]
    inside = (np.cumsum(in_a[order]) > 0.5) & (np.cumsum(in_b[order]) > 0.5)   # counts reset each row
    keep = inside[:-1] & (k[:-1] == k[1:]) & (x[1:] > x[:-1])
    return k[:-1][keep], x[:-1][keep], x[1:][keep]


def overlaps(scene, plane, depsgraph):
    """Every pair of visible closed objects that interpenetrate on the section:
    [(name A, name B, rows, area mm²)], largest first. Cached until something moves."""
    filled = []
    for obj in section.visible_meshes(scene):
        result = cut(obj, plane, depsgraph)
        if result.loops:
            filled.append((obj, result))
    key = tuple(_sections[obj.name][0] + (obj.name,) for obj, _result in filled)
    cached = _overlaps.get('all')
    if cached and cached[0] == key:
        return cached[1]
    rows = [(obj.name, object_rows(obj, plane, result)) for obj, result in filled]
    found = []
    for i, (name_a, rows_a) in enumerate(rows):
        for name_b, rows_b in rows[i + 1:]:
            both = rows_intersection(rows_a, rows_b)
            area = rows_area(both)
            if area >= params.SECTION_OVERLAP_MIN_AREA:
                found.append((name_a, name_b, both, area))
    found.sort(key=lambda f: -f[3])
    _overlaps['all'] = (key, found)
    return found


def forget(name):
    _overlaps.clear()
    for cache in (_meshes, _heights, _sections, _rows):
        cache.pop(name, None)
    for key in [k for k in _batches if name in k[0].split("\t")]:   # its own batches and its pairs'
        del _batches[key]


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """An edited mesh (or a modifier change) must be read again."""
    for u in depsgraph.updates:
        if u.is_updated_geometry and isinstance(u.id, bpy.types.Object):
            forget(u.id.original.name)


@bpy.app.handlers.persistent
def _on_load(_dummy):
    for cache in (_meshes, _heights, _sections, _rows, _overlaps, _batches):
        cache.clear()


# --- Drawing ----------------------------------------------------------------------

def cached_batch(name, what, key, make):
    cached = _batches.get((name, what))
    if cached is None or cached[0] != key:
        cached = _batches[(name, what)] = (key, make())
    return cached[1]


OVERLAP_COLOUR = (1.0, 0.0, 0.0, 0.85)


def cap_colour(obj):
    return (*(v * 0.7 for v in obj.color[:3]), 1.0)


def quads_batch(shader, plane, rows):
    points, tris = rows_quads(plane, rows)
    return batch_for_shader(shader, 'TRIS', {"pos": points}, indices=tris) if len(tris) else None


def _draw_contours():
    context = bpy.context
    scene = context.scene
    props = scene.omfs_section
    plane = section.active_plane(scene)
    if plane is None or not (props.show_contours or props.show_caps or props.show_overlap):
        return
    depsgraph = context.view_layer.depsgraph
    cuts = [(obj, cut(obj, plane, depsgraph)) for obj in section.visible_meshes(scene)]
    cuts = [(obj, result, _sections[obj.name][0]) for obj, result in cuts if len(result.segments)]
    fill = gpu.shader.from_builtin('UNIFORM_COLOR')
    if props.show_caps:
        gpu.state.depth_test_set('LESS_EQUAL')   # hidden by what is in front, inside the slab
        for obj, result, state in cuts:
            if result.loops:
                batch = cached_batch(obj.name, 'caps', state,
                                     lambda: quads_batch(fill, plane, object_rows(obj, plane, result)))
                if batch:
                    fill.uniform_float("color", cap_colour(obj))
                    batch.draw(fill)
    gpu.state.depth_test_set('NONE')   # analysis drawn on top: always readable
    if props.show_overlap:
        gpu.state.blend_set('ALPHA')
        fill.uniform_float("color", OVERLAP_COLOUR)
        for name_a, name_b, rows, _area in overlaps(scene, plane, depsgraph):
            state = (_sections[name_a][0], _sections[name_b][0])
            batch = cached_batch(f"{name_a}\t{name_b}", 'overlap', state, lambda: quads_batch(fill, plane, rows))
            if batch:
                batch.draw(fill)
        gpu.state.blend_set('NONE')
    if props.show_contours:
        line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        line.uniform_float("viewportSize", (context.region.width, context.region.height))
        line.uniform_float("lineWidth", params.SECTION_LINE_WIDTH)
        for obj, result, state in cuts:
            batch = cached_batch(obj.name, 'lines', state, lambda: batch_for_shader(
                line, 'LINES', {"pos": result.segments.reshape(-1, 3)}))
            line.uniform_float("color", (*obj.color[:3], 1.0))
            batch.draw(line)


_draw_handle = None


def register():
    global _draw_handle
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load)
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_contours, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    bpy.app.handlers.load_post.remove(_on_load)
    bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _on_load(None)
