"""Object creator: Splint / Chape. Intermediate or final occlusal splint, or a single-arch cap.

Built in steps (docs/workflow.md, "Object creator"):
1. Arch curve: points clicked on the arches, joined by a smooth curve (done).
2. Parametric blank swept along the curve, as wide as the crowns plus margins (done).
3. Tooth imprints by projection onto the arches, 0.10 mm clearance (to come).
4. Engraved text and print checks (to come).

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
from mathutils import Vector

from . import landmarks, locking, params, reference_frame

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
    """Occlusal "up": the reference frame's, else the scanner's Z."""
    frame = reference_frame.find_frame(scene)
    return frame.matrix_world.col[2].xyz.normalized() if frame else Vector((0, 0, 1))


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


# --- Blank ----------------------------------------------------------------------------------
#
# The blank is a closed slab following the arch curve: as wide as the crowns at each station
# (measured on the scans, both arches together) plus a lingual and a buccal margin, reaching
# "depth" above and below the curve (the middle of the bite), and past the last tooth at each end.
# Its top and bottom faces are fine grids (vertex groups "Top" / "Bottom"): step 3 projects them
# onto the teeth.

BLANK_NAME = "Splint"
BLANK_TAG = "splint blank"      # made_by of the blank
BLANK_COLOUR = (0.95, 0.70, 0.30, 0.55)   # guide orange, see-through
TOP_GROUP, BOTTOM_GROUP = "Top", "Bottom"
END_REACH = 20.0                # mm past each end of the curve searched for the last tooth
END_SIDEWAYS = 6.0              # mm on each side of the end cross-section, in that search (a Bezier end leaves the arch a little)

_measures = {}                  # measurement key -> raw extents (lingual, buccal, end extensions)


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
    """Crown extents at each station: (lingual, buccal) in mm from the curve, and how far the last
    tooth reaches past each end. Only the crown band the splint will cover is measured (no palate,
    no gingiva below it)."""
    up = np.array(up)
    verts = np.concatenate([world_vertices(o) for o in arches(scene)])
    heights = verts @ up
    band = (heights > (points @ up).min() - depth_low) & (heights < (points @ up).max() + depth_up)
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
        ok = (np.abs(along) <= 0.75 * step) & (h <= depth_up) & (h >= -depth_low)
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
    ends = []
    for i, direction in ((0, -tangents[0]), (-1, tangents[-1])):
        rel = verts - points[i]
        past = rel @ direction
        lat = rel @ normals[i]
        h = rel @ up
        # Generous sideways window: past its end the arch bends away from the curve's last tangent.
        ok = (past > 0) & (past < END_REACH) & (lat > -lingual[i] - END_SIDEWAYS) & \
             (lat < buccal[i] + END_SIDEWAYS) & (h <= depth_up) & (h >= -depth_low)
        ends.append(float(past[ok].max()) if ok.any() else 0.0)
    return lingual, buccal, ends


def curve_key(obj):
    return repr([tuple(round(c, 4) for c in p) for p in curve_points(obj)])


def blank_key(scene):
    props = scene.omfs_splint
    return repr((curve_key(find_curve(scene)), [o.name for o in arches(scene)],
                 [tuple(map(tuple, o.matrix_world)) for o in arches(scene)],
                 round(props.depth_up, 3), round(props.depth_low, 3)))


def blank_data(scene, depsgraph):
    """Vertices, faces, and the top / bottom vertex indices of the blank, or an error message."""
    props = scene.omfs_splint
    curve = find_curve(scene)
    if curve is None or len(curve_points(curve)) < 2:
        return "Cliquez d'abord la courbe d'arcade (2 points au moins)"
    if not arches(scene):
        return "Choisissez les arcades"
    up = up_axis(scene)
    step = params.SPLINT_MEASURE_STEP
    points, tangents, normals, length = stations(curve, depsgraph, step, up)
    key = blank_key(scene)
    if key not in _measures:
        _measures.clear()
        _measures[key] = measure(scene, points, tangents, normals, up, props.depth_up, props.depth_low, step)
    measured = _measures[key]
    if measured is None:
        return "Aucune dent trouvée le long de la courbe: vérifiez les arcades et la profondeur"
    lingual, buccal, ends = measured
    window = int(round(params.SPLINT_SMOOTH / step))
    lingual = smooth(lingual, window) + props.lingual_margin
    buccal = smooth(buccal, window) + props.buccal_margin
    ext = [e + props.distal_margin if e > 0 else props.distal_margin for e in ends]

    # Rows every `resolution` mm, from past the first end to past the last one.
    res = props.resolution
    station_s = np.linspace(0.0, length, len(points))
    rows_s = np.arange(-ext[0], length + ext[1] + 1e-6, res)
    inside = np.clip(rows_s, 0.0, length)
    c = np.stack([np.interp(inside, station_s, points[:, i]) for i in range(3)], axis=1)
    t = np.stack([np.interp(inside, station_s, tangents[:, i]) for i in range(3)], axis=1)
    n = np.stack([np.interp(inside, station_s, normals[:, i]) for i in range(3)], axis=1)
    t /= np.linalg.norm(t, axis=1)[:, None]
    n /= np.linalg.norm(n, axis=1)[:, None]
    c += t * (rows_s - inside)[:, None]                         # straight on past the ends
    lin = np.interp(inside, station_s, lingual)
    buc = np.interp(inside, station_s, buccal)

    cols = int(np.ceil((lin + buc).max() / res))
    f = np.linspace(0.0, 1.0, cols + 1)
    left = c - n * lin[:, None]
    right = c + n * buc[:, None]
    grid = left[:, None, :] + (right - left)[:, None, :] * f[None, :, None]   # rows x cols x 3
    up = np.array(up)
    top = (grid + up * props.depth_up).reshape(-1, 3)
    bottom = (grid - up * props.depth_low).reshape(-1, 3)
    verts = np.concatenate([top, bottom])

    rows, w = len(rows_s), cols + 1
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
    info = {"length": float(rows_s[-1] - rows_s[0]), "width_min": float((lin + buc).min()),
            "width_max": float((lin + buc).max())}
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
    return obj


def blank_info(obj):
    return json.loads(obj.get("omfs_blank_info", "{}"))


def blank_stale(scene, obj):
    return obj.get("omfs_blank_key") != blank_key(scene)


def _on_blank_param(props, context):
    if find_blank(context.scene) is not None:
        build_blank(context.scene, context.evaluated_depsgraph_get())


# --- Properties and operators ---------------------------------------------------------

class OMFS_SplintProperties(bpy.types.PropertyGroup):
    kind: bpy.props.EnumProperty(name="Type", items=KINDS, default='FINAL')
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
    distal_margin: bpy.props.FloatProperty(
        name="Distal", subtype='DISTANCE', default=params.SPLINT_DISTAL_MARGIN, min=0.0, soft_max=10.0,
        step=10, update=_on_blank_param, description="Débord au-delà de la dernière dent, à chaque bout")
    depth_up: bpy.props.FloatProperty(
        name="Haut", subtype='DISTANCE', default=params.SPLINT_DEPTH, min=0.5, soft_max=8.0, step=10,
        update=_on_blank_param,
        description="Jusqu'où la gouttière monte sur les dents du haut, depuis le milieu de l'occlusion")
    depth_low: bpy.props.FloatProperty(
        name="Bas", subtype='DISTANCE', default=params.SPLINT_DEPTH, min=0.5, soft_max=8.0, step=10,
        update=_on_blank_param,
        description="Jusqu'où la gouttière descend sur les dents du bas, depuis le milieu de l'occlusion")
    resolution: bpy.props.FloatProperty(
        name="Résolution", subtype='DISTANCE', default=params.SPLINT_RESOLUTION, min=0.1, soft_max=1.0,
        step=5, update=_on_blank_param,
        description="Distance entre les sommets du bloc: l'empreinte des dents ne sera pas plus fine que ça")
    centre: bpy.props.BoolProperty(
        name="Centrer entre les arcades", default=True,
        description="Chaque point cliqué est placé à mi-hauteur entre les faces occlusales du haut et du bas")


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
    """Build (or update) the splint blank along the arch curve: crown width plus the margins,
    from the depth above to the depth below the middle of the bite"""
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

classes = (OMFS_SplintProperties, OMFS_OT_arch_curve_click, OMFS_OT_arch_curve_undo,
           OMFS_OT_arch_curve_clear, OMFS_OT_arch_curve_edit, OMFS_OT_build_blank)


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
