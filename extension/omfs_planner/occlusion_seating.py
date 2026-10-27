"""Occlusion seating (part 2): the lower arch is pulled onto the fixed upper arch by springs.

Like IPS CaseDesigner's virtual occlusion: the surgeon clicks corresponding points on both arches
(e.g. a cusp tip on the upper arch and the fossa it goes into on the lower arch). Each pair is a
spring. The solver moves the lower block, step by step (shown as an animation), so the springs
shorten, while the teeth resist interpenetration beyond a tolerated overlap. Only the block's
transform changes: rigid, nothing touches the mesh data.

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
        ('SEQUENCE_COLOR_07', f"Chevauchement > {c:g} mm"),
        ('SEQUENCE_COLOR_01', f"Chevauchement {b:g} – {c:g} mm"),
        ('SEQUENCE_COLOR_02', f"Chevauchement {a:g} – {b:g} mm"),
        ('SEQUENCE_COLOR_03', f"Chevauchement 0 – {a:g} mm"),
        ('SEQUENCE_COLOR_04', f"Contact (écart < {contact:g} mm)"),
        ('SEQUENCE_COLOR_05', f"Écart {contact:g} – {far:g} mm"),
        ('SEQUENCE_COLOR_09', f"Écart > {far:g} mm"),
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
        d[i] = dist if (p - loc).dot(nor) >= 0 else -dist
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


# --- Point pairs (springs) -------------------------------------------------------------

def occlusion_points(scene):
    return [o for o in scene.objects if o.omfs.made_by == MADE_BY and o.omfs.landmark.startswith("OCC_")]


def point_pairs(scene):
    """[(upper point, lower point)] in order. Each point is a child of its arch scan."""
    points = {o.omfs.landmark: o for o in occlusion_points(scene)}
    numbers = sorted({int(lm[5:]) for lm in points})
    return [(points[f"OCC_U{i}"], points[f"OCC_L{i}"]) for i in numbers
            if f"OCC_U{i}" in points and f"OCC_L{i}" in points]


def next_pair_number(scene):
    return max((int(o.omfs.landmark[5:]) for o in occlusion_points(scene)), default=0) + 1


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


# --- Solver --------------------------------------------------------------------------

def skew(v):
    """Matrix S with S @ w = v x w."""
    x, y, z = v
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def seat_step(scene, tolerance):
    """One iteration: a small rigid motion of the lower block that shortens the springs and pushes
    overlaps deeper than the tolerance back out. Returns (translation mm, rotation deg) applied."""
    props = scene.omfs_occlusion
    upper, lower = props.upper, props.lower
    block = occlusion.find_block(scene, 'LOWER')
    centre = np.array(block.matrix_world.translation)
    rows, rhs = [], []

    # Springs: each pulls its lower point a fraction of the way towards its upper point.
    for up, low in point_pairs(scene):
        p, q = np.array(low.matrix_world.translation), np.array(up.matrix_world.translation)
        rows.append(np.hstack((-skew(p - centre), np.eye(3))))
        rhs.append((q - p) * params.OCC_SPRING_STEP)

    # Lower vertices inside the upper surface: move them out along the upper normal.
    points = world_points(lower, samples(lower))
    d, _closest, normals = signed_distances(points, upper, params.OCC_SEARCH_DISTANCE)
    deep = d < -tolerance
    v = points[deep] - centre
    rows.append(np.hstack((np.cross(v, normals[deep]), normals[deep])))
    rhs.append(-tolerance - d[deep])

    # Upper vertices inside the lower surface: move the lower surface away (against its normal).
    points = world_points(upper, samples(upper))
    d, closest, normals = signed_distances(points, lower, params.OCC_SEARCH_DISTANCE)
    deep = d < -tolerance
    v = closest[deep] - centre
    rows.append(np.hstack((np.cross(v, normals[deep]), normals[deep])))
    rhs.append(d[deep] + tolerance)

    a = np.vstack([np.atleast_2d(r) for r in rows if np.size(r)] + [1e-3 * np.eye(6)])
    b = np.concatenate([np.ravel(r) for r in rhs] + [np.zeros(6)])
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    omega, t = Vector(x[:3]), Vector(x[3:])
    angle = min(omega.length, np.radians(MAX_STEP))
    if t.length > MAX_STEP:
        t *= MAX_STEP / t.length
    rotation = Matrix.Rotation(angle, 4, omega.normalized()) if angle > 1e-12 else Matrix()
    c = Vector(centre)
    motion = Matrix.Translation(c + t) @ rotation @ Matrix.Translation(-c)
    block.matrix_world = registration.as_rigid(motion @ block.matrix_world)
    bpy.context.view_layer.update()   # so the arch and its points report their new position
    return t.length, np.degrees(angle)


def converged(step):
    shift, angle = step
    return shift < CONVERGED and angle < CONVERGED


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


def point_fit(scene):
    """World matrix of the lower block after a pure least-squares fit of the springs (Kabsch):
    gets there from any starting pose, even a scan exported upside down."""
    block = occlusion.find_block(scene, 'LOWER')
    lower, upper = pair_arrays(scene)
    fit = Matrix(registration.rigid_fit(lower, upper).tolist())
    return registration.as_rigid(fit @ block.matrix_world)


def seat(scene, tolerance, max_iterations=params.OCC_MAX_ITERATIONS):
    """Run the solver to the end (no animation): point fit, then springs against contact.
    Returns the number of contact iterations."""
    block = occlusion.find_block(scene, 'LOWER')
    block.matrix_world = point_fit(scene)
    bpy.context.view_layer.update()
    for i in range(1, max_iterations + 1):
        if converged(seat_step(scene, tolerance)):
            return i
    return max_iterations


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


def write_overlap_map(scene):
    """Colour both arches by their signed distance to the other one; store the summary."""
    props = scene.omfs_occlusion
    upper, lower = props.upper, props.lower
    deepest = 0.0
    for obj, other in ((lower, upper), (upper, lower)):
        points = registration.vertex_coords(obj)
        d, _closest, _normals = signed_distances(points, other, params.OCC_SEARCH_DISTANCE)
        mesh = obj.data
        attr = mesh.color_attributes.get(MAP_ATTRIBUTE) or mesh.color_attributes.new(
            MAP_ATTRIBUTE, 'FLOAT_COLOR', 'POINT')
        attr.data.foreach_set("color", overlap_colours(d).ravel())
        mesh.color_attributes.active_color = attr
        if np.any(d < 0):
            deepest = max(deepest, float(-np.nanmin(d)))
        if obj is lower:
            contact_areas(scene, lower, points, d)
    props.max_overlap = deepest
    props.map_state = repr(map_key(scene))


def contact_areas(scene, lower, points, d):
    """Contact area (mm²) on the lower arch: patient's right, anterior, left of the block's midline."""
    props = scene.omfs_occlusion
    block = occlusion.find_block(scene, 'LOWER')
    m = block.matrix_world
    x = (points - np.array(m.translation)) @ np.array(m.to_3x3().col[0])   # block X = patient left
    with np.errstate(invalid='ignore'):
        touching = d < params.OCC_CONTACT
    area = vertex_areas(lower)
    w = params.OCC_ANTERIOR_HALF_WIDTH
    props.area_right = float(area[touching & (x < -w)].sum())
    props.area_anterior = float(area[touching & (np.abs(x) <= w)].sum())
    props.area_left = float(area[touching & (x > w)].sum())


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
        shader.uniform_float("lineWidth", 2.0)
        shader.uniform_float("color", spring_colour((q - p).length))
        batch.draw(shader)


_draw_handle = None


# --- Operators -----------------------------------------------------------------------

def arches_ready(scene):
    props = scene.omfs_occlusion
    return props.upper is not None and props.lower is not None and occlusion.find_block(scene, 'LOWER')


class OMFS_OT_add_occlusion_pair(bpy.types.Operator):
    """Cliquez un point sur l'arcade du haut (ex. pointe de cuspide), puis le point correspondant sur
    l'arcade du bas (ex. fosse où elle doit s'asseoir). Échap / clic droit : annuler"""
    bl_idname = "omfs.add_occlusion_pair"
    bl_label = "Ajouter une paire"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if not arches_ready(context.scene):
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return {'CANCELLED'}
        props = context.scene.omfs_occlusion
        self.arches = (props.upper, props.lower)
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        self.number = next_pair_number(context.scene)
        self.upper_point = None
        set_points_hidden(context.scene, False)
        self.prompt()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self):
        which = "l'arcade du HAUT" if self.upper_point is None else "l'arcade du BAS (point correspondant)"
        self.area.header_text_set(f"Ressort {self.number} : cliquez sur {which}   |   Échap / clic droit : annuler")

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
            if self.upper_point is None:
                self.upper_point = add_point(context.scene, f"OCC_U{self.number}", f"Haut {self.number}",
                                             location, arch)
                self.prompt()
                return {'RUNNING_MODAL'}
            add_point(context.scene, f"OCC_L{self.number}", f"Bas {self.number}", location, arch)
            self.finish(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class OMFS_OT_delete_occlusion_pair(bpy.types.Operator):
    """Supprimer ce ressort (ses deux points)"""
    bl_idname = "omfs.delete_occlusion_pair"
    bl_label = "Supprimer le ressort"
    bl_options = {'REGISTER', 'UNDO'}

    number: bpy.props.IntProperty()

    def execute(self, context):
        for obj in [o for o in occlusion_points(context.scene)
                    if o.omfs.landmark in (f"OCC_U{self.number}", f"OCC_L{self.number}")]:
            bpy.data.objects.remove(obj)
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
    """Les ressorts tirent l'arcade du bas (et son bloc) vers l'arcade du haut fixe ; les dents résistent
    au-delà de la pénétration tolérée. Animé dans la vue ; Échap : annuler et revenir au départ"""
    bl_idname = "omfs.seat_occlusion"
    bl_label = "Asseoir (ressorts)"
    bl_options = {'REGISTER', 'UNDO'}

    def ready(self, context):
        if not arches_ready(context.scene):
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return False
        if len(point_pairs(context.scene)) < 3:
            self.report({'ERROR'}, "Placez au moins 3 paires de points")
            return False
        if points_spread(context.scene) < MIN_SPREAD:
            self.report({'ERROR'}, "Points presque alignés : étalez-les (ex. incisives + molaires gauche et droite)")
            return False
        return True

    def execute(self, context):   # scripts and tests: no animation
        if not self.ready(context):
            return {'CANCELLED'}
        scene = context.scene
        iterations = seat(scene, scene.omfs_occlusion.tolerance)
        write_overlap_map(scene)
        self.report({'INFO'}, f"Assise : {iterations} itérations")
        return {'FINISHED'}

    def invoke(self, context, event):
        global _seating
        if not self.ready(context):
            return {'CANCELLED'}
        self.block = occlusion.find_block(context.scene, 'LOWER')
        self.start = self.block.matrix_world.copy()
        self.fit = point_fit(context.scene)   # phase 1: glide to the point fit, phase 2: contact
        self.frame = 0
        self.iteration = 0
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
        step = seat_step(scene, scene.omfs_occlusion.tolerance)
        longest = max(spring_lengths(scene), default=0.0)
        if self.area:
            self.area.header_text_set(f"Assise : itération {self.iteration}, ressort le plus long "
                                      f"{longest:.2f} mm   |   Échap : annuler")
            self.area.tag_redraw()
        if converged(step) or self.iteration >= params.OCC_MAX_ITERATIONS:
            self.finish(context)
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
