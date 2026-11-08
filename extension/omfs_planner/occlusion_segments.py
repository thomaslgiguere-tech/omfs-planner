"""Segmental surgery of the upper arch, for occlusion seating: interdental (and palatal) osteotomy lines
divide the upper scan into segments, each seated on its own (e.g. transverse expansion of the posterior
segments for a better intercuspation).

Gesture: click-drag a line across the arch, like the guide's plane cut, then slide each end of the line
on the scan to put it exactly in the embrasure. A line is finite: it cuts only along its own length
(plus a small overshoot), so an interdental line does not cross the other side of the arch. It cuts
through the full height, along the occlusal axis of the arch. The regions the lines close off become
segments.

The scan is never modified: each segment is a new mesh holding a copy of the scan's faces of that region
(same vertices, same coordinates), with its own mobile block (occlusion.py). The scan is hidden while the
arch is segmented and comes back when the segments are joined again.
"""

import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, occlusion, occlusion_seating, params, registration

MADE_BY = occlusion.SEGMENT_MADE_BY
TRAIT_PREFIX = "SEGCUT_"
HANDLE_SIZE = 0.5        # mm, radius of the end spheres
PICK_PIXELS = 20         # a press this close to an end, on screen, grabs it
LINE_COLOUR = (1.0, 0.55, 0.1, 1.0)
SEGMENT_TINTS = ((0.95, 0.60, 0.60, 1.0), (0.60, 0.75, 0.95, 1.0), (0.70, 0.90, 0.60, 1.0),
                 (0.95, 0.85, 0.55, 1.0), (0.80, 0.65, 0.95, 1.0), (0.60, 0.90, 0.90, 1.0))


# --- Osteotomy lines ------------------------------------------------------------------------

def trait_ends(scene):
    """{number: (end A, end B)} of the lines drawn on the upper arch (Empties, children of the scan)."""
    ends = {}
    for obj in scene.objects:
        lm = obj.omfs.landmark
        if obj.omfs.made_by == MADE_BY and lm.startswith(TRAIT_PREFIX):
            number, end = lm[len(TRAIT_PREFIX):].split("_")
            ends.setdefault(int(number), {})[end] = obj
    return {n: (e["A"], e["B"]) for n, e in sorted(ends.items()) if "A" in e and "B" in e}


def add_end(scene, upper, number, end, location):
    obj = bpy.data.objects.new(f"Trait {number} {end}", None)
    occlusion.occlusion_collection(scene).objects.link(obj)
    obj.empty_display_type = 'SPHERE'
    obj.empty_display_size = HANDLE_SIZE
    obj.show_in_front = True
    obj.omfs.role = 'MARKER'
    obj.omfs.landmark = f"{TRAIT_PREFIX}{number}_{end}"
    obj.omfs.made_by = MADE_BY
    obj.parent = upper
    obj.matrix_parent_inverse = upper.matrix_world.inverted()
    obj.matrix_world = Matrix.Translation(location)
    locking.set_locked(obj, True)   # moved by "Adjust the lines" only, slid on the scan
    return obj


def add_trait(scene, upper, a, b):
    number = max(trait_ends(scene), default=0) + 1
    add_end(scene, upper, number, "A", a)
    add_end(scene, upper, number, "B", b)
    return number


def remove_trait(scene, number):
    for obj in trait_ends(scene).get(number, ()):
        bpy.data.objects.remove(obj)


def set_traits_hidden(scene, hide):
    for ends in trait_ends(scene).values():
        for obj in ends:
            obj.hide_set(hide)


def occlusal_axis_local(upper):
    co = occlusion_seating.samples(upper)
    return np.linalg.svd(co - co.mean(axis=0), full_matrices=False)[2][2]


def trait_frame(upper, a, b):
    """(origin, along, normal, length) in the scan's local coordinates: the cutting plane holds the line
    and the occlusal axis; along = the line's direction in the occlusal plane."""
    to_local = upper.matrix_world.inverted()
    a, b = np.array(to_local @ a), np.array(to_local @ b)
    axis = occlusal_axis_local(upper)
    along = (b - a) - ((b - a) @ axis) * axis
    length = np.linalg.norm(along)
    along = along / max(length, 1e-9)
    return a, along, np.cross(along, axis), length


def frames(scene, upper):
    return [trait_frame(upper, a.matrix_world.translation, b.matrix_world.translation)
            for a, b in trait_ends(scene).values()]


# --- Mesh connectivity ----------------------------------------------------------------------

_faces = {}


def face_data(mesh):
    """(face centres, face areas, pairs of faces sharing an edge), local, cached per mesh."""
    key = (mesh.name, len(mesh.vertices), len(mesh.polygons))
    if key not in _faces:
        n = len(mesh.polygons)
        centres = np.empty(n * 3)
        mesh.polygons.foreach_get("center", centres)
        areas = np.empty(n)
        mesh.polygons.foreach_get("area", areas)
        edge_of_loop = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("edge_index", edge_of_loop)
        totals = np.empty(n, dtype=np.int64)
        mesh.polygons.foreach_get("loop_total", totals)
        poly_of_loop = np.repeat(np.arange(n), totals)
        order = np.argsort(edge_of_loop, kind='stable')
        edges, polys = edge_of_loop[order], poly_of_loop[order]
        same = edges[1:] == edges[:-1]          # consecutive loops on one edge: two faces sharing it
        pairs = np.stack((polys[:-1][same], polys[1:][same]), axis=1)
        _faces[key] = (centres.reshape(-1, 3), areas, pairs)
    return _faces[key]


def components(n, pairs):
    """Label of each of n nodes: nodes linked by pairs share a label (smallest index of the group)."""
    labels = np.arange(n)
    while True:
        a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
        low = np.minimum(a, b)
        hooked = labels.copy()
        np.minimum.at(hooked, a, low)
        np.minimum.at(hooked, b, low)
        while True:                              # pointer jumping: every node to its group's root
            jumped = hooked[hooked]
            if np.array_equal(jumped, hooked):
                break
            hooked = jumped
        if np.array_equal(hooked, labels):
            return labels
        labels = hooked


def face_labels(upper, cut_frames):
    """Segment label per face of the scan: two neighbouring faces are cut apart when both lie within a
    line's length (plus overshoot) and on opposite sides of its plane."""
    centres, _areas, pairs = face_data(upper.data)
    keep = np.ones(len(pairs), dtype=bool)
    o = params.SEG_CUT_OVERSHOOT
    for origin, along, normal, length in cut_frames:
        s = (centres - origin) @ along
        side = (centres - origin) @ normal > 0
        inside = (s >= -o) & (s <= length + o)
        f, g = pairs[:, 0], pairs[:, 1]
        keep &= ~(inside[f] & inside[g] & (side[f] != side[g]))
    return components(len(centres), pairs[keep])


_cut_points = {}


def cut_points(upper, frame):
    """World points where the scan's edges cross a line's plane within its length: the cut, drawn on the scan."""
    mesh = upper.data
    key = (mesh.name, len(mesh.vertices), tuple(np.round(np.concatenate((frame[0], frame[1])), 4)), round(frame[3], 4))
    if key not in _cut_points:
        if len(_cut_points) > 32:
            _cut_points.clear()
        co = registration.vertex_coords(upper, world=False)
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        origin, along, normal, length = frame
        h = (co - origin) @ normal
        hu, hv = h[edges[:, 0]], h[edges[:, 1]]
        crossing = (hu > 0) != (hv > 0)
        u, v, hu, hv = co[edges[crossing, 0]], co[edges[crossing, 1]], hu[crossing], hv[crossing]
        points = u + (v - u) * (hu / (hu - hv))[:, None]
        s = (points - origin) @ along
        o = params.SEG_CUT_OVERSHOOT
        _cut_points[key] = points[(s >= -o) & (s <= length + o)]
    return occlusion_seating.world_points(upper, _cut_points[key])


# --- Segments --------------------------------------------------------------------------------

def is_segment(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.made_by == MADE_BY


def segments(scene):
    """The upper arch's segments, in the order they were made."""
    found = [o for o in scene.objects if is_segment(o)]
    return sorted(found, key=lambda o: int(o.omfs.landmark[3:]))


def block_of(segment):
    return occlusion.find_block(bpy.context.scene, segment.omfs.landmark)


def reference(scene):
    """The segment the lower arch is seated on first (it sets the Class and the midline)."""
    chosen = scene.omfs_occlusion.reference_segment
    parts = segments(scene)
    return chosen if chosen in parts else (parts[0] if parts else None)


def submesh(mesh, faces, name):
    """New mesh with a copy of the given faces of mesh (same vertex coordinates, nothing moved)."""
    totals = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    loop_vert = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    faces = np.flatnonzero(faces)
    sizes = totals[faces]
    loops = np.repeat(starts[faces] - np.cumsum(sizes) + sizes, sizes) + np.arange(sizes.sum())
    used, index = np.unique(loop_vert[loops], return_inverse=True)
    new = bpy.data.meshes.new(name)
    new.vertices.add(len(used))
    new.vertices.foreach_set("co", co[used].ravel())
    new.loops.add(len(loops))
    new.loops.foreach_set("vertex_index", index.astype(np.int32))
    new.polygons.add(len(faces))
    new.polygons.foreach_set("loop_start", (np.cumsum(sizes) - sizes).astype(np.int32))
    new.update(calc_edges=True)
    new.validate(clean_customdata=False)
    return new


def segment_name(scene, centre, taken):
    """Antérieur / postérieur droit / gauche, from the centre's distance to the midline."""
    x = float(occlusion_seating.midline_offset(scene, np.array([centre]))[0])
    if abs(x) <= params.SEG_MIDLINE_BAND:
        name = "Segment antérieur"
    else:
        name = "Segment postérieur " + ("gauche" if x > 0 else "droit")   # + = patient's left
    base, k = name, 2
    while name in taken:
        name, k = f"{base} {k}", k + 1
    taken.add(name)
    return name


def separate(scene):
    """Divide the upper scan along the lines into segments, each with its own block. Returns an error
    message, or None. The scan itself is only hidden."""
    props = scene.omfs_occlusion
    upper = props.upper
    if segments(scene):
        return "L'arcade est déjà segmentée : réunissez-la d'abord"
    cut_frames = frames(scene, upper)
    if not cut_frames:
        return "Tracez d'abord au moins un trait d'ostéotomie"
    labels = face_labels(upper, cut_frames)
    _centres, areas, _pairs = face_data(upper.data)
    totals = np.bincount(labels, weights=areas, minlength=len(areas))
    kept = [k for k in np.flatnonzero(totals >= params.SEG_MIN_AREA)]
    if len(kept) < 2:
        return ("Les traits ne séparent pas l'arcade : prolongez-les au-delà du bord du scan, et faites "
                "se rejoindre les traits qui doivent fermer un segment")
    bpy.context.view_layer.update()
    upper_block = occlusion.find_block(scene, 'UPPER')
    rotation = (upper_block or upper).matrix_world.to_quaternion()
    taken, made = set(), []
    for number, label in enumerate(sorted(kept, key=lambda k: -totals[k]), start=1):
        faces = labels == label
        mesh = submesh(upper.data, faces, "Segment")
        obj = bpy.data.objects.new("Segment", mesh)
        occlusion.occlusion_collection(scene).objects.link(obj)
        obj.matrix_world = upper.matrix_world
        obj.omfs.role = 'SCAN'
        obj.omfs.made_by = MADE_BY
        obj.omfs.landmark = f"SEG{number}"
        obj.color = SEGMENT_TINTS[(number - 1) % len(SEGMENT_TINTS)]
        centre = obj.matrix_world @ Vector(registration.vertex_coords(obj, world=False).mean(axis=0))
        obj.name = mesh.name = segment_name(scene, centre, taken)
        block = bpy.data.objects.new(f"Bloc {obj.name[8:]}", None)
        occlusion.occlusion_collection(scene).objects.link(block)
        block.empty_display_type = 'ARROWS'
        block.empty_display_size = occlusion.BLOCK_SIZE / 2
        block.show_in_front = True
        block.omfs.role = 'BLOCK'
        block.omfs.landmark = obj.omfs.landmark
        block.omfs.made_by = MADE_BY
        matrix = Matrix.LocRotScale(centre, rotation, None)
        block.matrix_world = matrix
        block.omfs_occl.preop = matrix
        block.lock_scale = (True,) * 3
        occlusion.add_member(block, obj)
        locking.set_locked(obj, True)   # moves only with its block
        made.append(obj)
    bpy.context.view_layer.update()
    move_upper_points(scene, made)
    anterior = next((o for o in made if o.name.startswith("Segment antérieur")), None)
    props.reference_segment = anterior or segment_with_most_springs(scene, made)
    upper.hide_set(True)
    set_traits_hidden(scene, True)
    return None


def nearest_object(objects, point):
    """The object whose surface is closest to a world point."""
    best, best_d = None, np.inf
    for obj in objects:
        bvh = occlusion_seating.surface(obj)[0]
        hit = bvh.find_nearest(obj.matrix_world.inverted() @ point)
        if hit[0] is not None and hit[3] < best_d:
            best, best_d = obj, hit[3]
    return best


def reparent(point, new_parent, world):
    point.parent = new_parent
    point.matrix_parent_inverse = new_parent.matrix_world.inverted()
    point.matrix_world = Matrix.Translation(world)


def move_upper_points(scene, parts):
    """Each upper spring point now belongs to the segment it sits on (same place, new parent)."""
    for point in occlusion_seating.occlusion_points(scene):
        if point.omfs.landmark.startswith("OCC_U"):
            world = point.matrix_world.translation.copy()
            reparent(point, nearest_object(parts, world), world)


def segment_with_most_springs(scene, parts):
    count = {p.name: 0 for p in parts}
    for up, _low in occlusion_seating.point_pairs(scene):
        if up.parent is not None and up.parent.name in count:
            count[up.parent.name] += 1
    return max(parts, key=lambda p: count[p.name])


def join(scene):
    """Back to the whole scan: segments and their blocks removed, the upper points back on the scan at
    the same place on the teeth (where their segment had moved them is dropped)."""
    upper = scene.omfs_occlusion.upper
    parts = segments(scene)
    for point in occlusion_seating.occlusion_points(scene):
        if point.parent in parts:   # same spot on the teeth, the segment's movement undone
            local = point.parent.matrix_world.inverted() @ point.matrix_world.translation
            reparent(point, upper, upper.matrix_world @ local)
    for obj in parts:
        block = block_of(obj)
        occlusion.remove_ghost(obj)
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        if block is not None:
            bpy.data.objects.remove(block)
    if upper is not None:
        upper.hide_set(False)
    set_traits_hidden(scene, False)


def moved(scene, segment):
    block = block_of(segment)
    return block is not None and occlusion.movement_summary(scene, block) is not None


def across_direction(scene):
    """Unit vector towards the patient's left: the line of the lower canine points, else the lower block's X."""
    points = {o.omfs.landmark: o.matrix_world.translation for o in occlusion_seating.occlusion_points(scene)}
    right, left = points.get("OCC_LCR"), points.get("OCC_LCL")
    if right is not None and left is not None:
        return (left - right).normalized()
    return occlusion.find_block(scene, 'LOWER').matrix_world.to_3x3().col[0].normalized()


def expansion(scene, segment):
    """mm the segment's centre moved away from the midline (negative: towards it)."""
    block = block_of(segment)
    preop = occlusion.preop_matrix(block).translation
    side = 1.0 if occlusion_seating.midline_offset(scene, np.array([preop]))[0] > 0 else -1.0
    return side * (block.matrix_world.translation - preop) @ across_direction(scene)


# --- Drawing ---------------------------------------------------------------------------------

def _draw_traits():
    scene = bpy.context.scene
    upper = scene.omfs_occlusion.upper
    if upper is None or segments(scene) or upper.hide_get():
        return
    ends = trait_ends(scene)
    if not ends:
        return
    region = bpy.context.region
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    dots = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('NONE')
    gpu.state.point_size_set(3.0)
    for a, b in ends.values():
        p, q = a.matrix_world.translation, b.matrix_world.translation
        batch = batch_for_shader(line, 'LINES', {"pos": [p, q]})
        line.uniform_float("viewportSize", (region.width, region.height))
        line.uniform_float("lineWidth", 1.5)
        line.uniform_float("color", (*LINE_COLOUR[:3], 0.6))
        batch.draw(line)
        points = cut_points(upper, trait_frame(upper, p, q))
        if len(points):
            batch = batch_for_shader(dots, 'POINTS', {"pos": points.astype(np.float32)})
            dots.uniform_float("color", LINE_COLOUR)
            batch.draw(dots)
    gpu.state.point_size_set(1.0)


def _draw_stroke(op):
    if op.start is None or op.end is None:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": [op.start, op.end]})
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", 2.0)
    shader.uniform_float("color", LINE_COLOUR)
    batch.draw(shader)


_draw_handle = None


# --- Operators -------------------------------------------------------------------------------

def view_area(context):
    return context.area if context.area and context.area.type == 'VIEW_3D' else next(
        (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)


def point_under_mouse(context, region, mouse, upper, depth):
    """On the scan under the mouse, else in the air at the depth of a point (the end of a line can go
    past the scan's border, so the cut goes through it)."""
    hit = landmarks.object_hit(context, region, mouse, upper)
    if hit is not None:
        return hit
    return view3d_utils.region_2d_to_location_3d(region, region.data, mouse, depth)


def can_edit_traits(context):
    props = context.scene.omfs_occlusion
    if props.upper is None:
        return "Choisissez d'abord l'arcade du haut"
    if segments(context.scene):
        return "L'arcade est segmentée : réunissez-la pour modifier les traits"
    return None


class OMFS_OT_draw_segment_line(bpy.types.Operator):
    """Cliquez-glissez un trait d'ostéotomie à travers l'arcade du haut, en vue occlusale (ex. dans
    l'embrasure canine / 1re prémolaire, du vestibule au palais). Le trait coupe sur toute la hauteur,
    seulement sur sa longueur : dépassez le bord du scan. Plusieurs traits de suite ; Entrée ou clic
    droit : terminer"""
    bl_idname = "omfs.draw_segment_line"
    bl_label = "Tracer un trait"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        error = can_edit_traits(context)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        self.area = view_area(context)
        if self.area is None:
            return {'CANCELLED'}
        self.upper = context.scene.omfs_occlusion.upper
        self.start = self.end = self.a = None
        self.count = 0
        self.handle = bpy.types.SpaceView3D.draw_handler_add(_draw_stroke, (self,), 'WINDOW', 'POST_PIXEL')
        self.area.header_text_set("Trait d'ostéotomie : cliquez-glissez à travers l'arcade, relâchez   |   "
                                  "Entrée / clic droit : terminer   |   Échap : annuler les traits de cette fois")
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context, result):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        self.area.tag_redraw()
        return {result}

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.finish(context, 'FINISHED' if self.count else 'CANCELLED')
        if event.value == 'PRESS' and event.type == 'ESC':
            for number in sorted(trait_ends(context.scene))[len(trait_ends(context.scene)) - self.count:]:
                remove_trait(context.scene, number)
            return self.finish(context, 'CANCELLED')
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
            if under_mouse is None:
                return {'PASS_THROUGH'}
            self.region, self.start = under_mouse
            self.end = self.start
            self.a = point_under_mouse(context, self.region, self.start, self.upper,
                                       occlusion.world_bbox_centre(self.upper))
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self.start is not None:
            self.end = (event.mouse_x - self.region.x, event.mouse_y - self.region.y)
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.start is not None:
            if (Vector(self.end) - Vector(self.start)).length < 10:
                self.report({'WARNING'}, "Tracez un trait plus long")
            else:
                b = point_under_mouse(context, self.region, self.end, self.upper, self.a)
                add_trait(context.scene, self.upper, self.a, b)
                self.count += 1
            self.start = self.end = self.a = None
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}   # orbit, pan and zoom keep working between the lines


class OMFS_OT_adjust_segment_lines(bpy.types.Operator):
    """Glisser les bouts des traits sur le scan pour placer chaque coupe exactement dans l'embrasure :
    cliquez près d'un bout et glissez ; la coupe (points orange sur le scan) suit en direct. Entrée ou
    clic droit : terminer ; Échap : annuler"""
    bl_idname = "omfs.adjust_segment_lines"
    bl_label = "Ajuster les traits"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        error = can_edit_traits(context)
        if error or not trait_ends(context.scene):
            self.report({'ERROR'}, error or "Tracez d'abord un trait")
            return {'CANCELLED'}
        self.area = view_area(context)
        if self.area is None:
            return {'CANCELLED'}
        self.upper = context.scene.omfs_occlusion.upper
        self.ends = [e for pair in trait_ends(context.scene).values() for e in pair]
        self.starts = [e.matrix_world.copy() for e in self.ends]
        self.dragging = None
        set_traits_hidden(context.scene, False)
        self.area.header_text_set("Ajuster les traits : cliquez près d'un bout et glissez sur le scan   |   "
                                  "Entrée / clic droit : terminer   |   Échap : annuler")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context, result):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        self.area.tag_redraw()
        return {result}

    def nearest_end(self, region, mouse):
        best, best_d = None, PICK_PIXELS
        for end in self.ends:
            p = view3d_utils.location_3d_to_region_2d(region, region.data, end.matrix_world.translation)
            if p is not None and (p - Vector(mouse)).length < best_d:
                best, best_d = end, (p - Vector(mouse)).length
        return best

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.finish(context, 'FINISHED')
        if event.value == 'PRESS' and event.type == 'ESC':
            for end, matrix in zip(self.ends, self.starts):
                end.matrix_world = matrix
            return self.finish(context, 'CANCELLED')
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
            if under_mouse is None:
                return {'PASS_THROUGH'}
            self.region, mouse = under_mouse
            self.dragging = self.nearest_end(self.region, mouse)
            if self.dragging is None:
                return {'PASS_THROUGH'}
            context.window.cursor_modal_set('HAND')
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self.dragging is not None:
            mouse = (event.mouse_x - self.region.x, event.mouse_y - self.region.y)
            location = point_under_mouse(context, self.region, mouse, self.upper,
                                         self.dragging.matrix_world.translation)
            self.dragging.matrix_world = Matrix.Translation(location)
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.dragging is not None:
            self.dragging = None
            context.window.cursor_modal_restore()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


class OMFS_OT_delete_segment_line(bpy.types.Operator):
    """Supprimer ce trait d'ostéotomie"""
    bl_idname = "omfs.delete_segment_line"
    bl_label = "Supprimer le trait"
    bl_options = {'REGISTER', 'UNDO'}

    number: bpy.props.IntProperty()

    def execute(self, context):
        error = can_edit_traits(context)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        remove_trait(context.scene, self.number)
        return {'FINISHED'}


class OMFS_OT_separate_segments(bpy.types.Operator):
    """Diviser l'arcade du haut le long des traits : chaque région fermée devient un segment, avec son
    propre bloc mobile. Le scan n'est pas modifié (copies de ses faces) ; il est caché. Les points des
    ressorts du haut passent au segment où ils sont"""
    bl_idname = "omfs.separate_segments"
    bl_label = "Séparer les segments"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.scene.omfs_occlusion.upper is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        error = separate(context.scene)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        parts = segments(context.scene)
        self.report({'INFO'}, f"{len(parts)} segments : " + ", ".join(p.name for p in parts))
        if context.scene.omfs_occlusion.show_map:
            occlusion_seating.write_overlap_map(context.scene)
        return {'FINISHED'}


class OMFS_OT_join_segments(bpy.types.Operator):
    """Réunir l'arcade du haut : les segments et leurs mouvements sont supprimés, le scan entier revient.
    Les traits restent, pour les ajuster et séparer de nouveau"""
    bl_idname = "omfs.join_segments"
    bl_label = "Réunir l'arcade"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if any(moved(context.scene, s) for s in segments(context.scene)):
            return context.window_manager.invoke_confirm(
                self, event, title="Réunir l'arcade ?", message="Les mouvements des segments seront perdus.",
                confirm_text="Réunir")
        return self.execute(context)

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        join(context.scene)
        if context.scene.omfs_occlusion.show_map:
            occlusion_seating.write_overlap_map(context.scene)
        return {'FINISHED'}


classes = (OMFS_OT_draw_segment_line, OMFS_OT_adjust_segment_lines, OMFS_OT_delete_segment_line,
           OMFS_OT_separate_segments, OMFS_OT_join_segments)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_traits, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
