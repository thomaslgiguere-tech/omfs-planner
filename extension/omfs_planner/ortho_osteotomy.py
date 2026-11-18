"""Orthognathic osteotomies: Le Fort I and mandibular (BSSO) cuts, separate from the genioplasty plane.

Like Dolphin 3D's osteotomy tool: the surgeon clicks a few points on the bone where the cut should pass, in the
view that looks through the bone the way the saw goes (Le Fort I: from the front, the cut runs back to the
pterygoid plates; BSSO: across the ramus / body). The line through the points is pushed along that view
direction into a cutting sheet ("nappe de coupe"): a ruled surface from the clicked line to a back line. Each
back point hangs from its front point and can be dragged on its own, so the cut is adjusted at a distance
(e.g. lower behind, at the pterygomaxillary junction). Front points can be dragged too; the sheet follows.

"Séparer" cuts a copy of the bone with every sheet of that bone (thickened to a saw kerf, Boolean difference) and
splits the result into its loose pieces: the distal segment (teeth-bearing: the Le Fort I maxilla, the
mandibular body) and the proximal parts (skull base, ramus + condyle each side). The bone's own mesh is never
touched (CBCT data): it is only hidden, and "Réunir" brings it back.

Generic on purpose: a cut is a line + a direction + a depth on any bone. Le Fort / BSSO are only the defaults
(depth, names, which piece is distal). Later, the interdental lines of a segmental Le Fort (occlusion_segments.py)
can become cuts of the same kind.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, params, reference_frame, scene_setup

MADE_BY = "Ortho osteotomy"
COLLECTION = "Ostéotomies orthognathiques"
KINDS = (
    ('LEFORT', "Le Fort I", "Trait au-dessus des apex, de part et d'autre, poussé vers l'arrière jusqu'aux ptérygoïdes"),
    ('BSSO_R', "BSSO droite", "Ostéotomie sagittale de la branche droite (trait médial, bord antérieur, vestibulaire)"),
    ('BSSO_L', "BSSO gauche", "Ostéotomie sagittale de la branche gauche"),
)
NAMES = {'LEFORT': "Trait Le Fort I", 'BSSO_R': "Trait BSSO droite", 'BSSO_L': "Trait BSSO gauche"}
SHEET_COLOUR = (1.0, 0.85, 0.1, 0.45)
FRONT_COLOUR, BACK_COLOUR = (1.0, 0.75, 0.0, 1.0), (0.2, 0.8, 1.0, 1.0)


# --- Axes ---------------------------------------------------------------------------------------

def clinical_axes(scene):
    """(left, posterior, up) unit vectors: the reference frame's if built, else the patient's (LPS) axes."""
    frame = reference_frame.find_frame(scene)
    if frame is None:
        return Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    m = frame.matrix_world.to_3x3().normalized()
    return m.col[0].normalized(), m.col[1].normalized(), m.col[2].normalized()


# --- Cut objects --------------------------------------------------------------------------------

def cuts(scene, source=None):
    found = [o for o in scene.objects if o.omfs.made_by == MADE_BY and o.type == 'MESH' and o.omfs.role == 'CUT']
    return [o for o in found if source is None or o.omfs_ortho.source == source]


def find_cut(scene, kind):
    return next((o for o in cuts(scene) if o.omfs_ortho.kind == kind), None)


def front_points(cut):
    """The clicked points (Empties), in order."""
    return sorted((o for o in cut.children if o.omfs.landmark.startswith("ORTHO_F")), key=lambda o: int(o.omfs.landmark[7:]))


def back_point(front):
    return next((o for o in front.children if o.omfs.landmark.startswith("ORTHO_B")), None)


def collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def new_point(scene, cut, name, tag, location, colour, size, parent=None):
    obj = bpy.data.objects.new(name, None)
    collection(scene).objects.link(obj)
    obj.empty_display_type, obj.empty_display_size = 'SPHERE', size
    obj.color = colour
    obj.show_in_front = True
    obj.omfs.made_by, obj.omfs.landmark, obj.omfs.role = MADE_BY, tag, 'MARKER'
    obj.parent = parent or cut
    obj.matrix_parent_inverse = Matrix()
    if parent is None:
        obj.location = location            # the cut object stays at the world origin
    else:
        obj.location = location            # local: the offset from its front point
    for lock in ("lock_rotation", "lock_scale"):
        setattr(obj, lock, (True, True, True))
    return obj


def make_cut(scene, kind, source, points, direction, depth=None):
    """A cut on a bone: clicked points (world), the direction the saw goes (world), and its depth (mm), measured
    from the clicked point furthest forward: the back points line up on one plane. Replaces a cut of the same
    kind. Returns the cut object (its mesh is the sheet)."""
    old = find_cut(scene, kind)
    if old is not None:
        remove_cut(old)
    depth = depth if depth is not None else (params.ORTHO_DEPTH_LEFORT if kind == 'LEFORT' else params.ORTHO_DEPTH_BSSO)
    mesh = bpy.data.meshes.new(NAMES[kind])
    cut = bpy.data.objects.new(NAMES[kind], mesh)
    collection(scene).objects.link(cut)
    cut.omfs.role, cut.omfs.made_by = 'CUT', MADE_BY
    cut.color = SHEET_COLOUR
    cut.show_transparent = True
    cut.hide_select = True
    props = cut.omfs_ortho
    props.kind, props.source, props.depth = kind, source, depth
    props.direction = Vector(direction).normalized()
    d = Vector(props.direction)
    nearest = min(d.dot(Vector(p)) for p in points)             # the clicked point furthest forward
    for i, p in enumerate(points, 1):
        front = new_point(scene, cut, f"{NAMES[kind]} {i}", f"ORTHO_F{i}", Vector(p), FRONT_COLOUR, 1.0)
        # every back point on one plane, depth behind the furthest-forward point: a straight back edge
        reach = depth - (d.dot(Vector(p)) - nearest)
        new_point(scene, cut, f"{NAMES[kind]} {i} arrière", f"ORTHO_B{i}", d * max(reach, 2.0), BACK_COLOUR, 1.2, front)
    bpy.context.view_layer.update()
    rebuild_sheet(cut)
    return cut


def remove_cut(cut):
    for obj in [c for f in front_points(cut) for c in f.children] + front_points(cut):
        bpy.data.objects.remove(obj)
    mesh = cut.data
    bpy.data.objects.remove(cut)
    bpy.data.meshes.remove(mesh)


def set_depth(cut, depth):
    """Move every back point along the cut's direction so the cut reaches depth, keeping each one's own adjustment."""
    props = cut.omfs_ortho
    d = Vector(props.direction)
    for front in front_points(cut):
        back = back_point(front)
        if back is not None:
            back.location = back.location + d * (depth - props.depth)
    props.depth = depth
    bpy.context.view_layer.update()      # the points' world positions, then the sheet
    rebuild_sheet(cut)


# --- The sheet ----------------------------------------------------------------------------------

def smooth_line(points, steps):
    """Catmull-Rom curve through the points (n, 3), steps samples per segment, ends included."""
    p = np.asarray(points, float)
    if len(p) < 3:
        t = np.linspace(0, 1, steps + 1)[:, None]
        return p[0] + t * (p[-1] - p[0])
    ext = np.vstack((2 * p[0] - p[1], p, 2 * p[-1] - p[-2]))
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, steps, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1])
    return np.array(out)


def extend(line, length):
    """The line with its ends pushed out along their tangents."""
    start = line[0] - (line[1] - line[0]) / max(np.linalg.norm(line[1] - line[0]), 1e-9) * length
    end = line[-1] + (line[-1] - line[-2]) / max(np.linalg.norm(line[-1] - line[-2]), 1e-9) * length
    return np.vstack((start, line, end))


def sheet_rows(cut):
    """(front row, back row) of the sheet (k, 3) world: the smooth clicked line pulled MARGIN back out of the bone
    (against the direction), and the smooth back line; both extended past their ends. None with < 2 points."""
    fronts = front_points(cut)
    if len(fronts) < 2:
        return None
    d = np.array(cut.omfs_ortho.direction)
    front = np.array([f.matrix_world.translation for f in fronts])
    back = np.array([back_point(f).matrix_world.translation if back_point(f) else f.matrix_world.translation + d * cut.omfs_ortho.depth
                     for f in fronts])
    steps = params.ORTHO_SMOOTH_STEPS
    a = extend(smooth_line(front - d * params.ORTHO_MARGIN, steps), params.ORTHO_END_EXTENSION)
    b = extend(smooth_line(back, steps), params.ORTHO_END_EXTENSION)
    return a, b


def rebuild_sheet(cut):
    """The cut object's mesh: the open sheet between the front and back rows (drawn translucent)."""
    rows = sheet_rows(cut)
    mesh = cut.data
    mesh.clear_geometry()
    if rows is None:
        return
    a, b = rows
    k = len(a)
    mesh.from_pydata(np.vstack((a, b)).tolist(), [], [(i, i + 1, k + i + 1, k + i) for i in range(k - 1)])
    mesh.update()


def cutter_mesh(cut):
    """The sheet thickened to the saw kerf: a closed slab, for the Boolean."""
    rows = sheet_rows(cut)
    a, b = rows
    k = len(a)
    bm = bmesh.new()
    verts = [bm.verts.new(v) for v in np.vstack((a, b))]
    for i in range(k - 1):
        bm.faces.new((verts[i], verts[i + 1], verts[k + i + 1], verts[k + i]))
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * params.ORTHO_KERF / 2      # solidify grows backwards: the slab ends centred on the sheet
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=params.ORTHO_KERF)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new("cutter")
    bm.to_mesh(mesh)
    bm.free()
    return mesh


_dirty = set()


@bpy.app.handlers.persistent
def _on_depsgraph(scene, depsgraph):
    """A point was moved: redraw its sheet just after (never inside the handler: that would loop)."""
    for update in depsgraph.updates:
        obj = update.id
        if isinstance(obj, bpy.types.Object) and update.is_updated_transform and obj.omfs.made_by == MADE_BY and obj.type == 'EMPTY':
            cut = obj.parent if obj.parent is not None and obj.parent.type == 'MESH' else (obj.parent.parent if obj.parent else None)
            if cut is not None:
                _dirty.add(cut.name)
    if _dirty and not bpy.app.timers.is_registered(_redraw):
        bpy.app.timers.register(_redraw, first_interval=0.05)


def _redraw():
    for name in list(_dirty):
        cut = bpy.data.objects.get(name)
        if cut is not None:
            rebuild_sheet(cut)
    _dirty.clear()
    return None


# --- Separation ---------------------------------------------------------------------------------

def mesh_volume(mesh):
    """Signed volume (mm3) and centroid of a closed mesh, from its triangles."""
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    a, b, c = (co[tris.reshape(-1, 3)[:, k]] for k in range(3))
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    total = v.sum()
    centre = ((a + b + c) / 4 * v[:, None]).sum(axis=0) / total if abs(total) > 1e-9 else co.mean(axis=0)
    return float(total), centre


def boolean_cut(context, source, cutters):
    """A new mesh (world coordinates): the source minus the cutter slabs. Fast solver for closed meshes (bones from
    the Segmentation workspace), exact solver otherwise."""
    scene = context.scene
    temp = bpy.data.objects.new("ortho temp", source.data)
    scene.collection.objects.link(temp)
    temp.matrix_world = source.matrix_world
    tool = bpy.data.objects.new("ortho cutter", cutters)
    scene.collection.objects.link(tool)
    mod = temp.modifiers.new("Cut", 'BOOLEAN')
    mod.operation, mod.object = 'DIFFERENCE', tool
    result = None
    for solver in ('MANIFOLD', 'EXACT'):
        try:
            mod.solver = solver
        except TypeError:
            continue
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()
        mesh = bpy.data.meshes.new_from_object(temp.evaluated_get(depsgraph))
        if len(mesh.polygons):
            result = mesh
            break
        bpy.data.meshes.remove(mesh)
    bpy.data.objects.remove(temp)
    bpy.data.objects.remove(tool)
    if result is not None:
        result.transform(source.matrix_world)
    return result


def loose_pieces(context, mesh, name):
    """Split a mesh into its loose parts (Edit Mode > Separate > By Loose Parts, in C): new objects, world coords."""
    scene = context.scene
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    with context.temp_override(active_object=obj, object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
        for other in context.view_layer.objects.selected:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
    found = [o for o in scene.objects if o.type == 'MESH' and (o == obj or o.name.startswith(name + "."))]
    for o in found:
        o.select_set(False)
    return found


def merge(context, target, others):
    """Join chips into a piece (Object > Join)."""
    if not others:
        return
    with context.temp_override(active_object=target, selected_editable_objects=[target] + others, selected_objects=[target] + others):
        bpy.ops.object.join()


def cutter_slabs(found):
    """Every cut of a bone as one closed mesh (the slabs side by side)."""
    slabs = bmesh.new()
    for cut in found:
        part = cutter_mesh(cut)
        slabs.from_mesh(part)
        bpy.data.meshes.remove(part)
    mesh = bpy.data.meshes.new("ortho cutters")
    slabs.to_mesh(mesh)
    slabs.free()
    return mesh


def touches(obj, tree, box):
    """Some vertex of the piece lies on a cut face (within the kerf): the cut went through it. Only the vertices
    inside the cutters' box are tested (the cut faces are a few hundred among a million vertices)."""
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co = co[np.all((co >= box[0]) & (co <= box[1]), axis=1)]
    reach = params.ORTHO_KERF + 0.3
    return any(tree.find_nearest(Vector(p), reach)[0] is not None for p in co)


def pieces_of(scene, source):
    return [o for o in scene.objects if o.omfs.made_by == f"{MADE_BY}: {source.name}"]


def rejoin(scene, source):
    for obj in pieces_of(scene, source):
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    source.hide_set(False)


def separate(context, source):
    """Cut the bone with all its cuts and name the pieces. Returns (pieces, warning or None)."""
    scene = context.scene
    found = cuts(scene, source)
    if not found:
        return [], "Tracez d'abord un trait sur cet os"
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    rejoin(scene, source)
    cutters = cutter_slabs(found)
    mesh = boolean_cut(context, source, cutters)
    bpy.data.meshes.remove(cutters)
    if mesh is None:
        return [], "La coupe a échoué (maillage de l'os non fermé ?)"
    base = f"{source.name} coupe"
    objects = loose_pieces(context, mesh, base)
    slab = cutter_slabs(found)
    tree = BVHTree.FromPolygons([v.co for v in slab.vertices], [p.vertices for p in slab.polygons])
    corners = np.array([v.co for v in slab.vertices])
    box = (corners.min(axis=0) - 1.0, corners.max(axis=0) + 1.0)
    bpy.data.meshes.remove(slab)
    sizes = {o.name: mesh_volume(o.data) for o in objects}
    touched = {o.name for o in objects if touches(o, tree, box)}
    # pieces the cut went through: what was cut; the others (bones already apart, e.g. vertebrae) stay fixed
    cut_big = [o for o in objects if o.name in touched and sizes[o.name][0] >= params.ORTHO_MIN_PIECE]
    if not cut_big:
        cut_big = [max(objects, key=lambda o: sizes[o.name][0])]
    kinds = {c.omfs_ortho.kind for c in found}
    left, posterior, up = (np.array(v) for v in clinical_axes(scene))
    centre = sum(sizes[o.name][1] * sizes[o.name][0] for o in cut_big) / sum(sizes[o.name][0] for o in cut_big)
    if 'LEFORT' in kinds:
        distal = min(cut_big, key=lambda o: sizes[o.name][1] @ up)                      # the maxilla, below the cut
    else:
        distal = min(cut_big, key=lambda o: abs((sizes[o.name][1] - centre) @ left))    # the body, across the midline
    keep = cut_big
    for other in [o for o in objects if o not in keep]:                     # chips and other bones join a fixed piece
        c = sizes[other.name][1]
        fixed = [o for o in keep if o != distal] or [distal]
        target = distal if other.name in touched and sizes[other.name][0] < params.ORTHO_MIN_PIECE and \
            np.linalg.norm(sizes[distal.name][1] - c) < min(np.linalg.norm(sizes[o.name][1] - c) for o in fixed) else \
            min(fixed, key=lambda o: np.linalg.norm(sizes[o.name][1] - c))
        merge(context, target, [other])
    big = keep
    centres = {o: mesh_volume(o.data)[1] for o in big}       # by object: the names change below
    expected = 2 if 'LEFORT' in kinds else 1 + len(kinds & {'BSSO_R', 'BSSO_L'})
    for o in big:
        o.omfs.made_by = f"{MADE_BY}: {source.name}"
        if o == distal:
            o.name = f"{source.name} distal (Le Fort I)" if 'LEFORT' in kinds else f"{source.name} distale"
            o.omfs.role = 'SEGMENT'
            locking.set_locked(o, False)
        else:
            side = "gauche" if (centres[o] - centres[distal]) @ left > 0 else "droit"
            o.name = f"{source.name} proximal" if 'LEFORT' in kinds else f"{source.name} proximal {side}"
            o.omfs.role = 'BONE'
            locking.set_locked(o, True)
        o.data.name = o.name
        o.color = (0.55, 0.85, 0.55, 1.0) if o == distal else (0.90, 0.84, 0.70, 1.0)
        if o.name not in collection(scene).objects:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            collection(scene).objects.link(o)
    source.hide_set(True)
    warning = None
    if len(big) < expected:
        warning = (f"Coupe incomplète: {len(big)} morceau(x) au lieu de {expected}. Prolongez le trait au-delà de l'os "
                   "(points d'extrémité) ou augmentez la profondeur")
    return big, warning


# --- Drawing ------------------------------------------------------------------------------------

class OMFS_OT_ortho_draw(bpy.types.Operator):
    """Cliquez des points sur l'os où l'ostéotomie doit passer, dans la vue qui regarde dans le sens de la scie
    (Le Fort I: de face; BSSO: de côté). Entrée ou clic droit: terminer (le trait est poussé dans l'os selon la
    direction de la vue). Retour arrière: retirer le dernier point. Échap: annuler"""
    bl_idname = "omfs.ortho_draw"
    bl_label = "Tracer le trait"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(items=KINDS)

    def invoke(self, context, event):
        source = bone_for(context.scene, self.kind)
        if source is None:
            self.report({'ERROR'}, "Choisissez d'abord l'os à couper")
            return {'CANCELLED'}
        self.source = source
        self.points = []
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        self.markers = []
        self.prompt()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self):
        self.area.header_text_set(f"{NAMES[self.kind]}: cliquez sur {self.source.name} ({len(self.points)} point(s))   |   "
                                  "Entrée / clic droit: terminer   |   Retour arrière: dernier point   |   Échap: annuler")

    def clear(self, context):
        for marker in self.markers:
            bpy.data.objects.remove(marker)
        self.markers = []
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self.clear(context)
            return {'CANCELLED'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS' and self.points:
            self.points.pop()
            bpy.data.objects.remove(self.markers.pop())
            self.prompt()
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if len(self.points) < 2:
                self.report({'WARNING'}, "Au moins 2 points")
                return {'RUNNING_MODAL'}
            view = next(r.data for r in self.area.regions if r.type == 'WINDOW')
            direction = view.view_rotation @ Vector((0.0, 0.0, -1.0))
            points = list(self.points)
            self.clear(context)
            make_cut(context.scene, self.kind, self.source, points, direction)
            return {'FINISHED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under = landmarks.viewport_region_under_mouse(self.area, event)
            if under is None:
                return {'PASS_THROUGH'}
            hit = landmarks.object_hit(context, under[0], under[1], self.source)
            if hit is None:
                return {'RUNNING_MODAL'}
            self.points.append(hit.copy())
            marker = bpy.data.objects.new("point", None)
            context.scene.collection.objects.link(marker)
            marker.empty_display_type, marker.empty_display_size, marker.location = 'SPHERE', 1.0, hit
            marker.show_in_front = True
            self.markers.append(marker)
            self.prompt()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


def bone_for(scene, kind):
    """The bone a cut of this kind goes through: the one chosen, else "Maxillaire" / "Mandibule" (the names the
    Segmentation workspace gives)."""
    return bone(scene, 'MAXILLA' if kind == 'LEFORT' else 'MANDIBLE')


def bone(scene, which):
    props = scene.omfs_ortho_scene
    chosen = props.maxilla if which == 'MAXILLA' else props.mandible
    if chosen is not None and chosen.name in scene.objects:
        return chosen
    return _default(scene, "Maxillaire" if which == 'MAXILLA' else "Mandibule")


class OMFS_OT_ortho_remove(bpy.types.Operator):
    """Effacer ce trait (ses points et sa nappe)"""
    bl_idname = "omfs.ortho_remove"
    bl_label = "Effacer le trait"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(items=KINDS)

    def execute(self, context):
        cut = find_cut(context.scene, self.kind)
        if cut is not None:
            remove_cut(cut)
        return {'FINISHED'}


class OMFS_OT_ortho_separate(bpy.types.Operator):
    """Couper une copie de l'os avec ses traits et séparer les segments (distal, proximal). L'os d'origine n'est
    jamais modifié: il est seulement caché"""
    bl_idname = "omfs.ortho_separate"
    bl_label = "Séparer les segments"
    bl_options = {'REGISTER', 'UNDO'}

    which: bpy.props.EnumProperty(items=(('MAXILLA', "Maxillaire", ""), ('MANDIBLE', "Mandibule", "")))

    def execute(self, context):
        source = bone(context.scene, self.which)
        if source is None:
            self.report({'ERROR'}, "Choisissez l'os")
            return {'CANCELLED'}
        pieces, warning = separate(context, source)
        if not pieces:
            self.report({'ERROR'}, warning)
            return {'CANCELLED'}
        if warning:
            self.report({'WARNING'}, warning)
        else:
            self.report({'INFO'}, f"{len(pieces)} segments: " + ", ".join(p.name for p in pieces))
        return {'FINISHED'}


class OMFS_OT_ortho_rejoin(bpy.types.Operator):
    """Effacer les segments et revenir à l'os entier (les traits restent, pour ajuster et séparer de nouveau)"""
    bl_idname = "omfs.ortho_rejoin"
    bl_label = "Réunir"
    bl_options = {'REGISTER', 'UNDO'}

    which: bpy.props.EnumProperty(items=(('MAXILLA', "Maxillaire", ""), ('MANDIBLE', "Mandibule", "")))

    def execute(self, context):
        source = bone(context.scene, self.which)
        if source is not None:
            rejoin(context.scene, source)
        return {'FINISHED'}


# --- Properties ---------------------------------------------------------------------------------

def _is_bone(_self, obj):
    return obj.type == 'MESH' and obj.omfs.role == 'BONE' and not obj.omfs.made_by.startswith(MADE_BY)


class OMFS_OrthoCut(bpy.types.PropertyGroup):
    kind: bpy.props.StringProperty()
    source: bpy.props.PointerProperty(type=bpy.types.Object)
    direction: bpy.props.FloatVectorProperty(size=3)
    depth: bpy.props.FloatProperty()
    depth_ui: bpy.props.FloatProperty(
        name="Profondeur", min=2.0, max=120.0, step=100, precision=0, unit='LENGTH',
        get=lambda self: self.depth, set=lambda self, value: set_depth(self.id_data, value),
        description="Jusqu'où la coupe s'enfonce derrière le trait (chaque point arrière garde son propre réglage)")


def _default(scene, name):
    obj = scene.objects.get(name)
    return obj if obj is not None and obj.type == 'MESH' else None


class OMFS_OrthoScene(bpy.types.PropertyGroup):
    maxilla: bpy.props.PointerProperty(name="Maxillaire", type=bpy.types.Object, poll=_is_bone,
                                       description="L'os coupé par le Le Fort I (ex. « Maxillaire » de la segmentation)")
    mandible: bpy.props.PointerProperty(name="Mandibule", type=bpy.types.Object, poll=_is_bone,
                                        description="L'os coupé par les BSSO")


classes = (OMFS_OrthoCut, OMFS_OrthoScene, OMFS_OT_ortho_draw, OMFS_OT_ortho_remove, OMFS_OT_ortho_separate,
           OMFS_OT_ortho_rejoin)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_ortho = bpy.props.PointerProperty(type=OMFS_OrthoCut)
    bpy.types.Scene.omfs_ortho_scene = bpy.props.PointerProperty(type=OMFS_OrthoScene)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    del bpy.types.Scene.omfs_ortho_scene
    del bpy.types.Object.omfs_ortho
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
