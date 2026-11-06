"""Plane cut and Erase & Fill (guide workshop): trim a printed part and keep it closed.

Plane cut, as in Meshmixer: drag a line across the part on the screen; the cutting plane goes
through that line, edge-on to you, and the smaller side is removed. The plane stays a live
Boolean (a half-space box parented to a visible plane): move and turn it with the gizmo, flip the
side, then apply. A Boolean Difference with a closed box keeps a closed part closed: the cut face
is filled by the Boolean itself.

Erase & Fill: the faces selected in Edit Mode are deleted and each hole left is filled with a
smooth patch. Only the new vertices of the patch move (a membrane stretched on the hole's rim): the
rest of the mesh, rim included, is not touched.
"""

import bmesh
import bpy
import gpu
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from . import guide_workshop, landmarks, params, pins

TRIM_MODIFIER = guide_workshop.TRIM_MODIFIER


# --- Plane cut --------------------------------------------------------------------------------

def plane_name(obj):
    return f"{TRIM_MODIFIER} ({obj.name})"


def is_closed(mesh):
    """Every edge shared by exactly two faces (no open border)."""
    uses = {}
    for p in mesh.polygons:
        for key in p.edge_keys:
            uses[key] = uses.get(key, 0) + 1
    return bool(uses) and all(n == 2 for n in uses.values())


def plane_mesh(name, size):
    """A square grid, drawn as wire: the handle the surgeon moves and turns."""
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=8, y_segments=8, size=size / 2)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def box_mesh(name, size):
    """Half-space box: its face z = 0 lies on the plane, it fills the +Z side (the side removed)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.translate(bm, vec=(0, 0, size / 2), verts=bm.verts)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def find_cut(obj):
    """(plane, modifier) of the plane cut waiting on obj, or (None, None)."""
    mod = obj.modifiers.get(TRIM_MODIFIER)
    plane = bpy.data.objects.get(plane_name(obj))
    if mod is None or plane is None:
        return None, None
    return plane, mod


def surface_centre(obj):
    """World centre of obj's evaluated surface (triangles weighted by their area): it lies in the bigger
    part, whatever the mesh density."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = obj.evaluated_get(depsgraph).to_mesh()
    mesh.calc_loop_triangles()
    total, area = Vector(), 0.0
    for t in mesh.loop_triangles:
        total += t.center * t.area
        area += t.area
    obj.evaluated_get(depsgraph).to_mesh_clear()
    return obj.matrix_world @ (total / area if area else Vector())


def start_cut(context, obj, origin, along, normal):
    """Plane through origin containing `along`, cutting off the +normal side (flipped to the smaller
    side of obj). Live Boolean on obj, placed before its other Booleans. Returns the plane."""
    cancel_cut(context, obj)
    normal = normal.normalized()
    if (surface_centre(obj) - origin).dot(normal) > 0:   # the smaller side goes
        normal = -normal
    x = (along - along.dot(normal) * normal).normalized()
    y = normal.cross(x)
    matrix = Matrix((x, y, normal)).transposed().to_4x4()
    matrix.translation = origin
    coll = guide_workshop.workshop_collection(context.scene)
    name = plane_name(obj)
    plane = bpy.data.objects.new(name, plane_mesh(name, params.TRIM_PLANE_SIZE))
    box = bpy.data.objects.new(f"{name}: côté retiré", box_mesh(name, params.TRIM_BOX_SIZE))
    for helper in (plane, box):
        coll.objects.link(helper)
        helper.omfs.role = 'MARKER'           # landmark and surface clicks go through it
        helper.display_type = 'WIRE'
    plane.show_in_front = True
    plane.lock_scale = (True, True, True)     # move and turn only: the box stays a half-space
    plane.matrix_world = matrix
    box.parent = plane                        # follows the plane
    box.hide_set(True)                        # still used by the Boolean while hidden
    mod = obj.modifiers.new(TRIM_MODIFIER, 'BOOLEAN')
    mod.operation, mod.object = 'DIFFERENCE', box
    # Manifold needs a closed part; a shell that a Solidify modifier closes is cut after it.
    closed = is_closed(obj.data) or any(m.type == 'SOLIDIFY' for m in obj.modifiers)
    mod.solver = guide_workshop.SOLVER if closed else 'EXACT'
    first_boolean = next((i for i, m in enumerate(obj.modifiers) if m.type == 'BOOLEAN'), len(obj.modifiers) - 1)
    obj.modifiers.move(len(obj.modifiers) - 1, first_boolean)
    return plane


def flip_cut(obj):
    plane, _mod = find_cut(obj)
    if plane is None:
        return False
    plane.matrix_world = plane.matrix_world @ Matrix.Rotation(3.141592653589793, 4, 'X')
    return True


def remove_helpers(obj):
    plane = bpy.data.objects.get(plane_name(obj))
    if plane is None:
        return
    for helper in list(plane.children) + [plane]:
        mesh = helper.data
        bpy.data.objects.remove(helper)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def cancel_cut(context, obj):
    mod = obj.modifiers.get(TRIM_MODIFIER)
    if mod is not None:
        obj.modifiers.remove(mod)
    remove_helpers(obj)


def apply_cut(context, obj):
    """The cut written into the mesh: the modifiers above it (e.g. the thickness) and the cut itself
    are applied, the Booleans below it stay live. Returns an error message or None."""
    plane, mod = find_cut(obj)
    if plane is None:
        return "Pas de coupe plane en attente sur cet objet"
    guide_workshop.make_active(context, obj)
    for name in [m.name for m in obj.modifiers][:obj.modifiers.find(TRIM_MODIFIER) + 1]:
        bpy.ops.object.modifier_apply(modifier=name)
    remove_helpers(obj)
    return None


def cut_target(context):
    """The part the plane cut works on: the one with a cut waiting if its plane is active, else the
    object the workshop tools act on."""
    active = context.active_object
    if active is not None and active.name.startswith(TRIM_MODIFIER + " ("):
        target = bpy.data.objects.get(active.name[len(TRIM_MODIFIER) + 2:-1])
        if target is not None:
            return target, None
    return guide_workshop.editable_active(context)


# --- Erase & Fill -----------------------------------------------------------------------------

def erase_and_fill(bm):
    """Delete the selected faces of bm and fill each hole left with a smooth patch, its rim kept as
    it is. Returns (holes filled, rims left open). Rim and old elements are marked in integer layers
    (bmesh operators clear the tag flags, and Python wrappers are not stable set members)."""
    old_v, old_e, old_f = (layers.int.new("omfs_old") for layers in (bm.verts.layers, bm.edges.layers,
                                                                      bm.faces.layers))
    selected = [f for f in bm.faces if f.select]
    for f in selected:
        for v in f.verts:
            v[old_v] = any(not g.select for g in v.link_faces)   # rim: stays exactly where it is
    bmesh.ops.delete(bm, geom=selected, context='FACES')
    border = [e for e in bm.edges if e.is_boundary and e.verts[0][old_v] and e.verts[1][old_v]]
    for e in border:
        e[old_e] = 1                                             # rim edges: never split
    for f in bm.faces:
        f[old_f] = 1                                             # old faces: never touched
    filled = bmesh.ops.edgeloop_fill(bm, edges=border)["faces"] if border else []
    if filled:
        bmesh.ops.triangulate(bm, faces=filled, quad_method='BEAUTY', ngon_method='BEAUTY')
    target = sum(e.calc_length() for e in border) / max(len(border), 1)

    def patch():
        return [f for f in bm.faces if not f[old_f]]

    for _ in range(params.ERASE_FILL_REFINE_STEPS):          # as fine as the rim: smooth curvature
        long = [e for e in bm.edges if not e[old_e] and any(not f[old_f] for f in e.link_faces)
                and e.calc_length() > params.ERASE_FILL_EDGE_FACTOR * target]
        if not long:
            break
        bmesh.ops.subdivide_edges(bm, edges=long, cuts=1)
        bmesh.ops.triangulate(bm, faces=[f for f in patch() if len(f.verts) > 3], quad_method='BEAUTY',
                              ngon_method='BEAUTY')
    inner = [e for e in bm.edges if not e[old_e] and e.link_faces and all(not f[old_f] for f in e.link_faces)]
    bmesh.ops.beautify_fill(bm, faces=patch(), edges=inner)
    inside = [v for v in bm.verts if v.link_faces and all(not f[old_f] for f in v.link_faces)]   # new ones
    for _ in range(params.ERASE_FILL_SMOOTH_ITERATIONS):     # membrane: each point to its neighbours' mean
        new = [sum((e.other_vert(v).co for e in v.link_edges), Vector()) / len(v.link_edges) for v in inside]
        for v, co in zip(inside, new):
            v.co = co
    bm.normal_update()
    left_open = sum(1 for e in bm.edges if e.is_boundary and e.verts[0][old_v] and e.verts[1][old_v])
    for layers in (bm.verts.layers, bm.edges.layers, bm.faces.layers):
        layers.int.remove(layers.int["omfs_old"])
    return len(filled), left_open


# --- Operators --------------------------------------------------------------------------------

def draw_line(op):
    if op.start is None or op.end is None:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": [op.start, op.end]})
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", 2.0)
    shader.uniform_float("color", (1.0, 0.55, 0.1, 1.0))
    batch.draw(shader)


class OMFS_OT_plane_cut(bpy.types.Operator):
    """Drag a line across the part: the cutting plane goes through it, edge-on to you"""
    bl_idname = "omfs.plane_cut"
    bl_label = "Coupe plane (tracer une ligne)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS['PLANE_CUT'])

    def invoke(self, context, event):
        self.target, error = cut_target(context)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        pins.take_over(self)
        self.start = self.end = None
        self.region = None
        self.handle = bpy.types.SpaceView3D.draw_handler_add(draw_line, (self,), 'WINDOW', 'POST_PIXEL')
        self.area.header_text_set(f"Coupe plane de {self.target.name}: cliquez-glissez une ligne à travers "
                                  "la pièce, relâchez   |   Échap / clic droit: annuler")
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
        if pins.superseded(self):
            self.finish(context, 'CANCELLED')
            return {'PASS_THROUGH'}
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.finish(context, 'CANCELLED')
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
            if under_mouse is None:
                return {'PASS_THROUGH'}
            self.region, self.start = under_mouse
            self.end = self.start
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self.start is not None:
            self.end = (event.mouse_x - self.region.x, event.mouse_y - self.region.y)
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.start is not None:
            if (Vector(self.end) - Vector(self.start)).length < 10:
                self.start = self.end = None
                self.report({'WARNING'}, "Tracez une ligne plus longue")
                return {'RUNNING_MODAL'}
            self.place(context)
            return self.finish(context, 'FINISHED')
        return {'PASS_THROUGH'}   # orbit, pan and zoom keep working before the line

    def place(self, context):
        rv3d = self.region.data
        corners = [self.target.matrix_world @ Vector(c) for c in self.target.bound_box]
        depth = sum(corners, Vector()) / 8
        a = view3d_utils.region_2d_to_location_3d(self.region, rv3d, self.start, depth)
        b = view3d_utils.region_2d_to_location_3d(self.region, rv3d, self.end, depth)
        middle = ((Vector(self.start) + Vector(self.end)) / 2)
        sight = view3d_utils.region_2d_to_vector_3d(self.region, rv3d, middle)
        origin = (a + b) / 2
        plane = start_cut(context, self.target, origin, b - a, (b - a).cross(sight))
        guide_workshop.make_active(context, plane)
        with guide_workshop.viewport(context):
            bpy.ops.wm.tool_set_by_id(name="builtin.transform")   # arrows + rings on the plane


class TrimOperator:
    bl_options = {'REGISTER', 'UNDO'}
    TOOL = ''

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS[cls.TOOL])

    def execute(self, context):
        target, error = cut_target(context)
        error = error or self.run(context, target)
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        return {'FINISHED'}


class OMFS_OT_plane_cut_flip(TrimOperator, bpy.types.Operator):
    """Remove the other side of the plane"""
    bl_idname = "omfs.plane_cut_flip"
    bl_label = "Inverser le côté coupé"
    TOOL = 'PLANE_CUT_FLIP'

    def run(self, context, target):
        return None if flip_cut(target) else "Tracez d'abord une coupe plane"


class OMFS_OT_plane_cut_apply(TrimOperator, bpy.types.Operator):
    """Write the cut into the mesh"""
    bl_idname = "omfs.plane_cut_apply"
    bl_label = "Appliquer la coupe"
    TOOL = 'PLANE_CUT_APPLY'

    def run(self, context, target):
        return apply_cut(context, target)


class OMFS_OT_plane_cut_cancel(TrimOperator, bpy.types.Operator):
    """Remove the plane, the part is back as it was"""
    bl_idname = "omfs.plane_cut_cancel"
    bl_label = "Annuler la coupe"
    TOOL = 'PLANE_CUT_CANCEL'

    def run(self, context, target):
        if find_cut(target)[0] is None:
            return "Pas de coupe plane en attente"
        cancel_cut(context, target)
        guide_workshop.make_active(context, target)
        return None


class OMFS_OT_erase_fill(bpy.types.Operator):
    """Delete the selected faces and fill the holes with a smooth patch (the rest is not touched)"""
    bl_idname = "omfs.erase_fill"
    bl_label = "Effacer et combler"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS['ERASE_FILL'])

    def execute(self, context):
        error = guide_workshop.need_mode(context, 'EDIT_MESH')
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        mesh = context.edit_object.data
        bm = bmesh.from_edit_mesh(mesh)
        count = sum(1 for f in bm.faces if f.select)
        if count == 0 or count == len(bm.faces):
            self.report({'WARNING'}, "Sélectionnez la partie à effacer (pas tout l'objet)")
            return {'CANCELLED'}
        holes, left_open = erase_and_fill(bm)
        bmesh.update_edit_mesh(mesh)
        if left_open:
            self.report({'WARNING'}, f"{holes} trou(s) comblé(s); la sélection touchait un bord ouvert: "
                                     "ce bord est seulement retiré")
        else:
            self.report({'INFO'}, f"{holes} trou(s) comblé(s), la surface reste fermée")
        return {'FINISHED'}


classes = (OMFS_OT_plane_cut, OMFS_OT_plane_cut_flip, OMFS_OT_plane_cut_apply, OMFS_OT_plane_cut_cancel,
           OMFS_OT_erase_fill)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
