"""Text engraved into (or embossed on) a printed part: splint, cap, later the bone guides.

A label is placed by two clicks on the part's surface: where the text starts and where it ends.
That gives its place, its reading direction and its length in one gesture; the letters fit between
the clicks (up to a maximum height) and read from the outside, left to right from the first click.

The label is a small closed mesh (the "cutter"): the letters, laid on the surface along the line of
sight, pushed into the part (engraved) or out of it (embossed). A Boolean modifier on the part
subtracts or adds it, so the part's mesh is never modified until the modifiers are applied at export,
and the text, size or depth can change at any time. The cutter is drawn as a wireframe.
"""

import bmesh
import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, params

TEXT_TAG = "text label"         # made_by of a cutter
MODES = (
    ('ENGRAVE', "Gravé", "Lettres creusées dans la pièce"),
    ('EMBOSS', "En relief", "Lettres en relief sur la pièce"),
)
REACH = 5.0                     # mm: rays onto the surface start this far outside it

_surfaces = {}                  # part -> BVH tree of its own mesh (without the text Booleans)


def is_label(obj):
    return isinstance(obj, bpy.types.Object) and obj.type == 'MESH' and obj.omfs.made_by == TEXT_TAG


def labels_on(part):
    return [m.object for m in part.modifiers if m.type == 'BOOLEAN' and is_label(m.object)]


def surface_of(part):
    """BVH tree of the part's own mesh (local coordinates): the surface before any text is cut,
    so a label never follows its own groove."""
    mesh = part.data
    first = tuple(mesh.vertices[0].co) if len(mesh.vertices) else ()
    key = (part.name, len(mesh.vertices), len(mesh.polygons), first)
    if _surfaces.get(part.name, (None,))[0] != key:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        _surfaces[part.name] = (key, BVHTree.FromBMesh(bm))
        bm.free()
    return _surfaces[part.name][1]


def glyphs(text):
    """The letters as a flat mesh, 1 unit font size, centred on the origin: (xy vertices, faces)."""
    curve = bpy.data.curves.new("omfs text", 'FONT')
    curve.body = text
    curve.align_x, curve.align_y = 'CENTER', 'CENTER'
    temp = bpy.data.objects.new("omfs text", curve)
    bpy.context.scene.collection.objects.link(temp)
    mesh = bpy.data.meshes.new_from_object(temp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(temp)
    bpy.data.curves.remove(curve)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bpy.data.meshes.remove(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    xy = np.array([v.co.xy[:] for v in bm.verts]).reshape(-1, 2)
    faces = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    if len(xy):
        xy -= (xy.min(axis=0) + xy.max(axis=0)) / 2
    return xy, faces


def frame(start, end, normal):
    """Text axes: x from the first click to the second (along the surface), z out of the surface."""
    z = Vector(normal).normalized()
    x = Vector(end) - Vector(start)
    x = (x - z * x.dot(z)).normalized()
    return x, z.cross(x), z


def build(label):
    """(Re)make the cutter mesh from the label's settings. Returns a warning, or None."""
    t = label.omfs_text
    part = t.part
    if part is None:
        return "Pièce introuvable"
    xy, faces = glyphs(t.text or " ")
    mesh = label.data
    mesh.clear_geometry()
    if not faces:
        return "Texte vide"
    x, y, z = frame(t.start, t.end, t.normal)
    length = (Vector(t.end) - Vector(t.start)).length
    size = np.ptp(xy, axis=0)
    scale = min(length / size[0], t.height / size[1])
    t.letter_height = size[1] * scale
    centre = (Vector(t.start) + Vector(t.end)) / 2

    # Lay each letter vertex on the surface along -z (the part's own mesh, without the text).
    tree = surface_of(part)
    to_local = part.matrix_world.inverted()
    down = (to_local.to_3x3() @ -z).normalized()
    surface, missed = [], 0
    for u, v in xy * scale:
        p = centre + x * u + y * v
        hit = tree.ray_cast(to_local @ (p + z * REACH), down, 2 * REACH)[0]
        if hit is None:
            missed += 1
            surface.append(p)
        else:
            surface.append(part.matrix_world @ hit)
    surface = np.array([s[:] for s in surface])
    z_ = np.array(z[:])
    over = params.TEXT_OVERLAP
    if t.mode == 'ENGRAVE':
        outer, inner = surface + z_ * over, surface - z_ * t.depth
    else:
        outer, inner = surface + z_ * t.depth, surface - z_ * over

    # Closed cutter: outer letters, inner letters (reversed), walls along the letters' outlines.
    n = len(xy)
    edges = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    outline = [(a, b) for (a, b) in edges if (b, a) not in edges]
    polys = [list(f) for f in faces] + [[i + n for i in reversed(f)] for f in faces]
    polys += [[b, a, a + n, b + n] for a, b in outline]
    mesh.from_pydata(np.concatenate([outer, inner]).tolist(), [], polys)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    label.matrix_world.identity()
    if missed:
        return f"Le texte déborde de la surface ({missed} points)"
    return None


def boolean_solver():
    return 'MANIFOLD'           # both the part and the cutter are closed: fast enough to stay live


def add_label(part, start, end, normal, settings):
    """New label on a part, from two surface points. `settings` gives text, height, depth, mode."""
    label = bpy.data.objects.new(f"Texte: {settings.text}", bpy.data.meshes.new("Texte"))
    for coll in part.users_collection:
        coll.objects.link(label)
    label.omfs.made_by = TEXT_TAG
    label.display_type = 'WIRE'
    label.hide_render = True
    t = label.omfs_text
    t.part, t.start, t.end, t.normal = part, start, end, normal
    t.text, t.height, t.depth, t.mode = settings.text, settings.height, settings.depth, settings.mode
    warning = build(label)
    attach(label)
    return label, warning


def attach(label):
    """The Boolean modifier on the part: subtract the letters (engraved) or add them (embossed)."""
    t = label.omfs_text
    mod = next((m for m in t.part.modifiers if m.type == 'BOOLEAN' and m.object == label), None)
    if mod is None:
        mod = t.part.modifiers.new(label.name, 'BOOLEAN')
        mod.object = label
        mod.solver = boolean_solver()
    mod.operation = 'DIFFERENCE' if t.mode == 'ENGRAVE' else 'UNION'


def rebuild_on(part):
    """The part's surface changed (a rebuilt blank): lay its labels on it again."""
    _surfaces.pop(part.name, None)
    for label in labels_on(part):
        if label.omfs_text.part == part:
            build(label)


def remove_label(label):
    for obj in bpy.data.objects:
        for mod in [m for m in obj.modifiers if m.type == 'BOOLEAN' and m.object == label]:
            obj.modifiers.remove(mod)
    mesh = label.data
    bpy.data.objects.remove(label)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def wall_under(label):
    """Thinnest part left under the engraved letters (mm), along the line of sight; None if embossed
    or if nothing is found under the letters."""
    t = label.omfs_text
    if t.mode != 'ENGRAVE' or t.part is None:
        return None
    x, y, z = frame(t.start, t.end, t.normal)
    tree = surface_of(t.part)
    to_local = t.part.matrix_world.inverted()
    down = (to_local.to_3x3() @ -z).normalized()
    n = len(label.data.vertices) // 2
    thinnest = None
    for v in label.data.vertices[n:]:            # bottom of the letters, inside the part
        p = label.matrix_world @ v.co
        hit = tree.ray_cast(to_local @ p, down, 20.0)[0]
        if hit is not None:
            left = ((t.part.matrix_world @ hit) - p).length
            thinnest = left if thinnest is None else min(thinnest, left)
    return thinnest


# --- Properties -------------------------------------------------------------------------------

def _on_label(t, context):
    label = t.id_data
    if is_label(label) and t.part is not None:
        build(label)
        attach(label)
        label.name = f"Texte: {t.text}"


class OMFS_TextSettings(bpy.types.PropertyGroup):
    """A label's own settings (on the cutter object), or the settings for the next label (scene)."""
    text: bpy.props.StringProperty(name="Texte", default="", update=_on_label,
                                   description="Le texte à graver: initiales, date, côté, type de gouttière")
    height: bpy.props.FloatProperty(
        name="Hauteur max", subtype='DISTANCE', default=params.TEXT_HEIGHT, min=0.5, soft_max=10.0, step=10,
        update=_on_label,
        description="Hauteur maximale des lettres: le texte remplit l'espace entre les deux clics, sans dépasser")
    depth: bpy.props.FloatProperty(
        name="Profondeur", subtype='DISTANCE', default=params.TEXT_DEPTH, min=0.1, soft_max=2.0, step=5,
        update=_on_label, description="Profondeur de la gravure, ou hauteur du relief")
    mode: bpy.props.EnumProperty(name="Mode", items=MODES, default='ENGRAVE', update=_on_label)
    part: bpy.props.PointerProperty(type=bpy.types.Object, name="Pièce")
    start: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    end: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    normal: bpy.props.FloatVectorProperty(size=3, subtype='XYZ', default=(0, 0, 1))
    letter_height: bpy.props.FloatProperty(name="Hauteur des lettres", subtype='DISTANCE')


# --- Placing ----------------------------------------------------------------------------------

def surface_hit(context, region, mouse, part):
    """World point and normal under the mouse on the part, or None."""
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
    to_local = part.matrix_world.inverted()
    depsgraph = context.evaluated_depsgraph_get()
    hit, location, normal, _index = part.ray_cast(to_local @ origin, to_local.to_3x3() @ direction,
                                                  depsgraph=depsgraph)
    if not hit:
        return None
    return part.matrix_world @ location, (part.matrix_world.to_3x3() @ normal).normalized()


class OMFS_OT_place_text(bpy.types.Operator):
    """Click where the text starts, then where it ends, on the part's surface. The letters fit
    between the two clicks and read from the outside, left to right from the first click"""
    bl_idname = "omfs.place_text"
    bl_label = "Placer le texte (2 clics)"
    bl_options = {'REGISTER', 'UNDO'}

    part_name: bpy.props.StringProperty()
    settings_path: bpy.props.StringProperty(description="Data path (from the scene) of the settings to use")

    def invoke(self, context, event):
        self.part = bpy.data.objects.get(self.part_name)
        if self.part is None:
            return {'CANCELLED'}
        self.settings = context.scene.path_resolve(self.settings_path)
        if not self.settings.text.strip():
            self.report({'ERROR'}, "Écrivez d'abord le texte")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.first = None
        self.area.header_text_set(f"Texte « {self.settings.text} »: cliquez où il commence sur {self.part.name}"
                                  "   |   Clic droit / Échap: annuler")
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'CANCELLED'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        hit = surface_hit(context, *under_mouse, self.part)
        if hit is None:
            self.report({'WARNING'}, f"Cliquez sur {self.part.name}")
            return {'RUNNING_MODAL'}
        if self.first is None:
            self.first = hit
            self.area.header_text_set(f"Texte « {self.settings.text} »: cliquez où il finit"
                                      "   |   Clic droit / Échap: annuler")
            return {'RUNNING_MODAL'}
        self.finish(context)
        (start, n1), (end, n2) = self.first, hit
        if (end - start).length < 1.0:
            self.report({'ERROR'}, "Les deux clics sont trop proches")
            return {'CANCELLED'}
        label, warning = add_label(self.part, start, end, (n1 + n2).normalized(), self.settings)
        for other in context.selected_objects:
            other.select_set(False)
        context.view_layer.objects.active = label
        label.select_set(True)
        if warning:
            self.report({'WARNING'}, warning)
        return {'FINISHED'}


class OMFS_OT_remove_text(bpy.types.Operator):
    """Remove this text and its Boolean modifier (the part itself is untouched)"""
    bl_idname = "omfs.remove_text"
    bl_label = "Retirer le texte"
    bl_options = {'REGISTER', 'UNDO'}

    label_name: bpy.props.StringProperty()

    def execute(self, context):
        label = bpy.data.objects.get(self.label_name)
        if is_label(label):
            remove_label(label)
        return {'FINISHED'}


classes = (OMFS_TextSettings, OMFS_OT_place_text, OMFS_OT_remove_text)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_text = bpy.props.PointerProperty(type=OMFS_TextSettings)


def unregister():
    del bpy.types.Object.omfs_text
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
