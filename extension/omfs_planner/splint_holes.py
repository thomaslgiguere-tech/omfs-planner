"""Splint option: holes for the osteosynthesis wire, through the buccal flange, between the teeth.

Gesture, like the guide's sleeves: a click on the splint drops a hole, the hole follows the mouse,
a second click fixes it; again for the next one. Each hole is a cylinder along the insertion axis
(the splint's "up"), through the whole splint. All the holes sit in one collection, subtracted by a
single Boolean modifier ("one big Boolean"): the splint mesh is untouched until export.

Constraint (the most important one, surgeon 2026-11-08): a hole never touches the tooth imprints,
upper or lower. It lives only in the buccal flange: on the buccal side of the arch curve, at least
SPLINT_WIRE_HOLE_TEETH_GAP mm from every tooth of either arch (measured across the insertion axis,
over the splint's height), and SPLINT_WIRE_HOLE_WALL mm inside the splint's outline. Dragged onto
the teeth, the hole is pushed buccally until it clears them; if the flange is too narrow, it turns
red and cannot be fixed.
"""

import bmesh
import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import guide_workshop, landmarks, params, splint

COLLECTION = "Trous pour fil"
MODIFIER = "Trous pour fil"
TAG = "wire hole"                 # made_by of a hole cylinder
SEGMENTS = 32
PUSH_STEP = 0.1                   # mm per step when a hole is pushed buccally off the teeth
PUSH_MAX = 10.0                   # mm: farther than this, the flange is not there
RIM_POINTS = 16
OK_COLOUR = (0.2, 0.8, 0.3, 1.0)
BAD_COLOUR = (0.9, 0.15, 0.15, 1.0)


def hole_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    parent = splint.splint_collection(scene)
    if coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def holes(scene):
    coll = bpy.data.collections.get(COLLECTION)
    return [o for o in coll.objects if o.omfs.made_by == TAG] if coll else []


def shape(mesh, diameter, length):
    """Closed cylinder along Z, centred on the origin."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=SEGMENTS,
                          radius1=diameter / 2, radius2=diameter / 2, depth=length)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def set_diameter(scene, diameter):
    for obj in holes(scene):
        shape(obj.data, diameter, obj["omfs_length"])


def add_boolean(part, scene):
    """The single Boolean subtracting every hole (the collection) from the part."""
    mod = part.modifiers.get(MODIFIER)
    if mod is None:
        mod = part.modifiers.new(MODIFIER, 'BOOLEAN')
    mod.operation, mod.operand_type, mod.solver = 'DIFFERENCE', 'COLLECTION', guide_workshop.SOLVER
    mod.collection = hole_collection(scene)
    return mod


def remove_hole(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)


# --- Where a hole may go ----------------------------------------------------------------------

class Flange:
    """What the constraint needs, gathered once per placing session: insertion axis, the arch curve
    (stations and buccal normals), the splint's height, the teeth seen along the axis, the outline."""

    def __init__(self, scene, part, depsgraph):
        self.up = np.array(splint.up_axis(scene))
        line = splint.centreline(scene, depsgraph, 0.5)          # the cap's curve or the splint's automatic arch
        if isinstance(line, str):
            raise ValueError(line)
        self.points, _tangents, self.normals, _length, self.above, self.below = line
        self.depth_up, self.depth_low = float(self.above.max()), float(self.below.max())
        heights = self.points @ self.up
        verts = np.concatenate([splint.world_vertices(o) for o in splint.arches(scene)])
        h = verts @ self.up
        band = (h > (heights - self.below).min() - 0.5) & (h < (heights + self.above).max() + 0.5)
        flat = self.flatten(verts[band])
        self.teeth = KDTree(len(flat))
        for i, v in enumerate(flat):
            self.teeth.insert(v, i)
        self.teeth.balance()
        co = splint.world_vertices(part)
        polys = [tuple(p.vertices) for p in part.data.polygons]
        self.outline = BVHTree.FromPolygons(co.tolist(), polys)

    def flatten(self, points):
        """Points moved along the insertion axis onto the plane through the origin: distances across it."""
        return points - np.outer(points @ self.up, self.up)

    def station(self, p):
        flat = self.flatten(self.points)
        return int(np.argmin(np.linalg.norm(flat - self.flatten(p[None])[0], axis=1)))

    def teeth_distance(self, q):
        _co, _index, dist = self.teeth.find(self.flatten(q[None])[0])
        return dist if dist is not None else np.inf

    def inside_outline(self, q, radius):
        """Every point of a circle around q (across the axis) is over the splint."""
        top = q + self.up * (self.depth_up + 5.0)
        e1 = np.cross(self.up, (1.0, 0.0, 0.0))
        if np.linalg.norm(e1) < 0.1:
            e1 = np.cross(self.up, (0.0, 1.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.up, e1)
        down = Vector(-self.up)
        for a in np.linspace(0, 2 * np.pi, RIM_POINTS, endpoint=False):
            start = Vector(top + radius * (np.cos(a) * e1 + np.sin(a) * e2))
            if self.outline.ray_cast(start, down, self.depth_up + self.depth_low + 10.0)[0] is None:
                return False
        return True

    def place(self, point, diameter):
        """(hole centre, error message or None) for a point clicked on the splint."""
        p = np.array(point)
        i = self.station(p)
        c, n = self.points[i], self.normals[i]
        lateral = (p - c) @ n
        if lateral <= 0:
            return None, "Côté lingual : les trous vont seulement dans le débord vestibulaire"
        # On the curve's height, between the splint's top and bottom faces.
        q = p - self.up * ((p - c) @ self.up) + self.up * (self.above[i] - self.below[i]) / 2
        need = diameter / 2 + params.SPLINT_WIRE_HOLE_TEETH_GAP
        pushed = 0.0
        while self.teeth_distance(q) < need:
            if pushed >= PUSH_MAX:
                return Vector(q), "Pas de place hors des dents ici"
            q = q + n * PUSH_STEP
            pushed += PUSH_STEP
        if not self.inside_outline(q, diameter / 2 + params.SPLINT_WIRE_HOLE_WALL):
            return Vector(q), ("Trop près du bord : élargissez le débord vestibulaire (bloc), ou déplacez "
                               "le trou entre deux dents")
        return Vector(q), None

    def length(self):
        return self.depth_up + self.depth_low + 2 * params.SPLINT_WIRE_HOLE_OVERSHOOT

    def matrix(self, centre):
        return Matrix.LocRotScale(centre, Vector(self.up).to_track_quat('Z', 'Y'), None)


def new_hole(scene, flange, centre, diameter):
    mesh = bpy.data.meshes.new("Trou pour fil")
    obj = bpy.data.objects.new("Trou pour fil", mesh)
    splint.splint_collection(scene).objects.link(obj)   # into the hole collection once fixed
    obj.omfs.role = 'MARKER'                             # not a surface: clicks go through it
    obj.omfs.made_by = TAG
    obj["omfs_length"] = flange.length()
    shape(mesh, diameter, flange.length())
    obj.matrix_world = flange.matrix(centre)
    obj.show_in_front = True
    obj.hide_render = True
    return obj


def fix_hole(scene, part, obj):
    """Into the collection the Boolean subtracts; drawn as a wireframe from now on."""
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    hole_collection(scene).objects.link(obj)
    obj.display_type = 'WIRE'
    obj.color = (1.0, 1.0, 1.0, 1.0)
    add_boolean(part, scene)


def place_hole(scene, part, point, diameter, depsgraph):
    """Scripts and tests: one hole fixed at once. Returns (hole or None, error or None)."""
    flange = Flange(scene, part, depsgraph)
    centre, error = flange.place(point, diameter)
    if error:
        return None, error
    obj = new_hole(scene, flange, centre, diameter)
    fix_hole(scene, part, obj)
    return obj, None


# --- Operators ----------------------------------------------------------------------------------

class OMFS_OT_add_wire_hole(bpy.types.Operator):
    """Trous pour le fil d'ostéosynthèse, dans le débord vestibulaire : clic sur la gouttière, la souris
    déplace le trou (repoussé hors des dents), clic : fixer ; encore pour le suivant. Tous soustraits
    par un seul Boolean. Échap : retirer celui qui bouge ; clic droit / Entrée : terminer"""
    bl_idname = "omfs.add_wire_hole"
    bl_label = "Placer des trous"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self.part = splint.current_part(context)
        if self.part is None:
            self.report({'ERROR'}, "Construisez d'abord la gouttière (bloc ou solide)")
            return {'CANCELLED'}
        line = splint.centreline(context.scene, context.evaluated_depsgraph_get(), 0.5)
        if isinstance(line, str):
            self.report({'ERROR'}, line)
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.flange = Flange(context.scene, self.part, context.evaluated_depsgraph_get())
        self.sliding = None
        self.error = None
        self.added = 0
        self.header()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def header(self):
        if self.sliding is None:
            text = (f"Trous pour fil sur {self.part.name} : cliquez dans le débord vestibulaire "
                    f"({self.added} placés)   |   Clic droit / Entrée : terminer")
        elif self.error:
            text = f"{self.error}   |   Échap : retirer ce trou"
        else:
            text = "Bougez la souris pour ajuster, clic : fixer   |   Échap : retirer ce trou"
        self.area.header_text_set(text)

    def drop(self):
        if self.sliding is not None:
            remove_hole(self.sliding)
            self.sliding = None

    def finish(self, context):
        self.drop()
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        return {'FINISHED'} if self.added else {'CANCELLED'}

    def hit(self, context, event):
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return None
        return landmarks.object_hit(context, *under_mouse, self.part)

    def follow(self, context, point):
        diameter = context.scene.omfs_splint.wire_hole_diameter
        centre, self.error = self.flange.place(point, diameter)
        if centre is None:
            return
        if self.sliding is None:
            self.sliding = new_hole(context.scene, self.flange, centre, diameter)
        self.sliding.matrix_world = self.flange.matrix(centre)
        self.sliding.color = BAD_COLOUR if self.error else OK_COLOUR

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.finish(context)
        if event.value == 'PRESS' and event.type == 'ESC':
            if self.sliding is None:
                return self.finish(context)
            self.drop()
            self.header()
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self.sliding is not None:
            point = self.hit(context, event)
            if point is not None:
                self.follow(context, point)
                self.header()
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        if landmarks.viewport_region_under_mouse(self.area, event) is None:
            return {'PASS_THROUGH'}
        if self.sliding is None:
            point = self.hit(context, event)
            if point is None:
                self.report({'WARNING'}, f"Cliquez sur {self.part.name}")
                return {'RUNNING_MODAL'}
            self.follow(context, point)
            if self.sliding is None:
                self.report({'WARNING'}, self.error)
            self.header()
            return {'RUNNING_MODAL'}
        if self.error:
            self.report({'WARNING'}, self.error)
            return {'RUNNING_MODAL'}
        fix_hole(context.scene, self.part, self.sliding)
        self.sliding = None
        self.added += 1
        self.header()
        return {'RUNNING_MODAL'}


class OMFS_OT_remove_wire_hole(bpy.types.Operator):
    """Supprimer ce trou (vide : tous les trous)"""
    bl_idname = "omfs.remove_wire_hole"
    bl_label = "Supprimer le trou"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty()

    def execute(self, context):
        for obj in holes(context.scene):
            if not self.name or obj.name == self.name:
                remove_hole(obj)
        return {'FINISHED'}


classes = (OMFS_OT_add_wire_hole, OMFS_OT_remove_wire_hole)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
