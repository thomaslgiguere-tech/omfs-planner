"""Occlusion seating, first step: only the two arch scans on screen, their borders cleaned.

- Isolate: every other object in the scene is hidden (eye icon, nothing deleted), without
  exception; the names are remembered so "Show everything" brings back exactly what was visible.
- Clean borders: intraoral scans end in a ragged open border (thin triangles, spikes, crumbs).
  A thin band along the border is removed, then the spikes left on it and the small loose
  pieces. The scan stays open (no hole is filled). The original mesh data is kept, so the
  original borders can be put back. The teeth are far from the border and never touched.
"""

import json
from math import radians

import bmesh
import bpy
from mathutils.kdtree import KDTree

from . import occlusion, params

SPIKE_PASSES = 20   # rounds of spike removal along the new border (each round peels one layer)


# --- Isolate ------------------------------------------------------------------------

def hidden_names(scene):
    try:
        return json.loads(scene.omfs_isolated or "[]")
    except ValueError:
        return []


def is_isolated(scene):
    return bool(scene.omfs_isolated)


def isolate(scene, view_layer, keep):
    """Hide every visible object except `keep`; remember which ones were hidden."""
    hidden = []
    for obj in scene.objects:
        if obj in keep:
            obj.hide_set(False)
            obj.hide_viewport = False
        elif obj.visible_get(view_layer=view_layer):
            obj.hide_set(True)
            hidden.append(obj.name)
    scene.omfs_isolated = json.dumps(sorted(set(hidden_names(scene)) | set(hidden)))


def show_hidden(scene):
    for name in hidden_names(scene):
        obj = scene.objects.get(name)
        if obj is not None:
            obj.hide_set(False)
    scene.omfs_isolated = ""


# --- Clean borders ------------------------------------------------------------------

def _is_spike(face, sharp):
    """A face hanging off the border: all edges free, or two free edges meeting at a sharp tip."""
    free = [e for e in face.edges if e.is_boundary]
    if len(free) >= 3:
        return True
    if len(free) < 2:
        return False
    tip = set(free[0].verts) & set(free[1].verts)
    return any(loop.vert in tip and loop.calc_angle() < sharp for loop in face.loops)


def _spikes(bm):
    """Spikes on the border, and the faces around a bow-tie vertex (two fans joined at one point)."""
    sharp = radians(params.OCC_TRIM_SPIKE_ANGLE)
    faces = {f for f in bm.faces if _is_spike(f, sharp)}
    for v in bm.verts:
        if v.is_boundary and sum(e.is_boundary for e in v.link_edges) > 2:
            faces.update(v.link_faces)
    return list(faces)


def _islands(bm):
    """Groups of faces connected by edges."""
    seen, groups = set(), []
    for start in bm.faces:
        if start in seen:
            continue
        seen.add(start)
        group, stack = [], [start]
        while stack:
            f = stack.pop()
            group.append(f)
            for e in f.edges:
                for g in e.link_faces:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        groups.append(group)
    return groups


def clean_borders(mesh, band, min_island):
    """Remove a band of `band` mm along the open border, then spikes and pieces smaller than
    `min_island` of the total area. Vertices kept are never moved."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    border = [v.co.copy() for v in bm.verts if v.is_boundary]
    if border and band > 0:
        tree = KDTree(len(border))
        for i, co in enumerate(border):
            tree.insert(co, i)
        tree.balance()
        near = [f for f in bm.faces if any(tree.find(v.co)[2] <= band for v in f.verts)]
        bmesh.ops.delete(bm, geom=near, context='FACES')
    for _ in range(SPIKE_PASSES):
        spikes = _spikes(bm)
        if not spikes:
            break
        bmesh.ops.delete(bm, geom=spikes, context='FACES')
    groups = _islands(bm)
    total = sum(f.calc_area() for f in bm.faces) or 1.0
    crumbs = [f for g in groups if sum(f.calc_area() for f in g) < min_island * total for f in g]
    if crumbs:
        bmesh.ops.delete(bm, geom=crumbs, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def users_of(mesh):
    return [o for o in bpy.data.objects if o.data is mesh]


def swap_mesh(obj, mesh):
    """Give obj (and every object sharing its mesh, e.g. a pre-op ghost) another mesh data-block."""
    for user in users_of(obj.data):
        user.data = mesh
    if obj.omfs_occl.fingerprint:   # in an occlusion block: the new borders are the reference shape
        occlusion.remember_shape(obj)


def trim_arch(obj, band, min_island):
    """Clean the borders on a copy of the mesh data; the untouched original is kept on the object."""
    original = obj.omfs_untrimmed or obj.data
    trimmed = original.copy()
    clean_borders(trimmed, band, min_island)
    if obj.omfs_untrimmed is None:
        obj.omfs_untrimmed = original
        name = original.name
        original.name = f"{name} (bords d'origine)"
        trimmed.name = name
    old = obj.data
    swap_mesh(obj, trimmed)
    if old is not original and old.users == 0:
        bpy.data.meshes.remove(old)   # an earlier trim, replaced
    return len(original.vertices) - len(trimmed.vertices)


def restore_arch(obj):
    original = obj.omfs_untrimmed
    if original is None:
        return
    old = obj.data
    obj.omfs_untrimmed = None
    name = old.name
    old.name = f"{name} (nettoyé)"
    original.name = name
    swap_mesh(obj, original)
    if old.users == 0:
        bpy.data.meshes.remove(old)


# --- Operators ----------------------------------------------------------------------

def arches(scene):
    props = scene.omfs_occlusion
    return [a for a in (props.upper, props.lower) if a is not None]


class OMFS_OT_prepare_arches(bpy.types.Operator):
    """Cacher tout le reste de la scène (sans exception) sauf l'arcade du haut et l'arcade du bas, et
    nettoyer leurs bordures : une fine bande le long du bord ouvert, les pointes et les miettes sont
    retirées. Les scans restent ouverts ; les bords d'origine sont gardés"""
    bl_idname = "omfs.prepare_arches"
    bl_label = "Isoler et nettoyer les arcades"
    bl_options = {'REGISTER', 'UNDO'}

    band: bpy.props.FloatProperty(name="Bande retirée au bord", subtype='DISTANCE', default=params.OCC_TRIM_BAND,
                                  min=0.0, soft_max=3.0, step=10, precision=1)
    min_island: bpy.props.FloatProperty(name="Miettes sous", subtype='PERCENTAGE', default=params.OCC_TRIM_MIN_ISLAND,
                                        min=0.0, max=50.0, description="Morceaux détachés plus petits que cette "
                                        "part de la surface de l'arcade : retirés")

    def execute(self, context):
        scene = context.scene
        pair = arches(scene)
        if len(pair) < 2:
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        isolate(scene, context.view_layer, pair)
        removed = [trim_arch(a, self.band, self.min_island / 100.0) for a in pair]
        self.report({'INFO'}, "Bords nettoyés : " + ", ".join(f"{a.name} -{n} sommets" for a, n in zip(pair, removed)))
        return {'FINISHED'}


class OMFS_OT_show_hidden(bpy.types.Operator):
    """Réafficher exactement les objets cachés par « Isoler et nettoyer les arcades »"""
    bl_idname = "omfs.show_hidden"
    bl_label = "Réafficher le reste"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        show_hidden(context.scene)
        return {'FINISHED'}


class OMFS_OT_restore_borders(bpy.types.Operator):
    """Remettre les bords d'origine des deux arcades (annule le nettoyage)"""
    bl_idname = "omfs.restore_borders"
    bl_label = "Bords d'origine"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for arch in arches(context.scene):
            restore_arch(arch)
        return {'FINISHED'}


classes = (OMFS_OT_prepare_arches, OMFS_OT_show_hidden, OMFS_OT_restore_borders)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_isolated = bpy.props.StringProperty()
    bpy.types.Object.omfs_untrimmed = bpy.props.PointerProperty(type=bpy.types.Mesh)


def unregister():
    del bpy.types.Object.omfs_untrimmed
    del bpy.types.Scene.omfs_isolated
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
