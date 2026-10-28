"""Section & Analysis, 3D tools: exact intersection volume of two objects.

Non-destructive by construction: the intersection is an overlay object that *shares* object A's
mesh data (like a linked duplicate) and carries a Boolean modifier, Intersect with B. A modifier
only changes what its own object shows, so A's mesh is never written. The overlay follows A
(Copy Transforms constraint) and updates live when A or B moves. Deleting the overlay removes
everything; A and B are exactly as before.

Two modes, chosen from the meshes:
- both closed: Manifold solver, fast (0.1 s on a CT mandible), the overlay is live;
- nearly closed (a few open edges, common in segmentations): the Manifold solver refuses them,
  the Exact solver copes but takes seconds, too slow to follow a drag. The result is computed once
  into the overlay's own mesh (a snapshot) and the panel offers Refresh once A or B has moved;
- open (scan): no inside, no intersection volume: refused.
"""

import bmesh
import bpy
import numpy as np

from . import locking, params, roles

INTERSECTION = "Intersection"   # made_by tag of the overlay
INTERSECTION_COLOUR = (1.0, 0.1, 0.1, 1.0)

_volumes = {}   # overlay name -> (volume mm³, closed?)


def is_analysable(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role not in roles.NOT_SURFACES


def open_edges(mesh):
    """Number of edges not shared by exactly two faces (0: the mesh is closed, it has an inside)."""
    edge_of_loop = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("edge_index", edge_of_loop)
    return int((np.bincount(edge_of_loop, minlength=len(mesh.edges)) != 2).sum())


def is_closed(mesh):
    return len(mesh.edges) > 0 and open_edges(mesh) == 0


def nearly_closed(mesh):
    return len(mesh.edges) > 0 and open_edges(mesh) <= params.ANALYSIS_MAX_OPEN_EDGES * len(mesh.edges)


def find_intersection(scene):
    return next((o for o in scene.objects if o.omfs.made_by == INTERSECTION), None)


def remove_intersection(scene):
    overlay = find_intersection(scene)
    if overlay is not None:
        _volumes.pop(overlay.name, None)
        mesh = overlay.data
        bpy.data.objects.remove(overlay)
        if mesh.users == 0:   # a snapshot's own mesh; a shared mesh is still A's
            bpy.data.meshes.remove(mesh)


def boolean_intersect(obj, other, solver):
    boolean = obj.modifiers.new("Intersect", 'BOOLEAN')
    boolean.operation = 'INTERSECT'
    boolean.object = other
    boolean.solver = solver
    return boolean


def snapshot_mesh(scene, a, b):
    """A ∩ B computed once with the Exact solver, as a new mesh (A's mesh is only read)."""
    temp = bpy.data.objects.new("Intersection (computing)", a.data)
    scene.collection.objects.link(temp)
    temp.matrix_world = a.matrix_world
    boolean_intersect(temp, b, 'EXACT')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(temp.evaluated_get(depsgraph))
    bpy.data.objects.remove(temp)
    return mesh


def pose_key(a, b):
    return repr([tuple(map(tuple, o.matrix_world)) for o in (a, b)])


def make_intersection(scene, a, b):
    """The overlay showing A ∩ B. Returns it, or an error message if a mesh is open."""
    open_meshes = [o.name for o in (a, b) if not nearly_closed(o.data)]
    if open_meshes:
        return f"Open mesh, no inside: {', '.join(open_meshes)}"
    remove_intersection(scene)
    live = is_closed(a.data) and is_closed(b.data)
    # Live: shares A's mesh, nothing copied, the modifier does the work. Snapshot: its own mesh.
    overlay = bpy.data.objects.new(f"{a.name} ∩ {b.name}", a.data if live else snapshot_mesh(scene, a, b))
    scene.collection.objects.link(overlay)
    overlay.omfs.role = 'ANALYSIS'
    overlay.omfs.made_by = INTERSECTION
    overlay.omfs_analysis.a, overlay.omfs_analysis.b = a, b
    overlay.constraints.new('COPY_TRANSFORMS').target = a
    if live:
        boolean_intersect(overlay, b, 'MANIFOLD')
    else:
        overlay.omfs_analysis.snapshot_pose = pose_key(a, b)
    overlay.color = INTERSECTION_COLOUR
    overlay.show_in_front = True   # seen through the bone
    locking.set_locked(overlay, True)
    return overlay


def is_snapshot(overlay):
    return bool(overlay.omfs_analysis.snapshot_pose)


def snapshot_stale(overlay):
    props = overlay.omfs_analysis
    return props.a is None or props.b is None or props.snapshot_pose != pose_key(props.a, props.b)


def intersection_volume(overlay, depsgraph):
    """(volume in mm³, closed?) of the evaluated overlay, i.e. after the Boolean."""
    evaluated = overlay.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    scale = abs(evaluated.matrix_world.to_3x3().determinant())
    volume = bm.calc_volume(signed=False) * scale
    closed = all(e.is_manifold for e in bm.edges)
    bm.free()
    evaluated.to_mesh_clear()
    return volume, closed


def cached_volume(overlay):
    return _volumes.get(overlay.name)


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """The Boolean is re-evaluated when A or B moves: measure the new volume."""
    overlay = find_intersection(scene)
    if overlay is None:
        return
    if overlay.name not in _volumes or any(
            u.is_updated_geometry and isinstance(u.id, bpy.types.Object) and u.id.original == overlay
            for u in depsgraph.updates):
        _volumes[overlay.name] = intersection_volume(overlay, depsgraph)


class OMFS_AnalysisObjectProperties(bpy.types.PropertyGroup):
    a: bpy.props.PointerProperty(type=bpy.types.Object)
    b: bpy.props.PointerProperty(type=bpy.types.Object)
    snapshot_pose: bpy.props.StringProperty()   # A and B poses of a snapshot; empty when live


class OMFS_OT_show_intersection(bpy.types.Operator):
    """Show the exact common volume of the two objects in red (live), with its volume in mm³.
    Click again to remove it. The objects themselves are never modified"""
    bl_idname = "omfs.show_intersection"
    bl_label = "Show Intersection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if find_intersection(scene):
            remove_intersection(scene)
            return {'FINISHED'}
        props = scene.omfs_section
        a, b = props.inter_a, props.inter_b
        if not (is_analysable(a) and is_analysable(b)) or a == b:
            self.report({'ERROR'}, "Pick two different objects")
            return {'CANCELLED'}
        if context.window:
            context.window.cursor_set('WAIT')
        result = make_intersection(scene, a, b)
        if context.window:
            context.window.cursor_set('DEFAULT')
        if isinstance(result, str):
            self.report({'ERROR'}, result)
            return {'CANCELLED'}
        _volumes[result.name] = intersection_volume(result, context.evaluated_depsgraph_get())
        return {'FINISHED'}


class OMFS_OT_refresh_intersection(bpy.types.Operator):
    """Compute the intersection again for the current positions (Exact solver: a few seconds)"""
    bl_idname = "omfs.refresh_intersection"
    bl_label = "Refresh Intersection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        overlay = find_intersection(context.scene)
        if overlay is None or overlay.omfs_analysis.a is None or overlay.omfs_analysis.b is None:
            return {'CANCELLED'}
        props = context.scene.omfs_section
        props.inter_a, props.inter_b = overlay.omfs_analysis.a, overlay.omfs_analysis.b
        remove_intersection(context.scene)
        return bpy.ops.omfs.show_intersection()


classes = (OMFS_AnalysisObjectProperties, OMFS_OT_show_intersection, OMFS_OT_refresh_intersection)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_analysis = bpy.props.PointerProperty(type=OMFS_AnalysisObjectProperties)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister():
    bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    del bpy.types.Object.omfs_analysis
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
