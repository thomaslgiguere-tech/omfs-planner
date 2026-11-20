"""Teeth registration, last step: "Modifier la géométrie des dents CBCT".

Once the scan's teeth position is canned (registration.freeze_scan), the crowns of the CBCT teeth
(imprecise: voxel size, metal scatter) are brought onto the scan's crowns (precise), so the scan with
its gingiva can be set aside. The roots of the CBCT teeth never move.

How (a "magnet" that knows which part of each mesh is which):
1. Scan side: only the scan's crowns (the "Crowns" vertex group of the registration, gingiva excluded).
2. CBCT side: a tooth vertex is a crown vertex when the scan crowns are within FUSE_SNAP_MAX of it,
   facing the same way (so the buccal side never snaps onto the lingual side), and not at the edge of
   the scan crown patch (the gum line: there the scan stops, the tooth goes on into the gingiva).
3. Those crown areas are first subdivided down to FUSE_MAX_EDGE, then every crown vertex is moved onto
   the closest point of the scan crown surface: the crown now lies on the scan's own surface.
   Grooves and pits narrower than the CBCT could see: a reverse pull (each scan crown point pulls
   its closest crown vertex, then all go back onto the scan surface), FUSE_GROOVE_PASSES rounds.
4. A transition band (FUSE_BLEND mm) below the crown follows smoothly (displacement diffused along the
   mesh); every vertex beyond it, the roots, keeps its exact CBCT position.
5. Report: distance from the scan's crown points to the new crowns (mean, 95 %, max), on the crown
   proper (points more than FUSE_REPORT_MARGIN from the gum line, where the transition is).
The scan's crown selection is first "closed" (grown then shrunk by FUSE_CROWN_CLOSE rings): the
suggested / painted selection has small holes, each one a border where nothing would snap.

Nothing is changed in place: the teeth object receives a new mesh data-block; the CBCT mesh data-block
is kept untouched on the object (omfs_fusion.original) and "Dents CBCT d'origine" puts it back.
"""

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import locking, occlusion_prepare, params, registration

CROWN_ATTRIBUTE = "Couronne STL"   # boolean point attribute on the new mesh: True on the crowns now from the scan
BLEND_ITERATIONS = 80


# --- Geometry -----------------------------------------------------------------------

def close_mask(mask, edges, rings):
    """Morphological closing on the mesh: grow the selection by `rings` rings, then shrink it back.
    Fills the small holes and notches of a painted / suggested selection; its outer edge stays."""
    a, b = edges[:, 0], edges[:, 1]
    n = len(mask)
    for _ in range(rings):   # grow: a vertex joins if a neighbour is in
        hit = np.bincount(a, weights=mask[b], minlength=n) + np.bincount(b, weights=mask[a], minlength=n)
        mask = mask | (hit > 0)
    degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    for _ in range(rings):   # shrink: a vertex stays if all its neighbours are in
        inside = np.bincount(a, weights=mask[b], minlength=n) + np.bincount(b, weights=mask[a], minlength=n)
        mask = mask & (inside >= degree)
    return mask


def scan_crown_surface(scan, to_teeth):
    """(BVH of the scan's crown faces, KDTree of the crown patch's border vertices, crown vertex
    coordinates away from that border), all in the teeth's local coordinates."""
    indices = registration.crown_indices(scan)
    co = registration.vertex_coords(scan, world=False)
    co = co @ to_teeth[:3, :3].T + to_teeth[:3, 3]
    is_crown = np.zeros(len(co), dtype=bool)
    is_crown[indices] = True
    edges = np.empty(len(scan.data.edges) * 2, dtype=np.int64)
    scan.data.edges.foreach_get("vertices", edges)
    is_crown = close_mask(is_crown, edges.reshape(-1, 2), params.FUSE_CROWN_CLOSE)
    tris = registration.triangles(scan.data)
    tris = tris[is_crown[tris].all(axis=1)]
    bvh = BVHTree.FromPolygons(co.tolist(), tris.tolist())
    edges = np.sort(np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = np.unique(unique[counts == 1])
    tree = KDTree(max(len(border), 1))
    for i, p in enumerate(co[border]):
        tree.insert(p, i)
    tree.balance()
    points = co[is_crown]
    if len(border):   # the crown proper: the deviation is measured away from the gum line
        points = np.array([p for p in points if tree.find(p)[2] > params.FUSE_REPORT_MARGIN])
    return bvh, (tree if len(border) else None), points


def snap_targets(co, normals, bvh, border_tree):
    """Closest scan crown point per vertex, and whether the vertex is a crown vertex (see module doc)."""
    closest, face_normals = registration.nearest(bvh, co, params.FUSE_SNAP_MAX)
    found = ~np.isnan(closest[:, 0])
    facing = np.zeros(len(co))
    facing[found] = np.einsum("ij,ij->i", normals[found], face_normals[found])
    if found.any() and np.median(facing[found]) < 0:   # scan normals pointing inwards: same test, flipped
        facing = -facing
    crown = found & (facing > np.cos(np.radians(params.FUSE_MAX_ANGLE)))
    if border_tree is not None:
        for i in np.flatnonzero(crown):
            crown[i] = border_tree.find(closest[i])[2] > params.FUSE_BORDER_GAP
    return closest, crown


def mesh_arrays(mesh):
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    normals = np.empty(len(mesh.vertices) * 3)
    mesh.vertex_normals.foreach_get("vector", normals)
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges)
    return co.reshape(-1, 3), normals.reshape(-1, 3), edges.reshape(-1, 2)


def subdivide_crowns(mesh, crown):
    """Split the crown edges longer than FUSE_MAX_EDGE (twice at most), so the new crowns sample the
    scan surface finely. Returns True if anything was split."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    is_crown = [bool(c) for c in crown]
    changed = False
    for _ in range(params.FUSE_SUBDIVIDE_PASSES):
        is_crown += [True] * (len(bm.verts) - len(is_crown))   # new vertices sit inside the crown area
        long_edges = [e for e in bm.edges if is_crown[e.verts[0].index] and is_crown[e.verts[1].index]
                      and e.calc_length() > params.FUSE_MAX_EDGE]
        if not long_edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        changed = True
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return changed


def blend(co, edges, crown, displacement):
    """Diffuse the crown displacement into the band below it; vertices beyond the band stay put."""
    if not crown.any():
        return displacement
    tree = KDTree(int(crown.sum()))
    for i, p in enumerate(co[crown]):
        tree.insert(p, i)
    tree.balance()
    candidates = np.flatnonzero(~crown)
    band = np.zeros(len(co), dtype=bool)
    for i in candidates:
        band[i] = tree.find(co[i])[2] <= params.FUSE_BLEND
    if not band.any():
        return displacement
    a, b = edges[:, 0], edges[:, 1]
    count = np.bincount(a, minlength=len(co)) + np.bincount(b, minlength=len(co))
    count[count == 0] = 1
    d = displacement.copy()
    for _ in range(BLEND_ITERATIONS):
        mean = np.column_stack([(np.bincount(a, weights=d[b, k], minlength=len(co))
                                 + np.bincount(b, weights=d[a, k], minlength=len(co))) / count for k in range(3)])
        d[band] = mean[band]
    return d


def neighbour_mean(co, edges):
    a, b = edges[:, 0], edges[:, 1]
    n = len(co)
    count = np.maximum(np.bincount(a, minlength=n) + np.bincount(b, minlength=n), 1)
    return np.column_stack([(np.bincount(a, weights=co[b, k], minlength=n)
                             + np.bincount(b, weights=co[a, k], minlength=n)) / count for k in range(3)])


def onto_scan(positions, scan_bvh):
    closest, _normals = registration.nearest(scan_bvh, positions, params.FUSE_SNAP_MAX)
    return np.where(np.isnan(closest[:, :1]), positions, closest)


def relax(co, crown, edges, scan_bvh):
    """Even out the crown triangles without leaving the scan surface: each crown vertex slides halfway
    to the mean of its neighbours along the surface (the part of the move across the surface is
    dropped, so it doesn't climb out of a groove), then goes back onto the scan. Closest-point
    projection bunches vertices in hollows and folds a few triangles: this unfolds them."""
    for _ in range(params.FUSE_RELAX_PASSES):
        p = co[crown]
        _closest, normals = registration.nearest(scan_bvh, p, params.FUSE_SNAP_MAX)
        normals = np.nan_to_num(normals)
        move = neighbour_mean(co, edges)[crown] - p
        move -= np.einsum("ij,ij->i", move, normals)[:, None] * normals
        co[crown] = onto_scan(p + 0.5 * move, scan_bvh)
    return co


def fill_grooves(co, crown, edges, scan_bvh, scan_points):
    """Crown vertices placed on the scan surface by closest point never reach the bottom of a groove
    or a pit narrower than the CBCT could see (they stop on its walls). Reverse pull: each scan crown
    point pulls the crown vertex closest to it, then every vertex goes back onto the scan surface,
    and the triangles are evened out. co: all vertices (crown ones already on the scan)."""
    co = relax(co, crown, edges, scan_bvh)
    if not crown.any() or not len(scan_points):
        return co
    for _ in range(params.FUSE_GROOVE_PASSES):
        positions = co[crown]
        tree = KDTree(len(positions))
        for i, p in enumerate(positions):
            tree.insert(p, i)
        tree.balance()
        owner = np.empty(len(scan_points), dtype=np.int64)
        near = np.zeros(len(scan_points), dtype=bool)
        for k, s in enumerate(scan_points):
            _co, owner[k], dist = tree.find(s)
            near[k] = dist <= params.FUSE_SNAP_MAX
        owner, pulled_by = owner[near], scan_points[near]
        count = np.bincount(owner, minlength=len(positions))
        pull = np.column_stack([np.bincount(owner, weights=pulled_by[:, k], minlength=len(positions))
                                for k in range(3)])
        has = count > 0
        target = positions.copy()
        target[has] = 0.5 * positions[has] + 0.5 * pull[has] / count[has, None]
        co[crown] = onto_scan(target, scan_bvh)
        co = relax(co, crown, edges, scan_bvh)
    return co


def fuse_mesh(mesh, scan_bvh, border_tree, scan_points=()):
    """Bring the crowns of this (teeth) mesh onto the scan crowns, in place. Returns the crown mask."""
    co, normals, _edges = mesh_arrays(mesh)
    _closest, crown = snap_targets(co, normals, scan_bvh, border_tree)
    subdivide_crowns(mesh, crown)
    co, normals, edges = mesh_arrays(mesh)
    closest, crown = snap_targets(co, normals, scan_bvh, border_tree)
    displacement = np.zeros_like(co)
    moved = co.copy()
    moved[crown] = closest[crown]
    moved = fill_grooves(moved, crown, edges, scan_bvh, np.asarray(scan_points, dtype=float).reshape(-1, 3))
    displacement[crown] = moved[crown] - co[crown]
    displacement = blend(co, edges, crown, displacement)
    mesh.vertices.foreach_set("co", (co + displacement).ravel())
    attribute = mesh.attributes.get(CROWN_ATTRIBUTE) or mesh.attributes.new(CROWN_ATTRIBUTE, 'BOOLEAN', 'POINT')
    attribute.data.foreach_set("value", crown.tolist())
    mesh.update()
    return crown


def deviation(mesh, scan_points):
    """Distances (mm) from the scan's crown points to the new teeth surface."""
    co, _normals, _edges = mesh_arrays(mesh)
    bvh = BVHTree.FromPolygons(co.tolist(), registration.triangles(mesh).tolist())
    closest, _ = registration.nearest(bvh, scan_points, params.FUSE_SNAP_MAX * 2)
    found = ~np.isnan(closest[:, 0])
    return np.linalg.norm(closest[found] - scan_points[found], axis=1)


# --- Objects ------------------------------------------------------------------------

def is_fused(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'TEETH' and obj.omfs_fusion.original is not None


def crown_mask(obj):
    """Per vertex: True on the crowns taken from the scan (fused teeth only)."""
    attribute = obj.data.attributes.get(CROWN_ATTRIBUTE)
    if attribute is None:
        return None
    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    attribute.data.foreach_get("value", mask)
    return mask


def fuse(scan):
    """Modify the geometry of the scan's CBCT teeth (registration target). Returns the teeth object."""
    teeth = scan.omfs_reg.target
    original = teeth.omfs_fusion.original or teeth.data
    to_teeth = np.array(teeth.matrix_world.inverted() @ scan.matrix_world)
    scan_bvh, border_tree, scan_points = scan_crown_surface(scan, to_teeth)
    fused = original.copy()
    crown = fuse_mesh(fused, scan_bvh, border_tree, scan_points)
    distances = deviation(fused, scan_points)
    if teeth.omfs_fusion.original is None:
        teeth.omfs_fusion.original = original
        name = original.name
        original.name = f"{name} (CBCT d'origine)"
        fused.name = name
    old = teeth.data
    occlusion_prepare.swap_mesh(teeth, fused)   # also its pre-op ghost; re-takes the block's shape proof
    if old is not original and old.users == 0:
        bpy.data.meshes.remove(old)
    props = teeth.omfs_fusion
    props.scan = scan
    props.crown_count = int(crown.sum())
    props.mean = float(distances.mean()) if len(distances) else -1.0
    props.p95 = float(np.percentile(distances, 95)) if len(distances) else -1.0
    props.max = float(distances.max()) if len(distances) else -1.0
    return teeth


def restore(teeth):
    original = teeth.omfs_fusion.original
    if original is None:
        return
    old = teeth.data
    teeth.omfs_fusion.original = None
    name = old.name
    old.name = f"{name} (couronnes STL)"
    original.name = name
    occlusion_prepare.swap_mesh(teeth, original)
    if old.users == 0:
        bpy.data.meshes.remove(old)
    teeth.omfs_fusion.crown_count = 0


# --- Operators ----------------------------------------------------------------------

class OMFS_OT_fuse_crowns(bpy.types.Operator):
    """Modifier la géométrie des dents CBCT : leurs couronnes sont ramenées exactement sur les couronnes
    du scan (gencive exclue) ; les racines ne bougent pas. Le maillage CBCT d'origine est gardé.
    Le scan est ensuite mis de côté (caché)"""
    bl_idname = "omfs.fuse_crowns"
    bl_label = "Modifier la géométrie des dents CBCT"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = registration.scan_for(context, self.scan_name)
        if scan is None or not scan.omfs_reg.frozen:
            self.report({'ERROR'}, "Cannez d'abord la position dentaire initiale")
            return {'CANCELLED'}
        if scan.omfs_reg.target is None or len(registration.crown_indices(scan)) < 100:
            self.report({'ERROR'}, "Il faut les dents CBCT et les couronnes du scan (étape 2 du recalage)")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        teeth = fuse(scan)
        scan.hide_set(True)   # set aside: the teeth now carry its crowns
        props = teeth.omfs_fusion
        self.report({'INFO'}, f"{teeth.name} : {props.crown_count} sommets de couronne sur le scan, "
                              f"écart moyen {props.mean:.3f} mm, 95 % sous {props.p95:.3f} mm")
        return {'FINISHED'}


class OMFS_OT_restore_cbct_teeth(bpy.types.Operator):
    """Remettre la géométrie CBCT d'origine des dents (annule la modification des couronnes)"""
    bl_idname = "omfs.restore_cbct_teeth"
    bl_label = "Dents CBCT d'origine"
    bl_options = {'REGISTER', 'UNDO'}

    teeth_name: bpy.props.StringProperty()

    def execute(self, context):
        teeth = context.scene.objects.get(self.teeth_name)
        if not is_fused(teeth):
            return {'CANCELLED'}
        restore(teeth)
        return {'FINISHED'}


class OMFS_OT_toggle_scan(bpy.types.Operator):
    """Montrer ou cacher le scan mis de côté"""
    bl_idname = "omfs.toggle_scan"
    bl_label = "Scan"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = context.scene.objects.get(self.scan_name)
        if scan is None:
            return {'CANCELLED'}
        scan.hide_set(not scan.hide_get())
        return {'FINISHED'}


class OMFS_FusionProperties(bpy.types.PropertyGroup):
    """Stored on the CBCT teeth object."""
    original: bpy.props.PointerProperty(type=bpy.types.Mesh, description="CBCT teeth mesh, untouched")
    scan: bpy.props.PointerProperty(type=bpy.types.Object, description="Scan whose crowns were taken")
    crown_count: bpy.props.IntProperty()
    mean: bpy.props.FloatProperty(default=-1.0)
    p95: bpy.props.FloatProperty(default=-1.0)
    max: bpy.props.FloatProperty(default=-1.0)


classes = (OMFS_FusionProperties, OMFS_OT_fuse_crowns, OMFS_OT_restore_cbct_teeth, OMFS_OT_toggle_scan)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_fusion = bpy.props.PointerProperty(type=OMFS_FusionProperties)


def unregister():
    del bpy.types.Object.omfs_fusion
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
