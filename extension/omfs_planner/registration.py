"""Register an intraoral scan onto the CBCT teeth: point pairs, then ICP on the crowns.

The CBCT teeth dictate the teeth position and never move; the scan dictates the true crown
shape. Only the scan's transform changes, always rigidly (rotation + translation, scale 1):
its vertices are never touched.

1. Point pairs: click matching points on the scan and the CBCT teeth (3 or more), then a
   least-squares rigid fit (Kabsch) brings the scan close.
2. Crowns: the "Crowns" vertex group on the scan says which scan vertices are used. It is
   suggested automatically (scan vertices close to the CBCT crowns) and can be painted by hand,
   e.g. to drop teeth with metal restorations.
3. ICP (iterative closest point), point-to-plane, on those crowns only, against the crown
   region of the CBCT teeth (outside the bone and away from it, so roots are ignored).
4. Report: mean / RMS distance on the crowns and a colour map.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, locking, params

COLLECTION = "Registration"
CROWNS_GROUP = "Crowns"
DISTANCE_ATTRIBUTE = "Registration distance"
POINT_SIZE = 0.8   # mm, radius of the point-pair spheres

GREY = (0.6, 0.6, 0.6, 1.0)


# --- Maths ---------------------------------------------------------------------------

def rigid_fit(source, target):
    """4x4 rotation + translation that best maps source points onto target points (Kabsch)."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))            # never a mirror image
    rotation = vt.T @ np.diag((1.0, 1.0, d)) @ u.T
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = ct - rotation @ cs
    return m


def as_rigid(matrix):
    """Remove any scale or shear left by floating-point drift: rotation + translation only."""
    return Matrix.LocRotScale(matrix.translation, matrix.to_quaternion(), None)


def point_to_plane_step(points, closest, normals):
    """Small rigid motion (4x4) reducing the point-to-plane distances, linearised."""
    centre = points.mean(axis=0)
    p = points - centre
    a = np.hstack((np.cross(p, normals), normals))
    b = -np.einsum("ij,ij->i", points - closest, normals)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    omega, t = Vector(x[:3]), Vector(x[3:])
    rotation = Matrix.Rotation(omega.length, 4, omega.normalized()) if omega.length > 1e-12 else Matrix()
    return Matrix.Translation(Vector(centre) + t) @ rotation @ Matrix.Translation(-Vector(centre))


# --- Mesh helpers --------------------------------------------------------------------

def vertex_coords(obj, world=True):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if not world:
        return co
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


def nearest(bvh, points, max_distance=None):
    """Closest surface points and normals for each point (NaN where none within max_distance)."""
    closest = np.full((len(points), 3), np.nan)
    normals = np.full((len(points), 3), np.nan)
    for i, p in enumerate(points):
        loc, nor, _index, _dist = (bvh.find_nearest(p, max_distance) if max_distance
                                   else bvh.find_nearest(p))
        if loc is not None:
            closest[i], normals[i] = loc, nor
    return closest, normals


def triangles(mesh):
    mesh.calc_loop_triangles()   # works for any mesh, not only all-triangle STL
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def teeth_bvh(teeth):
    """BVH of the whole CBCT teeth surface, in world coordinates (for measuring the fit)."""
    return BVHTree.FromPolygons(vertex_coords(teeth).tolist(), triangles(teeth.data).tolist())


def crown_region_bvh(teeth, bone):
    """BVH of the CBCT teeth faces outside the bone and > clearance away from it (the crowns)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bone_bvh = BVHTree.FromObject(bone, depsgraph)
    to_bone = np.array(bone.matrix_world.inverted())
    co = vertex_coords(teeth)
    co_bone = co @ to_bone[:3, :3].T + to_bone[:3, 3]
    crown = np.zeros(len(co), dtype=bool)
    for i, p in enumerate(co_bone):
        loc, nor, _index, dist = bone_bvh.find_nearest(p)
        crown[i] = dist > params.REG_CROWN_BONE_CLEARANCE and (p - np.array(loc)) @ np.array(nor) > 0
    polys = triangles(teeth.data)
    keep = crown[polys].all(axis=1)
    return BVHTree.FromPolygons(co.tolist(), polys[keep].tolist())


def crown_indices(scan):
    group = scan.vertex_groups.get(CROWNS_GROUP)
    if group is None:
        return np.array([], dtype=np.int64)
    gi = group.index
    return np.array([v.index for v in scan.data.vertices
                     if any(g.group == gi and g.weight > 0.5 for g in v.groups)], dtype=np.int64)


# --- Objects -------------------------------------------------------------------------

def registration_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def point_pairs(scene, scan):
    """[(scan point Empty, CBCT point Empty)], in order. Scan points are children of the scan."""
    points = {o.omfs.landmark: o for o in registration_points(scene, scan)}
    pairs, i = [], 1
    while f"REG_S{i}" in points or f"REG_T{i}" in points:
        if f"REG_S{i}" in points and f"REG_T{i}" in points:
            pairs.append((points[f"REG_S{i}"], points[f"REG_T{i}"]))
        i += 1
    return pairs


def registration_points(scene, scan):
    return [o for o in scene.objects if o.omfs.made_by == scan.name and o.omfs.landmark.startswith("REG_")]


def points_visible(scene, scan):
    return any(not o.hide_get() for o in registration_points(scene, scan))


def set_points_hidden(scene, scan, hide):
    for obj in registration_points(scene, scan):
        obj.hide_set(hide)


def add_point(scene, scan, point_id, name, location, parent=None):
    obj = bpy.data.objects.new(name, None)
    registration_collection(scene).objects.link(obj)
    obj.empty_display_type = 'SPHERE'
    obj.empty_display_size = POINT_SIZE
    obj.show_name = True
    obj.show_in_front = True
    obj.omfs.role = 'MARKER'
    obj.omfs.landmark = point_id
    obj.omfs.made_by = scan.name
    obj.matrix_world = Matrix.Translation(location)
    if parent is not None:
        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()
    locking.set_locked(obj, True)
    return obj


def next_pair_number(scene, scan):
    used = [int(o.omfs.landmark[5:]) for o in scene.objects
            if o.omfs.made_by == scan.name and o.omfs.landmark.startswith("REG_")]
    return max(used, default=0) + 1


def move_scan(scan, matrix):
    """Set the scan's transform (rigid only), whatever its lock state."""
    locked = locking.is_locked(scan)
    locking.set_locked(scan, False)
    scan.matrix_world = as_rigid(matrix)
    locking.set_locked(scan, locked)
    bpy.context.view_layer.update()   # so the scan's point-pair children report their new position


def align_points(scene, scan):
    """Rigid fit of the scan points onto the CBCT points. Returns the RMS residual in mm."""
    pairs = point_pairs(scene, scan)
    src = np.array([s.matrix_world.translation for s, _t in pairs])
    dst = np.array([t.matrix_world.translation for _s, t in pairs])
    fit = Matrix(rigid_fit(src, dst).tolist())
    move_scan(scan, fit @ scan.matrix_world)
    moved = np.array([s.matrix_world.translation for s, _t in pairs])
    return float(np.sqrt(((moved - dst) ** 2).sum(axis=1).mean()))


def suggest_crowns(scan, target_bvh):
    """Fill the Crowns group with scan vertices close to the CBCT crowns. Returns the count."""
    closest, _normals = nearest(target_bvh, vertex_coords(scan), params.REG_SUGGEST_DISTANCE)
    indices = np.flatnonzero(~np.isnan(closest[:, 0])).tolist()
    group = scan.vertex_groups.get(CROWNS_GROUP) or scan.vertex_groups.new(name=CROWNS_GROUP)
    group.remove(list(range(len(scan.data.vertices))))
    group.add(indices, 1.0, 'REPLACE')
    return len(indices)


def icp(scan, target_bvh, indices):
    """Point-to-plane ICP of the scan crowns onto the CBCT crowns. Returns iterations used."""
    rng = np.random.default_rng(0)   # same sample every time: reproducible result
    sample = indices if len(indices) <= params.ICP_SAMPLES else rng.choice(indices, params.ICP_SAMPLES, replace=False)
    local = vertex_coords(scan, world=False)[sample]
    matrix = scan.matrix_world.copy()
    for iteration in range(1, params.ICP_MAX_ITERATIONS + 1):
        m = np.array(matrix)
        points = local @ m[:3, :3].T + m[:3, 3]
        closest, normals = nearest(target_bvh, points, params.ICP_MAX_PAIR_DISTANCE)
        found = ~np.isnan(closest[:, 0])
        dist = np.linalg.norm(points[found] - closest[found], axis=1)
        if len(dist) < 10:
            break
        keep = dist <= max(3 * np.median(dist), 0.2)   # drop outliers (gingiva, artefacts)
        step = point_to_plane_step(points[found][keep], closest[found][keep], normals[found][keep])
        matrix = step @ matrix
        angle = step.to_quaternion().angle
        if step.translation.length < 1e-4 and angle < 1e-5:
            break
    move_scan(scan, matrix)
    return iteration


def crown_distances(scan, target_bvh, indices):
    """Unsigned distance (mm) from each crown vertex to the CBCT teeth.

    Pass the whole-teeth BVH: against the crown region only, scan vertices at its border
    would measure to the cut edge and look worse than they are.
    """
    points = vertex_coords(scan)[indices]
    closest, _normals = nearest(target_bvh, points)
    return np.linalg.norm(points - closest, axis=1)


def distance_colour(d):
    """Green up to the target error, yellow at the warning threshold, red at twice that."""
    target, warn = params.REG_TARGET_ERROR, params.REG_WARN_ERROR
    if d <= target:
        return (0.1, 0.8, 0.2, 1.0)
    if d <= warn:
        f = (d - target) / (warn - target)
        return (0.1 + 0.9 * f, 0.8 + 0.1 * f, 0.2 - 0.2 * f, 1.0)
    f = min((d - warn) / warn, 1.0)
    return (1.0, 0.9 - 0.8 * f, 0.0, 1.0)


def write_distance_map(scan, indices, distances):
    mesh = scan.data
    attr = mesh.color_attributes.get(DISTANCE_ATTRIBUTE) or mesh.color_attributes.new(
        DISTANCE_ATTRIBUTE, 'FLOAT_COLOR', 'POINT')
    colours = np.tile(GREY, (len(mesh.vertices), 1))
    colours[indices] = [distance_colour(d) for d in distances]
    attr.data.foreach_set("color", colours.ravel())
    mesh.color_attributes.active_color = attr


# --- Properties and operators ---------------------------------------------------------

def _is_role(role):
    return lambda _props, obj: obj.type == 'MESH' and obj.omfs.role == role and not obj.omfs.made_by


class OMFS_RegistrationProperties(bpy.types.PropertyGroup):
    """Stored on the scan object."""
    target: bpy.props.PointerProperty(name="CBCT teeth", type=bpy.types.Object, poll=_is_role('TEETH'))
    bone: bpy.props.PointerProperty(name="Bone", type=bpy.types.Object, poll=_is_role('BONE'),
                                    description="Bone around these teeth: crowns are what sticks out of it")
    points_rms: bpy.props.FloatProperty(default=-1.0)
    mean: bpy.props.FloatProperty(default=-1.0)
    rms: bpy.props.FloatProperty(default=-1.0)
    p95: bpy.props.FloatProperty(default=-1.0)
    crown_count: bpy.props.IntProperty()


def default_partners(scene, scan):
    """CBCT teeth and bone whose bounding boxes overlap the most with each other."""
    def bbox(obj):
        pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        return Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])

    def overlap(a, b):
        (alo, ahi), (blo, bhi) = bbox(a), bbox(b)
        size = [max(0.0, min(ahi[i], bhi[i]) - max(alo[i], blo[i])) for i in range(3)]
        return size[0] * size[1] * size[2]

    teeth = [o for o in scene.objects if _is_role('TEETH')(None, o)]
    bones = [o for o in scene.objects if _is_role('BONE')(None, o)]
    props = scan.omfs_reg
    if props.target is None and teeth:
        props.target = teeth[0]
    if props.bone is None and bones and props.target is not None:
        props.bone = max(bones, key=lambda b: overlap(b, props.target))


def scan_for(context, scan_name):
    scan = context.scene.objects.get(scan_name)
    if scan is None or scan.omfs.role != 'SCAN':
        return None
    default_partners(context.scene, scan)
    return scan


_target_cache = {}


def target_bvh(scan):
    """Crown-region BVH of the CBCT teeth, built once per (teeth, bone) pair."""
    props = scan.omfs_reg
    key = (props.target.name, props.bone.name, tuple(map(tuple, props.target.matrix_world)))
    if key not in _target_cache:
        _target_cache.clear()
        _target_cache[key] = crown_region_bvh(props.target, props.bone)
    return _target_cache[key]


class OMFS_OT_add_point_pair(bpy.types.Operator):
    """Click a point on the scan, then the same point on the CBCT teeth (Esc / right-click: cancel)"""
    bl_idname = "omfs.add_point_pair"
    bl_label = "Add Point Pair"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def invoke(self, context, event):
        self.scan = scan_for(context, self.scan_name)
        if self.scan is None or self.scan.omfs_reg.target is None:
            self.report({'ERROR'}, "Choose the scan and the CBCT teeth first")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        self.number = next_pair_number(context.scene, self.scan)
        self.scan_point = None
        self.prompt()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self):
        which = self.scan.name if self.scan_point is None else self.scan.omfs_reg.target.name
        self.area.header_text_set(f"Pair {self.number}: click the point on {which}   |   Esc / right-click: cancel")

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self.scan_point is not None:
                bpy.data.objects.remove(self.scan_point)
            self.finish(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
            if under_mouse is None:
                return {'PASS_THROUGH'}
            target = self.scan if self.scan_point is None else self.scan.omfs_reg.target
            location = landmarks.object_hit(context, *under_mouse, target)
            if location is None:
                self.report({'WARNING'}, f"Click on {target.name}")
                return {'RUNNING_MODAL'}
            if self.scan_point is None:
                self.scan_point = add_point(context.scene, self.scan, f"REG_S{self.number}",
                                            f"Scan pt {self.number}", location, parent=self.scan)
                self.prompt()
                return {'RUNNING_MODAL'}
            add_point(context.scene, self.scan, f"REG_T{self.number}", f"CBCT pt {self.number}", location)
            self.finish(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class OMFS_OT_delete_point_pair(bpy.types.Operator):
    """Delete this point pair"""
    bl_idname = "omfs.delete_point_pair"
    bl_label = "Delete Point Pair"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()
    number: bpy.props.IntProperty()

    def execute(self, context):
        for obj in [o for o in context.scene.objects if o.omfs.made_by == self.scan_name
                    and o.omfs.landmark in (f"REG_S{self.number}", f"REG_T{self.number}")]:
            bpy.data.objects.remove(obj)
        return {'FINISHED'}


class OMFS_OT_align_points(bpy.types.Operator):
    """Move the scan so its points best match the CBCT points (rigid, least squares)"""
    bl_idname = "omfs.align_points"
    bl_label = "Align by Points"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if scan is None or len(point_pairs(context.scene, scan)) < 3:
            self.report({'ERROR'}, "Place at least 3 point pairs")
            return {'CANCELLED'}
        scan.omfs_reg.points_rms = align_points(context.scene, scan)
        self.report({'INFO'}, f"Points aligned, RMS {scan.omfs_reg.points_rms:.2f} mm")
        return {'FINISHED'}


class OMFS_OT_suggest_crowns(bpy.types.Operator):
    """Select the scan vertices lying on the CBCT crowns (gingiva excluded). Edit with Paint Crowns"""
    bl_idname = "omfs.suggest_crowns"
    bl_label = "Suggest Crowns"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if scan is None or scan.omfs_reg.target is None or scan.omfs_reg.bone is None:
            self.report({'ERROR'}, "Choose the CBCT teeth and bone first")
            return {'CANCELLED'}
        count = suggest_crowns(scan, target_bvh(scan))
        scan.omfs_reg.crown_count = count
        if count == 0:
            self.report({'WARNING'}, "No scan vertex near the CBCT crowns: align by points first")
        return {'FINISHED'}


class OMFS_OT_paint_crowns(bpy.types.Operator):
    """Paint the crowns used for registration on the scan (red = used, blue = ignored)"""
    bl_idname = "omfs.paint_crowns"
    bl_label = "Paint Crowns"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()
    include: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if scan is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT' and context.active_object != scan:
            bpy.ops.object.mode_set(mode='OBJECT')
        locking.set_locked(scan, False)
        group = scan.vertex_groups.get(CROWNS_GROUP) or scan.vertex_groups.new(name=CROWNS_GROUP)
        scan.vertex_groups.active_index = group.index
        context.view_layer.objects.active = scan
        scan.select_set(True)
        if context.mode != 'PAINT_WEIGHT':
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        settings = context.tool_settings.weight_paint.unified_paint_settings   # Blender 5.x location
        settings.use_unified_weight = True
        settings.weight = 1.0 if self.include else 0.0
        return {'FINISHED'}


class OMFS_OT_finish_painting(bpy.types.Operator):
    """Leave paint mode"""
    bl_idname = "omfs.finish_painting"
    bl_label = "Done Painting"

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if scan is not None:
            scan.omfs_reg.crown_count = len(crown_indices(scan))
            scan.select_set(False)
        return {'FINISHED'}


class OMFS_OT_refine_icp(bpy.types.Operator):
    """Refine the scan position on the crowns (ICP, rigid) and measure the fit"""
    bl_idname = "omfs.refine_icp"
    bl_label = "Refine (ICP)"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        indices = crown_indices(scan) if scan is not None else []
        if len(indices) < 100:
            self.report({'ERROR'}, "Select the crowns first (Suggest Crowns)")
            return {'CANCELLED'}
        iterations = icp(scan, target_bvh(scan), indices)
        distances = crown_distances(scan, teeth_bvh(scan.omfs_reg.target), indices)
        props = scan.omfs_reg
        props.mean = float(distances.mean())
        props.rms = float(np.sqrt((distances ** 2).mean()))
        props.p95 = float(np.percentile(distances, 95))
        props.crown_count = len(indices)
        write_distance_map(scan, indices, distances)
        locking.set_locked(scan, True)
        set_points_hidden(context.scene, scan, True)   # registration done: hide the point pairs
        self.report({'INFO'}, f"ICP: {iterations} iterations, mean {props.mean:.2f} mm, RMS {props.rms:.2f} mm")
        return {'FINISHED'}


class OMFS_OT_reset_scan(bpy.types.Operator):
    """Put the scan back at its file coordinates (undo the registration)"""
    bl_idname = "omfs.reset_scan"
    bl_label = "Reset Scan Position"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if scan is not None:
            home = scan.parent.matrix_world @ scan.matrix_parent_inverse if scan.parent else Matrix()
            move_scan(scan, home)   # file coordinates, carried by its occlusion block if it has one
            scan.omfs_reg.points_rms = scan.omfs_reg.mean = scan.omfs_reg.rms = scan.omfs_reg.p95 = -1.0
        return {'FINISHED'}


class OMFS_OT_toggle_point_pairs(bpy.types.Operator):
    """Show or hide this scan's point pairs"""
    bl_idname = "omfs.toggle_point_pairs"
    bl_label = "Toggle Point Pairs"
    bl_options = {'REGISTER', 'UNDO'}

    scan_name: bpy.props.StringProperty()

    def execute(self, context):
        scan = scan_for(context, self.scan_name)
        if scan is None:
            return {'CANCELLED'}
        set_points_hidden(context.scene, scan, points_visible(context.scene, scan))
        return {'FINISHED'}


class OMFS_OT_toggle_distance_map(bpy.types.Operator):
    """Show the registration distance colours instead of the role colours"""
    bl_idname = "omfs.toggle_distance_map"
    bl_label = "Distance Map"

    def execute(self, context):
        shading = context.space_data.shading
        shading.color_type = 'OBJECT' if shading.color_type == 'VERTEX' else 'VERTEX'
        return {'FINISHED'}


classes = (OMFS_RegistrationProperties, OMFS_OT_add_point_pair, OMFS_OT_delete_point_pair,
           OMFS_OT_align_points, OMFS_OT_suggest_crowns, OMFS_OT_paint_crowns, OMFS_OT_finish_painting,
           OMFS_OT_refine_icp, OMFS_OT_reset_scan, OMFS_OT_toggle_point_pairs, OMFS_OT_toggle_distance_map)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_reg = bpy.props.PointerProperty(type=OMFS_RegistrationProperties)


def unregister():
    del bpy.types.Object.omfs_reg
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
