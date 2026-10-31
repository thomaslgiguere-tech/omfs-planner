"""Object creator: Splint / Chape. Intermediate or final occlusal splint, or a single-arch cap.

Built in steps (docs/workflow.md, "Object creator"):
1. Arch curve: points clicked on the arches, joined by a smooth curve (this module, done).
2. Parametric blank swept along the curve (to come).
3. Tooth imprints by projection onto the arches, 0.10 mm clearance (to come).
4. Engraved text and print checks (to come).

The arch curve is a Bezier curve object (role ARCH) whose points are the clicks, with automatic
handles so the curve passes through every point. For a splint between two arches, each click is
moved halfway between the upper and lower occlusal surfaces at that spot, so the curve runs
through the middle of the bite. The scans are only read, never modified.
"""

import blf
import bpy
import gpu
from bpy_extras import view3d_utils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import landmarks, locking, reference_frame

COLLECTION = "Splint"
CURVE_NAME = "Arch curve"
KINDS = (
    ('INTERMEDIATE', "Intermédiaire", "Gouttière entre l'arcade du haut déplacée et l'arcade du bas préop"),
    ('FINAL', "Finale", "Gouttière entre les deux arcades en occlusion finale"),
    ('CHAPE', "Chape", "Coiffe sur une seule arcade"),
)
CURVE_COLOUR = (1.0, 0.55, 0.1, 1.0)   # orange
CENTRE_REACH = 15.0                    # mm: how far up / down to look for the other arch's surface


# --- Arches and axes --------------------------------------------------------------------

def is_scan(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'SCAN'


def upper_arch(scene):
    """The upper scan: chosen here, else the one chosen for the occlusion."""
    chosen = scene.omfs_splint.upper
    return chosen if is_scan(chosen) else (scene.omfs_occlusion.upper if is_scan(scene.omfs_occlusion.upper) else None)


def lower_arch(scene):
    chosen = scene.omfs_splint.lower
    return chosen if is_scan(chosen) else (scene.omfs_occlusion.lower if is_scan(scene.omfs_occlusion.lower) else None)


def arches(scene):
    """The arch meshes the curve is drawn on: both arches for a splint, one for a cap."""
    props = scene.omfs_splint
    if props.kind == 'CHAPE':
        return [o for o in (props.cap_arch,) if is_scan(o)]
    return [o for o in (upper_arch(scene), lower_arch(scene)) if o is not None]


def up_axis(scene):
    """Occlusal "up": the reference frame's, else the scanner's Z."""
    frame = reference_frame.find_frame(scene)
    return frame.matrix_world.col[2].xyz.normalized() if frame else Vector((0, 0, 1))


def ray_hit(obj, origin, direction, depsgraph):
    """First hit of a world-space ray on one object (world location), or None."""
    to_local = obj.matrix_world.inverted()
    hit, location, _normal, _index = obj.ray_cast(to_local @ origin, to_local.to_3x3() @ direction,
                                                  depsgraph=depsgraph)
    return obj.matrix_world @ location if hit else None


def centre_between_arches(scene, point, depsgraph):
    """The point halfway between the lower arch's occlusal surface (seen from above) and the upper
    arch's (seen from below), on the vertical through the point. The point itself if either is missed."""
    up = up_axis(scene)
    lower_scan, upper_scan = lower_arch(scene), upper_arch(scene)
    lower = ray_hit(lower_scan, point + up * CENTRE_REACH, -up, depsgraph) if lower_scan else None
    upper = ray_hit(upper_scan, point - up * CENTRE_REACH, up, depsgraph) if upper_scan else None
    if lower is None or upper is None or (upper - lower).dot(up) < 0:
        return point
    return (lower + upper) / 2


# --- Arch curve ---------------------------------------------------------------------------

def splint_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def find_curve(scene):
    return next((o for o in scene.objects if o.type == 'CURVE' and o.omfs.role == 'ARCH'), None)


def ensure_curve(scene):
    obj = find_curve(scene)
    if obj is not None:
        return obj
    curve = bpy.data.curves.new(CURVE_NAME, 'CURVE')
    curve.dimensions = '3D'
    obj = bpy.data.objects.new(CURVE_NAME, curve)
    splint_collection(scene).objects.link(obj)
    obj.omfs.role = 'ARCH'
    obj.show_in_front = True
    return obj


def curve_points(obj):
    """World positions of the clicked points, in order."""
    if obj is None or not obj.data.splines:
        return []
    return [obj.matrix_world @ p.co for p in obj.data.splines[0].bezier_points]


def set_points(obj, points):
    """Rebuild the curve through these world points (automatic handles: passes through each)."""
    curve = obj.data
    curve.splines.clear()
    if not points:
        return
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    to_local = obj.matrix_world.inverted()
    for bp, p in zip(spline.bezier_points, points):
        bp.co = to_local @ p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    spline.resolution_u = 24


def add_point(scene, location, depsgraph):
    """Append a point (centred between the arches for a splint). Returns the point placed."""
    props = scene.omfs_splint
    if props.kind != 'CHAPE' and props.centre:
        location = centre_between_arches(scene, location, depsgraph)
    obj = ensure_curve(scene)
    set_points(obj, curve_points(obj) + [location])
    return location


def remove_last_point(scene):
    obj = find_curve(scene)
    if obj is not None:
        set_points(obj, curve_points(obj)[:-1])


def sampled_curve(obj, depsgraph):
    """World points along the evaluated curve (the smooth line, not only the clicks)."""
    if obj is None or not obj.data.splines:
        return []
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = [evaluated.matrix_world @ v.co for v in mesh.vertices]
    evaluated.to_mesh_clear()
    return points


def curve_length(obj):
    if obj is None or not obj.data.splines or len(obj.data.splines[0].bezier_points) < 2:
        return 0.0
    return obj.data.splines[0].calc_length()


# --- Properties and operators ---------------------------------------------------------

class OMFS_SplintProperties(bpy.types.PropertyGroup):
    kind: bpy.props.EnumProperty(name="Type", items=KINDS, default='FINAL')
    upper: bpy.props.PointerProperty(name="Arcade du haut", type=bpy.types.Object,
                                     poll=lambda _p, o: is_scan(o), description="Scan de l'arcade du haut (par défaut: celle de l'assise de l'occlusion)")
    lower: bpy.props.PointerProperty(name="Arcade du bas", type=bpy.types.Object,
                                     poll=lambda _p, o: is_scan(o), description="Scan de l'arcade du bas (par défaut: celle de l'assise de l'occlusion)")
    cap_arch: bpy.props.PointerProperty(name="Arcade", type=bpy.types.Object, poll=lambda _p, o: is_scan(o),
                                        description="Scan de l'arcade que la chape recouvre")
    centre: bpy.props.BoolProperty(
        name="Centrer entre les arcades", default=True,
        description="Chaque point cliqué est placé à mi-hauteur entre les faces occlusales du haut et du bas")


class OMFS_OT_arch_curve_click(bpy.types.Operator):
    """Click points along the arch, from the last molar on one side to the last molar on the other.
    Right-click, Enter or Esc to finish"""
    bl_idname = "omfs.arch_curve_click"
    bl_label = "Cliquer les points"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if not arches(context.scene):
            self.report({'ERROR'}, "Choisissez d'abord la ou les arcades (scans)")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.header(context)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def header(self, context):
        n = len(curve_points(find_curve(context.scene)))
        self.area.header_text_set(f"Cliquez sur les faces occlusales, d'une dernière molaire à l'autre ({n} points)"
                                  "   |   Retour arrière: retirer le dernier   |   Clic droit / Entrée / Échap: terminer")

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER', 'ESC'}:
            self.finish(context)
            return {'FINISHED'}
        if event.value == 'PRESS' and event.type == 'BACK_SPACE':
            remove_last_point(context.scene)
            self.header(context)
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        region, mouse = under_mouse
        origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
        direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
        depsgraph = context.evaluated_depsgraph_get()
        hits = [h for h in (ray_hit(o, origin, direction, depsgraph) for o in arches(context.scene)) if h]
        if not hits:
            self.report({'WARNING'}, "Cliquez sur une arcade")
            return {'RUNNING_MODAL'}
        add_point(context.scene, min(hits, key=lambda h: (h - origin).length), depsgraph)
        self.header(context)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}


class OMFS_OT_arch_curve_undo(bpy.types.Operator):
    """Remove the last point of the arch curve"""
    bl_idname = "omfs.arch_curve_undo"
    bl_label = "Retirer le dernier point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        remove_last_point(context.scene)
        return {'FINISHED'}


class OMFS_OT_arch_curve_clear(bpy.types.Operator):
    """Delete the arch curve to start again"""
    bl_idname = "omfs.arch_curve_clear"
    bl_label = "Effacer la courbe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = find_curve(context.scene)
        if obj is not None:
            curve = obj.data
            bpy.data.objects.remove(obj)
            if curve.users == 0:
                bpy.data.curves.remove(curve)
        return {'FINISHED'}


class OMFS_OT_arch_curve_edit(bpy.types.Operator):
    """Edit the curve's points: G to move (snapped onto the surfaces), X to delete,
    Ctrl + right-click to add one. Tab to finish"""
    bl_idname = "omfs.arch_curve_edit"
    bl_label = "Modifier les points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = find_curve(context.scene)
        if obj is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        locking.set_locked(obj, False)
        for other in context.selected_objects:
            other.select_set(False)
        context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        ts = context.scene.tool_settings
        ts.use_snap = True
        ts.snap_elements = {'FACE_PROJECT'}
        return {'FINISHED'}


# --- Drawing --------------------------------------------------------------------------------

def _draw_curve():
    """The arch curve, thick and on top, so it reads over the teeth."""
    context = bpy.context
    obj = find_curve(context.scene)
    if obj is None or not obj.visible_get():
        return
    line = sampled_curve(obj, context.evaluated_depsgraph_get())
    points = curve_points(obj)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    if len(line) > 1:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("viewportSize", (context.region.width, context.region.height))
        shader.uniform_float("lineWidth", 4.0)
        shader.uniform_float("color", CURVE_COLOUR)
        batch_for_shader(shader, 'LINE_STRIP', {"pos": line}).draw(shader)
    if points:
        dots = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
        gpu.state.point_size_set(11.0)
        dots.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_for_shader(dots, 'POINTS', {"pos": points}).draw(dots)
        gpu.state.point_size_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_numbers():
    context = bpy.context
    obj = find_curve(context.scene)
    if obj is None or not obj.visible_get():
        return
    font = 0
    blf.size(font, 13 * context.preferences.system.ui_scale)
    blf.color(font, *CURVE_COLOUR)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
    for i, p in enumerate(curve_points(obj), start=1):
        xy = location_3d_to_region_2d(context.region, context.region_data, p)
        if xy is not None:
            blf.position(font, xy.x + 8, xy.y + 8, 0)
            blf.draw(font, str(i))
    blf.disable(font, blf.SHADOW)


_draw_handles = []

classes = (OMFS_SplintProperties, OMFS_OT_arch_curve_click, OMFS_OT_arch_curve_undo,
           OMFS_OT_arch_curve_clear, OMFS_OT_arch_curve_edit)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_splint = bpy.props.PointerProperty(type=OMFS_SplintProperties)
    if not bpy.app.background:
        _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_curve, (), 'WINDOW', 'POST_VIEW'))
        _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_numbers, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    for handle in _draw_handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _draw_handles.clear()
    del bpy.types.Scene.omfs_splint
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
