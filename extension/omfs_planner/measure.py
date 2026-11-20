"""Section & Analysis: point-to-point measurements in the viewport, usable with or without the section.

Click a point on a surface and drag to a second point (or click, release, click the second point):
the straight 3D distance between the two points, in mm, is drawn on a line between them. Measurements
stay in the scene (two world positions each, nothing added to any mesh) until deleted, are drawn by a
viewport draw handler, and are included in the PNG snapshot. With the slab on, clicks go through the
part of the scene the slab hides. Meshmixer: Analysis > Measure (Distance).
"""

import blf
import bpy
import gpu
from bpy_extras import view3d_utils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import landmarks, params, roles

_preview = None   # (a, b) while a measurement is being drawn
_handles = []


# --- Picking ------------------------------------------------------------------------

def clipped(rv3d, point):
    """True if the view's clipping planes (the section slab) hide this point."""
    if not rv3d.use_clip_planes:
        return False
    return any(Vector(p[:3]).dot(point) + p[3] < 0 for p in rv3d.clip_planes)


def surface_point(context, region, mouse):
    """First visible surface point under the mouse (helpers and the slab's hidden part skipped), or None."""
    rv3d = region.data
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    depsgraph = context.evaluated_depsgraph_get()
    for _ in range(50):
        hit, location, _normal, _index, obj, _matrix = context.scene.ray_cast(depsgraph, origin, direction)
        if not hit:
            return None
        if obj.omfs.role not in roles.NOT_SURFACES and not clipped(rv3d, location):
            return location
        origin = location + direction * 0.01   # 0.01 mm past it
    return None


def add(scene, a, b):
    m = scene.omfs_measures.add()
    m.a, m.b = a, b
    return m


def distance(m):
    return (Vector(m.b) - Vector(m.a)).length


# --- Drawing ------------------------------------------------------------------------

def segments(scene):
    found = [(Vector(m.a), Vector(m.b)) for m in scene.omfs_measures] if scene.omfs_measure_show else []
    if _preview is not None:
        found.append(_preview)
    return found


def _draw_3d():
    context = bpy.context
    found = segments(context.scene)
    region = context.region
    if not found or region is None:
        return
    gpu.state.depth_test_set('NONE')   # always on top: a measurement is never hidden by the anatomy
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", params.MEASURE_LINE_WIDTH)
    shader.uniform_float("color", params.MEASURE_COLOUR)
    batch_for_shader(shader, 'LINES', {"pos": [p for seg in found for p in seg]}).draw(shader)
    points = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    gpu.state.point_size_set(params.MEASURE_POINT_SIZE * context.preferences.system.ui_scale)
    points.uniform_float("color", params.MEASURE_COLOUR)
    batch_for_shader(points, 'POINTS', {"pos": [p for seg in found for p in seg]}).draw(points)
    gpu.state.point_size_set(1.0)


def _draw_labels():
    context = bpy.context
    found = segments(context.scene)
    region, rv3d = context.region, context.region_data
    if not found or rv3d is None:
        return
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, 18 * scale)
    blf.color(font, *params.MEASURE_COLOUR)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
    for a, b in found:
        middle = location_3d_to_region_2d(region, rv3d, (a + b) / 2)
        if middle is None:
            continue
        text = f"{(b - a).length:.2f} mm"
        width, height = blf.dimensions(font, text)
        blf.position(font, middle.x - width / 2, middle.y + 8 * scale, 0)
        blf.draw(font, text)
    blf.disable(font, blf.SHADOW)


# --- Operators ----------------------------------------------------------------------

class OMFS_OT_measure(bpy.types.Operator):
    """Mesurer : cliquer un point et glisser jusqu'au second (ou cliquer, puis cliquer le second).
    Distance 3D en mm, visible dans la vue et la capture. Plusieurs mesures de suite ; Échap pour finir"""
    bl_idname = "omfs.measure"
    bl_label = "Mesurer"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        context.scene.omfs_measure_show = True
        self.start = self.start_mouse = None
        self.prompt()
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self, text=None):
        self.area.header_text_set(text or "Mesure : cliquer le 1er point et glisser jusqu'au 2e (ou cliquer les deux)"
                                          "   |   Échap / clic droit : terminer")

    def finish(self, context):
        global _preview
        _preview = None
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        self.area.tag_redraw()

    def modal(self, context, event):
        global _preview
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self.start is not None:   # drop the unfinished one, keep measuring
                self.start, _preview = None, None
                self.prompt()
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self.finish(context)
            return {'FINISHED'}
        under = landmarks.viewport_region_under_mouse(self.area, event)
        if event.type == 'MOUSEMOVE' and self.start is not None and under is not None:
            point = surface_point(context, *under)
            if point is not None:
                _preview = (self.start, point)
                self.prompt(f"{(point - self.start).length:.2f} mm")
                self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or under is None:
            return {'PASS_THROUGH'}   # navigation (middle mouse, wheel) and the sidebar keep working
        point = surface_point(context, *under)
        if event.value == 'PRESS':
            if self.start is None:
                if point is None:
                    self.prompt("Cliquer sur une surface   |   Échap : terminer")
                    return {'RUNNING_MODAL'}
                self.start, self.start_mouse = point, Vector(under[1])
                _preview = (point, point)
                return {'RUNNING_MODAL'}
            return self.end(context, point)
        if event.value == 'RELEASE' and self.start is not None:
            if (Vector(under[1]) - self.start_mouse).length < params.MEASURE_DRAG_PX:
                return {'RUNNING_MODAL'}   # a click, not a drag: wait for the second click
            return self.end(context, point)
        return {'RUNNING_MODAL'}

    def end(self, context, point):
        global _preview
        if point is None and _preview is not None:
            point = _preview[1]            # released over empty space: last point on a surface
        if point is None or (point - self.start).length < 1e-6:
            return {'RUNNING_MODAL'}
        m = add(context.scene, self.start, point)
        self.start, _preview = None, None
        self.prompt(f"{distance(m):.2f} mm   |   mesure suivante, ou Échap : terminer")
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}


class OMFS_OT_delete_measure(bpy.types.Operator):
    """Supprimer cette mesure (toutes si aucune n'est indiquée)"""
    bl_idname = "omfs.delete_measure"
    bl_label = "Supprimer la mesure"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        measures = context.scene.omfs_measures
        if self.index < 0:
            measures.clear()
        elif self.index < len(measures):
            measures.remove(self.index)
        return {'FINISHED'}


class OMFS_Measure(bpy.types.PropertyGroup):
    a: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    b: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')


classes = (OMFS_Measure, OMFS_OT_measure, OMFS_OT_delete_measure)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_measures = bpy.props.CollectionProperty(type=OMFS_Measure)
    bpy.types.Scene.omfs_measure_show = bpy.props.BoolProperty(name="Afficher les mesures", default=True)
    if not bpy.app.background:
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_3d, (), 'WINDOW', 'POST_VIEW'))
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_labels, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    for handle in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _handles.clear()
    del bpy.types.Scene.omfs_measure_show
    del bpy.types.Scene.omfs_measures
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
