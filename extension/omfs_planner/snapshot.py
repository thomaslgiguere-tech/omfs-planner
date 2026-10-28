"""Section & Analysis: caption, scale bar and PNG snapshot of the section view.

While the section tool is on, a caption (plane, position, slab thickness) and, in an orthographic
view, a scale bar are drawn in the corner of the viewport, like the legend of a radiograph.
Save Snapshot draws the viewport again into an offscreen buffer (GPUOffScreen): the scene as the
viewport shows it, slab clipping included, without Blender's interface (toolbar, sidebar, view
name, gizmos). Our own layers (contours, caps, overlap, distance marker, caption) are drawn on
top with the same view matrices, and the pixels are written as a PNG to a folder chosen outside
the repository (patient images must never land in git).
"""

import time
from pathlib import Path

import blf
import bpy
import gpu
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from . import distance_map, reference_frame, section, section_contours

NICE_LENGTHS = (0.5, 1, 2, 5, 10, 20, 50, 100)   # mm
BAR_TARGET_PX = 120
TEXT_COLOUR = (1.0, 1.0, 1.0, 1.0)
MARGIN = 20   # px from the visible edges

_snapshotting = False
# Blender overlays hidden in a snapshot
HIDDEN = ('show_floor', 'show_ortho_grid', 'show_axis_x', 'show_axis_y', 'show_axis_z', 'show_cursor',
          'show_object_origins', 'show_outline_selected')


def plane_label(scene):
    props = scene.omfs_section
    if props.orientation == 'CUSTOM':
        return f"Plane of {props.custom.name}" if props.custom else "Custom plane"
    name = {'AXIAL': "Axial", 'CORONAL': "Coronal", 'SAGITTAL': "Sagittal"}[props.orientation]
    frame = props.use_frame and reference_frame.find_frame(scene)
    return f"{name} ({'reference frame' if frame else 'scanner axes'})"


def caption(scene):
    props = scene.omfs_section
    return [plane_label(scene),
            f"Position {props.offset:+.1f} mm   Slab {props.thickness:.1f} mm"]


def bar_length(px_per_mm):
    """A round length (mm) whose bar is close to BAR_TARGET_PX on screen."""
    return min(NICE_LENGTHS, key=lambda mm: abs(mm * px_per_mm - BAR_TARGET_PX))


def px_per_mm(region, rv3d):
    """Screen pixels per millimetre in an orthographic view (None in perspective)."""
    if rv3d.is_perspective:
        return None
    centre = rv3d.view_location
    right = rv3d.view_rotation @ Vector((1, 0, 0))   # screen right, in world space
    a = location_3d_to_region_2d(region, rv3d, centre)
    b = location_3d_to_region_2d(region, rv3d, centre + right * 10.0)
    return (b - a).length / 10.0 if a is not None and b is not None else None


def visible_rect(area, region):
    """(left, bottom, right, top) of the window region not covered by the toolbar, sidebar or headers
    (region coordinates). With Region Overlap, those regions are drawn over the viewport."""
    left, bottom, right, top = 0, 0, region.width, region.height
    for other in area.regions:
        if other.width <= 1 or other.height <= 1 or other.type == 'WINDOW':
            continue
        if other.type == 'TOOLS' and other.x <= region.x + 1:
            left = max(left, other.x + other.width - region.x)
        elif other.type == 'UI' and other.x > region.x:
            right = min(right, other.x - region.x)
        elif other.type in {'HEADER', 'TOOL_HEADER'} and other.y > region.y:
            top = min(top, other.y - region.y)
    return left, bottom, right, top


def _draw_caption():
    context = bpy.context
    scene = context.scene
    props = scene.omfs_section
    if section.active_plane(scene) is None or not (props.show_caption or _snapshotting):
        return
    region, rv3d = context.region, context.region_data
    left = 0 if _snapshotting else visible_rect(context.area, region)[0]   # a snapshot has no toolbar
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, 15 * scale)
    blf.color(font, *TEXT_COLOUR)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
    x, y = left + MARGIN, MARGIN
    ppm = px_per_mm(region, rv3d)
    if ppm:
        mm = bar_length(ppm)
        length = mm * ppm
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", 3.0)
        shader.uniform_float("color", TEXT_COLOUR)
        tick = 6 * scale
        lines = [(x, y), (x + length, y), (x, y - tick / 2), (x, y + tick), (x + length, y - tick / 2),
                 (x + length, y + tick)]
        batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
        blf.position(font, x + length + 8 * scale, y - 5 * scale, 0)
        blf.draw(font, f"{mm:g} mm")
        y += 18 * scale
    else:
        blf.position(font, x, y, 0)
        blf.draw(font, "Perspective: no scale (Align View for a true-scale view)")
        y += 22 * scale
    for line in reversed(caption(scene)):
        blf.position(font, x, y, 0)
        blf.draw(font, line)
        y += 20 * scale
    blf.disable(font, blf.SHADOW)


_draw_handle = None


# --- Saving --------------------------------------------------------------------------

def repository_root():
    """The git repository this extension is developed in, or None once installed normally."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def folder_problem(folder):
    """Why this folder can't receive snapshots, or None."""
    if not folder:
        return "Choose a folder first"
    path = Path(bpy.path.abspath(folder)).resolve()
    if not path.is_dir():
        return f"Not a folder: {path}"
    repo = repository_root()
    if repo is not None and (path == repo or repo in path.parents):
        return "Inside the git repository: choose a folder outside it (patient images)"
    return None


def snapshot_name(scene):
    props = scene.omfs_section
    kind = props.orientation.lower()
    return f"section_{kind}_{props.offset:+.1f}mm_{props.thickness:.1f}mm_{time.strftime('%Y%m%d-%H%M%S')}.png"


def pixel_projection(width, height):
    """Projection for drawing in pixels (x right, y up) in a width x height buffer."""
    return Matrix(((2 / width, 0, 0, -1), (0, 2 / height, 0, -1), (0, 0, 1, 0), (0, 0, 0, 1)))


def render_view(context, area, region):
    """The view as RGBA floats (height, width, 4), rows from the bottom, with our layers on top."""
    space = area.spaces.active
    rv3d = space.region_3d
    width, height = region.width, region.height
    offscreen = gpu.types.GPUOffScreen(width, height)
    try:
        offscreen.draw_view3d(context.scene, context.view_layer, space, region,
                              rv3d.view_matrix, rv3d.window_matrix, do_color_management=True)
        with offscreen.bind(), context.temp_override(area=area, region=region):
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(rv3d.view_matrix)
                gpu.matrix.load_projection_matrix(rv3d.window_matrix)
                section_contours._draw_contours()
                distance_map._draw_marker_3d()
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(pixel_projection(width, height))
                distance_map._draw_marker_label()
                _draw_caption()
        buffer = offscreen.texture_color.read()
        buffer.dimensions = width * height * 4
        pixels = np.array(buffer, dtype=np.float32).reshape(height, width, 4)
    finally:
        offscreen.free()
    return pixels / 255.0 if pixels.max() > 1.0 else pixels   # 8-bit buffer


def save_png(pixels, target):
    height, width = pixels.shape[:2]
    image = bpy.data.images.new("OMFS snapshot", width, height)
    image.pixels.foreach_set(np.ascontiguousarray(pixels, np.float32).ravel())
    image.filepath_raw = target
    image.file_format = 'PNG'
    image.save()
    bpy.data.images.remove(image)


class OMFS_OT_section_snapshot(bpy.types.Operator):
    """Save the section view as a PNG (caption and scale bar included) in the chosen folder"""
    bl_idname = "omfs.section_snapshot"
    bl_label = "Save Snapshot"

    def execute(self, context):
        global _snapshotting
        scene = context.scene
        problem = folder_problem(scene.omfs_section.snapshot_dir)
        if problem:
            self.report({'ERROR'}, problem)
            return {'CANCELLED'}
        area = context.area if context.area and context.area.type == 'VIEW_3D' else None
        if area is None:
            return {'CANCELLED'}
        region = next(r for r in area.regions if r.type == 'WINDOW')
        overlay = area.spaces.active.overlay
        saved = {name: getattr(overlay, name) for name in HIDDEN}
        target = str(Path(bpy.path.abspath(scene.omfs_section.snapshot_dir)) / snapshot_name(scene))
        try:
            for name in HIDDEN:
                setattr(overlay, name, False)
            _snapshotting = True
            save_png(render_view(context, area, region), target)
        finally:
            for name, value in saved.items():
                setattr(overlay, name, value)
            _snapshotting = False
        self.report({'INFO'}, f"Saved {target}")
        return {'FINISHED'}


classes = (OMFS_OT_section_snapshot,)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_caption, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
