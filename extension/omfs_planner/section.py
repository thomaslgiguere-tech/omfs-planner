"""Section & Analysis: a section plane, a slab around it and a radiograph-like view.

Cross-cutting tool, usable at every planning step. Nothing here touches mesh data:
- the plane is drawn by a draw handler, it is not an object;
- "slab only" is Blender's clipping region (Alt + B), set to two parallel planes of any
  orientation in every 3D viewport. The GPU hides what is outside: instant on large meshes.

The plane is a 4x4 matrix (plane_matrix): Z = normal, X / Y = in-plane axes, translation =
a point of the plane. Axial / coronal / sagittal follow the clinical reference frame when it
is built (X left, Y posterior, Z up), the scanner axes otherwise. Positive offsets go up,
anterior and to the patient's left, like every clinical number in this extension.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from . import params, reference_frame, roles, scene_setup

# Plane axes (X, Y, Z = normal) in the frame's axes. Normals: up, anterior, left.
PLANE_AXES = {
    'AXIAL': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    'CORONAL': ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    'SAGITTAL': ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
}
# Viewer's axes (screen right, screen up, towards the viewer), radiological conventions:
# axial seen from the feet, coronal from the front, sagittal from the patient's right.
VIEW_AXES = {
    'AXIAL': ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    'CORONAL': ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    'SAGITTAL': ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
}
PLANE_COLOUR = (0.20, 0.85, 1.00)


def basis(columns):
    return Matrix([[c[i] for c in columns] for i in range(3)])


def frame_matrix(scene, props):
    """Axes the standard planes follow: the reference frame, or the scanner axes."""
    frame = reference_frame.find_frame(scene) if props.use_frame else None
    return frame.matrix_world.normalized() if frame else Matrix()


def base_matrix(scene, props):
    """The plane at offset 0, or None. One place decides the plane: a new kind of plane
    (e.g. a user-defined occlusal plane) is a new orientation item handled here."""
    if props.orientation == 'CUSTOM':
        obj = props.custom
        return obj.matrix_world.normalized() if obj else None   # its local Z is the normal
    frame = frame_matrix(scene, props)
    return frame @ basis(PLANE_AXES[props.orientation]).to_4x4()


def plane_matrix(scene):
    props = scene.omfs_section
    base = base_matrix(scene, props)
    if base is None:
        return None
    return Matrix.Translation(base.col[2].xyz * props.offset) @ base


def view_rotation(scene):
    """Rotation of a viewer looking perpendicular to the plane."""
    props = scene.omfs_section
    if props.orientation == 'CUSTOM':
        rot = plane_matrix(scene).to_3x3()   # looking down the object's Z
    else:
        rot = frame_matrix(scene, props).to_3x3() @ basis(VIEW_AXES[props.orientation])
    if props.flip_view:   # the other side: half a turn around screen up
        rot = rot @ Matrix.Rotation(3.141592653589793, 3, 'Y')
    return rot.to_quaternion()


def slab_planes(scene):
    """The slab as two clipping planes (a, b, c, d): a point p is kept when n . p + d >= 0."""
    m = plane_matrix(scene)
    if m is None:
        return None
    n, centre = m.col[2].xyz, m.translation.dot(m.col[2].xyz)
    half = scene.omfs_section.thickness / 2
    return [(*n, half - centre), (*(-n), half + centre)]


def in_slab(scene, point):
    return all(Vector(p[:3]).dot(point) + p[3] >= 0 for p in slab_planes(scene))


def visible_meshes(scene):
    """Anatomy, segments, guides...: visible meshes that are not helpers."""
    return [obj for obj in scene.objects
            if obj.type == 'MESH' and obj.omfs.role not in roles.NOT_SURFACES and obj.visible_get()]


def scene_bounds(scene):
    """World bounding box (min, max) of the visible meshes, or None."""
    corners = [obj.matrix_world @ Vector(c) for obj in visible_meshes(scene) for c in obj.bound_box]
    if not corners:
        return None
    return (Vector(min(c[i] for c in corners) for i in range(3)),
            Vector(max(c[i] for c in corners) for i in range(3)))


def centre_offset(scene):
    """Offset that puts the plane through the middle of the visible meshes."""
    props = scene.omfs_section
    base, bounds = base_matrix(scene, props), scene_bounds(scene)
    if base is None or bounds is None:
        return None
    return ((bounds[0] + bounds[1]) / 2 - base.translation).dot(base.col[2].xyz)


def plane_square(scene):
    """(centre, half size) of the square drawn for the plane: covers the visible meshes."""
    m, bounds = plane_matrix(scene), scene_bounds(scene)
    if bounds is None:
        return m.translation, params.SECTION_PLANE_SIZE / 2
    centre = (bounds[0] + bounds[1]) / 2
    n = m.col[2].xyz
    centre -= n * (centre - m.translation).dot(n)   # projected onto the plane
    return centre, max(params.SECTION_PLANE_SIZE, (bounds[1] - bounds[0]).length) / 2


# --- Slab: Blender's clipping region ------------------------------------------------

def set_clipping(context, on):
    """Clip every 3D viewport to the slab, or stop clipping."""
    planes = slab_planes(context.scene) if on else None
    for window, area, region, space in scene_setup.view3d_areas(context):
        rv3d = space.region_3d
        if planes is None:
            rv3d.use_clip_planes = False
            continue
        if not rv3d.use_clip_planes:
            # Alt + B prepares the clipping data; setting the flag alone draws nothing clipped.
            with context.temp_override(window=window, area=area, region=region):
                bpy.ops.view3d.clip_border(xmin=0, xmax=region.width, ymin=0, ymax=region.height)
        rv3d.clip_planes = planes + [(0, 0, 0, 0)] * 4   # zero planes keep everything
        area.tag_redraw()


def update_clipping(context):
    if context.window_manager is None or bpy.app.background:
        return
    set_clipping(context, context.scene.omfs_section.slab_only and plane_matrix(context.scene) is not None)


def _on_change(self, context):
    update_clipping(context)
    for _window, area, _region, _space in scene_setup.view3d_areas(context):
        area.tag_redraw()


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """Keep the slab on the plane when its custom object or the reference frame moves."""
    props = scene.omfs_section
    if not props.slab_only:
        return
    source = props.custom if props.orientation == 'CUSTOM' else reference_frame.find_frame(scene)
    if source and any(u.is_updated_transform and u.id.original == source for u in depsgraph.updates
                      if isinstance(u.id, bpy.types.Object)):
        update_clipping(bpy.context)


# --- Drawing ----------------------------------------------------------------------

def square_corners(m, centre, half):
    x, y = m.col[0].xyz * half, m.col[1].xyz * half
    return [centre - x - y, centre + x - y, centre + x + y, centre - x + y]


def _draw_plane():
    scene = bpy.context.scene
    props = scene.omfs_section
    if not props.show_plane:
        return
    m = plane_matrix(scene)
    if m is None:
        return
    centre, half = plane_square(scene)
    corners = square_corners(m, centre, half)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    towards_viewer = bpy.context.region_data.view_rotation @ Vector((0, 0, 1))
    if abs(towards_viewer.dot(m.col[2].xyz)) < 0.98:   # seen face on, the fill would tint everything
        fill = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(fill, 'TRIS', {"pos": corners}, indices=((0, 1, 2), (0, 2, 3)))
        fill.uniform_float("color", (*PLANE_COLOUR, 0.08))
        batch.draw(fill)
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    line.uniform_float("viewportSize", (region.width, region.height))
    gpu.state.depth_test_set('NONE')
    n = m.col[2].xyz * props.thickness / 2
    for shift, width, alpha in ((Vector(), 2.0, 0.9), (n, 1.0, 0.4), (-n, 1.0, 0.4)):
        loop = [c + shift for c in corners]
        batch = batch_for_shader(line, 'LINE_LOOP', {"pos": loop})
        line.uniform_float("lineWidth", width)
        line.uniform_float("color", (*PLANE_COLOUR, alpha))
        batch.draw(line)
    gpu.state.blend_set('NONE')


_draw_handle = None


# --- Settings and operators ---------------------------------------------------------

class OMFS_SectionProperties(bpy.types.PropertyGroup):
    orientation: bpy.props.EnumProperty(
        name="Plane", update=_on_change,
        items=(('AXIAL', "Axial", "Horizontal plane, offset + = up"),
               ('CORONAL', "Coronal", "Frontal plane, offset + = anterior"),
               ('SAGITTAL', "Sagittal", "Side plane, offset + = patient's left"),
               ('CUSTOM', "Object", "The local XY plane of an object of your choice (its Z is the normal)")))
    custom: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Plane object", update=_on_change,
        description="Any object: an Empty, a plane, a cut... The section is its local XY plane")
    use_frame: bpy.props.BoolProperty(
        name="Follow reference frame", default=True, update=_on_change,
        description="Axial / coronal / sagittal in the clinical frame (Frankfort) when it is built. "
                    "Off: the scanner axes")
    offset: bpy.props.FloatProperty(
        name="Position", subtype='DISTANCE', unit='LENGTH', default=0.0, soft_min=-150, soft_max=150,
        step=10, precision=1, update=_on_change, description="Moves the plane along its normal")
    thickness: bpy.props.FloatProperty(
        name="Thickness", subtype='DISTANCE', unit='LENGTH', default=params.SECTION_THICKNESS,
        min=params.SECTION_THICKNESS_MIN, max=params.SECTION_THICKNESS_MAX, step=10, precision=1,
        update=_on_change, description="Slab thickness, centred on the plane")
    slab_only: bpy.props.BoolProperty(
        name="Slab Only", update=_on_change,
        description="Hide everything outside the slab, in every 3D viewport (Blender's clipping region, "
                    "like Alt + B). The meshes are not changed")
    show_plane: bpy.props.BoolProperty(name="Show Plane", default=True, update=_on_change)
    flip_view: bpy.props.BoolProperty(
        name="Other Side", description="Align View looks at the plane from the other side")


class OMFS_OT_section_centre(bpy.types.Operator):
    """Put the plane through the middle of the visible meshes"""
    bl_idname = "omfs.section_centre"
    bl_label = "Centre"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        offset = centre_offset(context.scene)
        if offset is None:
            return {'CANCELLED'}
        context.scene.omfs_section.offset = offset
        return {'FINISHED'}


class OMFS_OT_section_view(bpy.types.Operator):
    """Orthographic view perpendicular to the plane, centred on it, like a radiograph"""
    bl_idname = "omfs.section_view"
    bl_label = "Align View"

    def execute(self, context):
        scene = context.scene
        area = context.area if context.area and context.area.type == 'VIEW_3D' else None
        if area is None or plane_matrix(scene) is None:
            return {'CANCELLED'}
        space = area.spaces.active
        space.lock_object = None
        rv3d = space.region_3d
        rv3d.view_rotation = view_rotation(scene)
        rv3d.view_perspective = 'ORTHO'
        rv3d.view_location = plane_square(scene)[0]
        bounds = scene_bounds(scene)
        if bounds:   # fit the visible meshes: an ortho view is view_distance * 36 / lens mm wide
            rv3d.view_distance = (bounds[1] - bounds[0]).length * space.lens / 36 * 0.8
        return {'FINISHED'}


classes = (OMFS_SectionProperties, OMFS_OT_section_centre, OMFS_OT_section_view)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_section = bpy.props.PointerProperty(type=OMFS_SectionProperties)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_plane, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    del bpy.types.Scene.omfs_section
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
