"""Handles on the osteotomy plane in the 3D view (like Meshmixer's transform arrows).

Each gizmo is bound to one cut parameter, so dragging it and typing in the panel are the
same thing:
- vertical arrow in front of the chin: height below the foramina
- two rings: profile angle (about the left-right axis) and frontal angle (about the AP axis)
- one horizontal arrow per side, at each foramen: posterior extent
"""

from math import pi

import bpy
from mathutils import Matrix, Vector

from . import landmarks, osteotomy

DEFAULT_ANTERIOR = 15.0   # mm in front of the foramina, if there is no pogonion
ARROW_COLOR = (1.0, 0.85, 0.1)
PROFILE_COLOR = (1.0, 0.35, 0.35)   # same colour as the midsagittal plane it turns in
FRONTAL_COLOR = (0.35, 0.85, 0.35)  # same colour as the coronal plane it turns in
HIGHLIGHT = (1.0, 1.0, 1.0)


def slope(props, x, y):
    """How much higher than "below foramina" the plane is at (x, y): height = -below + slope."""
    return osteotomy.height_on_plane(props, x, y) + props.below_foramina


def handle_y(scene, cut):
    """Local Y of the handles: at the pogonion if placed, else a bit in front of the foramina."""
    pog = landmarks.find(scene, 'POG')
    if pog is None:
        return -DEFAULT_ANTERIOR
    return (cut.matrix_world.inverted() @ pog.matrix_world.translation).y


def foramen_x(scene, cut, lm):
    obj = landmarks.find(scene, lm)
    return (cut.matrix_world.inverted() @ obj.matrix_world.translation).x if obj else 0.0


def place(gizmo, cut, local_point, z_axis):
    """Put a gizmo at a point in cut-local coordinates, its Z axis along a local direction."""
    rotation = Vector(z_axis).to_track_quat('Z', 'Y').to_matrix().to_4x4()
    gizmo.matrix_basis = cut.matrix_world @ Matrix.Translation(local_point) @ rotation


class OMFS_GGT_osteotomy(bpy.types.GizmoGroup):
    bl_idname = "OMFS_GGT_osteotomy"
    bl_label = "Osteotomy Handles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        cut = osteotomy.find_cut(context.scene)
        return context.mode == 'OBJECT' and cut is not None and cut.visible_get()

    def setup(self, context):
        self.height = self.arrow(ARROW_COLOR)
        self.extent_r = self.arrow(ARROW_COLOR)
        self.extent_l = self.arrow(ARROW_COLOR)
        self.profile = self.dial(PROFILE_COLOR)
        self.frontal = self.dial(FRONTAL_COLOR)

        def props():
            return osteotomy.find_cut(bpy.context.scene).omfs_cut

        def y():
            return handle_y(bpy.context.scene, osteotomy.find_cut(bpy.context.scene))

        def set_height(value):
            props().below_foramina = max(0.0, value - slope(props(), 0.0, y()))

        # Arrows are drawn shifted by their value along their axis: the height arrow points down
        # from the foramina level, so its value is the depth of the plane at the handle.
        self.height.target_set_handler("offset", get=lambda: props().below_foramina - slope(props(), 0.0, y()),
                                       set=set_height)
        self.extent_r.target_set_handler("offset", get=lambda: props().extent_r,
                                         set=lambda v: setattr(props(), "extent_r", max(0.0, v)))
        self.extent_l.target_set_handler("offset", get=lambda: props().extent_l,
                                         set=lambda v: setattr(props(), "extent_l", max(0.0, v)))
        # Dials turn counter-clockwise about their Z axis: the profile dial looks along -X
        # (patient's right side view) and the frontal dial along -Y (front view), so that
        # + matches "rises anteriorly" and "left side higher".
        self.profile.target_set_handler("offset", get=lambda: props().profile_angle,
                                        set=lambda v: setattr(props(), "profile_angle", wrap(v)))
        self.frontal.target_set_handler("offset", get=lambda: props().frontal_angle,
                                        set=lambda v: setattr(props(), "frontal_angle", wrap(v)))

    def arrow(self, color):
        gz = self.gizmos.new("GIZMO_GT_arrow_3d")
        gz.color, gz.color_highlight = color, HIGHLIGHT
        gz.alpha, gz.alpha_highlight = 0.9, 1.0
        gz.scale_basis = 0.8
        return gz

    def dial(self, color):
        gz = self.gizmos.new("GIZMO_GT_dial_3d")
        gz.draw_options = {'ANGLE_VALUE'}
        gz.color, gz.color_highlight = color, HIGHLIGHT
        gz.alpha, gz.alpha_highlight = 0.8, 1.0
        gz.line_width = 3
        gz.scale_basis = 1.1
        return gz

    def draw_prepare(self, context):
        scene = context.scene
        cut = osteotomy.find_cut(scene)
        props = cut.omfs_cut
        y = handle_y(scene, cut)
        on_plane = Vector((0.0, y, osteotomy.height_on_plane(props, 0.0, y)))

        place(self.height, cut, (0.0, y, 0.0), (0, 0, -1))
        place(self.profile, cut, on_plane, (-1, 0, 0))
        place(self.frontal, cut, on_plane, (0, -1, 0))
        for gz, lm in ((self.extent_r, 'MF_R'), (self.extent_l, 'MF_L')):
            x = foramen_x(scene, cut, lm)
            place(gz, cut, (x, 0.0, osteotomy.height_on_plane(props, x, 0.0)), (0, 1, 0))


def wrap(angle):
    """Dials report angles in 0..2pi: bring them to -pi..pi."""
    return (angle + pi) % (2 * pi) - pi


def register():
    bpy.utils.register_class(OMFS_GGT_osteotomy)


def unregister():
    bpy.utils.unregister_class(OMFS_GGT_osteotomy)
