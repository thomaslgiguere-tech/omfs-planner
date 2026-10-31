"""Drag handles for the segment movement (like Meshmixer's transform widget, but on the
reference frame's axes, not the object's or the world's).

Three arrows (left, anterior, up) and three rings (pitch, roll, yaw) at the pivot, where the
segment has taken it. Each handle is bound to one of the six movement values, so dragging it
and typing in the panel are the same thing. Colours follow the frame: X (left) red, Y
(anterior) green, Z (up) blue, like Blender's own axes and the plane each axis is normal to.
"""

from math import pi

import bpy
from mathutils import Matrix, Vector

from . import movement

X_COLOUR = (0.95, 0.30, 0.35)
Y_COLOUR = (0.45, 0.85, 0.30)
Z_COLOUR = (0.30, 0.55, 1.00)
HIGHLIGHT = (1.0, 1.0, 1.0)

# (value, frame-local direction of the arrow or the ring's axis, colour). Rings turn
# right-handed about their axis, so the axes are those of movement.rotation's signs.
ARROWS = (('left', (1, 0, 0), X_COLOUR), ('anterior', (0, -1, 0), Y_COLOUR), ('up', (0, 0, 1), Z_COLOUR))
RINGS = (('pitch', (-1, 0, 0), X_COLOUR), ('roll', (0, -1, 0), Y_COLOUR), ('yaw', (0, 0, 1), Z_COLOUR))


def wrap(angle):
    """Rings report angles in 0..2pi: bring them to -pi..pi."""
    return (angle + pi) % (2 * pi) - pi


def segment_props():
    obj = movement.active_segment(bpy.context.scene)
    return obj.omfs_move if obj else None


def bind(gizmo, name, wrap_angle=False):
    def get():
        props = segment_props()
        return getattr(props, name) if props else 0.0

    def set(value):
        props = segment_props()
        if props is not None:
            setattr(props, name, wrap(value) if wrap_angle else value)
    gizmo.target_set_handler("offset", get=get, set=set)


class OMFS_GGT_segment_movement(bpy.types.GizmoGroup):
    bl_idname = "OMFS_GGT_segment_movement"
    bl_label = "Segment Movement Handles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        obj = movement.active_segment(context.scene)
        return (context.mode == 'OBJECT' and obj is not None and obj.visible_get()
                and obj.omfs_move.has_preop and not obj.omfs_move.locked
                and movement.frame_matrix(context.scene) is not None)

    def setup(self, context):
        self.arrows = []
        for name, _axis, colour in ARROWS:
            gz = self.gizmos.new("GIZMO_GT_arrow_3d")
            gz.color, gz.color_highlight = colour, HIGHLIGHT
            gz.alpha, gz.alpha_highlight = 0.9, 1.0
            gz.scale_basis = 1.6
            bind(gz, name)
            self.arrows.append(gz)
        self.rings = []
        for name, _axis, colour in RINGS:
            gz = self.gizmos.new("GIZMO_GT_dial_3d")
            gz.draw_options = {'ANGLE_VALUE'}
            gz.color, gz.color_highlight = colour, HIGHLIGHT
            gz.alpha, gz.alpha_highlight = 0.8, 1.0
            gz.line_width = 3
            gz.scale_basis = 1.1
            bind(gz, name, wrap_angle=True)
            self.rings.append(gz)

    def draw_prepare(self, context):
        scene = context.scene
        obj = movement.active_segment(scene)
        props = obj.omfs_move
        f = movement.frame_matrix(scene)
        rotation = f.to_3x3()
        pivot = f @ (movement.local_movement(props) @ Vector(props.pivot))   # where the pivot is now
        for gz, (name, axis, _colour) in zip(self.arrows, ARROWS):
            direction = rotation @ Vector(axis)
            # An arrow is drawn shifted by its value along its axis: start it that far back,
            # so it always comes out of the pivot (and stays still while it is dragged).
            gz.matrix_basis = placed(direction, pivot - direction * getattr(props, name))
        for gz, (_name, axis, _colour) in zip(self.rings, RINGS):
            gz.matrix_basis = placed(rotation @ Vector(axis), pivot)


def placed(z_axis, location):
    """Gizmo matrix: its Z axis along a world direction, at a world point."""
    return Matrix.LocRotScale(location, z_axis.to_track_quat('Z', 'Y'), None)


def register():
    bpy.utils.register_class(OMFS_GGT_segment_movement)


def unregister():
    bpy.utils.unregister_class(OMFS_GGT_segment_movement)
