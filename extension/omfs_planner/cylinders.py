"""Adjustable cylinders for the guide workshop (drill-sleeve bore and tube, later screw channels).

A cylinder keeps its diameter and length as settings on the object; its mesh is rebuilt from them,
so typing 4.1 mm gives exactly 4.1 mm (scaling with S would only give a factor, and would stretch
the 64-sided circle). Its own axis is Z: the 3D cursor's orientation, along the bone normal.

Drag handles, like the move / rotate widgets: an arrow at each end changes the length and keeps the
other end where it is (the end in the bone stays put while the top is pulled out), a side arrow
changes the diameter. Typing in the panel changes the length about the centre.
"""

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import params

TAG = "workshop cylinder"       # made_by of an adjustable cylinder
SEGMENTS = 64
LENGTH_COLOUR = (0.30, 0.55, 1.00)
DIAMETER_COLOUR = (0.95, 0.60, 0.20)
HIGHLIGHT = (1.0, 1.0, 1.0)


def is_cylinder(obj):
    return isinstance(obj, bpy.types.Object) and obj.type == 'MESH' and obj.omfs.made_by == TAG


def rebuild(obj):
    """The mesh from the settings: a closed cylinder centred on the object's origin, along its Z."""
    c = obj.omfs_cylinder
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=SEGMENTS,
                          radius1=c.diameter / 2, radius2=c.diameter / 2, depth=c.length)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def axis(obj):
    """World direction of the cylinder's axis (its Z), unit."""
    return (obj.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()


def set_length(obj, length, keep):
    """New length; `keep` = 'BOTTOM' or 'TOP' end stays in place, 'CENTRE' grows both ways."""
    c = obj.omfs_cylinder
    length = max(params.CYLINDER_MIN_SIZE, length)
    shift = {'BOTTOM': 0.5, 'TOP': -0.5, 'CENTRE': 0.0}[keep] * (length - c.length)
    obj.location += axis(obj) * shift
    c["length"] = length          # no update: the location moved with it
    rebuild(obj)


def _on_size(c, context):
    if is_cylinder(c.id_data):
        rebuild(c.id_data)


class OMFS_CylinderProperties(bpy.types.PropertyGroup):
    diameter: bpy.props.FloatProperty(
        name="Diamètre", subtype='DISTANCE', default=4.1, min=params.CYLINDER_MIN_SIZE, soft_max=20.0,
        step=10, precision=2, update=_on_size, description="Diamètre exact du cylindre, en mm")
    length: bpy.props.FloatProperty(
        name="Longueur", subtype='DISTANCE', default=params.SLEEVE_TUBE_LENGTH, min=params.CYLINDER_MIN_SIZE,
        soft_max=40.0, step=10, precision=2, update=_on_size,
        description="Longueur du cylindre, en mm (tapée ici: grandit des deux côtés; "
                    "avec les flèches des bouts: l'autre bout reste en place)")


def add(context, diameter, length, name, collection):
    """New cylinder at the 3D cursor, along the cursor's Z (the surface normal)."""
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    collection.objects.link(obj)
    cursor = context.scene.cursor
    obj.matrix_world = Matrix.LocRotScale(cursor.location, cursor.rotation_euler, None)
    obj.omfs.made_by = TAG
    c = obj.omfs_cylinder
    c["diameter"], c["length"] = diameter, length
    rebuild(obj)
    for other in context.selected_objects:
        other.select_set(False)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


# --- Drag handles ---------------------------------------------------------------------------

def active_cylinder():
    obj = bpy.context.view_layer.objects.active
    return obj if is_cylinder(obj) else None


def placed(z_axis, location):
    """Gizmo matrix: its Z along a world direction, at a world point."""
    return Matrix.LocRotScale(location, z_axis.to_track_quat('Z', 'Y'), None)


class OMFS_GGT_cylinder(bpy.types.GizmoGroup):
    bl_idname = "OMFS_GGT_cylinder"
    bl_label = "Cylinder Size Handles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        obj = context.view_layer.objects.active
        return context.mode == 'OBJECT' and is_cylinder(obj) and obj.visible_get()

    def setup(self, context):
        def arrow(colour):
            gz = self.gizmos.new("GIZMO_GT_arrow_3d")
            gz.color, gz.color_highlight = colour, HIGHLIGHT
            gz.alpha, gz.alpha_highlight = 0.9, 1.0
            gz.scale_basis = 1.2
            return gz

        # Each end's arrow starts at the other end and reaches out by the length: dragging it moves
        # only that end.
        self.top, self.bottom = arrow(LENGTH_COLOUR), arrow(LENGTH_COLOUR)
        for gz, keep in ((self.top, 'BOTTOM'), (self.bottom, 'TOP')):
            gz.target_set_handler("offset", get=lambda: (active_cylinder().omfs_cylinder.length
                                                         if active_cylinder() else 0.0),
                                  set=lambda value, keep=keep: active_cylinder() and set_length(
                                      active_cylinder(), value, keep))
        self.side = arrow(DIAMETER_COLOUR)
        self.side.target_set_handler(
            "offset", get=lambda: active_cylinder().omfs_cylinder.diameter / 2 if active_cylinder() else 0.0,
            set=lambda value: active_cylinder() and setattr(
                active_cylinder().omfs_cylinder, "diameter", max(params.CYLINDER_MIN_SIZE, 2 * value)))

    def draw_prepare(self, context):
        obj = context.view_layer.objects.active
        c = obj.omfs_cylinder
        centre = obj.matrix_world.translation
        z = axis(obj)
        x = (obj.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
        self.top.matrix_basis = placed(z, centre - z * c.length / 2)        # from the bottom end, up
        self.bottom.matrix_basis = placed(-z, centre + z * c.length / 2)    # from the top end, down
        self.side.matrix_basis = placed(x, centre)


classes = (OMFS_CylinderProperties, OMFS_GGT_cylinder)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_cylinder = bpy.props.PointerProperty(type=OMFS_CylinderProperties)


def unregister():
    del bpy.types.Object.omfs_cylinder
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
