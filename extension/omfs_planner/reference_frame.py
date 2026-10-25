"""Clinical reference frame: three perpendicular planes built from landmarks.

- Horizontal: Frankfort horizontal, through both porions and the midpoint of the orbitales.
- Midsagittal: perpendicular to FH, through nasion and basion.
- Coronal: perpendicular to both, through nasion.
Origin: nasion projected onto FH, so all three planes meet there.

The frame is an Empty (role REFERENCE); the anatomy is never moved. Its local axes are
X = patient left, Y = posterior, Z = superior (right-handed, like Slicer's LPS), so Blender's
front view along the frame looks at the face. Clinical numbers are reported as
left +, anterior +, up + (see to_clinical).
"""

import bpy
from mathutils import Matrix, Vector

from . import landmarks, locking

COLLECTION = "Reference"
FRAME_NAME = "Reference frame"
REQUIRED = ('PO_R', 'PO_L', 'OR_R', 'OR_L', 'N', 'BA')

PLANE_SIZE = 140.0   # mm, side of the square drawn for each plane
PLANE_ALPHA = 0.15
# (name, colour, local axes spanning the plane)
PLANES = (
    ("Frankfort horizontal", (0.30, 0.55, 1.00), (0, 1)),
    ("Midsagittal plane", (1.00, 0.35, 0.35), (1, 2)),
    ("Coronal plane", (0.35, 0.85, 0.35), (0, 2)),
)


def frame_matrix(points):
    """4x4 world matrix of the frame from landmark positions {id: Vector}."""
    po_r, po_l, n, ba = points['PO_R'], points['PO_L'], points['N'], points['BA']
    or_mid = (points['OR_R'] + points['OR_L']) / 2
    po_mid = (po_r + po_l) / 2

    left = (po_l - po_r).normalized()
    anterior = or_mid - po_mid
    up = left.cross(-anterior).normalized()          # normal of FH: left x posterior = superior

    # Midsagittal plane contains the FH normal and the nasion-basion line: its normal is "left".
    x = up.cross(n - ba).normalized()
    if x.dot(left) < 0:
        x = -x
    z = up
    y = z.cross(x)                                   # posterior

    origin = n - (n - po_r).dot(z) * z               # nasion projected onto FH
    return Matrix((
        (x.x, y.x, z.x, origin.x),
        (x.y, y.y, z.y, origin.y),
        (x.z, y.z, z.z, origin.z),
        (0, 0, 0, 1),
    ))


def to_clinical(frame, world_vector):
    """A world-space displacement as (left, anterior, up) in mm, in the frame's axes."""
    local = frame.matrix_world.to_3x3().transposed() @ Vector(world_vector)
    return local.x, -local.y, local.z


def find_frame(scene):
    for obj in scene.objects:
        if obj.omfs.role == 'REFERENCE' and obj.type == 'EMPTY':
            return obj
    return None


def missing_landmarks(scene):
    return [lm for lm in REQUIRED if landmarks.find(scene, lm) is None]


def reference_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def make_plane(name, colour, axes, frame, coll):
    """A square in the frame's local plane spanned by two axes, parented to the frame."""
    h = PLANE_SIZE / 2
    a, b = (Vector([1 if i == axis else 0 for i in range(3)]) for axis in axes)
    verts = [(sa * a + sb * b) * h for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.parent = frame
    obj.omfs.role = 'REFERENCE'
    obj.color = (*colour, PLANE_ALPHA)   # alpha shows as transparency in Solid shading
    locking.set_locked(obj, True)
    return obj


def build(scene, points):
    """Create or update the frame and its planes. Returns the frame Empty."""
    frame = find_frame(scene)
    coll = reference_collection(scene)
    if frame is None:
        frame = bpy.data.objects.new(FRAME_NAME, None)
        coll.objects.link(frame)
        frame.empty_display_type = 'ARROWS'
        frame.empty_display_size = 30.0
        frame.show_name = True
        frame.show_in_front = True
        frame.omfs.role = 'REFERENCE'
        for name, colour, axes in PLANES:
            make_plane(name, colour, axes, frame, coll)
    locking.set_locked(frame, False)
    frame.matrix_world = frame_matrix(points)
    locking.set_locked(frame, True)
    return frame


def planes_visible(frame):
    return any(not child.hide_get() for child in frame.children)


class OMFS_OT_build_reference_frame(bpy.types.Operator):
    """Build (or update) the reference frame from porions, orbitales, nasion and basion"""
    bl_idname = "omfs.build_reference_frame"
    bl_label = "Build Reference Frame"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        missing = missing_landmarks(context.scene)
        if missing:
            names = ", ".join(landmarks.BY_ID[lm][1] for lm in missing)
            self.report({'ERROR'}, f"Place these landmarks first: {names}")
            return {'CANCELLED'}
        points = {lm: landmarks.find(context.scene, lm).matrix_world.translation.copy() for lm in REQUIRED}
        build(context.scene, points)
        self.report({'INFO'}, "Reference frame updated")
        return {'FINISHED'}


class OMFS_OT_toggle_reference_planes(bpy.types.Operator):
    """Show or hide the three reference planes"""
    bl_idname = "omfs.toggle_reference_planes"
    bl_label = "Toggle Reference Planes"

    def execute(self, context):
        frame = find_frame(context.scene)
        if frame is None:
            return {'CANCELLED'}
        hide = planes_visible(frame)
        for child in frame.children:
            child.hide_set(hide)
        return {'FINISHED'}


classes = (OMFS_OT_build_reference_frame, OMFS_OT_toggle_reference_planes)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
