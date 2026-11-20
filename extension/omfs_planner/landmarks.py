"""Anatomical landmarks: named points clicked on a mesh surface (porion, nasion, mental foramen, ...).

Each landmark is an Empty (an object with no mesh: just a position) with role LANDMARK and
`obj.omfs.landmark` set to its id. Placing one never changes the mesh it sits on.
Landmarks are the building block for the reference frame, the osteotomy safety distances
and the movement measurements.
"""

import bpy
from bpy_extras import view3d_utils

from . import locking, roles

COLLECTION = "Landmarks"

# (id, name, description, group). Groups are only for the panel layout.
LANDMARKS = (
    ('PO_R', "Porion R", "Highest point of the right external auditory meatus", "Reference frame"),
    ('PO_L', "Porion L", "Highest point of the left external auditory meatus", "Reference frame"),
    ('OR_R', "Orbitale R", "Lowest point of the right infraorbital margin", "Reference frame"),
    ('OR_L', "Orbitale L", "Lowest point of the left infraorbital margin", "Reference frame"),
    ('N', "Nasion", "Midline point of the frontonasal suture", "Reference frame"),
    ('BA', "Basion", "Midline point on the anterior margin of the foramen magnum", "Reference frame"),
    ('MF_R', "Mental foramen R", "Centre of the right mental foramen", "Mandible"),
    ('MF_L', "Mental foramen L", "Centre of the left mental foramen", "Mandible"),
    ('CA_R', "Canine apex R", "Root apex of the right mandibular canine", "Mandible"),
    ('CA_L', "Canine apex L", "Root apex of the left mandibular canine", "Mandible"),
    ('POG', "Pogonion", "Most anterior point of the bony chin", "Mandible"),
    ('ME', "Menton", "Lowest point of the mandibular symphysis", "Mandible"),
)
BY_ID = {lm[0]: lm for lm in LANDMARKS}

DISPLAY_SIZE = 1.5  # mm, radius of the sphere drawn around each landmark


def find(scene, landmark_id):
    for obj in scene.objects:
        if obj.omfs.role == 'LANDMARK' and obj.omfs.landmark == landmark_id:
            return obj
    return None


def all_landmarks(scene):
    return [obj for obj in scene.objects if obj.omfs.role == 'LANDMARK']


def landmarks_visible(scene):
    return any(not obj.hide_get() for obj in all_landmarks(scene))


def landmark_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def place(scene, landmark_id, location, surface=None):
    """Create the landmark, or move it if it already exists. Returns the Empty."""
    obj = find(scene, landmark_id)
    if obj is None:
        name = BY_ID[landmark_id][1]
        obj = bpy.data.objects.new(name, None)   # None data = Empty
        landmark_collection(scene).objects.link(obj)
        obj.empty_display_type = 'SPHERE'
        obj.empty_display_size = DISPLAY_SIZE
        obj.show_name = True       # label in the viewport
        obj.show_in_front = True   # visible through bone
        obj.omfs.role = 'LANDMARK'
        obj.omfs.landmark = landmark_id
    locking.set_locked(obj, False)
    obj.hide_set(False)   # placing a landmark shows it, even if the others are hidden
    obj.location = location
    obj.omfs.surface = surface.name if surface else ""
    locking.set_locked(obj, True)  # moved only by placing it again
    return obj


def viewport_region_under_mouse(area, event):
    """The 3D view's main region and the mouse position inside it, or None if the mouse is elsewhere.

    Uses window coordinates: when the operator starts from a sidebar button, context.region
    is the sidebar, not the 3D view.
    """
    for region in area.regions:
        if region.type == 'WINDOW':
            x, y = event.mouse_x - region.x, event.mouse_y - region.y
            if 0 <= x < region.width and 0 <= y < region.height:
                return region, (x, y)
    return None


def surface_hit(context, region, mouse):
    """Raycast from the mouse into the scene: (location, object) of the first anatomy hit, or None.

    Helpers such as the reference planes are skipped: the ray continues past them.
    """
    rv3d = region.data
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    depsgraph = context.evaluated_depsgraph_get()
    for _ in range(10):
        hit, location, _normal, _index, obj, _matrix = context.scene.ray_cast(depsgraph, origin, direction)
        if not hit:
            return None
        if obj.omfs.role not in roles.NOT_SURFACES:
            return location, obj
        origin = location + direction * 0.01   # 0.01 mm past the helper
    return None


def object_hit(context, region, mouse, obj):
    """Raycast from the mouse against one object only: world location of the hit, or None (also when the object is
    not in the current view layer: no evaluated mesh to hit)."""
    if obj is None or obj.name not in context.view_layer.objects:
        return None
    rv3d = region.data
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    to_local = obj.matrix_world.inverted()
    hit, location, _normal, _index = obj.ray_cast(
        to_local @ origin, to_local.to_3x3() @ direction, depsgraph=context.evaluated_depsgraph_get())
    return obj.matrix_world @ location if hit else None


def object_face_hit(context, region, mouse, obj):
    """Raycast from the mouse against one object only: index of the face hit, or None."""
    rv3d = region.data
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    to_local = obj.matrix_world.inverted()
    hit, _location, _normal, index = obj.ray_cast(
        to_local @ origin, to_local.to_3x3() @ direction, depsgraph=context.evaluated_depsgraph_get())
    return index if hit else None


class OMFS_OT_place_landmark(bpy.types.Operator):
    """Click on a surface to place this landmark (Esc or right-click to cancel)"""
    bl_idname = "omfs.place_landmark"
    bl_label = "Place Landmark"
    bl_options = {'REGISTER', 'UNDO'}

    landmark_id: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        lm = BY_ID.get(properties.landmark_id)
        return f"Click on the surface to place {lm[1]}: {lm[2]}" if lm else cls.__doc__

    def invoke(self, context, event):
        if self.landmark_id not in BY_ID:
            self.report({'ERROR'}, f"Unknown landmark {self.landmark_id!r}")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            self.report({'ERROR'}, "No 3D viewport")
            return {'CANCELLED'}
        name = BY_ID[self.landmark_id][1]
        self.area.header_text_set(f"Click on the surface to place {name}   |   Esc / right-click: cancel")
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.finish(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            under_mouse = viewport_region_under_mouse(self.area, event)
            if under_mouse is None:
                return {'PASS_THROUGH'}   # click in the sidebar or elsewhere: let it through
            result = surface_hit(context, *under_mouse)
            if result is None:
                self.report({'WARNING'}, "No surface under the mouse: click on a mesh")
                return {'RUNNING_MODAL'}
            location, surface = result
            place(context.scene, self.landmark_id, location, surface)
            self.finish(context)
            self.report({'INFO'}, f"{BY_ID[self.landmark_id][1]} placed on {surface.name}")
            return {'FINISHED'}
        return {'PASS_THROUGH'}   # orbit, pan and zoom keep working while placing


class OMFS_OT_delete_landmark(bpy.types.Operator):
    """Delete this landmark"""
    bl_idname = "omfs.delete_landmark"
    bl_label = "Delete Landmark"
    bl_options = {'REGISTER', 'UNDO'}

    landmark_id: bpy.props.StringProperty()

    def execute(self, context):
        obj = find(context.scene, self.landmark_id)
        if obj is not None:
            bpy.data.objects.remove(obj)
        return {'FINISHED'}


class OMFS_OT_toggle_landmarks(bpy.types.Operator):
    """Show or hide all landmarks (e.g. once the reference frame is approved)"""
    bl_idname = "omfs.toggle_landmarks"
    bl_label = "Toggle Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        hide = landmarks_visible(context.scene)
        for obj in all_landmarks(context.scene):
            obj.hide_set(hide)
        return {'FINISHED'}


classes = (OMFS_OT_place_landmark, OMFS_OT_delete_landmark, OMFS_OT_toggle_landmarks)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
