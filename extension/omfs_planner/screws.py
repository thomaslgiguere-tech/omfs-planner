"""Bicortical screws ("Positionnement de vis bicorticale", guide workshop section 4).

Each screw is a cylinder of the screw's diameter (Synthes 1.85 mm) from its entry point to its exit
point, both clicked on the bone surface: usually the entry on the buccal cortex of the genial segment
and the exit on the lingual cortex of the proximal segment, two different objects. Its length is the
exact distance between them: the panel lists "Vis 1: 14.32 mm"...

An end can be moved afterwards: it follows the mouse on the bone surface (the surface is the
absolute reference: an entry always sits on the bone), a click fixes it, and the length follows.

The screw keeps its two ends on the object (world coordinates); its mesh is rebuilt from them, its own
Z running from the entry to the exit. The anatomy is only clicked on, never modified.
"""

import bmesh
import bpy
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from . import landmarks, params

TAG = "bicortical screw"        # made_by of a screw
COLLECTION = "Vis bicorticales"
BONES = {'BONE', 'SEGMENT'}     # what the ends are clicked on
SEGMENTS = 24
PREVIEW = "Vis: aperçu"


def is_screw(obj):
    return isinstance(obj, bpy.types.Object) and obj.type == 'MESH' and obj.omfs.made_by == TAG


def screws(scene):
    return sorted((o for o in scene.objects if is_screw(o)), key=lambda o: o.omfs_screw.number)


def length(screw):
    s = screw.omfs_screw
    return (Vector(s.exit) - Vector(s.entry)).length


def collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def bone_hit(context, region, mouse):
    """(world point, object) of the first bone or segment surface under the mouse; whatever stands in
    front (guide, teeth, scans, helpers) is passed through. None if no bone is under the mouse."""
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
    return first_bone(context, origin, direction)


def first_bone(context, origin, direction):
    """(world point, object) where a ray first meets a bone or segment, passing through the rest."""
    depsgraph = context.evaluated_depsgraph_get()
    for _ in range(20):
        hit, location, _normal, _index, obj, _matrix = context.scene.ray_cast(depsgraph, origin, direction)
        if not hit:
            return None
        if obj.omfs.role in BONES:
            return location, obj
        origin = location + direction * 0.01           # past whatever is in front
    return None


def shape(mesh, entry, exit_, diameter):
    """Cylinder mesh from the origin up its Z, as long as the screw (object placed at the entry)."""
    bm = bmesh.new()
    depth = max((Vector(exit_) - Vector(entry)).length, 0.01)
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=SEGMENTS,
                          radius1=diameter / 2, radius2=diameter / 2, depth=depth,
                          matrix=Matrix.Translation((0, 0, depth / 2)))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def placed(entry, exit_):
    """Object matrix: at the entry, its Z towards the exit."""
    axis = Vector(exit_) - Vector(entry)
    rotation = axis.to_track_quat('Z', 'Y') if axis.length > 1e-6 else Vector((0, 0, 1)).to_track_quat('Z', 'Y')
    return Matrix.LocRotScale(Vector(entry), rotation, None)


def rebuild(screw):
    s = screw.omfs_screw
    shape(screw.data, s.entry, s.exit, bpy.context.scene.omfs_screws.diameter)
    screw.matrix_world = placed(s.entry, s.exit)


def add_screw(scene, entry, entry_on, exit_, exit_on):
    """New screw between two bone points (with the names of the objects clicked)."""
    number = max((o.omfs_screw.number for o in screws(scene)), default=0) + 1
    name = f"Vis {number}"
    screw = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    collection(scene).objects.link(screw)
    screw.omfs.made_by = TAG
    screw.color = params.SCREW_COLOUR
    s = screw.omfs_screw
    s.number, s.entry, s.exit, s.entry_on, s.exit_on = number, entry, exit_, entry_on, exit_on
    rebuild(screw)
    return screw


def remove_screw(screw):
    mesh = screw.data
    bpy.data.objects.remove(screw)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def preview(scene, entry, exit_):
    """A temporary cylinder while the second end follows the mouse."""
    obj = bpy.data.objects.get(PREVIEW)
    if obj is None:
        obj = bpy.data.objects.new(PREVIEW, bpy.data.meshes.new(PREVIEW))
        collection(scene).objects.link(obj)
        obj.display_type = 'WIRE'
    shape(obj.data, entry, exit_, scene.omfs_screws.diameter)
    obj.matrix_world = placed(entry, exit_)


def drop_preview():
    obj = bpy.data.objects.get(PREVIEW)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)


# --- Properties -------------------------------------------------------------------------------

def _on_diameter(settings, context):
    for screw in screws(context.scene):
        rebuild(screw)


class OMFS_ScrewSettings(bpy.types.PropertyGroup):
    diameter: bpy.props.FloatProperty(
        name="Diamètre de la vis", subtype='DISTANCE', default=params.SCREW_DIAMETER, min=0.5, soft_max=4.0,
        step=5, precision=2, update=_on_diameter, description="Diamètre des cylindres-vis (Synthes 1,85 mm)")


class OMFS_Screw(bpy.types.PropertyGroup):
    number: bpy.props.IntProperty()
    entry: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    exit: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    entry_on: bpy.props.StringProperty()       # name of the object the entry was clicked on
    exit_on: bpy.props.StringProperty()


# --- Operators --------------------------------------------------------------------------------

class ModalOnBone:
    """Modal tool on the bone surface, with a header; orbit, pan and zoom keep working."""
    def begin(self, context):
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        self.area.header_text_set(self.header(context))
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def end(self, context, result):
        drop_preview()
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        for area in context.screen.areas:              # the list of lengths in the sidebar
            area.tag_redraw()
        return result


class OMFS_OT_place_screws(ModalOnBone, bpy.types.Operator):
    """Place bicortical screws: click the entry on the bone (usually the buccal cortex of the genial
    segment), then the exit (usually the lingual cortex of the proximal segment); the screw follows
    the mouse between the clicks with its length. Again for the next one; right-click or Enter: done"""
    bl_idname = "omfs.place_screws"
    bl_label = "Poser des vis (2 clics par vis)"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self.entry = None
        self.added = 0
        self.last = 0.0
        return self.begin(context)

    def header(self, context):
        n = len(screws(context.scene)) + 1
        if self.entry is None:
            return (f"Vis {n}: cliquez le point d'entrée sur l'os ({self.added} posées)"
                    "   |   Clic droit / Entrée: terminer")
        return (f"Vis {n}: cliquez la sortie (corticale linguale) — {self.last:.2f} mm"
                "   |   Échap: reprendre l'entrée")

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.end(context, {'FINISHED'})
        if event.value == 'PRESS' and event.type == 'ESC':
            if self.entry is None:
                return self.end(context, {'FINISHED'})
            self.entry = None
            drop_preview()
            self.area.header_text_set(self.header(context))
            return {'RUNNING_MODAL'}
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE' and self.entry is not None:
            hit = bone_hit(context, *under_mouse)
            if hit is not None:
                preview(context.scene, self.entry[0], hit[0])
                self.last = (hit[0] - self.entry[0]).length
                self.area.header_text_set(self.header(context))
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}
        hit = bone_hit(context, *under_mouse)
        if hit is None:
            self.report({'WARNING'}, "Cliquez sur l'os ou le segment")
            return {'RUNNING_MODAL'}
        if self.entry is None:
            self.entry = (hit[0], hit[1].name)
            self.last = 0.0
        elif (hit[0] - self.entry[0]).length < 1.0:
            self.report({'WARNING'}, "Sortie trop proche de l'entrée")
            return {'RUNNING_MODAL'}
        else:
            drop_preview()
            add_screw(context.scene, self.entry[0], self.entry[1], hit[0], hit[1].name)
            self.entry = None
            self.added += 1
        self.area.header_text_set(self.header(context))
        return {'RUNNING_MODAL'}


class OMFS_OT_move_screw_end(ModalOnBone, bpy.types.Operator):
    """Move the entry or the exit of this screw: it follows the mouse on the bone surface, a click
    fixes it and the length follows. Esc or right-click: put it back"""
    bl_idname = "omfs.move_screw_end"
    bl_label = "Déplacer un bout de la vis"
    bl_options = {'REGISTER', 'UNDO'}

    screw_name: bpy.props.StringProperty()
    end: bpy.props.EnumProperty(items=(('ENTRY', "Entrée", ""), ('EXIT', "Sortie", "")))

    def invoke(self, context, event):
        self.screw = bpy.data.objects.get(self.screw_name)
        if not is_screw(self.screw):
            return {'CANCELLED'}
        s = self.screw.omfs_screw
        self.attribute = "entry" if self.end == 'ENTRY' else "exit"
        self.before = (Vector(getattr(s, self.attribute)), getattr(s, self.attribute + "_on"))
        return self.begin(context)

    def header(self, context):
        which = "l'entrée" if self.end == 'ENTRY' else "la sortie"
        return (f"{self.screw.name}: bougez {which} sur l'os — {length(self.screw):.2f} mm, clic: fixer"
                "   |   Échap / clic droit: annuler")

    def set_end(self, point, on):
        s = self.screw.omfs_screw
        setattr(s, self.attribute, point)
        setattr(s, self.attribute + "_on", on)
        rebuild(self.screw)

    def modal(self, context, event):
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC'}:
            self.set_end(*self.before)
            return self.end(context, {'CANCELLED'})
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            hit = bone_hit(context, *under_mouse)
            if hit is not None:
                self.set_end(hit[0], hit[1].name)
                self.area.header_text_set(self.header(context))
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = bone_hit(context, *under_mouse)
            if hit is not None:
                self.set_end(hit[0], hit[1].name)
            return self.end(context, {'FINISHED'})
        return {'PASS_THROUGH'}


class OMFS_OT_remove_screw(bpy.types.Operator):
    """Remove this screw"""
    bl_idname = "omfs.remove_screw"
    bl_label = "Retirer la vis"
    bl_options = {'REGISTER', 'UNDO'}

    screw_name: bpy.props.StringProperty()

    def execute(self, context):
        screw = bpy.data.objects.get(self.screw_name)
        if is_screw(screw):
            remove_screw(screw)
        return {'FINISHED'}


classes = (OMFS_ScrewSettings, OMFS_Screw, OMFS_OT_place_screws, OMFS_OT_move_screw_end, OMFS_OT_remove_screw)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_screws = bpy.props.PointerProperty(type=OMFS_ScrewSettings)
    bpy.types.Object.omfs_screw = bpy.props.PointerProperty(type=OMFS_Screw)


def unregister():
    del bpy.types.Object.omfs_screw
    del bpy.types.Scene.omfs_screws
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
