"""Occlusion seating (part 1): which scans are the upper and lower arches, and what moves with them.

Each arch gets a mobile block: an Empty (role BLOCK) that carries the arch scan and everything
associated with it (CBCT teeth, bone segment, nerve, ...) as children. Moving the block moves
them all in one rigid motion; their mesh data is never touched.

Invariant: each member's own transform (matrix_basis) stays its pre-op pose, and its parent
inverse is the block's pre-op matrix. So:
- block at its pre-op matrix  -> everything exactly pre-op;
- planned movement             = block.matrix_world @ preop^-1;
- removing a member from a block puts it back at its pre-op pose.

A shape fingerprint (hash of the vertices and faces) is taken when an object joins a block and
checked again whenever its geometry changes, as proof that only the position changed.
"""

import hashlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import locking, reference_frame, roles

COLLECTION = "Occlusion"
ARCHES = (   # (id, block name, label)
    ('UPPER', "Bloc supérieur", "l'arcade du haut"),
    ('LOWER', "Bloc inférieur", "l'arcade du bas"),
)
BLOCK_SIZE = 12.0     # mm, size of the block's axes in the viewport
GHOST_ALPHA = 0.25
GHOST_COLOR = (0.75, 0.75, 0.75)
MOVED_TOLERANCE = 1e-4   # mm / rad: below this, a block counts as pre-op


# --- Shape fingerprint ---------------------------------------------------------------

def shape_fingerprint(obj):
    """Hash of the mesh data: changes if any vertex moves or any face changes, never with the transform."""
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    sizes = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    h = hashlib.sha1()
    for array in (co, loops, sizes):
        h.update(array.tobytes())
    return h.hexdigest()


_intact = {}   # object name -> bool, cleared when its geometry changes


def shape_intact(obj):
    """True if the mesh data still matches the fingerprint taken when it joined a block."""
    stored = obj.omfs_occl.fingerprint
    if obj.type != 'MESH' or not stored:
        return True
    if obj.name not in _intact:
        _intact[obj.name] = shape_fingerprint(obj) == stored
    return _intact[obj.name]


def remember_shape(obj):
    if obj.type == 'MESH':
        obj.omfs_occl.fingerprint = shape_fingerprint(obj)
        _intact.pop(obj.name, None)


@bpy.app.handlers.persistent
def _on_depsgraph_update(_scene, depsgraph):
    for update in depsgraph.updates:
        if update.is_updated_geometry and isinstance(update.id, bpy.types.Object):
            _intact.pop(update.id.original.name, None)


@bpy.app.handlers.persistent
def _on_load(_dummy):
    _intact.clear()


# --- Blocks --------------------------------------------------------------------------

def occlusion_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def find_block(scene, arch_id):
    for obj in scene.objects:
        if obj.type == 'EMPTY' and obj.omfs.role == 'BLOCK' and obj.omfs.landmark == arch_id:
            return obj
    return None


def is_block(obj):
    return obj is not None and obj.type == 'EMPTY' and obj.omfs.role == 'BLOCK'


def arch_of(scene, arch_id):
    props = scene.omfs_occlusion
    return props.upper if arch_id == 'UPPER' else props.lower


def preop_matrix(block):
    return block.omfs_occl.preop.copy()


def world_bbox_centre(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(pts, Vector()) / len(pts)


def ensure_block(scene, arch_id, arch):
    """The arch's block, created at the arch centre, oriented like the reference frame if built
    (so the block's local axes, used by G / R + X/Y/Z, are the clinical axes)."""
    block = find_block(scene, arch_id)
    if block is not None:
        return block
    frame = reference_frame.find_frame(scene)
    rotation = frame.matrix_world.to_quaternion() if frame else None
    matrix = Matrix.LocRotScale(world_bbox_centre(arch), rotation, None)
    block = bpy.data.objects.new(dict((a, n) for a, n, _l in ARCHES)[arch_id], None)
    occlusion_collection(scene).objects.link(block)
    block.empty_display_type = 'ARROWS'
    block.empty_display_size = BLOCK_SIZE
    block.show_in_front = True
    block.omfs.role = 'BLOCK'
    block.omfs.landmark = arch_id
    block.matrix_world = matrix
    block.omfs_occl.preop = matrix
    block.lock_scale = (True,) * 3    # rigid only: move and rotate, never scale
    return block


def members(block):
    """Objects moving with this block."""
    return list(block.children)


def can_join(obj):
    """Meshes that are anatomy (not helpers), not already in a block or parented to something else."""
    return (obj.type == 'MESH' and obj.omfs.role not in roles.NOT_SURFACES
            and obj.parent is None and not obj.omfs.made_by)


def add_member(block, obj):
    """Parent obj to the block without changing its own transform: it lands at the planned position."""
    world = obj.matrix_world.copy()
    if obj.parent is not None:      # e.g. was in the other block: back to its pre-op pose first
        remove_member(obj)
        world = obj.matrix_world.copy()
    obj.parent = block
    obj.matrix_parent_inverse = preop_matrix(block).inverted()
    obj.matrix_basis = world
    remember_shape(obj)


def remove_member(obj):
    """Unparent: the object goes back to its pre-op pose (its own transform was never changed)."""
    basis = obj.matrix_basis.copy()
    remove_ghost(obj)
    obj.parent = None
    obj.matrix_basis = basis
    obj.omfs_occl.fingerprint = ""


def set_arch(scene, arch_id, arch):
    """Called when the arch field changes: the new scan joins the block, the old one leaves it."""
    block = find_block(scene, arch_id)
    if block is not None:
        for obj in members(block):
            if obj.omfs.role == 'SCAN' and obj is not arch:
                remove_member(obj)
    if arch is None:
        return
    block = ensure_block(scene, arch_id, arch)
    if arch.parent is not block:
        add_member(block, arch)
    locking.set_locked(arch, True)   # the scan only moves with its block, never on its own


def movement(block):
    """Planned rigid movement of the block (world 4x4) since pre-op."""
    return block.matrix_world @ preop_matrix(block).inverted()


def movement_summary(scene, block):
    """(left, anterior, up) mm of the block centre, total rotation in degrees; None if pre-op."""
    preop = preop_matrix(block)
    shift = block.matrix_world.translation - preop.translation
    angle = movement(block).to_quaternion().angle
    if shift.length < MOVED_TOLERANCE and angle < MOVED_TOLERANCE:
        return None
    frame = reference_frame.find_frame(scene)
    clinical = reference_frame.to_clinical(frame, shift) if frame else tuple(shift)
    return clinical, angle * 57.29577951308232, frame is not None


def back_to_preop(block):
    block.matrix_world = preop_matrix(block)


# --- Pre-op ghost --------------------------------------------------------------------

def all_ghosts(scene):
    return [o for o in scene.objects if o.omfs.role == 'MARKER' and o.omfs.landmark == "GHOST"]


def ghost_of(obj):
    return next((g for g in all_ghosts(bpy.context.scene) if g.omfs.made_by == obj.name), None)


def ghosts(block):
    return [g for g in (ghost_of(o) for o in members(block)) if g is not None]


def remove_ghost(obj):
    ghost = ghost_of(obj)
    if ghost is not None:
        bpy.data.objects.remove(ghost)   # the mesh data stays: the original still uses it


def set_ghosts(scene, block, show):
    """Linked duplicates (same mesh data, nothing copied) frozen at the pre-op pose, semi-transparent.
    Not parented: they stay put while the block moves."""
    for obj in members(block):
        if not show:
            remove_ghost(obj)
            continue
        if ghost_of(obj) is not None:
            continue
        ghost = bpy.data.objects.new(f"{obj.name} (préop)", obj.data)
        occlusion_collection(scene).objects.link(ghost)
        ghost.omfs.role = 'MARKER'           # not a surface: clicks for landmarks go through it
        ghost.omfs.landmark = "GHOST"
        ghost.omfs.made_by = obj.name
        ghost.color = (*GHOST_COLOR, GHOST_ALPHA)
        ghost.matrix_world = obj.matrix_basis   # the original's own transform = its pre-op pose
        locking.set_locked(ghost, True)


# --- Properties and operators ---------------------------------------------------------

def _poll_arch(other):
    def poll(props, obj):
        return obj.type == 'MESH' and obj.omfs.role == 'SCAN' and obj is not getattr(props, other)
    return poll


def _on_arch(arch_id):
    def update(props, context):
        set_arch(context.scene, arch_id, props.upper if arch_id == 'UPPER' else props.lower)
    return update


def _poll_pick(_props, obj):
    return can_join(obj)


def _on_pick(arch_id):
    attr = f"pick_{arch_id.lower()}"

    def update(props, context):
        obj = getattr(props, attr)
        if obj is None:
            return
        block = find_block(context.scene, arch_id)
        if block is not None:
            add_member(block, obj)
        setattr(props, attr, None)   # the field is only a picker: clear it for the next object
    return update


class OMFS_OcclusionProperties(bpy.types.PropertyGroup):
    """Stored on the scene."""
    upper: bpy.props.PointerProperty(name="Arcade du haut", type=bpy.types.Object, poll=_poll_arch("lower"),
                                     update=_on_arch('UPPER'), description="Empreinte optique du maxillaire")
    lower: bpy.props.PointerProperty(name="Arcade du bas", type=bpy.types.Object, poll=_poll_arch("upper"),
                                     update=_on_arch('LOWER'), description="Empreinte optique de la mandibule")
    pick_upper: bpy.props.PointerProperty(name="Ajouter", type=bpy.types.Object, poll=_poll_pick,
                                          update=_on_pick('UPPER'),
                                          description="Objet qui bougera avec l'arcade du haut")
    pick_lower: bpy.props.PointerProperty(name="Ajouter", type=bpy.types.Object, poll=_poll_pick,
                                          update=_on_pick('LOWER'),
                                          description="Objet qui bougera avec l'arcade du bas")


class OMFS_OcclusionObjectProperties(bpy.types.PropertyGroup):
    """Stored on block Empties (pre-op matrix) and on their members (shape fingerprint)."""
    preop: bpy.props.FloatVectorProperty(size=(4, 4), subtype='MATRIX')
    fingerprint: bpy.props.StringProperty()


class OMFS_OT_remove_from_block(bpy.types.Operator):
    """Ne plus bouger avec ce bloc : l'objet retourne à sa position préopératoire"""
    bl_idname = "omfs.remove_from_block"
    bl_label = "Retirer du bloc"
    bl_options = {'REGISTER', 'UNDO'}

    object_name: bpy.props.StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.object_name)
        if obj is None or not is_block(obj.parent):
            return {'CANCELLED'}
        remove_member(obj)
        return {'FINISHED'}


class OMFS_OT_block_preop(bpy.types.Operator):
    """Remettre ce bloc (et tout ce qui bouge avec lui) en position préopératoire"""
    bl_idname = "omfs.block_preop"
    bl_label = "Retour préop"
    bl_options = {'REGISTER', 'UNDO'}

    arch: bpy.props.StringProperty()

    def execute(self, context):
        block = find_block(context.scene, self.arch)
        if block is None:
            return {'CANCELLED'}
        back_to_preop(block)
        return {'FINISHED'}


class OMFS_OT_select_block(bpy.types.Operator):
    """Sélectionner ce bloc pour le déplacer (G) ou le tourner (R) dans la vue 3D"""
    bl_idname = "omfs.select_block"
    bl_label = "Sélectionner le bloc"
    bl_options = {'REGISTER', 'UNDO'}

    arch: bpy.props.StringProperty()

    def execute(self, context):
        block = find_block(context.scene, self.arch)
        if block is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        block.hide_set(False)
        block.select_set(True)
        context.view_layer.objects.active = block
        return {'FINISHED'}


class OMFS_OT_toggle_ghost(bpy.types.Operator):
    """Afficher ou cacher le fantôme préopératoire : copies liées (même mesh data) semi-transparentes"""
    bl_idname = "omfs.toggle_ghost"
    bl_label = "Fantôme préop"
    bl_options = {'REGISTER', 'UNDO'}

    arch: bpy.props.StringProperty()

    def execute(self, context):
        block = find_block(context.scene, self.arch)
        if block is None:
            return {'CANCELLED'}
        set_ghosts(context.scene, block, not ghosts(block))
        return {'FINISHED'}


classes = (OMFS_OcclusionProperties, OMFS_OcclusionObjectProperties, OMFS_OT_remove_from_block,
           OMFS_OT_block_preop, OMFS_OT_select_block, OMFS_OT_toggle_ghost)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_occlusion = bpy.props.PointerProperty(type=OMFS_OcclusionProperties)
    bpy.types.Object.omfs_occl = bpy.props.PointerProperty(type=OMFS_OcclusionObjectProperties)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    bpy.app.handlers.load_post.remove(_on_load)
    bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    del bpy.types.Object.omfs_occl
    del bpy.types.Scene.omfs_occlusion
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
