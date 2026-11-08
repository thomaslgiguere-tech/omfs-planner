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
from math import pi

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import locking, params, reference_frame, registration, roles

COLLECTION = "Occlusion"
ARCHES = (   # (id, block name, label)
    ('UPPER', "Bloc supérieur", "l'arcade du haut"),
    ('LOWER', "Bloc inférieur", "l'arcade du bas"),
)
BLOCK_SIZE = 12.0     # mm, size of the block's axes in the viewport
GHOST_ALPHA = 0.25
GHOST_COLOR = (0.75, 0.75, 0.75)
SEGMENT_MADE_BY = "Segmental"   # upper arch segments (occlusion_segments.py)
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


# --- Pre-positioning the upper arch -------------------------------------------------
#
# Scans often come in the scanner's own frame, each arch teeth up, one inside the other. This
# changes the upper scan's own transform (its pre-op pose), never its block: it is not a surgical
# movement, and "Back to pre-op" keeps it.

def flip_above(upper, lower, gap):
    """Pre-op pose of the upper scan: turned teeth down (about the front-back axis, so the incisors
    stay anterior) and placed gap mm above the lower arch, centred on it. Computed in the lower
    scan's own coordinates, where both scanner exports are teeth up, so it also works once the
    lower arch has been registered onto the CBCT."""
    up = registration.vertex_coords(upper, world=False)
    low = registration.vertex_coords(lower, world=False)
    flip = Matrix.Rotation(pi, 4, 'Y')
    flipped = up @ np.array(flip.to_3x3()).T
    shift = Vector((*(low[:, :2].mean(axis=0) - flipped[:, :2].mean(axis=0)),
                    low[:, 2].max() + gap - flipped[:, 2].min()))
    return lower.matrix_basis @ Matrix.Translation(shift) @ flip   # lower.matrix_basis = its pre-op pose


def recentre_block(block, arch):
    """Move a block that is still pre-op to the centre of its arch (the pivot for R), without
    moving anything: each member keeps its world position."""
    if movement_summary(bpy.context.scene, block) is not None:
        return
    bpy.context.view_layer.update()
    rotation = block.matrix_world.to_quaternion()
    matrix = Matrix.LocRotScale(world_bbox_centre(arch), rotation, None)
    block.matrix_world = matrix
    block.omfs_occl.preop = matrix
    for obj in members(block):
        obj.matrix_parent_inverse = matrix.inverted()


def set_upper_preop(scene, matrix):
    props = scene.omfs_occlusion
    upper = props.upper
    upper.matrix_basis = registration.as_rigid(matrix)
    bpy.context.view_layer.update()
    block = find_block(scene, 'UPPER')
    if block is not None:
        recentre_block(block, upper)


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
        return (obj.type == 'MESH' and obj.omfs.role == 'SCAN' and obj is not getattr(props, other)
                and obj.omfs.made_by != SEGMENT_MADE_BY)
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
    # Seating (occlusion_seating.py)
    tolerance_anterior: bpy.props.FloatProperty(
        name="Antérieur (canine à canine)", subtype='DISTANCE', default=params.OCC_TOLERATED_ANTERIOR,
        min=0.0, soft_max=0.75, step=5,
        description="Chevauchement permis de canine à canine. 0 = aucun meulage antérieur. Zone définie "
                    "par les ressorts canins")
    tolerance_posterior: bpy.props.FloatProperty(
        name="Postérieur (prémolaires, molaires)", subtype='DISTANCE', default=params.OCC_TOLERATED_POSTERIOR,
        min=0.0, soft_max=0.75, step=5,
        description="Chevauchement permis sur les dents multicuspidées, au-delà duquel les dents bloquent "
                    "le mouvement. 0 = s'arrête au contact")
    spring_share: bpy.props.FloatProperty(
        name="Part des ressorts", subtype='PERCENTAGE', default=params.OCC_SPRING_SHARE, min=0.0, max=100.0,
        description="Part du guidage faite par les ressorts ; le reste est l'aimant qui rapproche les arcades "
                    "jusqu'aux contacts stables. 100 % = ressorts seuls")
    canine_weight: bpy.props.FloatProperty(
        name="Poids des canines", default=params.OCC_CANINE_WEIGHT, min=1.0, soft_max=10.0, step=50,
        description="Un ressort canin tire autant de fois plus fort qu'un autre ressort : la relation canine "
                    "(classe I) prime, quitte à moins respecter les ressorts postérieurs")
    reference_segment: bpy.props.PointerProperty(
        name="Référence", type=bpy.types.Object,
        poll=lambda _props, obj: obj.type == 'MESH' and obj.omfs.made_by == SEGMENT_MADE_BY,
        description="Segment sur lequel le bas s'assied d'abord (classe, milieu) ; ensuite le bas est fixe et "
                    "chaque autre segment s'assied sur lui")
    show_map: bpy.props.BoolProperty()
    max_overlap: bpy.props.FloatProperty(default=-1.0)
    max_overlap_anterior: bpy.props.FloatProperty()
    max_overlap_posterior: bpy.props.FloatProperty()
    area_right: bpy.props.FloatProperty()
    area_anterior: bpy.props.FloatProperty()
    area_left: bpy.props.FloatProperty()
    map_state: bpy.props.StringProperty()


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


class OMFS_OT_flip_upper_arch(bpy.types.Operator):
    """Retourner l'arcade du haut (dents vers le bas) et la placer au-dessus de l'arcade du bas,
    centrée, avec un écart. Change la position de départ du scan, pas son bloc"""
    bl_idname = "omfs.flip_upper_arch"
    bl_label = "Retourner et placer au-dessus"
    bl_options = {'REGISTER', 'UNDO'}

    gap: bpy.props.FloatProperty(name="Écart", subtype='DISTANCE', default=params.OCC_PREPOSITION_GAP, min=0.0)

    def execute(self, context):
        props = context.scene.omfs_occlusion
        if props.upper is None or props.lower is None:
            self.report({'ERROR'}, "Choisissez d'abord l'arcade du haut et l'arcade du bas")
            return {'CANCELLED'}
        set_upper_preop(context.scene, flip_above(props.upper, props.lower, self.gap))
        return {'FINISHED'}


class OMFS_OT_adjust_upper_arch(bpy.types.Operator):
    """Déverrouiller et sélectionner l'arcade du haut pour l'ajuster à la main (G / R), puis la
    reverrouiller. Change sa position de départ, pas son bloc"""
    bl_idname = "omfs.adjust_upper_arch"
    bl_label = "Ajuster à la main"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        upper = scene.omfs_occlusion.upper
        if upper is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if locking.is_locked(upper):
            locking.set_locked(upper, False)
            upper.lock_scale = (True,) * 3   # rigid only
            for obj in context.selected_objects:
                obj.select_set(False)
            upper.select_set(True)
            context.view_layer.objects.active = upper
        else:
            locking.set_locked(upper, True)
            context.view_layer.objects.active = None
            block = find_block(scene, 'UPPER')
            if block is not None:
                recentre_block(block, upper)
        return {'FINISHED'}


classes = (OMFS_OcclusionProperties, OMFS_OcclusionObjectProperties, OMFS_OT_remove_from_block,
           OMFS_OT_block_preop, OMFS_OT_select_block, OMFS_OT_toggle_ghost,
           OMFS_OT_flip_upper_arch, OMFS_OT_adjust_upper_arch)


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
