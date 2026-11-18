"""Segmentation workspace: a DentalSeg .nrrd (probabilities or labels) checked and corrected in Blender, then
imported into the plan. It replaces 3D Slicer's Segment Editor + export, and the surfaces.py command.

Drop a <name>_probabilites.nrrd (or a .seg.nrrd) on the 3D view: a separate scene "Segmentation" shows one
surface per structure, in the patient's coordinates (LPS, like a 3D Slicer export).
- The surface of a structure passes where its probability equals the threshold (between the voxels); it is
  recomputed a moment after the threshold changes (openvdb, which ships with Blender, does the meshing).
- Small loose pieces and inner cavities are hidden live by a Geometry Nodes modifier (each piece's volume is
  summed over its faces); smoothing and simplification are ordinary modifiers. All non-destructive: the voxels
  are only changed by the two corrections below.
- Corrections, by clicking a piece: give it to another structure (e.g. a piece of "maxilla" that is really the
  mandibular ramus) or delete it. The voxels of that piece are found by flood fill from the click.
- "Valider et importer" turns what is shown into ordinary meshes of the plan, with their roles (bone and nerve
  locked), exactly like Import Case does with STL files.

The voxel data lives in memory only (it is patient data and large): a saved .blend keeps the surfaces, and
reopening the .nrrd starts again from the file.
"""

import gzip
import math
import time
from pathlib import Path

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

from . import locking, params, roles, scene_setup, view

SCENE_NAME = "Segmentation"
PREFIX = "Seg · "        # objects of the Segmentation scene: the plan's objects keep the plain names
GROUP = "OMFS Volume Pieces"
PIECES, SMOOTH, SIMPLIFY = "Morceaux", "Lissage", "Simplification"
ISLAND, ISLAND_VOLUME = "omfs_island", "omfs_island_volume"
# (file / object name, role in the plan, colour here: the bones differ so a wrong label shows)
STRUCTURES = (
    ("Maxillaire", 'BONE', (0.89, 0.87, 0.56, 1.0)),
    ("Mandibule", 'BONE', (0.83, 0.63, 0.90, 1.0)),
    ("Dents du haut", 'TEETH', (0.86, 0.58, 0.40, 1.0)),
    ("Dents du bas", 'TEETH', (0.55, 0.75, 0.95, 1.0)),
    ("Canal mandibulaire", 'NERVE', (0.85, 0.40, 0.31, 1.0)),
)


# --- NRRD ---------------------------------------------------------------------------------------

class Volume:
    """A segmentation in memory. data: probabilities (structure, x, y, z) uint8 0-255, or labels (x, y, z)
    uint8 0-5. directions: 3 x 3, one row per voxel axis (mm, LPS); origin: LPS position of voxel (0, 0, 0)."""

    def __init__(self, data, directions, origin, path=""):
        self.data, self.directions, self.origin, self.path = data, np.asarray(directions, float), np.asarray(origin, float), path
        self.probabilities = data.ndim == 4
        self.undo = []          # [(structure index or None, slices, saved voxels)] of the last correction

    @property
    def shape(self):
        return self.data.shape[-3:]

    def field(self, k):
        """Structure k (0-based) as a (x, y, z) array: its probabilities, or 255 where the label is k + 1."""
        return self.data[k] if self.probabilities else None

    def inside(self, k, threshold, crop=(slice(None),) * 3):
        """Boolean voxels of structure k above the threshold (0-255) in a crop."""
        if self.probabilities:
            return self.data[(k,) + tuple(crop)] > threshold
        return self.data[tuple(crop)] == k + 1

    def to_world(self, ijk):
        return np.asarray(ijk, float) @ self.directions + self.origin

    def to_voxel(self, xyz):
        return (np.asarray(xyz, float) - self.origin) @ np.linalg.inv(self.directions)


def read_nrrd(path):
    """A DentalSeg .nrrd (uint8, gzip or raw, 3D labels or 4D probabilities) as a Volume. Minimal reader: Blender
    has no NRRD library, and these files are written by our own segment.py (or by 3D Slicer for .seg.nrrd)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"\n\n")
    header = {}
    for line in raw[:end].decode("utf-8", "replace").splitlines()[1:]:
        if line.startswith("#") or ":" not in line:
            continue
        key, _sep, value = line.partition(":")
        header[key.strip()] = value.lstrip("=").strip()
    if header.get("type") not in ("uint8", "unsigned char", "uchar", "uint8_t"):
        raise ValueError(f"NRRD de type {header.get('type')}: attendu uint8 (fichier de DentalSeg)")
    sizes = [int(v) for v in header["sizes"].split()]
    body = raw[end + 2:]
    encoding = header.get("encoding", "raw")
    if encoding in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif encoding != "raw":
        raise ValueError(f"Encodage NRRD non pris en charge: {encoding}")
    data = np.frombuffer(bytearray(body), dtype=np.uint8, count=int(np.prod(sizes))).reshape(sizes, order='F')
    vectors = [v for v in header["space directions"].replace("none", "").split(")") if "(" in v]
    directions = np.array([[float(x) for x in v.split("(")[1].split(",")] for v in vectors])[-3:]
    origin = np.array([float(x) for x in header["space origin"].strip("()").split(",")])
    if data.ndim == 4 and sizes[0] != len(STRUCTURES):
        raise ValueError(f"{sizes[0]} structures dans le fichier: attendu {len(STRUCTURES)}")
    if data.ndim not in (3, 4):
        raise ValueError(f"NRRD à {data.ndim} dimensions: attendu 3 (étiquettes) ou 4 (probabilités)")
    return Volume(data, directions, origin, str(path))


def write_nrrd(path, volume):
    """Write a Volume as a gzip NRRD (tests, and saving a corrected segmentation)."""
    data = volume.data
    spatial = " ".join("(%r,%r,%r)" % tuple(float(v) for v in row) for row in volume.directions)
    lines = ["NRRD0005", "type: uint8", f"dimension: {data.ndim}", "space: left-posterior-superior",
             "sizes: " + " ".join(str(s) for s in data.shape),
             "space directions: " + ("none " if data.ndim == 4 else "") + spatial,
             "kinds: " + ("list " if data.ndim == 4 else "") + "domain domain domain", "encoding: gzip",
             "space origin: (%r,%r,%r)" % tuple(float(v) for v in volume.origin)]
    Path(path).write_bytes(("\n".join(lines) + "\n\n").encode() + gzip.compress(data.tobytes(order='F'), 1))


_volumes = {}     # scene name -> Volume (memory only)


def volume_of(scene):
    return _volumes.get(scene.name)


# --- Surfaces -----------------------------------------------------------------------------------

def isosurface(volume, k, threshold):
    """Surface of structure k where its probability equals threshold (0-255; labels: the voxel faces), in
    patient coordinates: (vertices, triangles, quads), or None if the structure is empty."""
    import openvdb
    inside = volume.inside(k, threshold)
    if not inside.any():
        return None
    spans = []
    for axis in range(3):
        on = np.flatnonzero(inside.any(axis=tuple(a for a in range(3) if a != axis)))
        spans.append(slice(max(on[0] - 1, 0), min(on[-1] + 2, inside.shape[axis])))
    if volume.probabilities:
        crop, level = volume.data[(k,) + tuple(spans)].astype(np.float32), float(threshold)
    else:
        crop, level = inside[tuple(spans)].astype(np.float32) * 255.0, 127.5
    grid = openvdb.FloatGrid()
    grid.copyFromArray(np.ascontiguousarray(np.pad(crop, 1)))
    points, triangles, quads = grid.convertToPolygons(isovalue=level, adaptivity=0.0)
    ijk = points + np.array([s.start for s in spans], float) - 1.0
    triangles, quads = triangles.reshape(-1, 3), quads.reshape(-1, 4)
    if np.linalg.det(volume.directions) > 0:        # openvdb's faces look inwards in a right-handed grid
        triangles, quads = triangles[:, ::-1], quads[:, ::-1]
    return volume.to_world(ijk), triangles, quads


def fill_mesh(mesh, surface):
    mesh.clear_geometry()
    if surface is None:
        return
    vertices, tris, quads = surface
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", np.asarray(vertices, np.float32).ravel())
    mesh.loops.add(3 * len(tris) + 4 * len(quads))
    mesh.loops.foreach_set("vertex_index", np.concatenate((tris.ravel(), quads.ravel())).astype(np.int32))
    mesh.polygons.add(len(tris) + len(quads))
    starts = np.concatenate((np.arange(len(tris)) * 3, 3 * len(tris) + np.arange(len(quads)) * 4))
    mesh.polygons.foreach_set("loop_start", starts.astype(np.int32))
    mesh.update()


def pieces_group():
    """Geometry Nodes group: each loose piece (Mesh Island) gets its volume, summed over its faces (area x
    centre . normal / 3); pieces under "Min volume" and, if asked, inner cavities (negative volume: their faces
    look inwards) are removed from the evaluated mesh. The piece number and volume stay as face attributes.
    In Blender: Mesh Island + Accumulate Field. Meshmixer: Edit > Separate Shells, then deleting by hand."""
    tree = bpy.data.node_groups.get(GROUP)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
    face = tree.interface
    face.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    face.new_socket("Min volume", in_out='INPUT', socket_type='NodeSocketFloat').default_value = params.VOL_MIN_PIECE
    face.new_socket("Remove cavities", in_out='INPUT', socket_type='NodeSocketBool').default_value = True
    face.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, link = tree.nodes, tree.links.new

    def node(kind, **settings):
        n = nodes.new(kind)
        for key, value in settings.items():
            setattr(n, key, value)
        return n
    group_in, group_out = node('NodeGroupInput'), node('NodeGroupOutput')
    island = node('GeometryNodeInputMeshIsland')
    dot = node('ShaderNodeVectorMath', operation='DOT_PRODUCT')
    link(node('GeometryNodeInputPosition').outputs[0], dot.inputs[0])
    link(node('GeometryNodeInputNormal').outputs[0], dot.inputs[1])
    cone = node('ShaderNodeMath', operation='MULTIPLY')
    link(node('GeometryNodeInputMeshFaceArea').outputs[0], cone.inputs[0])
    link(dot.outputs["Value"], cone.inputs[1])
    third = node('ShaderNodeMath', operation='DIVIDE')
    link(cone.outputs[0], third.inputs[0])
    third.inputs[1].default_value = 3.0
    total = node('GeometryNodeAccumulateField', data_type='FLOAT', domain='FACE')
    link(third.outputs[0], total.inputs["Value"])
    link(island.outputs["Island Index"], total.inputs["Group ID"])
    size = node('ShaderNodeMath', operation='ABSOLUTE')
    link(total.outputs["Total"], size.inputs[0])
    small = node('FunctionNodeCompare', data_type='FLOAT', operation='LESS_THAN')
    link(size.outputs[0], small.inputs["A"])
    link(group_in.outputs["Min volume"], small.inputs["B"])
    hollow = node('FunctionNodeCompare', data_type='FLOAT', operation='LESS_THAN')
    link(total.outputs["Total"], hollow.inputs["A"])
    hollow.inputs["B"].default_value = 0.0
    cavity = node('FunctionNodeBooleanMath', operation='AND')
    link(hollow.outputs["Result"], cavity.inputs[0])
    link(group_in.outputs["Remove cavities"], cavity.inputs[1])
    drop = node('FunctionNodeBooleanMath', operation='OR')
    link(small.outputs["Result"], drop.inputs[0])
    link(cavity.outputs[0], drop.inputs[1])
    geometry = group_in.outputs["Geometry"]
    for name, data_type, value in ((ISLAND, 'INT', island.outputs["Island Index"]), (ISLAND_VOLUME, 'FLOAT', total.outputs["Total"])):
        store = node('GeometryNodeStoreNamedAttribute', data_type=data_type, domain='FACE')
        store.inputs["Name"].default_value = name
        link(geometry, store.inputs["Geometry"])
        link(value, store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    delete = node('GeometryNodeDeleteGeometry', domain='FACE')
    link(geometry, delete.inputs["Geometry"])
    link(drop.outputs[0], delete.inputs["Selection"])
    link(delete.outputs["Geometry"], group_out.inputs["Geometry"])
    return tree


def set_group_input(mod, name, value):
    identifier = next(s.identifier for s in mod.node_group.interface.items_tree
                      if s.item_type == 'SOCKET' and s.in_out == 'INPUT' and s.name == name)
    getattr(mod.properties.inputs, identifier).value = value


# --- Properties ---------------------------------------------------------------------------------

def _settings_changed(self, context):
    obj = self.id_data
    apply_settings(obj)


class OMFS_VolumeStructure(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty(default=-1)          # 0-4: which structure of the file (-1: not a structure)
    threshold: bpy.props.FloatProperty(
        name="Seuil", default=params.VOL_THRESHOLD, min=5.0, max=95.0, step=100, precision=0, subtype='PERCENTAGE',
        description="La surface passe là où la probabilité de la structure vaut ce seuil. Plus bas: la structure "
                    "grossit; plus haut: elle s'amincit (comme Threshold dans 3D Slicer)")
    min_piece: bpy.props.FloatProperty(
        name="Morceaux min.", default=params.VOL_MIN_PIECE, min=0.0, soft_max=2000.0, step=1000, precision=0, update=_settings_changed,
        description="mm³. Les morceaux détachés plus petits sont cachés (comme Islands dans 3D Slicer)")
    cavities: bpy.props.BoolProperty(
        name="Retirer les cavités internes", default=True, update=_settings_changed,
        description="Enlever les poches fermées à l'intérieur de la structure (invisibles du dehors, gênantes à l'impression)")
    smooth: bpy.props.IntProperty(
        name="Lissage", default=0, min=0, max=60, update=_settings_changed,
        description="Nombre de passes de lissage (modificateur Smooth). Rétrécit légèrement les reliefs fins: rester bas")
    simplify: bpy.props.FloatProperty(
        name="Simplification", default=1.0, min=0.05, max=1.0, step=5, precision=2, subtype='FACTOR', update=_settings_changed,
        description="Part des faces gardées (modificateur Decimate). 1 = toutes. Allège le maillage pour la planification")
    made_for: bpy.props.FloatProperty(default=-1.0)   # threshold the mesh was computed for
    pieces: bpy.props.StringProperty()                # JSON-free: volumes of the shown pieces, mm3, largest first ("a;b;c")


class OMFS_VolumeSettings(bpy.types.PropertyGroup):
    is_volume_scene: bpy.props.BoolProperty()
    source_scene: bpy.props.PointerProperty(type=bpy.types.Scene)
    path: bpy.props.StringProperty(subtype='FILE_PATH')
    target: bpy.props.EnumProperty(
        name="Donner à", description="Ce que devient le morceau cliqué",
        items=[(str(i), name, "") for i, (name, _role, _colour) in enumerate(STRUCTURES)] + [('DELETE', "Supprimer", "Le morceau disparaît")],
        default='1')


def volume_scene():
    scene = bpy.data.scenes.get(SCENE_NAME)
    return scene if scene is not None and scene.omfs_volume.is_volume_scene else None


def structures(scene):
    return sorted((o for o in scene.objects if o.type == 'MESH' and o.omfs_vol.index >= 0), key=lambda o: o.omfs_vol.index)


def structure_name(obj):
    return STRUCTURES[obj.omfs_vol.index][0]


def structure(scene, k):
    return next((o for o in structures(scene) if o.omfs_vol.index == k), None)


def apply_settings(obj):
    """Push the structure's settings to its modifiers (pieces, smoothing, simplification)."""
    props = obj.omfs_vol
    mod = obj.modifiers.get(PIECES)
    if mod is not None:
        set_group_input(mod, "Min volume", props.min_piece)
        set_group_input(mod, "Remove cavities", props.cavities)
    smooth = obj.modifiers.get(SMOOTH)
    if smooth is not None:
        smooth.iterations = props.smooth
        smooth.show_viewport = smooth.show_render = props.smooth > 0
    simplify = obj.modifiers.get(SIMPLIFY)
    if simplify is not None:
        simplify.ratio = props.simplify
        simplify.show_viewport = simplify.show_render = props.simplify < 0.999
    obj.update_tag()
    props.pieces = ""        # to recount


def rebuild(scene, obj):
    """Recompute a structure's surface from the voxels, for its current threshold."""
    volume = volume_of(scene)
    if volume is None:
        return False
    props = obj.omfs_vol
    fill_mesh(obj.data, isosurface(volume, props.index, props.threshold * 2.55))
    props.made_for, props.pieces = props.threshold, ""
    return True


def count_pieces(scene, obj):
    """Volumes (mm3, largest first) of the pieces shown, read from the evaluated mesh; stored on the object."""
    depsgraph = bpy.context.evaluated_depsgraph_get() if bpy.context.scene == scene else scene.view_layers[0].depsgraph
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.data
    found = []
    ids, sizes = mesh.attributes.get(ISLAND), mesh.attributes.get(ISLAND_VOLUME)
    if ids is not None and sizes is not None and len(mesh.polygons):
        a = np.empty(len(mesh.polygons), np.int32)
        v = np.empty(len(mesh.polygons), np.float32)
        ids.data.foreach_get("value", a)
        sizes.data.foreach_get("value", v)
        _unique, first = np.unique(a, return_index=True)
        found = sorted((abs(float(x)) for x in v[first]), reverse=True)
    obj.omfs_vol.pieces = ";".join(f"{x:.0f}" for x in found) or "0"
    return found


def piece_volumes(obj):
    text = obj.omfs_vol.pieces
    return [float(x) for x in text.split(";") if x and x != "0"] if text else None


def tick():
    """While the Segmentation scene is shown: recompute a surface once its threshold has stopped changing,
    and recount the pieces after any change."""
    scene = volume_scene()
    window = bpy.context.window
    if scene is None or window is None or window.scene != scene:
        return 1.0
    now = time.monotonic()
    changed = False
    for obj in structures(scene):
        props = obj.omfs_vol
        if abs(props.made_for - props.threshold) > 1e-3 and volume_of(scene) is not None:
            seen = _moving.get(obj.name)
            if seen is None or seen[0] != props.threshold:
                _moving[obj.name] = (props.threshold, now)
            elif now - seen[1] >= params.VOL_UPDATE_DELAY:
                rebuild(scene, obj)
                changed = True
        elif not props.pieces and not obj.hide_get():
            count_pieces(scene, obj)
            changed = True
    if changed:
        for area in window.screen.areas:
            area.tag_redraw()
    return 0.15


_moving = {}


# --- Opening a file -----------------------------------------------------------------------------

def make_scene(context, volume):
    scene = volume_scene()
    if scene is None:
        scene = bpy.data.scenes.new(SCENE_NAME)
        scene.omfs_volume.is_volume_scene = True
        scene_setup.set_millimetres(scene)
    for obj in list(scene.objects):
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if isinstance(mesh, bpy.types.Mesh) and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    current = context.scene
    if current != scene and not current.omfs_volume.is_volume_scene and not getattr(current, "omfs_print").is_print_scene:
        scene.omfs_volume.source_scene = current
    scene.omfs_volume.path = volume.path
    _volumes[scene.name] = volume
    for k, (name, role, colour) in enumerate(STRUCTURES):
        obj = bpy.data.objects.new(PREFIX + name, bpy.data.meshes.new(PREFIX + name))
        scene.collection.objects.link(obj)
        obj.omfs.role = role            # (sets the role's colour: overwritten just below)
        obj.color = colour              # one colour per structure, so a wrong label shows
        props = obj.omfs_vol
        props.index = k
        if name == "Canal mandibulaire":
            props.min_piece = 0.0              # thin: every piece counts, as in the Slicer extension
        if not volume.probabilities:
            props.smooth = params.VOL_SMOOTH_LABELS
        mod = obj.modifiers.new(PIECES, 'NODES')
        mod.node_group = pieces_group()
        smooth = obj.modifiers.new(SMOOTH, 'SMOOTH')
        smooth.factor = params.VOL_SMOOTH_FACTOR
        obj.modifiers.new(SIMPLIFY, 'DECIMATE')
        apply_settings(obj)
        rebuild(scene, obj)
        for transform in ("lock_location", "lock_rotation", "lock_scale"):
            setattr(obj, transform, (True, True, True))      # the patient's coordinates: nothing moves here
    return scene


def show(context, scene):
    if context.window is None:
        return
    context.window.scene = scene
    for _window, _area, _region, space in scene_setup.view3d_areas(context):
        scene_setup.setup_viewport(space)
        space.show_region_ui = True
        space.show_gizmo_object_translate = space.show_gizmo_object_rotate = False
    scene_setup.frame_all(context)
    if not bpy.app.background:
        bpy.app.timers.register(show_tab, first_interval=0.3)


def show_tab(tries=[0]):
    """Open the sidebar on the "Segmentation" tab (its category exists only once the sidebar was drawn)."""
    window = bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None
    for area in (window.screen.areas if window else ()):
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'UI':
                    try:
                        region.active_panel_category = "Segmentation"
                        return None
                    except TypeError:
                        pass
    tries[0] += 1
    return 0.3 if tries[0] % 20 else None


class OMFS_OT_volume_open(bpy.types.Operator):
    """Ouvrir une segmentation DentalSeg (.nrrd de probabilités, ou .seg.nrrd d'étiquettes) pour la vérifier et la
    corriger avant de planifier. On peut aussi glisser le fichier dans la vue 3D"""
    bl_idname = "omfs.volume_open"
    bl_label = "Ouvrir une segmentation (.nrrd)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    filter_glob: bpy.props.StringProperty(default="*.nrrd", options={'HIDDEN'})

    def invoke(self, context, event):
        if self.filepath:
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        try:
            volume = read_nrrd(self.filepath)
        except (OSError, ValueError, KeyError) as error:
            self.report({'ERROR'}, f"Fichier illisible: {error}")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        scene = make_scene(context, volume)
        show(context, scene)
        kind = "probabilités" if volume.probabilities else "étiquettes (surface en escalier: lissage activé)"
        self.report({'INFO'}, f"Segmentation ouverte: {kind}")
        return {'FINISHED'}


class OMFS_FH_volume(bpy.types.FileHandler):
    bl_idname = "OMFS_FH_volume"
    bl_label = "Segmentation DentalSeg (.nrrd)"
    bl_import_operator = "omfs.volume_open"
    bl_file_extensions = ".nrrd"

    @classmethod
    def poll_drop(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'


# --- Corrections --------------------------------------------------------------------------------

def grow(inside_crop, seeds):
    """Voxels connected to the seeds (by faces) inside a boolean crop: flood fill by repeated dilation."""
    found = seeds & inside_crop
    while True:
        grown = found.copy()
        for axis in range(3):
            lead = (slice(None),) * axis + (slice(1, None),)
            lag = (slice(None),) * axis + (slice(None, -1),)
            grown[lead] |= found[lag]
            grown[lag] |= found[lead]
        grown &= inside_crop
        if grown.sum() == found.sum():
            return found
        found = grown


def piece_at(volume, k, threshold, ijk, box=None):
    """(slices, boolean crop) of the piece of structure k under a click (voxel coordinates), found by flood
    fill. box: (low, high) voxel corners of the piece's surface, so the fill starts in the right crop; the crop
    grows until the piece no longer touches its border. None if nothing is near the click."""
    shape = np.array(volume.shape)
    centre = np.rint(ijk).astype(int)
    r = params.VOL_PICK_RADIUS
    low, high = np.maximum(centre - r, 0), np.minimum(centre + r + 1, shape)
    near = volume.inside(k, threshold, tuple(slice(a, b) for a, b in zip(low, high)))
    if not near.any():
        return None
    hits = np.argwhere(near) + low
    seed = hits[np.argmin(np.linalg.norm(hits - ijk, axis=1))]
    half = 24
    if box is not None:
        low, high = np.maximum(np.floor(box[0]).astype(int) - 2, 0), np.minimum(np.ceil(box[1]).astype(int) + 3, shape)
        low, high = np.minimum(low, seed), np.maximum(high, seed + 1)
    while True:
        if box is None:
            low, high = np.maximum(seed - half, 0), np.minimum(seed + half + 1, shape)
        slices = tuple(slice(a, b) for a, b in zip(low, high))
        inside = volume.inside(k, threshold, slices)
        seeds = np.zeros_like(inside)
        seeds[tuple(seed - low)] = True
        found = grow(inside, seeds)
        touches = any(found.take(0, axis=a).any() and low[a] > 0 or found.take(-1, axis=a).any() and high[a] < shape[a] for a in range(3))
        if not touches:
            return slices, found
        box = None
        half = max(half * 2, int((high - low).max()))


def reassign(volume, k, threshold, ijk, target, box=None):
    """Give the piece of structure k under the click to structure target (None: delete it). The piece and one
    voxel around it (its soft edge) change hands, except voxels of another piece of k. Returns its volume in
    mm3, or None if nothing was under the click. Undone by undo()."""
    picked = piece_at(volume, k, threshold, ijk, box)
    if picked is None:
        return None
    slices, piece = picked
    voxel = abs(float(np.linalg.det(volume.directions)))
    if volume.probabilities:
        others = volume.inside(k, threshold, slices) & ~piece
        edge = piece.copy()
        for axis in range(3):
            lead = (slice(None),) * axis + (slice(1, None),)
            lag = (slice(None),) * axis + (slice(None, -1),)
            edge[lead] |= piece[lag]
            edge[lag] |= piece[lead]
        edge &= ~others
        saved = [(k, volume.data[(k,) + slices].copy())]
        source = volume.data[(k,) + slices]
        if target is not None:
            saved.append((target, volume.data[(target,) + slices].copy()))
            destination = volume.data[(target,) + slices]
            destination[edge] = np.maximum(destination[edge], source[edge])
        source[edge] = 0
    else:
        saved = [(None, volume.data[slices].copy())]
        volume.data[slices][piece] = 0 if target is None else target + 1
    volume.undo.append((slices, saved))
    return float(piece.sum()) * voxel


def undo(volume):
    """Put back the voxels of the last correction. Returns the structure indices to redraw (None: all)."""
    if not volume.undo:
        return []
    slices, saved = volume.undo.pop()
    touched = []
    for k, voxels in saved:
        if k is None:
            volume.data[slices] = voxels
            return None
        volume.data[(k,) + slices] = voxels
        touched.append(k)
    return touched


def click_point(context, event):
    """(structure object, world location) under the mouse in the 3D view, or (None, None)."""
    region = context.region
    if region is None or region.type != 'WINDOW':
        return None, None
    mouse = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
    hit, location, _normal, _index, obj, _matrix = context.scene.ray_cast(context.evaluated_depsgraph_get(), origin, direction)
    if not hit or obj is None:
        return None, None
    original = context.scene.objects.get(obj.name)
    return (original, location) if original is not None and original.omfs_vol.index >= 0 else (None, None)


class OMFS_OT_volume_reassign(bpy.types.Operator):
    """Cliquez un morceau mal étiqueté: il passe à la structure choisie dans « Donner à » (ou disparaît). Plusieurs
    clics possibles; Échap ou clic droit pour finir. Ex.: un bout de « maxillaire » qui est en fait la branche de
    la mandibule"""
    bl_idname = "omfs.volume_reassign"
    bl_label = "Corriger un morceau (cliquer)"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.omfs_volume.is_volume_scene and volume_of(context.scene) is not None

    def invoke(self, context, event):
        context.window.cursor_modal_set('EYEDROPPER')
        self.prompt(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def prompt(self, context, extra=""):
        target = context.scene.omfs_volume.target
        what = "supprimer" if target == 'DELETE' else f"donner à « {STRUCTURES[int(target)][0]} »"
        context.area.header_text_set(f"Cliquez un morceau pour le {what}   |   Échap / clic droit: finir   {extra}")

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            context.area.header_text_set(None)
            context.window.cursor_modal_restore()
            return {'FINISHED'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}
        obj, location = click_point(context, event)
        if obj is None:
            return {'PASS_THROUGH'}                    # e.g. a click in the sidebar, or on nothing
        target = context.scene.omfs_volume.target
        message = correct(context.scene, obj, location, None if target == 'DELETE' else int(target))
        self.prompt(context, f"|   {message}")
        return {'RUNNING_MODAL'}


def piece_box(scene, obj, location, volume):
    """(low, high) voxel corners of the shown piece nearest to a world location (its faces' bounding box), or
    None: the flood fill then starts in a box that fits the piece."""
    depsgraph = bpy.context.evaluated_depsgraph_get() if bpy.context.scene == scene else scene.view_layers[0].depsgraph
    mesh = obj.evaluated_get(depsgraph).data
    ids = mesh.attributes.get(ISLAND)
    if ids is None or not len(mesh.polygons):
        return None
    n = len(mesh.polygons)
    centres = np.empty(n * 3, np.float32)
    mesh.polygons.foreach_get("center", centres)
    centres = centres.reshape(-1, 3)
    island = np.empty(n, np.int32)
    ids.data.foreach_get("value", island)
    nearest = int(np.argmin(((centres - np.asarray(location, np.float32)) ** 2).sum(axis=1)))
    points = centres[island == island[nearest]]
    corners = np.array([[x, y, z] for x in (points[:, 0].min(), points[:, 0].max())
                        for y in (points[:, 1].min(), points[:, 1].max()) for z in (points[:, 2].min(), points[:, 2].max())])
    ijk = volume.to_voxel(corners)
    return ijk.min(axis=0), ijk.max(axis=0)


def correct(scene, obj, location, target):
    """Apply one correction from a click on a structure object; redraws what changed. Returns a message."""
    volume = volume_of(scene)
    k = obj.omfs_vol.index
    if target == k:
        return f"Ce morceau est déjà « {structure_name(obj)} »"
    size = reassign(volume, k, obj.omfs_vol.made_for * 2.55, volume.to_voxel(location), target,
                    piece_box(scene, obj, location, volume))
    if size is None:
        return "Rien sous le clic"
    for index in ([k] if target is None else [k, target]):
        rebuild(scene, structure(scene, index))
    where = "supprimé" if target is None else f"donné à « {STRUCTURES[target][0]} »"
    return f"Morceau de {size / 1000:.2f} cm³ {where}"


class OMFS_OT_volume_undo(bpy.types.Operator):
    """Annuler la dernière correction (réattribution ou suppression d'un morceau)"""
    bl_idname = "omfs.volume_undo"
    bl_label = "Annuler la dernière correction"

    @classmethod
    def poll(cls, context):
        volume = volume_of(context.scene)
        return volume is not None and bool(volume.undo)

    def execute(self, context):
        scene = context.scene
        touched = undo(volume_of(scene))
        for obj in structures(scene):
            if touched is None or obj.omfs_vol.index in touched:
                rebuild(scene, obj)
        return {'FINISHED'}


# --- Into the plan ------------------------------------------------------------------------------

def plan_scene(scene):
    source = scene.omfs_volume.source_scene
    if source is not None and source.name in bpy.data.scenes:
        return source
    return next((s for s in bpy.data.scenes if not s.omfs_volume.is_volume_scene and not s.omfs_print.is_print_scene), None)


def import_structures(context, scene):
    """Copy every shown structure into the plan as an ordinary mesh (what is seen: pieces filtered, smoothed,
    simplified), named, coloured and locked by its role like Import Case does. Returns the new objects."""
    target = plan_scene(scene)
    depsgraph = context.evaluated_depsgraph_get() if context.scene == scene else scene.view_layers[0].depsgraph
    depsgraph.update()
    made = []
    for obj in structures(scene):
        if obj.hide_get() or obj.hide_viewport:
            continue
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
        if not len(mesh.polygons):
            bpy.data.meshes.remove(mesh)
            continue
        for name in (ISLAND, ISLAND_VOLUME):
            if name in mesh.attributes:
                mesh.attributes.remove(mesh.attributes[name])
        name = structure_name(obj)
        old = target.objects.get(name)
        if old is not None and old.omfs.made_by == "Segmentation":
            old_mesh = old.data
            bpy.data.objects.remove(old)
            bpy.data.meshes.remove(old_mesh)
        new = bpy.data.objects.new(name, mesh)
        mesh.name = new.name
        target.collection.objects.link(new)
        new.omfs.role = obj.omfs.role
        new.omfs.made_by = "Segmentation"
        roles.apply_role_color(new)
        locking.set_locked(new, new.omfs.role in roles.LOCKED_BY_DEFAULT)
        made.append(new)
    return made


class OMFS_OT_volume_import(bpy.types.Operator):
    """Importer dans la planification ce qui est affiché: chaque structure visible devient un maillage (os et nerf
    verrouillés), dans les coordonnées du patient. Refaire remplace les maillages importés la fois précédente"""
    bl_idname = "omfs.volume_import"
    bl_label = "Valider et importer dans la planification"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return volume_scene() is not None

    def execute(self, context):
        scene = volume_scene()                    # whichever scene is shown
        target = plan_scene(scene)
        if target is None:
            self.report({'ERROR'}, "Aucune scène de planification")
            return {'CANCELLED'}
        made = import_structures(context, scene)
        if not made:
            self.report({'ERROR'}, "Aucune structure visible à importer")
            return {'CANCELLED'}
        if context.window is not None:
            context.window.scene = target
            scene_setup.setup_scene(context)
            scene_setup.frame_all(context)
            if not bpy.app.background:
                view.activate_navigate_tool(context)
        self.report({'INFO'}, f"{len(made)} structures importées dans la planification")
        return {'FINISHED'}


class OMFS_OT_volume_back(bpy.types.Operator):
    """Revenir à la planification sans importer"""
    bl_idname = "omfs.volume_back"
    bl_label = "Retour à la planification"

    def execute(self, context):
        target = plan_scene(context.scene)
        if target is None:
            return {'CANCELLED'}
        context.window.scene = target
        return {'FINISHED'}


class OMFS_OT_volume_show(bpy.types.Operator):
    """Revenir à la segmentation ouverte"""
    bl_idname = "omfs.volume_show"
    bl_label = "Revenir à la segmentation"

    def execute(self, context):
        scene = volume_scene()
        if scene is None:
            return {'CANCELLED'}
        show(context, scene)
        return {'FINISHED'}


classes = (OMFS_VolumeStructure, OMFS_VolumeSettings, OMFS_OT_volume_open, OMFS_FH_volume, OMFS_OT_volume_reassign,
           OMFS_OT_volume_undo, OMFS_OT_volume_import, OMFS_OT_volume_back, OMFS_OT_volume_show)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.omfs_vol = bpy.props.PointerProperty(type=OMFS_VolumeStructure)
    bpy.types.Scene.omfs_volume = bpy.props.PointerProperty(type=OMFS_VolumeSettings)
    if not bpy.app.background:
        bpy.app.timers.register(tick, first_interval=1.0, persistent=True)


def unregister():
    _volumes.clear()
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    del bpy.types.Scene.omfs_volume
    del bpy.types.Object.omfs_vol
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
