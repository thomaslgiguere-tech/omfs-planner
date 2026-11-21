"""Impression workspace: from the printable parts of a plan to a print-ready Elegoo .goo, inside Blender.

A separate Blender Scene ("Impression") holds *copies* of the parts on a virtual build plate: the planning
objects and their patient coordinates are never touched (the copies are evaluated meshes, every modifier and
Boolean applied; position and rotation on the plate live only here). A Scene is Blender's container for one
set of objects; Meshmixer has a single workspace, so there is no equivalent.

What the surgeon does: move and turn the parts (usual gizmos, G / R, or typed values). What the add-on does
by itself, a moment after each move: drop the part to its height, regenerate its supports (PrusaSlicer, in the
background), rebuild the raft, check the build volume. Export slices the plate and writes the .goo (UVtools),
then checks it. print_chain.py runs the external tools; this module is the scene and its operators.

Bearing surfaces (the side of a guide that fits the bone, of a splint that fits the teeth) never get
supports: their faces are found from the anatomy the part fits on, and a thin slab over them ("Zone sans
support", shown in teal) is given to PrusaSlicer as a support blocker.
"""

import json
import math
import shutil
import tempfile
import threading
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper
from mathutils import Euler, Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import convex_hull_2d

from . import guide_workshop, locking, params, print_chain, print_profiles, roles, scene_setup, text_labels

SCENE_NAME = "Impression"
BEARING = "omfs_bearing"          # face attribute (boolean): bearing surface, no support there
BASES = (
    ('RAFT', "Radeau", "Une plaque rigide sous toutes les pièces, polymérisée contre le plateau, avec un rebord en "
                       "biseau: la spatule glisse dessous et tout se décolle d'un coup"),
    ('PAD', "Socle par pièce", "Le socle de PrusaSlicer, un par pièce, sous ses supports"),
    ('NONE', "Aucune", "Les supports partent directement du plateau"),
)
SUPPORT_COLOUR = (0.45, 0.55, 0.75, 1.0)
BLOCKER_COLOUR = (0.0, 0.75, 0.70, 0.45)
RAFT_COLOUR = (0.35, 0.42, 0.55, 1.0)
PLATE_COLOUR = (0.16, 0.16, 0.18, 1.0)
OUTSIDE_COLOUR = (0.95, 0.15, 0.15, 1.0)
STATES = {'': ('BLANK1', ""), 'WAIT': ('SORTTIME', "en attente"), 'RUN': ('SORTTIME', "supports en calcul…"),
          'OK': ('CHECKMARK', "supports à jour"), 'ERROR': ('ERROR', "erreur")}


# --- Properties ---------------------------------------------------------------------------------

_items = {"printers": [], "resins": []}   # kept alive: Blender needs the strings of dynamic enums to stay


def _profile_items(kind):
    def items(_self, _context):
        found = print_profiles.profiles(kind)
        _items[kind] = [(key, ("★ " if p.get("favourite") else "") + p["name"], p.get("source", "")) for key, p in found.items()]
        return _items[kind] or [("", "Aucun profil", "")]
    return items


def _touch(_self, context):
    """A setting changed: every part's supports are out of date (the timer regenerates them)."""
    for area in (context.screen.areas if context.screen else ()):
        area.tag_redraw()


class OMFS_PrintSettings(bpy.types.PropertyGroup):
    is_print_scene: bpy.props.BoolProperty()
    source_scene: bpy.props.PointerProperty(type=bpy.types.Scene)
    printer: bpy.props.EnumProperty(name="Imprimante", items=_profile_items("printers"), update=_touch)
    resin: bpy.props.EnumProperty(name="Résine", items=_profile_items("resins"), update=_touch,
                                  description="Profil de résine: hauteur de couche, expositions, soulèvement")
    supports: bpy.props.BoolProperty(name="Supports", default=True, update=_touch,
                                     description="Générer les supports (PrusaSlicer). Ils se recalculent seuls après chaque déplacement")
    live: bpy.props.BoolProperty(name="Mise à jour automatique", default=True,
                                 description="Recalculer les supports d'une pièce dès qu'elle a fini de bouger")
    density: bpy.props.IntProperty(name="Densité", default=params.PRINT_SUPPORT_DENSITY, min=30, max=300, subtype='PERCENTAGE',
                                   update=_touch, description="Densité relative des points de support")
    tip: bpy.props.FloatProperty(name="Pointe", default=params.PRINT_SUPPORT_TIP, min=0.1, max=1.5, step=5, precision=2,
                                 unit='LENGTH', update=_touch, description="Diamètre de la pointe d'un support, là où elle touche la pièce")
    lift: bpy.props.FloatProperty(name="Soulèvement", default=params.PRINT_SUPPORT_LIFT, min=1.0, max=30.0, step=50, precision=1,
                                  unit='LENGTH', update=_touch, description="Hauteur du point le plus bas de la pièce au-dessus du plateau")
    base: bpy.props.EnumProperty(name="Base", items=BASES, default='RAFT', update=_touch)
    raft_thickness: bpy.props.FloatProperty(name="Épaisseur", default=params.PRINT_RAFT_THICKNESS, min=0.3, max=5.0, step=10,
                                            precision=1, unit='LENGTH', update=_touch)
    raft_margin: bpy.props.FloatProperty(name="Marge", default=params.PRINT_RAFT_MARGIN, min=0.0, max=15.0, step=50, precision=1,
                                         unit='LENGTH', update=_touch, description="Débord du radeau autour des pièces et des supports")
    raft_lip: bpy.props.FloatProperty(name="Biseau", default=params.PRINT_RAFT_LIP, min=0.0, max=3.0, step=10, precision=1,
                                      unit='LENGTH', update=_touch,
                                      description="Le dessus du radeau dépasse son pied de cette distance: la spatule glisse dessous")
    case_id: bpy.props.StringProperty(name="Cas", default="CAS01", description="Identifiant du cas dans le nom du fichier. Jamais le nom du patient")
    result: bpy.props.StringProperty()        # last export: JSON {"path", "layers", "resin_ml", "print_time", "problems"}
    busy: bpy.props.StringProperty()          # what the export is doing right now


class OMFS_PrintObject(bpy.types.PropertyGroup):
    kind: bpy.props.StringProperty()          # PART, SUPPORTS, BLOCKER, PLATE, VOLUME, RAFT ('' = not a print object)
    pick: bpy.props.BoolProperty(name="Imprimer", description="Inclure cette pièce dans l'impression")
    source: bpy.props.StringProperty()        # name of the planning object a part was copied from
    supports: bpy.props.PointerProperty(type=bpy.types.Object)
    state: bpy.props.StringProperty()
    message: bpy.props.StringProperty()
    done: bpy.props.StringProperty()          # signature the supports were made for
    outside: bpy.props.BoolProperty()
    lift: bpy.props.FloatProperty(default=-1.0)   # mm above the plate given by the last support run (-1: none yet)
    resin_ml: bpy.props.FloatProperty()
    layers: bpy.props.IntProperty()
    up: bpy.props.FloatVectorProperty(size=4, default=(1.0, 0.0, 0.0, 0.0))   # quaternion: bearing surface up


def settings(scene):
    return scene.omfs_print


def print_scene():
    scene = bpy.data.scenes.get(SCENE_NAME)
    return scene if scene is not None and scene.omfs_print.is_print_scene else None


def of_kind(scene, kind):
    return [o for o in scene.objects if o.omfs_print.kind == kind]


def parts(scene):
    return sorted(of_kind(scene, 'PART'), key=lambda o: o.name)


def printer_of(scene):
    return print_profiles.profiles("printers").get(scene.omfs_print.printer)


def resin_of(scene):
    return print_profiles.profiles("resins").get(scene.omfs_print.resin)


def support_settings(scene):
    props = scene.omfs_print
    return {"enable": props.supports, "pad": props.base == 'PAD', "lift": props.lift, "density": props.density, "tip": props.tip}


def raft_on(scene):
    return scene.omfs_print.supports and scene.omfs_print.base == 'RAFT'


def tool_path(name):
    """Path set in the add-on preferences, if any ('' = look in the usual places)."""
    for package in (__package__, "bl_ext.user_default.omfs_planner"):
        addon = bpy.context.preferences.addons.get(package)
        if addon is not None and hasattr(addon.preferences, "prusaslicer_path"):
            return addon.preferences.prusaslicer_path if name == "PrusaSlicer" else addon.preferences.uvtools_path
    return ""


def tools():
    """(PrusaSlicer path, UVtoolsCmd path); raises print_chain.ToolMissing with the download link."""
    return print_chain.find_tool("PrusaSlicer", tool_path("PrusaSlicer")), print_chain.find_tool("UVtools", tool_path("UVtools"))


# --- Plate and build volume ---------------------------------------------------------------------

def new_object(scene, name, mesh, kind, colour, selectable=False):
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.omfs_print.kind = kind
    obj.color = colour
    obj.hide_select = not selectable
    return obj


def box_mesh(name, low, high):
    mesh = bpy.data.meshes.new(name)
    (x0, y0, z0), (x1, y1, z1) = low, high
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.from_pydata(verts, [], faces)
    return mesh


def build_plate(scene):
    """The printer's build plate (a slab under z = 0, centred on the origin) and its build volume (a wire
    box): what the printer can reach. Rebuilt when the printer changes."""
    for obj in of_kind(scene, 'PLATE') + of_kind(scene, 'VOLUME'):
        bpy.data.objects.remove(obj)
    printer = printer_of(scene)
    if printer is None:
        return
    w, h, z = printer["width"] / 2, printer["height"] / 2, printer["z"]
    new_object(scene, "Plateau", box_mesh("Plateau", (-w, -h, -2.0), (w, h, 0.0)), 'PLATE', PLATE_COLOUR)
    volume = new_object(scene, "Volume d'impression", box_mesh("Volume", (-w, -h, 0.0), (w, h, z)), 'VOLUME', (0.6, 0.8, 1.0, 1.0))
    volume.display_type = 'WIRE'
    front = bpy.data.objects.new("Avant", None)        # the side facing the user at the printer
    scene.collection.objects.link(front)
    front.omfs_print.kind = 'VOLUME'
    front.empty_display_type, front.empty_display_size = 'SINGLE_ARROW', 8.0
    front.location, front.rotation_euler = (0.0, -h - 10.0, 0.0), (math.radians(-90), 0.0, 0.0)
    front.show_name, front.hide_select = True, True


# --- Parts --------------------------------------------------------------------------------------

def world_mesh(obj, depsgraph):
    """A new mesh: the object as it will be printed (every modifier applied), in world coordinates."""
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    mesh.transform(obj.matrix_world)
    return mesh


def bearing_faces(mesh, surfaces, depsgraph):
    """Boolean per face of a world-coordinates mesh: all its vertices closer than PRINT_BEARING_DISTANCE to
    one of the surfaces the part fits on (bone, segment, teeth, scan), and the face looks at that surface
    (the two normals oppose each other): a side wall that starts on the bone is not a bearing surface."""
    near = np.zeros(len(mesh.vertices), bool)
    facing = np.zeros((len(mesh.vertices), 3))          # the surface's normal at the closest point, world
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    for surface in surfaces:
        tree = BVHTree.FromObject(surface, depsgraph, deform=False)
        inverse = np.array(surface.matrix_world.inverted())
        turn = np.array(surface.matrix_world.to_3x3())
        local = co @ inverse[:3, :3].T + inverse[:3, 3]
        for i in np.flatnonzero(~near):
            _location, normal, _index, distance = tree.find_nearest(local[i], params.PRINT_BEARING_DISTANCE)
            if distance is not None:
                near[i], facing[i] = True, turn @ np.array(normal)
    faces = np.zeros(len(mesh.polygons), bool)
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        faces[poly.index] = near[verts].all() and float(facing[verts].mean(axis=0) @ np.array(poly.normal)) < -0.5
    return faces


def set_bearing(mesh, faces):
    attribute = mesh.attributes.get(BEARING) or mesh.attributes.new(BEARING, 'BOOLEAN', 'FACE')
    attribute.data.foreach_set("value", np.asarray(faces, bool))


def get_bearing(mesh):
    attribute = mesh.attributes.get(BEARING)
    faces = np.zeros(len(mesh.polygons), bool)
    if attribute is not None:
        attribute.data.foreach_get("value", faces)
    return faces


def blocker_mesh(mesh):
    """The slab given to PrusaSlicer as a support blocker: the bearing faces and the ring of small faces around
    them (it should overlap past the edges, or a support lands on the rim), thickened PRINT_BLOCKER_HALF each
    side. None without bearing faces."""
    flagged = get_bearing(mesh)
    if not flagged.any():
        return None
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    keep = {f for f in bm.faces if flagged[f.index]}
    reach = 4 * params.PRINT_BLOCKER_HALF     # only small neighbours: on a coarse mesh a neighbour is a whole wall
    keep |= {f for face in list(keep) for v in face.verts for f in v.link_faces
             if max(e.calc_length() for e in f.edges) <= reach}
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f not in keep], context='FACES')
    bm.normal_update()
    for v in bm.verts:
        v.co -= v.normal * params.PRINT_BLOCKER_HALF
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=-2 * params.PRINT_BLOCKER_HALF)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    out = bpy.data.meshes.new("Zone sans support")
    bm.to_mesh(out)
    bm.free()
    return out


def refresh_blocker(scene, part):
    for child in [c for c in part.children if c.omfs_print.kind == 'BLOCKER']:
        mesh = child.data
        bpy.data.objects.remove(child)
        bpy.data.meshes.remove(mesh)
    mesh = blocker_mesh(part.data)
    if mesh is None:
        return None
    blocker = new_object(scene, f"Zone sans support ({part.name})", mesh, 'BLOCKER', BLOCKER_COLOUR)
    blocker.parent = part
    blocker.show_transparent = True
    return blocker


def bearing_up(mesh):
    """Quaternion turning the part so its bearing surface faces up (area-weighted mean of the bearing
    normals); identity without bearing faces."""
    flagged = get_bearing(mesh)
    if not flagged.any():
        return Quaternion()
    normal = Vector()
    for poly in mesh.polygons:
        if flagged[poly.index]:
            normal += poly.normal * poly.area
    if normal.length < 1e-9:
        return Quaternion()
    return normal.normalized().rotation_difference(Vector((0.0, 0.0, 1.0)))


def add_part(scene, name, mesh, source="", role='GUIDE'):
    """A part of the print scene from a world-coordinates mesh (taken over): centred on its bounding box so
    it turns about its middle, bearing surface up and tilted, with its no-support slab."""
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mesh.transform(Matrix.Translation(-Vector((co.min(axis=0) + co.max(axis=0)) / 2)))
    mesh.name = name
    part = new_object(scene, name, mesh, 'PART', roles.COLORS.get(role, roles.COLORS['GUIDE']), selectable=True)
    part.omfs.role = role
    part.omfs_print.source = source
    up = bearing_up(mesh)
    part.omfs_print.up = up
    part.rotation_mode = 'XYZ'
    tilt = params.PRINT_TILT_DEFAULT if get_bearing(mesh).any() else 0.0
    part.rotation_euler = (Euler((math.radians(tilt), 0.0, 0.0)).to_quaternion() @ up).to_euler('XYZ')
    part.lock_location[2] = True              # the height is set by the supports (or the plate)
    part.lock_scale = (True, True, True)      # a guide printed at another scale no longer fits
    refresh_blocker(scene, part)
    return part


def rotation_of(part):
    return np.array(part.matrix_basis.to_3x3())


def turned_vertices(part, mesh=None):
    """Vertices of the part (or of another mesh in its local coordinates) turned as on the plate, origin at
    the part's centre."""
    mesh = mesh or part.data
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3) @ rotation_of(part).T


def triangles_of(mesh):
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def chain_part(scene, part):
    """The part as print_chain wants it: turned as on the plate, with its blockers, at its place on the bed."""
    printer = printer_of(scene)
    blockers = [(turned_vertices(part, c.data), triangles_of(c.data)) for c in part.children if c.omfs_print.kind == 'BLOCKER']
    bed = (part.location.x + printer["width"] / 2, part.location.y + printer["height"] / 2)
    return print_chain.Part(part.name, turned_vertices(part), triangles_of(part.data), blockers, bed)


def base_height(scene, part):
    """How high the part's lowest point sits above the plate."""
    if not scene.omfs_print.supports:
        return 0.0
    if part.omfs_print.lift >= 0:
        return part.omfs_print.lift
    return scene.omfs_print.lift + (2.0 if scene.omfs_print.base == 'PAD' else 0.0)


def drop(scene, part):
    """Set the part's height: its lowest point at base_height above the plate."""
    z = base_height(scene, part) - float(turned_vertices(part)[:, 2].min())
    if abs(part.location.z - z) > 1e-4:
        part.location.z = z


def signature(scene, part):
    """What the supports of a part depend on: its orientation and the support, resin and printer settings."""
    props = scene.omfs_print
    rotation = tuple(round(v, 4) for row in rotation_of(part) for v in row)
    return repr((rotation, len(part.data.vertices), props.supports, props.base, round(props.lift, 2), props.density,
                 round(props.tip, 2), props.printer, props.resin))


# --- Supports -----------------------------------------------------------------------------------

def set_supports(scene, part, preview):
    """Show the supports of a part (a print_chain.SupportPreview; None removes them)."""
    old = part.omfs_print.supports
    if old is not None:
        mesh = old.data
        bpy.data.objects.remove(old)
        bpy.data.meshes.remove(mesh)
    if preview is None or not len(preview.vertices):
        return None
    mesh = bpy.data.meshes.new(f"Supports ({part.name})")
    tris, quads = np.asarray(preview.triangles), np.asarray(preview.quads)
    mesh.vertices.add(len(preview.vertices))
    mesh.vertices.foreach_set("co", np.asarray(preview.vertices, np.float32).ravel())
    mesh.loops.add(3 * len(tris) + 4 * len(quads))
    mesh.loops.foreach_set("vertex_index", np.concatenate((tris.ravel(), quads.ravel())).astype(np.int32))
    mesh.polygons.add(len(tris) + len(quads))
    starts = np.concatenate((np.arange(len(tris)) * 3, 3 * len(tris) + np.arange(len(quads)) * 4))
    mesh.polygons.foreach_set("loop_start", starts.astype(np.int32))
    mesh.update()
    mesh.validate()
    supports = new_object(scene, mesh.name, mesh, 'SUPPORTS', SUPPORT_COLOUR)
    supports.location = (part.location.x, part.location.y, 0.0)
    part.omfs_print.supports = supports
    return supports


def apply_preview(scene, part, preview, made_for):
    props = part.omfs_print
    set_supports(scene, part, preview)
    props.lift, props.resin_ml, props.layers = preview.lift, preview.resin_ml, preview.layers
    props.done = made_for
    props.state, props.message = ('OK', " · ".join(preview.warnings)) if not preview.warnings else ('ERROR', " · ".join(preview.warnings))
    drop(scene, part)


def update_supports_now(scene, part):
    """Regenerate one part's supports and wait for them (tests, scripts). The UI uses the background jobs."""
    made_for = signature(scene, part)
    if not scene.omfs_print.supports:
        clear_supports(scene, part, made_for)
        return None
    work = tempfile.mkdtemp(prefix="omfs_print_")
    try:
        preview = print_chain.support_preview(chain_part(scene, part), printer_of(scene), resin_of(scene),
                                              support_settings(scene), tools()[0], work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    apply_preview(scene, part, preview, made_for)
    return preview


def clear_supports(scene, part, made_for):
    set_supports(scene, part, None)
    props = part.omfs_print
    props.lift, props.resin_ml, props.layers, props.done, props.state, props.message = -1.0, 0.0, 0, made_for, '', ""
    drop(scene, part)


class Job:
    """One support run in a worker thread: only arrays and files there, never bpy."""

    def __init__(self, scene, part):
        self.part, self.made_for = part.name, signature(scene, part)
        self.cancel, self.result, self.error = threading.Event(), None, None
        self.work = tempfile.mkdtemp(prefix="omfs_print_")
        args = (chain_part(scene, part), printer_of(scene), resin_of(scene), support_settings(scene), tools()[0], self.work, self.cancel)
        self.thread = threading.Thread(target=self.run, args=args, daemon=True)
        self.thread.start()

    def run(self, *args):
        try:
            self.result = print_chain.support_preview(*args)
        except print_chain.Cancelled:
            pass
        except Exception as error:   # shown on the part, never a crash of the timer
            self.error = str(error)
        finally:
            shutil.rmtree(self.work, ignore_errors=True)


_jobs = {}        # part name -> Job
_changed = {}     # part name -> (signature last seen, time it was first seen)
_raft_key = [None]


def tick_part(scene, part, now):
    """One part, a few times a second: follow it with its supports, and once it has stopped moving, set its
    height and start (or restart) its support run. Returns True if something visible changed."""
    props = part.omfs_print
    changed = False
    supports = props.supports
    if supports is not None and (abs(supports.location.x - part.location.x) > 1e-5 or abs(supports.location.y - part.location.y) > 1e-5):
        supports.location.x, supports.location.y = part.location.x, part.location.y
        changed = True
    current = signature(scene, part)
    seen = _changed.get(part.name)
    if seen is None or seen[0] != current:
        _changed[part.name] = (current, now)
        job = _jobs.pop(part.name, None)
        if job is not None:
            job.cancel.set()                  # it moved again: the run in progress is for an old position
        if props.done != current and scene.omfs_print.supports:
            props.state, changed = 'WAIT', True
        return changed
    still = now - seen[1] >= params.PRINT_UPDATE_DELAY
    job = _jobs.get(part.name)
    if job is not None and not job.thread.is_alive():
        del _jobs[part.name]
        if job.error:
            props.state, props.message, props.done = 'ERROR', job.error, current
        elif job.result is not None and job.made_for == current:
            apply_preview(scene, part, job.result, current)
        return True
    if props.done == current or not still:
        return changed
    if not scene.omfs_print.supports:
        clear_supports(scene, part, current)
        return True
    drop(scene, part)
    if job is None and scene.omfs_print.live:
        try:
            _jobs[part.name] = Job(scene, part)
            props.state, props.message = 'RUN', ""
        except print_chain.ToolMissing as error:
            props.state, props.message, props.done = 'ERROR', str(error), current
        return True
    return changed


def world_box(obj):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_basis)     # print objects have no parent; matrix_world lags until the next redraw
    co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    return co.min(axis=0), co.max(axis=0)


def check_volume(scene):
    """Mark the parts that leave the build volume (with their supports): they turn red."""
    printer = printer_of(scene)
    if printer is None:
        return False
    high = np.array((printer["width"] / 2, printer["height"] / 2, printer["z"]))
    low = np.array((-high[0], -high[1], -params.PRINT_PREVIEW_PIXEL))   # the preview mesh rounds to a voxel
    changed = False
    for part in parts(scene):
        outside = False
        for obj in (part, part.omfs_print.supports):
            if obj is not None and len(obj.data.vertices):
                a, b = world_box(obj)
                outside |= bool((a < low).any() or (b > high).any())
        if outside != part.omfs_print.outside:
            part.omfs_print.outside = outside
            changed = True
        colour = OUTSIDE_COLOUR if outside else roles.COLORS.get(part.omfs.role, roles.COLORS['GUIDE'])
        if max(abs(x - y) for x, y in zip(part.color, colour)) > 1e-3:      # colours are stored as 32-bit floats
            part.color = colour
    return changed


# --- Raft ---------------------------------------------------------------------------------------

def grow(points, distance):
    """Convex outline of 2D points pushed outwards by distance, corners rounded."""
    if distance > 0:
        ring = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]) * distance
        points = (points[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    return points[convex_hull_2d(points.tolist())]


def raft_outline(scene):
    """2D outline (world X, Y) around every part and its supports, grown by the raft margin; None if empty."""
    found = []
    for part in parts(scene):
        co = turned_vertices(part)[:, :2] + np.array(part.location[:2])
        found.append(co[convex_hull_2d(co.tolist())])
        supports = part.omfs_print.supports
        if supports is not None and len(supports.data.vertices):
            a, b = world_box(supports)
            found.append(np.array(((a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1]))))
    if not found:
        return None
    return grow(np.vstack(found), scene.omfs_print.raft_margin)


def raft_arrays(scene):
    """(vertices, triangles) of the raft: its foot on the plate, its top wider by the bevel (raft_lip), so the
    edge overhangs all around and a scraper slides under it. None without parts."""
    foot = raft_outline(scene)
    if foot is None:
        return None
    props = scene.omfs_print
    top = foot
    if props.raft_lip > 0:
        edges = np.roll(foot, -1, axis=0) - foot
        normals = np.stack((edges[:, 1], -edges[:, 0]), axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-9)[:, None]
        if np.cross(edges[0], edges[1]) < 0:
            normals = -normals
        corner = normals + np.roll(normals, 1, axis=0)                 # bisector of the two edges at each vertex
        corner /= np.maximum((corner * normals).sum(axis=1), 0.3)[:, None]
        top = foot + corner * props.raft_lip
    n = len(foot)
    vertices = np.vstack((np.column_stack((foot, np.zeros(n))), np.column_stack((top, np.full(n, props.raft_thickness)))))
    i = np.arange(n)
    j = (i + 1) % n
    sides = np.vstack((np.column_stack((i, j, j + n)), np.column_stack((i, j + n, i + n))))
    bottom = np.column_stack((np.zeros(n - 2, int), np.arange(2, n), np.arange(1, n - 1)))
    cap = np.column_stack((np.full(n - 2, n), np.arange(n + 1, 2 * n - 1), np.arange(n + 2, 2 * n)))
    triangles = np.vstack((sides, bottom, cap))
    if np.cross(foot[1] - foot[0], foot[2] - foot[1]) < 0:           # clockwise outline: flip to keep normals outside
        triangles = triangles[:, ::-1]
    return vertices, triangles


def update_raft(scene, force=False):
    """Rebuild the raft object when the parts, their supports or the raft settings changed."""
    props = scene.omfs_print
    key = (raft_on(scene), round(props.raft_thickness, 2), round(props.raft_margin, 2), round(props.raft_lip, 2),
           tuple((p.name, round(p.location.x, 2), round(p.location.y, 2), p.omfs_print.done, signature(scene, p)) for p in parts(scene)))
    if key == _raft_key[0] and not force and (of_kind(scene, 'RAFT') or not raft_on(scene)):
        return False
    _raft_key[0] = key
    for obj in of_kind(scene, 'RAFT'):
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
    arrays = raft_arrays(scene) if raft_on(scene) else None
    if arrays is None:
        return True
    mesh = bpy.data.meshes.new("Radeau")
    mesh.from_pydata(arrays[0].tolist(), [], arrays[1].tolist())
    new_object(scene, "Radeau", mesh, 'RAFT', RAFT_COLOUR)
    return True


def raft_part(scene):
    """The raft as a printed part of the plate (print_chain.Part), or None."""
    arrays = raft_arrays(scene) if raft_on(scene) else None
    if arrays is None:
        return None
    printer = printer_of(scene)
    vertices, triangles = arrays
    centre = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    return print_chain.Part("Radeau", vertices, triangles, [], (centre[0] + printer["width"] / 2, centre[1] + printer["height"] / 2),
                            {"supports_enable": 0, "pad_enable": 0})   # flat on the plate: it is the base itself


# --- The timer ----------------------------------------------------------------------------------

def tick():
    """A few times a second while the print scene is shown: supports follow, heights, runs, raft, volume."""
    scene = print_scene()
    window = bpy.context.window
    if scene is None or window is None or window.scene != scene:
        return 1.0
    now = time.monotonic()
    changed = False
    for part in parts(scene):
        changed |= tick_part(scene, part, now)
    if not any(now - seen[1] < params.PRINT_UPDATE_DELAY for seen in _changed.values()):
        changed |= update_raft(scene)
    changed |= check_volume(scene)
    if changed or _jobs:
        for area in window.screen.areas:
            area.tag_redraw()
    return 0.2


def update_all_now(scene):
    """Everything the timer does, at once and waiting for the supports (tests, scripts, before an export)."""
    for part in parts(scene):
        if part.omfs_print.done != signature(scene, part):
            update_supports_now(scene, part)
        drop(scene, part)
    update_raft(scene, force=True)
    check_volume(scene)


# --- Estimates and checks -----------------------------------------------------------------------

def estimates(scene):
    """{"layers", "time" (s), "resin_ml", "size" (x, y, z mm)} of the plate as it stands (from the last
    support runs: the export gives the exact values)."""
    found = parts(scene)
    resin = resin_of(scene)
    if not found or resin is None:
        return None
    high = 0.0
    low_xy, high_xy = np.full(2, np.inf), np.full(2, -np.inf)
    volume = 0.0
    for part in found:
        for obj in (part, part.omfs_print.supports):
            if obj is not None and len(obj.data.vertices):
                a, b = world_box(obj)
                high = max(high, b[2])
                low_xy, high_xy = np.minimum(low_xy, a[:2]), np.maximum(high_xy, b[:2])
        volume += part.omfs_print.resin_ml or mesh_volume(part.data) / 1000
    if raft_on(scene):
        outline = raft_outline(scene)
        x, y = outline[:, 0], outline[:, 1]
        volume += abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2 * scene.omfs_print.raft_thickness / 1000
    layers = int(math.ceil(high / resin["layer_height"]))
    size = (*(high_xy - low_xy), high)
    return {"layers": layers, "time": print_chain.print_time(layers, resin), "resin_ml": volume, "size": size}


def mesh_volume(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = abs(bm.calc_volume(signed=True))
    bm.free()
    return volume


def open_edges(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    count = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return count


def checks(scene):
    """[(ok, text)] shown in the panel before export; the export is refused while one is red."""
    found = parts(scene)
    out = [(bool(found), "Au moins une pièce sur le plateau")]
    if not found:
        return out
    leaking = [p.name for p in found if open_edges(p.data)]
    out.append((not leaking, "Pièces fermées (manifold)" if not leaking else "Pièce ouverte: " + ", ".join(leaking)))
    outside = [p.name for p in found if p.omfs_print.outside]
    out.append((not outside, "Dans le volume d'impression" if not outside else "Hors du volume: " + ", ".join(outside)))
    if scene.omfs_print.supports:
        stale = [p.name for p in found if p.omfs_print.done != signature(scene, p) or p.omfs_print.state in ('WAIT', 'RUN')]
        failed = [p.name for p in found if p.omfs_print.state == 'ERROR']
        out.append((not stale and not failed, "Supports à jour" if not stale and not failed else
                    ("Supports en erreur: " + ", ".join(failed) if failed else "Supports à recalculer: " + ", ".join(stale))))
    try:
        tools()
        out.append((True, "PrusaSlicer et UVtools trouvés"))
    except print_chain.ToolMissing as error:
        out.append((False, str(error)))
    return out


def file_name(scene):
    """Case ID + part names + resin: never a patient name."""
    resin = resin_of(scene)
    names = "_".join(bpy.path.clean_name(p.name) for p in parts(scene))[:60]
    return f"{bpy.path.clean_name(scene.omfs_print.case_id)}_{names}_{print_profiles.slug(resin['name']) if resin else ''}.goo"


def setup_record(scene, result):
    """What was printed and how, kept next to the .goo and in the scene (traceability, methods section)."""
    return {"date": time.strftime("%Y-%m-%d %H:%M"), "case": scene.omfs_print.case_id, "file": Path(result.path).name,
            "printer": printer_of(scene)["name"], "resin": {k: v for k, v in resin_of(scene).items() if k != "user"},
            "supports": dict(support_settings(scene), base=scene.omfs_print.base,
                             raft=[scene.omfs_print.raft_thickness, scene.omfs_print.raft_margin, scene.omfs_print.raft_lip]),
            "layers": result.layers, "resin_ml": round(result.resin_ml, 2), "print_time_s": round(result.print_time),
            "parts": [{"name": p.name, "source": p.omfs_print.source, "location_mm": [round(v, 3) for v in p.location],
                       "rotation_deg": [round(math.degrees(v), 2) for v in p.rotation_euler]} for p in parts(scene)]}


def export_now(scene, path):
    """Slice the plate and write the .goo at path, waiting for it. Returns (print_chain.ExportResult, [problems]):
    the file is kept only when there is no problem."""
    slicer, uvtools = tools()
    plate = [chain_part(scene, p) for p in parts(scene)]
    raft = raft_part(scene)
    work = tempfile.mkdtemp(prefix="omfs_print_")
    try:
        result = print_chain.export_goo(plate + ([raft] if raft else []), printer_of(scene), resin_of(scene),
                                        support_settings(scene), Path(work) / "plate.goo", slicer, uvtools, work)
        if not result.problems:
            shutil.move(str(result.path), str(path))
            result = result._replace(path=Path(path))
            Path(path).with_suffix(".json").write_text(json.dumps(setup_record(scene, result), indent=1, ensure_ascii=False))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


# --- Operators ----------------------------------------------------------------------------------

def printable(scene):
    """Objects of a planning scene that can be printed: any free mesh, never raw anatomy."""
    return [o for o in scene.objects if text_labels.can_carry(o) and not o.omfs_print.kind]


def arrange(scene):
    """Parts side by side in rows, centred on the plate, PRINT_PART_GAP apart (supports included)."""
    printer = printer_of(scene)
    gap = params.PRINT_PART_GAP + 2 * (scene.omfs_print.raft_margin if raft_on(scene) else 0.0) / 2
    sizes = []
    for part in parts(scene):
        co = turned_vertices(part)
        low, high = co[:, :2].min(axis=0), co[:, :2].max(axis=0)
        supports = part.omfs_print.supports
        if supports is not None and len(supports.data.vertices):
            a, b = world_box(supports)
            low = np.minimum(low, a[:2] - np.array(part.location[:2]))
            high = np.maximum(high, b[:2] - np.array(part.location[:2]))
        sizes.append((part, low, high))
    x = y = row = 0.0
    placed = []
    for part, low, high in sizes:
        w, h = high - low
        if x > 0 and x + w > printer["width"] - 2 * gap:
            x, y, row = 0.0, y + row + gap, 0.0
        placed.append((part, x - low[0], y - low[1], low, high))
        x, row = x + w + gap, max(row, h)
    if not placed:
        return
    xs = [(px + low[0], px + high[0]) for _p, px, _py, low, high in placed]
    ys = [(py + low[1], py + high[1]) for _p, _px, py, low, high in placed]
    cx = (min(a for a, _b in xs) + max(b for _a, b in xs)) / 2
    cy = (min(a for a, _b in ys) + max(b for _a, b in ys)) / 2
    for part, px, py, _low, _high in placed:
        part.location.x, part.location.y = px - cx, py - cy


def set_handles(space, on):
    """The move arrows and rotation rings on the selected part (Blender's object gizmos), whatever the tool:
    drag an arrow to slide the part on the plate, a ring to turn it. Meshmixer: the Transform widget."""
    space.show_gizmo = True
    space.show_gizmo_object_translate = on
    space.show_gizmo_object_rotate = on
    space.show_gizmo_object_scale = False


def show_scene(context, scene):
    if context.window is None:        # no window (tests, scripts): nothing to show
        return
    context.window.scene = scene
    for _window, area, region, space in scene_setup.view3d_areas(context):
        scene_setup.setup_viewport(space)
        space.overlay.show_floor = False
        space.overlay.show_axis_x = space.overlay.show_axis_y = False
        space.show_region_ui = True
        set_handles(space, scene.omfs_print.is_print_scene)


def frame_plate(context):
    """Look at the plate from the front, a little above, like standing at the printer."""
    if context.window is None:
        return
    scene = context.scene
    for window, area, region, space in scene_setup.view3d_areas(context):
        rv3d = space.region_3d
        rv3d.view_perspective = 'PERSP'
        rv3d.view_rotation = Euler((math.radians(62), 0.0, math.radians(18))).to_quaternion()
        printer = printer_of(scene)
        rv3d.view_location = (0.0, 0.0, 10.0)
        rv3d.view_distance = (printer["width"] if printer else 150.0) * 0.95


def make_print_scene(context):
    scene = print_scene()
    if scene is None:
        scene = bpy.data.scenes.new(SCENE_NAME)
        scene.omfs_print.is_print_scene = True
        scene_setup.set_millimetres(scene)
        props = scene.omfs_print
        printers = print_profiles.profiles("printers")
        if printers:
            props.printer = next(iter(printers))
        favourite = print_profiles.favourite_resin()
        if favourite:
            props.resin = favourite
    if context.scene != scene and not context.scene.omfs_print.is_print_scene:
        scene.omfs_print.source_scene = context.scene
    if not of_kind(scene, 'PLATE'):
        build_plate(scene)
    return scene


def copy_to_print(context, scene, obj):
    """Copy a planning object into the print scene (the object itself is not touched)."""
    depsgraph = context.evaluated_depsgraph_get()
    mesh = world_mesh(obj, depsgraph)
    surfaces = guide_workshop.fitted_surfaces(context, obj)
    set_bearing(mesh, bearing_faces(mesh, surfaces, depsgraph) if surfaces else np.zeros(len(mesh.polygons), bool))
    return add_part(scene, obj.name, mesh, source=obj.name, role=obj.omfs.role)


class OMFS_OT_print_prepare(bpy.types.Operator):
    """Copier les pièces cochées sur le plateau virtuel de l'imprimante (scène « Impression »). Les objets de
    la planification ne sont jamais modifiés: l'impression travaille sur des copies"""
    bl_idname = "omfs.print_prepare"
    bl_label = "Préparer l'impression"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        source = context.scene
        chosen = [o for o in printable(source) if o.omfs_print.pick]
        if not chosen:
            self.report({'ERROR'}, "Cochez au moins une pièce à imprimer")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        scene = make_print_scene(context)
        present = {p.omfs_print.source for p in parts(scene)}
        for obj in chosen:
            if obj.name not in present:
                copy_to_print(context, scene, obj)
        arrange(scene)
        show_scene(context, scene)
        frame_plate(context)
        return {'FINISHED'}


class OMFS_OT_print_back(bpy.types.Operator):
    """Revenir à la planification. Le plateau reste tel quel: « Préparer l'impression » y revient"""
    bl_idname = "omfs.print_back"
    bl_label = "Retour à la planification"

    def execute(self, context):
        source = context.scene.omfs_print.source_scene
        if source is None:
            source = next((s for s in bpy.data.scenes if not s.omfs_print.is_print_scene), None)
        if source is None:
            return {'CANCELLED'}
        context.window.scene = source
        for _window, _area, _region, space in scene_setup.view3d_areas(context):
            set_handles(space, False)
            space.overlay.show_floor = True
            space.overlay.show_axis_x = space.overlay.show_axis_y = True
        return {'FINISHED'}


class OMFS_OT_print_show(bpy.types.Operator):
    """Ouvrir le plateau d'impression"""
    bl_idname = "omfs.print_show"
    bl_label = "Ouvrir le plateau"

    def execute(self, context):
        scene = make_print_scene(context)
        show_scene(context, scene)
        frame_plate(context)
        return {'FINISHED'}


def active_part(context):
    obj = context.view_layer.objects.active
    if obj is not None and obj.omfs_print.kind == 'PART' and obj.name in context.scene.objects:
        return obj
    found = parts(context.scene)
    return found[0] if len(found) == 1 else None


class OMFS_OT_print_tilt(bpy.types.Operator):
    """Orienter la pièce active: surface d'appui vers le haut, puis inclinée de cet angle (une face plate
    parallèle au plateau s'imprime mal: l'incliner répartit l'effort de décollement)"""
    bl_idname = "omfs.print_tilt"
    bl_label = "Incliner"
    bl_options = {'REGISTER', 'UNDO'}

    angle: bpy.props.FloatProperty(name="Angle", subtype='ANGLE', default=math.radians(30))
    axis: bpy.props.EnumProperty(name="Axe", items=(('X', "Avant / arrière", ""), ('Y', "Gauche / droite", "")), default='X')
    flip: bpy.props.BoolProperty(name="Surface d'appui vers le bas", default=False)

    def execute(self, context):
        part = active_part(context)
        if part is None:
            self.report({'ERROR'}, "Sélectionnez une pièce du plateau")
            return {'CANCELLED'}
        tilt = Euler((self.angle, 0.0, 0.0) if self.axis == 'X' else (0.0, self.angle, 0.0)).to_quaternion()
        turn = Euler((math.pi, 0.0, 0.0)).to_quaternion() if self.flip else Quaternion()
        part.rotation_euler = (tilt @ turn @ Quaternion(part.omfs_print.up)).to_euler('XYZ')
        return {'FINISHED'}


class OMFS_OT_print_arrange(bpy.types.Operator):
    """Disposer les pièces côte à côte, centrées sur le plateau (leurs rotations ne changent pas)"""
    bl_idname = "omfs.print_arrange"
    bl_label = "Disposer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arrange(context.scene)
        return {'FINISHED'}


class OMFS_OT_print_select(bpy.types.Operator):
    """Sélectionner cette pièce (puis G pour la déplacer, R pour la tourner, ou les poignées)"""
    bl_idname = "omfs.print_select"
    bl_label = "Sélectionner"

    name: bpy.props.StringProperty()

    def execute(self, context):
        part = context.scene.objects.get(self.name)
        if part is None:
            return {'CANCELLED'}
        for obj in context.selected_objects:
            obj.select_set(False)
        part.select_set(True)
        context.view_layer.objects.active = part
        return {'FINISHED'}


def remove_part(scene, part):
    job = _jobs.pop(part.name, None)
    if job is not None:
        job.cancel.set()
    _changed.pop(part.name, None)
    for obj in [part.omfs_print.supports] + [c for c in part.children if c.omfs_print.kind == 'BLOCKER'] + [part]:
        if obj is not None:
            mesh = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)


class OMFS_OT_print_remove(bpy.types.Operator):
    """Retirer cette pièce du plateau (l'objet de la planification n'est pas touché)"""
    bl_idname = "omfs.print_remove"
    bl_label = "Retirer du plateau"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty()

    def execute(self, context):
        part = context.scene.objects.get(self.name)
        if part is None or part.omfs_print.kind != 'PART':
            return {'CANCELLED'}
        remove_part(context.scene, part)
        return {'FINISHED'}


class OMFS_OT_print_update_part(bpy.types.Operator):
    """Reprendre la pièce depuis la planification (après une modification du guide), en gardant sa position et
    son orientation sur le plateau"""
    bl_idname = "omfs.print_update_part"
    bl_label = "Mettre à jour la pièce dans l'impression"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty()

    def execute(self, context):
        scene = context.scene
        part = scene.objects.get(self.name)
        source_scene = scene.omfs_print.source_scene
        source = source_scene.objects.get(part.omfs_print.source) if part and source_scene else None
        if source is None:
            self.report({'ERROR'}, "La pièce d'origine n'existe plus dans la planification")
            return {'CANCELLED'}
        with context.temp_override(scene=source_scene, view_layer=source_scene.view_layers[0]):
            depsgraph = source_scene.view_layers[0].depsgraph
            depsgraph.update()
            mesh = world_mesh(source, depsgraph)
            surfaces = guide_workshop.fitted_surfaces(bpy.context, source)
            set_bearing(mesh, bearing_faces(mesh, surfaces, depsgraph) if surfaces else np.zeros(len(mesh.polygons), bool))
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mesh.transform(Matrix.Translation(-Vector((co.min(axis=0) + co.max(axis=0)) / 2)))
        old = part.data
        mesh.name = old.name
        part.data = mesh
        bpy.data.meshes.remove(old)
        refresh_blocker(scene, part)
        part.omfs_print.done = ""           # supports to regenerate
        return {'FINISHED'}


class OMFS_OT_print_refresh(bpy.types.Operator):
    """Recalculer maintenant les supports de toutes les pièces"""
    bl_idname = "omfs.print_refresh"
    bl_label = "Recalculer les supports"

    def execute(self, context):
        for part in parts(context.scene):
            part.omfs_print.done = ""
        _changed.clear()
        return {'FINISHED'}


class OMFS_OT_print_export(bpy.types.Operator, ExportHelper):
    """Découper le plateau à la résolution de l'imprimante et écrire le fichier .goo, puis le vérifier: réglages
    de la résine présents dans le fichier, aucune zone sans appui. Le fichier n'est gardé que si tout est vert"""
    bl_idname = "omfs.print_export"
    bl_label = "Exporter .goo"
    filename_ext = ".goo"
    filter_glob: bpy.props.StringProperty(default="*.goo", options={'HIDDEN'})

    def invoke(self, context, event):
        red = [text for ok, text in checks(context.scene) if not ok]
        if red:
            self.report({'ERROR'}, "Export refusé: " + " · ".join(red))
            return {'CANCELLED'}
        self.filepath = file_name(context.scene)
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        scene = context.scene
        red = [text for ok, text in checks(scene) if not ok]
        if red and not bpy.app.background:
            self.report({'ERROR'}, "Export refusé: " + " · ".join(red))
            return {'CANCELLED'}
        if bpy.app.background:               # scripts and tests: wait for it
            return self.finish(context, export_now(scene, self.filepath), None)
        scene.omfs_print.busy = "Découpe du plateau et écriture du .goo…"
        self.outcome = {}
        args = (scene, self.filepath)
        self.plate = [chain_part(scene, p) for p in parts(scene)]

        def work():
            try:
                self.outcome["result"] = export_now(*args)
            except Exception as error:
                self.outcome["error"] = str(error)
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        self.timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER' or self.thread.is_alive():
            return {'PASS_THROUGH'}
        context.window_manager.event_timer_remove(self.timer)
        context.scene.omfs_print.busy = ""
        return self.finish(context, self.outcome.get("result"), self.outcome.get("error"))

    def finish(self, context, result, error):
        props = context.scene.omfs_print
        if error or result is None:
            props.result = json.dumps({"problems": [error or "Export interrompu"]})
            self.report({'ERROR'}, error or "Export interrompu")
            return {'CANCELLED'}
        props.result = json.dumps({"path": str(result.path) if not result.problems else "", "layers": result.layers,
                                   "resin_ml": result.resin_ml, "print_time": result.print_time, "problems": result.problems})
        for area in (context.screen.areas if context.screen else ()):
            area.tag_redraw()
        if result.problems:
            self.report({'ERROR'}, "Fichier non gardé: " + " · ".join(result.problems))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exporté: {result.path}")
        return {'FINISHED'}


# --- Resin profiles -----------------------------------------------------------------------------

RESIN_FIELDS = (   # (key in the profile or in its "elegoo" block, label, property)
    ("layer_height", "Hauteur de couche (mm)"), ("bottom_layers", "Couches de base"),
    ("bottom_exposure_time", "Exposition des couches de base (s)"), ("exposure_time", "Exposition (s)"),
    ("TransitionLayerCount", "Couches de transition"), ("LightOffDelay", "Délai lumière éteinte (s)"),
    ("BottomWaitTimeBeforeCure", "Attente avant exposition, base (s)"), ("WaitTimeBeforeCure", "Attente avant exposition (s)"),
    ("LiftHeight", "Soulèvement 1 (mm)"), ("LiftSpeed", "Vitesse de soulèvement 1 (mm/min)"),
    ("LiftHeight2", "Soulèvement 2 (mm)"), ("LiftSpeed2", "Vitesse de soulèvement 2 (mm/min)"),
    ("RetractSpeed", "Vitesse de descente 1 (mm/min)"), ("RetractHeight2", "Descente 2 (mm)"),
    ("RetractSpeed2", "Vitesse de descente 2 (mm/min)"),
)


def resin_value(resin, key):
    return resin[key] if key in resin else resin["elegoo"].get(key, 0)


class OMFS_OT_print_edit_resin(bpy.types.Operator):
    """Modifier ce profil de résine, ou en créer un nouveau à partir de lui (changez le nom). Les valeurs
    viennent de la fiche du fabricant ou de votre trancheur habituel. Les réglages des couches de base suivent
    ceux des couches normales pour le soulèvement"""
    bl_idname = "omfs.print_edit_resin"
    bl_label = "Profil de résine"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty(name="Nom")
    favourite: bpy.props.BoolProperty(name="Favorite (choisie par défaut)")
    values: bpy.props.FloatVectorProperty(size=len(RESIN_FIELDS), precision=3)

    def invoke(self, context, event):
        resin = resin_of(context.scene)
        if resin is None:
            return {'CANCELLED'}
        self.name, self.favourite = resin["name"], bool(resin.get("favourite"))
        self.values = [float(resin_value(resin, key)) for key, _label in RESIN_FIELDS]
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "name")
        layout.prop(self, "favourite")
        col = layout.column(align=True)
        for i, (_key, label) in enumerate(RESIN_FIELDS):
            col.prop(self, "values", index=i, text=label)

    def execute(self, context):
        resin = {k: v for k, v in resin_of(context.scene).items() if k != "user"}
        resin["elegoo"] = dict(resin["elegoo"])
        resin["name"], resin["favourite"] = self.name.strip() or resin["name"], self.favourite
        resin["source"] = f"Modifié dans OMFS Planner le {time.strftime('%Y-%m-%d')}"
        for (key, _label), value in zip(RESIN_FIELDS, self.values):
            value = int(round(value)) if key in ("bottom_layers", "TransitionLayerCount") else round(float(value), 3)
            if key in resin:
                resin[key] = value
            else:
                resin["elegoo"][key] = value
                if f"Bottom{key}" in resin["elegoo"] and key.startswith(("Lift", "Retract")):
                    resin["elegoo"][f"Bottom{key}"] = value
        context.scene.omfs_print.resin = print_profiles.save_resin(resin, favourite=self.favourite)
        return {'FINISHED'}


# --- Synthetic test piece -----------------------------------------------------------------------

def cylinder(bm, radius, z0, z1, centre, segments=48):
    result = bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=z1 - z0)
    bmesh.ops.translate(bm, verts=result["verts"], vec=(centre[0], centre[1], (z0 + z1) / 2))


def test_piece_mesh():
    """One synthetic part that answers everything a first print must answer, in one print:
    - which way round it prints: the plate has one corner cut and a raised "R" (a mirrored print shows it);
    - the scale in X, Y and Z: plate 30 x 20 mm, steps 1.0 / 1.5 / 2.0 / 2.5 mm thick (calipers);
    - the drill-sleeve bore: a tube with the bore the guides use (sleeve diameter + fit clearance);
    - the fixation hole for the 1.85 mm screw, as in the guides;
    - the supports: the whole underside is a bearing surface (no support may touch it).
    Returns a closed mesh, its bearing faces flagged."""
    w, d, t = params.PRINT_TEST_PLATE
    cut = 6.0
    outline = [(0, 0), (w, 0), (w, d - cut), (w - cut, d), (0, d)]        # one corner cut: asymmetric
    bm = bmesh.new()
    bottom = [bm.verts.new((x, y, 0.0)) for x, y in outline]
    face = bm.faces.new(bottom)
    top = bmesh.ops.extrude_face_region(bm, geom=[face])
    bmesh.ops.translate(bm, verts=[v for v in top["geom"] if isinstance(v, bmesh.types.BMVert)], vec=(0, 0, t))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    plate = bpy.data.meshes.new("Pièce test")
    bm.to_mesh(plate)
    bm.free()
    holder = bpy.data.objects.new("test piece (building)", plate)
    bpy.context.scene.collection.objects.link(holder)
    made = [holder]

    def helper(build):
        b = bmesh.new()
        build(b)
        mesh = bpy.data.meshes.new("test piece helper")
        b.to_mesh(mesh)
        b.free()
        obj = bpy.data.objects.new("test piece helper", mesh)
        bpy.context.scene.collection.objects.link(obj)
        made.append(obj)
        return obj

    def boolean(other, operation):
        mod = holder.modifiers.new(operation, 'BOOLEAN')
        mod.operation, mod.object, mod.solver = operation, other, 'EXACT'

    bore = params.SLEEVE_OUTER_DIAMETER + params.SLEEVE_FIT_CLEARANCE
    tube_at, hole_at = (8.0, 12.0), (8.0, 4.0)
    boolean(helper(lambda b: cylinder(b, params.SLEEVE_TUBE_DIAMETER / 2, t - 0.2, t + params.SLEEVE_TUBE_LENGTH, tube_at)), 'UNION')
    for k, height in enumerate((1.0, 1.5, 2.0)):                         # steps cut down from the top: 1.0, 1.5, 2.0 (plate: 2.5)
        x0 = w - 12.0 + 4.0 * k

        def step(b, x0=x0, height=height):
            box = bmesh.ops.create_cube(b, size=1.0)
            bmesh.ops.scale(b, verts=box["verts"], vec=(4.0, 6.0, t))
            bmesh.ops.translate(b, verts=box["verts"], vec=(x0 + 2.0, 3.0 - 0.01, height + t / 2))
        boolean(helper(step), 'DIFFERENCE')
    boolean(helper(lambda b: cylinder(b, bore / 2, -1.0, t + params.SLEEVE_TUBE_LENGTH + 1.0, tube_at)), 'DIFFERENCE')
    boolean(helper(lambda b: cylinder(b, params.FIXATION_HOLE_DIAMETER / 2, -1.0, t + 1.0, hole_at)), 'DIFFERENCE')
    curve = bpy.data.curves.new("test piece R", 'FONT')
    curve.body, curve.size, curve.extrude = "R", 9.0, 0.4
    letter = bpy.data.objects.new("test piece helper", curve)
    bpy.context.scene.collection.objects.link(letter)
    made.append(letter)
    letter.location = (14.0, 8.5, t + 0.35)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    letter_mesh = bpy.data.meshes.new_from_object(letter.evaluated_get(depsgraph))
    letter_mesh.transform(letter.matrix_world)
    solid = bpy.data.objects.new("test piece helper", letter_mesh)
    bpy.context.scene.collection.objects.link(solid)
    made.append(solid)
    boolean(solid, 'UNION')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    mesh = bpy.data.meshes.new_from_object(holder.evaluated_get(depsgraph))
    for obj in made:
        data = obj.data
        bpy.data.objects.remove(obj)
        (bpy.data.curves if isinstance(data, bpy.types.Curve) else bpy.data.meshes).remove(data)
    set_bearing(mesh, [abs(p.center.z) < 1e-4 and p.normal.z < -0.9 for p in mesh.polygons])
    return mesh


class OMFS_OT_print_test_piece(bpy.types.Operator):
    """Ajouter la pièce test synthétique au plateau: une seule impression valide le sens (coin coupé + « R »),
    l'échelle (plaque 30 x 20 mm, marches de 1 à 2,5 mm), l'alésage de la douille, le trou de vis de fixation et
    les supports (dessous = surface d'appui, sans support)"""
    bl_idname = "omfs.print_test_piece"
    bl_label = "Ajouter la pièce test"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        mesh = test_piece_mesh()
        scene = make_print_scene(context)
        add_part(scene, "Piece_test", mesh, role='GUIDE')
        arrange(scene)
        if context.window is not None:
            show_scene(context, scene)
            frame_plate(context)
        return {'FINISHED'}


class OMFS_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    prusaslicer_path: bpy.props.StringProperty(name="PrusaSlicer", subtype='FILE_PATH',
                                               description="Vide: /Applications/PrusaSlicer.app (cherché automatiquement)")
    uvtools_path: bpy.props.StringProperty(name="UVtoolsCmd", subtype='FILE_PATH',
                                           description="Vide: /Applications/UVtools.app (cherché automatiquement)")

    segmentation_python: bpy.props.StringProperty(
        name="Python de segmentation", subtype='FILE_PATH',
        description="Python de l'environnement dentalseg (SimpleITK, nnU-Net). Vide: " + params.DICOM_PYTHON)
    segmentation_folder: bpy.props.StringProperty(
        name="Dossier segmentation", subtype='DIR_PATH',
        description="Dossier qui contient segment.py, dicom_to_nrrd.py et model/. Vide: le dossier segmentation "
                    "du dépôt (à côté de l'extension)")

    def draw(self, context):
        self.layout.label(text="Nouveau cas: lecture DICOM et segmentation (laisser vide pour les valeurs par défaut)")
        self.layout.prop(self, "segmentation_python")
        self.layout.prop(self, "segmentation_folder")
        self.layout.label(text="Impression: outils externes (laisser vide pour la détection automatique)")
        self.layout.prop(self, "prusaslicer_path")
        self.layout.prop(self, "uvtools_path")


classes = (OMFS_PrintSettings, OMFS_PrintObject, OMFS_OT_print_prepare, OMFS_OT_print_back, OMFS_OT_print_show,
           OMFS_OT_print_tilt, OMFS_OT_print_arrange, OMFS_OT_print_select, OMFS_OT_print_remove,
           OMFS_OT_print_update_part, OMFS_OT_print_refresh, OMFS_OT_print_export, OMFS_OT_print_edit_resin,
           OMFS_OT_print_test_piece)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    try:
        bpy.utils.register_class(OMFS_Preferences)
    except (ValueError, RuntimeError):
        pass   # not installed as an add-on (tests): paths are found automatically
    bpy.types.Scene.omfs_print = bpy.props.PointerProperty(type=OMFS_PrintSettings)
    bpy.types.Object.omfs_print = bpy.props.PointerProperty(type=OMFS_PrintObject)
    if not bpy.app.background:
        bpy.app.timers.register(tick, first_interval=1.0, persistent=True)


def unregister():
    for job in _jobs.values():
        job.cancel.set()
    _jobs.clear()
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    del bpy.types.Object.omfs_print
    del bpy.types.Scene.omfs_print
    try:
        bpy.utils.unregister_class(OMFS_Preferences)
    except (ValueError, RuntimeError):
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
