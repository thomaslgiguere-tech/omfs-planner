"""Tubes and pins: the cutting guide and the cap (chape) plug into each other.

For each connection (usually 3):
- Female, in the cutting guide: a boss (Union) with a conical bore (Difference). Its axis is
  perpendicular to the osteotomy plane and its top face lies exactly in that plane: the male's
  shoulder rests on it, so the axial stop is the osteotomy plane.
- Male: a conical pin, the bore's shape minus the clearance, whose tip stays short of the bore's
  bottom (the shoulder is the only stop), and a shoulder and back above the plane.
- Arm, from the cap: a tube leaving the cap at 90 degrees to its surface where clicked and arriving
  along the pin's axis on its back, joined smoothly (a cubic Bezier, like plugging in a cable).
  Pin and arm are added to the cap (Union).

Every part is a small closed mesh generated from the settings, used by a Boolean modifier: the
guide and the cap meshes are untouched until the modifiers are applied, and changing a setting
rebuilds every part. Each connection is an Empty ("Pin 1"...) at its stop, Z along the pin axis
(towards the teeth): the female, male and arm are rebuilt from it, wherever the Empty is now. When
the guide follows the segment (movement.py), its Empties move with it and every part is rebuilt:
the arms then reach the pins where they are, not where they were placed.

The same pins join two printed dental models (model_base.py), so the casts find the planned
occlusion again by plugging them together (kind 'MODEL'): the female stands on the lower model's
base plane (its boss flat on the build plate), slid with the mouse in that plane, and a stem joins it
to the model; the arm leaves the upper model.
"""

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import landmarks, movement, osteotomy, params, reference_frame

SITE_TAG = "pin site"           # made_by of a connection's Empty
PART_TAG = "pin part"           # made_by of its generated parts
SEGMENTS = 48                   # sides of the round parts
OVERSHOOT = 1.0                 # mm a cutter reaches past the surface it opens, for a clean Boolean
SINK = 0.5                      # mm the arm's ends go into the cap and into the pin's back


# --- Geometry -----------------------------------------------------------------------------

def basis(u):
    """Two unit vectors perpendicular to u (and to each other)."""
    u = np.asarray(u, float)
    a = np.array((1.0, 0, 0)) if abs(u[0]) < 0.9 else np.array((0, 1.0, 0))
    v = np.cross(u, a)
    v /= np.linalg.norm(v)
    return v, np.cross(u, v)


def revolve(profile, origin, u):
    """Closed solid of revolution about the axis (origin, u): profile = [(height along u, radius)...]
    from bottom to top. Returns (vertices, faces)."""
    v, w = basis(u)
    angles = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    ring = np.outer(np.cos(angles), v) + np.outer(np.sin(angles), w)
    origin, u = np.asarray(origin, float), np.asarray(u, float)
    verts = [origin + u * h + ring * r for h, r in profile]
    verts = np.concatenate(verts + [[origin + u * profile[0][0]], [origin + u * profile[-1][0]]])
    n, k = SEGMENTS, len(profile)
    bottom, top = k * n, k * n + 1
    faces = []
    for i in range(k - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    faces += [(bottom, (j + 1) % n, j) for j in range(n)]
    faces += [(top, (k - 1) * n + j, (k - 1) * n + (j + 1) % n) for j in range(n)]
    return verts, faces


def bezier(p0, p1, p2, p3, count):
    t = np.linspace(0, 1, count)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def tube(path, radius):
    """Closed round tube along a polyline (rotation-minimising frames, so it never twists)."""
    tangents = np.gradient(path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    v, _w = basis(tangents[0])
    angles = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    rings = []
    for t in tangents:
        v = v - t * v.dot(t)                       # carry the frame along, minimal rotation
        v /= np.linalg.norm(v)
        w = np.cross(t, v)
        rings.append(np.outer(np.cos(angles), v) + np.outer(np.sin(angles), w))
    verts = np.concatenate([p + ring * radius for p, ring in zip(path, rings)] + [path[:1], path[-1:]])
    n, k = SEGMENTS, len(path)
    start, end = k * n, k * n + 1
    faces = [(i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j)
             for i in range(k - 1) for j in range(n)]
    faces += [(start, (j + 1) % n, j) for j in range(n)]
    faces += [(end, (k - 1) * n + j, (k - 1) * n + (j + 1) % n) for j in range(n)]
    return verts, faces


def slab(start, end, axis, half_width, depth):
    """Closed box from `start` to `end` (both on the top face), `half_width` either side across, and
    `depth` down `-axis`: the bridge from a female to the guide, its top flush with the plane."""
    start, end, axis = (np.asarray(x, float) for x in (start, end, axis))
    along = end - start
    across = np.cross(axis, along)
    across *= half_width / np.linalg.norm(across)
    down = -axis * depth
    top = [start - across, end - across, end + across, start + across]
    verts = np.array(top + [t + down for t in top])
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return verts, faces


def arm_path(start, normal, end, axis):
    """Centre line of the arm: leaves `start` along `normal`, arrives at `end` along -`axis`
    (down the pin's axis). A cubic Bezier whose handles are a third of the distance long."""
    start, normal, end, axis = (np.asarray(x, float) for x in (start, normal, end, axis))
    reach = np.linalg.norm(end - start) / 3
    count = max(16, int(np.linalg.norm(end - start) / 0.5))
    return bezier(start - normal * SINK, start + normal * reach, end + axis * reach, end - axis * SINK, count)


def tightest_bend(path):
    """Smallest radius of curvature along a polyline (mm): below the tube's radius it folds on itself."""
    d1, d2 = np.gradient(path, axis=0), np.gradient(np.gradient(path, axis=0), axis=0)
    speed = np.linalg.norm(d1, axis=1)
    curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(speed ** 3, 1e-12)
    return float(1 / curvature.max()) if curvature.max() > 1e-12 else float('inf')


# --- Dimensions ---------------------------------------------------------------------------

def dims(scene):
    p = scene.omfs_pins
    r_base, r_tip = p.base_diameter / 2, p.tip_diameter / 2
    return {
        "male": [(-p.cone_length + p.tip_gap, r_tip + (r_base - r_tip) * p.tip_gap / p.cone_length),
                 (0.0, r_base), (0.0, p.shoulder_diameter / 2), (p.back_length, p.shoulder_diameter / 2)],
        "bore": [(-p.cone_length, r_tip + p.clearance), (0.0, r_base + p.clearance),
                 (OVERSHOOT, r_base + p.clearance)],
        "boss": [(-p.cone_length - p.floor, r_base + p.wall), (0.0, r_base + p.wall)],
    }


# --- Connections ---------------------------------------------------------------------------

def is_site(obj):
    return isinstance(obj, bpy.types.Object) and obj.omfs.made_by == SITE_TAG


def sites(scene, kind='GUIDE'):
    """The connections of the cutting guide ('GUIDE') or of the dental models ('MODEL'), in order."""
    return sorted((o for o in scene.objects if is_site(o) and o.omfs_pin.kind == kind),
                  key=lambda o: o.omfs_pin.number)


def osteotomy_plane(scene, guide):
    """(point, unit normal towards the teeth) of the osteotomy plane where the guide is: moved with it
    if the guide followed the segment. None without a cut."""
    cut = osteotomy.find_cut(scene)
    if cut is None:
        return None
    c = cut.omfs_cut
    point = cut.matrix_world @ Vector((0, 0, -c.below_foramina))
    normal = (cut.matrix_world.to_3x3().normalized() @ osteotomy.plane_normal(c)).normalized()
    if guide is not None and movement.guide_moved(guide):
        moved = guide.matrix_world @ movement.preop_of(guide).inverted()
        point, normal = moved @ point, (moved.to_3x3() @ normal).normalized()
    frame = reference_frame.find_frame(scene)
    up = frame.matrix_world.col[2].xyz if frame else Vector((0, 0, 1))
    return point, (normal if normal.dot(up) >= 0 else -normal)


def stop_point(click, surface_normal, plane_point, axis, offset):
    """Where the pin's axis meets the plane: the axis runs `offset` mm out from the clicked surface
    point (sideways, perpendicular to the axis), parallel to `axis`."""
    out = surface_normal - axis * surface_normal.dot(axis)
    out = out.normalized() if out.length > 1e-6 else Vector((0, 0, 0))
    through = click + out * offset
    return through + axis * (plane_point - through).dot(axis)


def make_part(name, verts, faces, collection):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        collection.objects.link(obj)
        obj.omfs.made_by = PART_TAG
        obj.display_type = 'WIRE'
        obj.hide_render = True
    obj.matrix_world = Matrix.Identity(4)         # vertices in world coordinates (older parts were moved
    obj.pop(movement.GUIDE_PREOP, None)           # with the guide: the Empty moves now, they are rebuilt)
    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(np.asarray(verts).tolist(), [], faces)
    mesh.update()
    return obj


def boolean_on(target, part, operation, solver):
    mod = next((m for m in target.modifiers if m.type == 'BOOLEAN' and m.object == part), None)
    if mod is None:
        mod = target.modifiers.new(part.name, 'BOOLEAN')
        mod.object, mod.operation = part, operation
    mod.solver = solver                           # also on older parts made with another solver
    return mod


def part_name(site, kind):
    return f"{site.name}: {kind}"


def stem_ends(site):
    """World ends of the bridge / stem, stored in the Empty's coordinates (they move with it)."""
    s = site.omfs_pin
    if not s.stem_local:                          # stored in world coordinates before 2026-11-04
        to_site = site.matrix_world.inverted()
        s.stem_start, s.stem_end = to_site @ Vector(s.stem_start), to_site @ Vector(s.stem_end)
        s.stem_local = True
    return site.matrix_world @ Vector(s.stem_start), site.matrix_world @ Vector(s.stem_end)


def set_stem(site, start, end):
    s = site.omfs_pin
    to_site = site.matrix_world.inverted()
    s.stem_start, s.stem_end, s.stem_local = to_site @ start, to_site @ end, True


def build(scene, site):
    """(Re)make the parts of one connection from its Empty and the settings. Returns a warning or None."""
    from . import guide_workshop                  # guide_workshop imports this module's neighbours
    s = site.omfs_pin
    d = dims(scene)
    origin = site.matrix_world.translation
    axis = (site.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    coll = site.users_collection[0]
    female_solver = guide_workshop.SOLVER        # Exact left open edges (case1: 252 on a model, a few on the guide)
    warning = None
    if s.guide is not None:
        boss = make_part(part_name(site, "bossage"), *revolve(d["boss"], origin, axis), coll)
        bore = make_part(part_name(site, "logement"), *revolve(d["bore"], origin, axis), coll)
        boolean_on(s.guide, boss, 'UNION', female_solver)
        kind = "pont" if s.kind == 'GUIDE' else "tige"
        stem = bpy.data.objects.get(part_name(site, kind))
        if s.has_stem:
            p = scene.omfs_pins
            start, end = stem_ends(site)
            if s.kind == 'GUIDE':                        # bridge: the boss's guide-side half, out to the guide
                shape = slab(start, end, axis, p.base_diameter / 2 + p.wall, p.cone_length + p.floor)
            else:                                        # stem: a rod flat on the model's base plane
                shape = tube(np.linspace(np.array(start), np.array(end), 8), p.stem_diameter / 2)
            stem = make_part(part_name(site, kind), *shape, coll)
            boolean_on(s.guide, stem, 'UNION', female_solver)
        elif stem is not None:                           # slid back against the guide: none any more
            remove_part(stem)
        boolean_on(s.guide, bore, 'DIFFERENCE', female_solver)
        guide_workshop.order_booleans(s.guide)
    if s.cap is not None and s.has_male:
        male = make_part(part_name(site, "mâle"), *revolve(d["male"], origin, axis), coll)
        boolean_on(s.cap, male, 'UNION', 'MANIFOLD')
    if s.cap is not None and s.has_arm:
        back = origin + axis * scene.omfs_pins.back_length
        path = arm_path(s.arm_start, s.arm_normal, back, axis)
        radius = scene.omfs_pins.arm_diameter / 2
        arm = make_part(part_name(site, "bras"), *tube(path, radius), coll)
        boolean_on(s.cap, arm, 'UNION', 'MANIFOLD')
        if tightest_bend(path) < radius:
            warning = f"{site.name}: le bras se plie trop serré (départ trop proche de la pin)"
    return warning


def build_all(scene):
    warnings = [w for kind in ('GUIDE', 'MODEL') for w in (build(scene, site) for site in sites(scene, kind)) if w]
    return "; ".join(warnings) or None


def guide_start(scene, guide, click, surface_normal):
    """Where a female first appears for a click on the guide: its axis `embed` mm inside the boss's
    edge from the clicked surface, its stop on the osteotomy plane. None without a cut."""
    plane = osteotomy_plane(scene, guide)
    if plane is None:
        return None
    plane_point, axis = plane
    p = scene.omfs_pins
    return stop_point(click, surface_normal, plane_point, axis, p.base_diameter / 2 + p.wall - p.embed)


def guide_stop(scene, guide, foot):
    """(stop, axis) for a female whose axis passes through `foot`: the stop is where that axis meets
    the osteotomy plane, so the axial stop stays exactly in the plane wherever the pin slides."""
    plane_point, axis = osteotomy_plane(scene, guide)
    return foot + axis * (plane_point - foot).dot(axis), axis


def guide_stem(scene, guide, stop, axis):
    """(start, end, gap) of the bridge joining a female slid away from the guide: from the pin's axis,
    in the osteotomy plane, towards the closest point of the guide, out to PIN_STEM_SINK into it, as
    wide as the boss and as deep (its top in the plane: more cutting-plane surface for the blade).
    No bridge (start None) while the boss still reaches into the guide. `gap` = distance from the
    axis to the guide, across the axis."""
    p = scene.omfs_pins
    end, gap = stem_to(guide, stop - axis * (p.cone_length + p.floor) / 2, axis)
    if end is None or gap <= p.base_diameter / 2 + p.wall - p.embed:
        return None, None, gap
    return stop, end + axis * (p.cone_length + p.floor) / 2, gap   # both on the top face, in the plane


def add_site(scene, guide, foot):
    """New female on the guide, its axis through `foot`, its stop on the osteotomy plane; a bridge joins
    it to the guide if it was slid away. Returns the Empty, or an error message."""
    if osteotomy_plane(scene, guide) is None:
        return "Il faut le plan d'ostéotomie (panneau Osteotomy)"
    stop, axis = guide_stop(scene, guide, foot)
    number = max((o.omfs_pin.number for o in sites(scene)), default=0) + 1
    site = bpy.data.objects.new(f"Pin {number}", None)
    site.empty_display_type = 'SINGLE_ARROW'
    site.empty_display_size = 6.0
    guide.users_collection[0].objects.link(site)
    site.omfs.made_by = SITE_TAG
    site.matrix_world = Matrix.LocRotScale(stop, axis.to_track_quat('Z', 'Y'), None)
    s = site.omfs_pin
    s.number, s.guide = number, guide
    start, end, _gap = guide_stem(scene, guide, stop, axis)
    s.has_stem = start is not None
    if start is not None:
        set_stem(site, start, end)
    build(scene, site)
    return site


def add_males(scene, cap):
    count = 0
    for site in sites(scene):
        if not site.omfs_pin.has_male:
            site.omfs_pin.cap, site.omfs_pin.has_male = cap, True
            build(scene, site)
            count += 1
    return count


def set_arm(scene, site, cap, start, normal):
    s = site.omfs_pin
    s.cap, s.has_male, s.has_arm = cap, True, True
    s.arm_start, s.arm_normal = start, normal
    return build(scene, site)


def remove_part(part):
    """A part and the Boolean modifiers using it."""
    for obj in bpy.data.objects:
        for mod in [m for m in obj.modifiers if m.type == 'BOOLEAN' and m.object == part]:
            obj.modifiers.remove(mod)
    mesh = part.data
    bpy.data.objects.remove(part)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def remove_site(site):
    for kind in ("bossage", "logement", "tige", "pont", "mâle", "bras"):
        part = bpy.data.objects.get(part_name(site, kind))
        if part is not None:
            remove_part(part)
    bpy.data.objects.remove(site)


# --- Pins between two dental models -------------------------------------------------------------

def model_base_plane(model):
    """(point, unit up) of a model's flat base, in world coordinates: up points at the teeth."""
    from . import model_base
    report = model_base.last_report(model)
    up_local = Vector(report["up_vector"]).normalized()
    co = np.empty(len(model.data.vertices) * 3)
    model.data.vertices.foreach_get("co", co)
    lowest = float((co.reshape(-1, 3) @ np.array(up_local)).min())   # the base is the lowest along up
    point = model.matrix_world @ (up_local * lowest)
    return point, (model.matrix_world.to_3x3() @ up_local).normalized()


def model_stop(scene, lower, foot):
    """Pin stop above a point of the lower model's base plane: the boss stands on the plane."""
    _point, up = model_base_plane(lower)
    p = scene.omfs_pins
    return foot + up * (p.cone_length + p.floor), up


def stem_to(target, start, axis=None):
    """(end, gap) of a stem from `start` to the target's closest surface point, sunk PIN_STEM_SINK
    into it; with `axis`, the stem stays across it (flat on a model's base plane). `gap` = the
    stem's length to the surface."""
    to_local = target.matrix_world.inverted()
    tree = BVHTree.FromPolygons([tuple(v.co) for v in target.data.vertices],
                                [tuple(f.vertices) for f in target.data.polygons])
    location = tree.find_nearest(to_local @ start)[0]
    if location is None:
        return None, None
    flat = target.matrix_world @ location - start
    if axis is not None:
        flat -= axis * flat.dot(axis)                    # across the axis, never along it
    gap = flat.length
    if gap < 1e-6:
        return start, 0.0
    return start + flat.normalized() * (gap + params.PIN_STEM_SINK), gap


def stem_end(scene, lower, foot, up):
    """Where the stem from the pin (at its foot, on the base plane) enters the lower model."""
    return stem_to(lower, foot + up * scene.omfs_pins.stem_diameter / 2, up)


def add_model_site(scene, lower, upper, foot):
    """New connection standing on the lower model's base plane at `foot`. Returns the Empty."""
    stop, up = model_stop(scene, lower, foot)
    number = max((o.omfs_pin.number for o in sites(scene, 'MODEL')), default=0) + 1
    site = bpy.data.objects.new(f"Pin modèle {number}", None)
    site.empty_display_type = 'SINGLE_ARROW'
    site.empty_display_size = 6.0
    lower.users_collection[0].objects.link(site)
    site.omfs.made_by = SITE_TAG
    site.matrix_world = Matrix.LocRotScale(stop, up.to_track_quat('Z', 'Y'), None)
    s = site.omfs_pin
    s.kind, s.number, s.guide, s.cap = 'MODEL', number, lower, upper
    end, _distance = stem_end(scene, lower, foot, up)
    s.has_stem = end is not None
    if end is not None:
        set_stem(site, foot + up * scene.omfs_pins.stem_diameter / 2, end)
    build(scene, site)
    return site


# --- Properties -------------------------------------------------------------------------------

def _on_dims(props, context):
    build_all(context.scene)


def _dim(name, default, description, minimum=0.0):
    return bpy.props.FloatProperty(name=name, subtype='DISTANCE', default=default, min=minimum, soft_max=20.0,
                                   step=10, precision=2, update=_on_dims, description=description)


class OMFS_PinSettings(bpy.types.PropertyGroup):
    """Dimensions shared by every connection (scene.omfs_pins)."""
    base_diameter: _dim("Ø au plan", params.PIN_BASE_DIAMETER, "Diamètre du cône de la pin au plan d'ostéotomie", 0.5)
    tip_diameter: _dim("Ø pointe", params.PIN_TIP_DIAMETER, "Diamètre du cône à la pointe (plus petit: conique)", 0.3)
    cone_length: _dim("Longueur du cône", params.PIN_CONE_LENGTH, "Profondeur d'emboîtement sous le plan", 0.5)
    clearance: _dim("Jeu", params.PIN_CLEARANCE, "Jeu radial entre la pin et son logement")
    tip_gap: _dim("Écart en pointe", params.PIN_TIP_GAP,
                  "La pointe s'arrête avant le fond: l'épaulement sur le plan est la seule butée")
    wall: _dim("Paroi du bossage", params.PIN_WALL, "Épaisseur autour du logement", 0.3)
    floor: _dim("Fond du bossage", params.PIN_FLOOR, "Épaisseur sous la pointe du logement", 0.3)
    embed: _dim("Ancrage dans le guide", params.PIN_EMBED, "Profondeur du bossage dans la surface du guide")
    shoulder_diameter: _dim("Ø épaulement", params.PIN_SHOULDER_DIAMETER,
                            "Diamètre de la pin au-dessus du plan: il appuie sur le dessus du bossage", 0.5)
    back_length: _dim("Dos de la pin", params.PIN_BACK_LENGTH, "Longueur au-dessus du plan, jusqu'au bras", 0.3)
    arm_diameter: _dim("Ø du bras", params.PIN_ARM_DIAMETER, "Diamètre du tube qui part de la chape", 0.5)
    stem_diameter: _dim("Ø de la tige", params.PIN_STEM_DIAMETER,
                        "Modèles: diamètre de la tige qui relie la femelle au modèle du bas, au ras de la base", 0.5)
    cap: bpy.props.PointerProperty(name="Chape", type=bpy.types.Object,
                                   description="La chape (ou gouttière) qui porte les pins mâles et leurs bras")


class OMFS_PinSite(bpy.types.PropertyGroup):
    """One connection (on its Empty)."""
    number: bpy.props.IntProperty()
    kind: bpy.props.EnumProperty(items=(('GUIDE', "Guide", "Guide de coupe et chape"),
                                        ('MODEL', "Modèles", "Modèle du bas (femelle) et du haut (bras)")))
    guide: bpy.props.PointerProperty(type=bpy.types.Object)   # the female's object (guide, lower model)
    cap: bpy.props.PointerProperty(type=bpy.types.Object)
    has_male: bpy.props.BoolProperty()
    has_arm: bpy.props.BoolProperty()
    arm_start: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    arm_normal: bpy.props.FloatVectorProperty(size=3, subtype='XYZ', default=(0, 0, 1))
    has_stem: bpy.props.BoolProperty()
    stem_start: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    stem_end: bpy.props.FloatVectorProperty(size=3, subtype='XYZ')
    stem_local: bpy.props.BoolProperty()          # stem ends in the Empty's coordinates (see stem_ends)


# --- Clicking ---------------------------------------------------------------------------------

def surface_hit(context, region, mouse, obj):
    """World point and normal under the mouse on one object, or None."""
    origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
    to_local = obj.matrix_world.inverted()
    hit, location, normal, _index = obj.ray_cast(to_local @ origin, to_local.to_3x3() @ direction,
                                                 depsgraph=context.evaluated_depsgraph_get())
    if not hit:
        return None
    return obj.matrix_world @ location, (obj.matrix_world.to_3x3() @ normal).normalized()


_running = []                   # the click tool in use: starting another one ends it


def take_over(operator):
    """`operator` becomes the click tool in use; the one running before ends at its next event
    (otherwise it keeps eating the clicks: case1, the females tool still running asked for the lower
    model while the surgeon wanted to click the arms)."""
    _running[:] = [operator]


def superseded(operator):
    return not _running or _running[0] is not operator


class ClickOperator:
    """Modal clicks on one object, with a header; subclasses say what a click does."""
    def start(self, context, target):
        take_over(self)
        self.target = target
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

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        return {'FINISHED'}

    def modal(self, context, event):
        if superseded(self):
            self.finish(context)
            return {'PASS_THROUGH'}
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC', 'RET', 'NUMPAD_ENTER'}:
            return self.finish(context)
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        hit = surface_hit(context, *under_mouse, self.target)
        if hit is None:
            self.report({'WARNING'}, f"Cliquez sur {self.target.name}")
            return {'RUNNING_MODAL'}
        if self.clicked(context, *hit):
            return self.finish(context)
        self.area.header_text_set(self.header(context))
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}


class OMFS_OT_add_pin_arm(ClickOperator, bpy.types.Operator):
    """For each pin in turn, click on the cap (or the upper model) where its arm starts: the tube
    leaves at 90 degrees and arrives on the pin's back along its axis, joined by a smooth curve"""
    bl_idname = "omfs.add_pin_arm"
    bl_label = "Bras depuis la chape (un clic par pin)"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(items=(('GUIDE', "Guide", ""), ('MODEL', "Modèles", "")))

    def invoke(self, context, event):
        self.queue = [s for s in sites(context.scene, self.kind) if not s.omfs_pin.has_arm]
        if self.kind == 'MODEL':
            cap = self.queue[0].omfs_pin.cap if self.queue else None
        else:
            cap = context.scene.omfs_pins.cap
        if not self.queue:
            self.report({'ERROR'}, "Toutes les pins ont déjà un bras (ou ajoutez d'abord les femelles)")
            return {'CANCELLED'}
        if cap is None:
            self.report({'ERROR'}, "Choisissez la chape" if self.kind == 'GUIDE' else "Choisissez le modèle du haut")
            return {'CANCELLED'}
        cap.hide_set(False)
        return self.start(context, cap)

    def header(self, context):
        return (f"Cliquez sur {self.target.name} le départ du bras de {self.queue[0].name}"
                "   |   Clic droit / Entrée: terminer")

    def clicked(self, context, point, normal):
        warning = set_arm(context.scene, self.queue.pop(0), self.target, point, normal)
        if warning:
            self.report({'WARNING'}, warning)
        return not self.queue


class SlideOperator:
    """Click the part, slide the female with the mouse (it stays in its plane, the distance to the
    part shows in the header), click to confirm; again for the next one. Right-click or Enter: done.
    Subclasses give the part, the plane, the first position, the stop for a point of the plane and
    what a confirmed female becomes."""
    PREVIEW = "Pin: aperçu"

    def invoke(self, context, event):
        error = self.setup(context)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        take_over(self)
        self.part.hide_set(False)
        self.sliding = None                              # offset from the mouse to the pin while sliding
        self.added = 0
        self.preview = None
        self.area.header_text_set(self.header(None))
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def header(self, distance):
        if self.sliding is None:
            return (f"Cliquez sur {self.part.name} près de l'endroit voulu ({self.added} femelles placées)"
                    f"   |   Clic droit / Entrée: terminer (ou lancez « {self.next_step} »)")
        where = "dans la pièce" if distance is None else f"à {distance:.1f} mm de la pièce"
        return f"Bougez la souris pour placer la femelle ({where}), clic: confirmer   |   Échap: annuler celle-ci"

    def on_plane(self, region, mouse):
        """Where the mouse ray meets the plane, or None."""
        origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
        direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
        point, normal = self.plane
        along = direction.dot(normal)
        if abs(along) < 1e-6:
            return None
        return origin + direction * ((point - origin).dot(normal) / along)

    def show_preview(self, context, foot):
        scene = context.scene
        stop, axis = self.stop_of(scene, foot)
        self.preview = make_part(self.PREVIEW, *revolve(dims(scene)["boss"], stop, axis), self.part.users_collection[0])
        self.area.header_text_set(self.header(self.distance_of(scene, foot)))

    def drop_preview(self):
        if self.preview is not None:
            mesh = self.preview.data
            bpy.data.objects.remove(self.preview)
            bpy.data.meshes.remove(mesh)
            self.preview = None

    def finish(self, context):
        self.drop_preview()
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        return {'FINISHED'}

    def modal(self, context, event):
        if superseded(self):
            self.finish(context)
            return {'PASS_THROUGH'}
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.finish(context)
        if event.value == 'PRESS' and event.type == 'ESC':
            if self.sliding is None:
                return self.finish(context)
            self.sliding = None
            self.drop_preview()
            self.area.header_text_set(self.header(None))
            return {'RUNNING_MODAL'}
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE' and self.sliding is not None:
            mouse = self.on_plane(*under_mouse)
            if mouse is not None:
                self.show_preview(context, mouse + self.sliding)
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        mouse = self.on_plane(*under_mouse)
        if self.sliding is None:
            hit = surface_hit(context, *under_mouse, self.part)
            if hit is None or mouse is None:
                self.report({'WARNING'}, f"Cliquez sur {self.part.name}")
                return {'RUNNING_MODAL'}
            start = self.first_position(context.scene, *hit)
            self.sliding = start - mouse                 # the pin starts there and follows the mouse
            self.show_preview(context, start)
            return {'RUNNING_MODAL'}
        if mouse is None:
            return {'RUNNING_MODAL'}
        self.drop_preview()
        self.add(context.scene, mouse + self.sliding)
        self.added += 1
        self.sliding = None
        self.area.header_text_set(self.header(None))
        return {'RUNNING_MODAL'}


class OMFS_OT_add_pin_female(SlideOperator, bpy.types.Operator):
    """Place the females on the cutting guide (usually 3): click the guide, slide the pin with the
    mouse in the osteotomy plane (its axial stop always exactly in the plane), click to confirm. A
    bridge joins it to the guide if it is slid away. Right-click or Enter: done"""
    bl_idname = "omfs.add_pin_female"
    bl_label = "Ajouter les femelles (clic, glisser, clic)"
    bl_options = {'REGISTER', 'UNDO'}
    next_step = "Bras depuis la chape"

    def setup(self, context):
        from . import guide_workshop
        self.part = guide_workshop.target_guide(context)
        if self.part is None:
            return "Pas encore de guide de coupe"
        self.plane = osteotomy_plane(context.scene, self.part)
        if self.plane is None:
            return "Il faut le plan d'ostéotomie (panneau Osteotomy)"
        return None

    def first_position(self, scene, point, normal):
        return guide_start(scene, self.part, point, normal)

    def stop_of(self, scene, foot):
        return guide_stop(scene, self.part, foot)

    def distance_of(self, scene, foot):
        stop, axis = guide_stop(scene, self.part, foot)
        p = scene.omfs_pins
        _start, _end, gap = guide_stem(scene, self.part, stop, axis)
        return None if gap is None else max(0.0, gap - (p.base_diameter / 2 + p.wall))

    def add(self, scene, foot):
        add_site(scene, self.part, foot)


class OMFS_OT_add_model_pin(SlideOperator, bpy.types.Operator):
    """Place the females on the lower model's base: click the lower model, slide the pin with the
    mouse (it stays on the base plane, the distance to the model shows in the header), click to
    confirm; a stem joins it to the model. Right-click or Enter: done"""
    bl_idname = "omfs.add_model_pin"
    bl_label = "Ajouter les femelles (base du modèle du bas)"
    bl_options = {'REGISTER', 'UNDO'}
    next_step = "Bras depuis le modèle du haut"

    def setup(self, context):
        from . import model_base
        self.part, self.upper = model_base.pin_models(context.scene)
        if self.part is None or self.upper is None:
            return "Il faut le modèle du bas et le modèle du haut (Créer le modèle)"
        self.plane = model_base_plane(self.part)
        return None

    def first_position(self, scene, point, normal):
        base, up = self.plane
        return point - up * (point - base).dot(up)       # straight down onto the base plane

    def stop_of(self, scene, foot):
        return model_stop(scene, self.part, foot)

    def distance_of(self, scene, foot):
        _end, distance = stem_end(scene, self.part, foot, self.plane[1])
        return distance

    def add(self, scene, foot):
        add_model_site(scene, self.part, self.upper, foot)


class OMFS_OT_remove_pin(bpy.types.Operator):
    """Remove this connection: its female, male and arm, and their Boolean modifiers"""
    bl_idname = "omfs.remove_pin"
    bl_label = "Retirer la pin"
    bl_options = {'REGISTER', 'UNDO'}

    site_name: bpy.props.StringProperty()

    def execute(self, context):
        site = bpy.data.objects.get(self.site_name)
        if is_site(site):
            remove_site(site)
        return {'FINISHED'}


classes = (OMFS_PinSite, OMFS_PinSettings, OMFS_OT_add_pin_female, OMFS_OT_add_pin_arm, OMFS_OT_add_model_pin,
           OMFS_OT_remove_pin)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_pins = bpy.props.PointerProperty(type=OMFS_PinSettings)
    bpy.types.Object.omfs_pin = bpy.props.PointerProperty(type=OMFS_PinSite)


def unregister():
    del bpy.types.Object.omfs_pin
    del bpy.types.Scene.omfs_pins
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
