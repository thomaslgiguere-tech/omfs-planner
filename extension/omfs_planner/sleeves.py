"""Drill-sleeve cylinders on the final screws, guide fixation holes, and the Mega Boolean
(guide workshop section 4).

Sleeve cylinders: the bicortical screws (screws.py), placed with the segment in its planned position,
give the direction; the surgeon gives the position along it. Click near a screw: the cylinder
appears on that screw's axis and slides along it with the mouse (it never leaves the axis); a
click fixes it. The tube is united to the guide, the bore subtracted from it, at once (Boolean
modifiers, so the arrows and the diameter still change it until the Mega Boolean).

Fixation holes: a click on the guide drills a hole for the Synthes 1.85 mm screw that holds the guide
to the bone, through the guide along the surface normal there.

Mega Boolean: every Boolean of the guide applied in order (unions, then differences, then the cut
edge, the fixation holes last). A pin may sit right over the guide's body: where a male overlaps the
guide itself (its own volume, before any Boolean), the guide is hollowed for it (pins.male_channel),
with the other differences. Where a sleeve tube or hole crosses a pin of the cap (male or arm),
it is subtracted from the cap too, so the drill and the sleeve pass. The cap's own Booleans are
applied after (its unions first).
"""

import bpy
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import cylinders, guide_workshop, landmarks, model_base, params, pins, screws

REACH = 30.0                    # mm past a screw's ends where a click still picks it (sleeves sit outside the bone)
PICK_DISTANCE = 10.0            # mm: a click farther than this from every screw axis picks none


# --- Geometry -----------------------------------------------------------------------------------

def along_axis(entry, exit_, origin, direction):
    """(t, distance): t = mm along the screw from its entry to the point of its axis closest to the
    mouse ray; distance between the axis there and the ray."""
    u = (Vector(exit_) - Vector(entry)).normalized()
    d = direction.normalized()
    w = Vector(entry) - origin
    b, e, f = u.dot(d), u.dot(w), d.dot(w)
    denominator = 1 - b * b
    t = 0.0 if denominator < 1e-9 else (b * f - e) / denominator
    on_axis = Vector(entry) + u * t
    s = (on_axis - origin).dot(d)
    return t, (on_axis - (origin + d * s)).length


def nearest_screw(scene, origin, direction):
    """(screw, t) whose axis passes closest to the mouse ray (within REACH of its ends), or (None, None)."""
    best = (PICK_DISTANCE, None, None)
    for screw in screws.screws(scene):
        s = screw.omfs_screw
        t, _distance = along_axis(s.entry, s.exit, origin, direction)
        t = min(max(t, -REACH), screws.length(screw) + REACH)
        u = (Vector(s.exit) - Vector(s.entry)).normalized()
        point = Vector(s.entry) + u * t
        gap = (point - origin - direction.normalized() * (point - origin).dot(direction.normalized())).length
        if gap < best[0]:
            best = (gap, screw, t)
    return best[1], best[2]


def on_screw(screw, t):
    """Matrix of a cylinder centred t mm from the screw's entry, its Z along the screw."""
    s = screw.omfs_screw
    u = (Vector(s.exit) - Vector(s.entry)).normalized()
    return Matrix.LocRotScale(Vector(s.entry) + u * t, u.to_track_quat('Z', 'Y'), None)


SLEEVES = {   # kind -> (name, diameter, length, Boolean on the guide)
    'TUBE': ("Tube manchon", lambda: params.SLEEVE_TUBE_DIAMETER,
             lambda: params.SLEEVE_TUBE_LENGTH, 'UNION'),
    'BORE': ("Trou manchon", guide_workshop.sleeve_bore_diameter,
             lambda: params.SLEEVE_TUBE_LENGTH + 2 * params.SLEEVE_BORE_OVERSHOOT, 'DIFFERENCE'),
}


def add_boolean(guide, cutter, operation):
    """Boolean of a cylinder on the guide, in order (see guide_workshop.boolean_rank)."""
    label = "Union" if operation == 'UNION' else "Trou"
    mod = guide.modifiers.new(f"{label} {cutter.name}", 'BOOLEAN')
    mod.operation, mod.object, mod.solver = operation, cutter, guide_workshop.SOLVER
    cutter.display_type = 'WIRE'
    guide_workshop.order_booleans(guide)
    return mod


def add_sleeve(context, guide, kind, screw, t):
    """New sleeve cylinder on a screw's axis, t mm from its entry (not yet on the guide)."""
    name, diameter, length, _operation = SLEEVES[kind]
    obj = cylinders.add(context, diameter(), length(), f"{name} ({screw.name})",
                        guide_workshop.workshop_collection(context.scene), on_screw(screw, t), kind)
    obj.omfs_cylinder.screw = screw.name
    return obj


def add_fixation_hole(context, guide, point, normal):
    """Hole for a guide fixation screw where the guide was clicked: through it along the normal."""
    depth = params.GUIDE_THICKNESS + 2 * params.FIXATION_HOLE_OVERSHOOT
    centre = point - normal * (params.GUIDE_THICKNESS / 2)
    matrix = Matrix.LocRotScale(centre, (-normal).to_track_quat('Z', 'Y'), None)
    obj = cylinders.add(context, params.FIXATION_HOLE_DIAMETER, depth, "Trou de fixation",
                        guide_workshop.workshop_collection(context.scene), matrix, 'FIXATION')
    add_boolean(guide, obj, 'DIFFERENCE')
    return obj


# --- Mega Boolean ---------------------------------------------------------------------------------

def world_tree(obj):
    m = obj.matrix_world
    return BVHTree.FromPolygons([tuple(m @ v.co) for v in obj.data.vertices],
                                [tuple(f.vertices) for f in obj.data.polygons])


def overlaps(a, b):
    return bool(world_tree(a).overlap(world_tree(b)))


def mother_tree(context, guide):
    """BVH of the guide itself in world coordinates: its stack without the Booleans (no boss, no tube)."""
    booleans = [m for m in guide.modifiers if m.type == 'BOOLEAN' and m.show_viewport]
    for mod in booleans:
        mod.show_viewport = False
    depsgraph = context.evaluated_depsgraph_get()
    depsgraph.update()
    mesh = guide.evaluated_get(depsgraph).to_mesh()
    m = guide.matrix_world
    tree = BVHTree.FromPolygons([tuple(m @ v.co) for v in mesh.vertices], [tuple(f.vertices) for f in mesh.polygons])
    guide.evaluated_get(depsgraph).to_mesh_clear()
    for mod in booleans:
        mod.show_viewport = True
    return tree


def in_guide(tree, obj):
    """Does obj overlap the guide's own volume (surfaces crossing, or obj inside it)?"""
    if world_tree(obj).overlap(tree):
        return True
    return model_base.inside(tree, obj.matrix_world @ obj.data.vertices[0].co)


def hollow_for_males(context, guide):
    """Channel subtracted from the guide for each male overlapping its own volume. Returns their pins."""
    tree = mother_tree(context, guide)
    done = []
    for site in pins.sites(context.scene):
        male = bpy.data.objects.get(pins.part_name(site, "mâle"))
        if site.omfs_pin.guide == guide and site.omfs_pin.has_male and male is not None and in_guide(tree, male):
            pins.male_channel(context.scene, site)
            done.append(site.name)
    return done


def cap_pins(cap):
    """The pin parts (males, arms) united to the cap."""
    return [m.object for m in cap.modifiers if m.type == 'BOOLEAN' and m.object is not None
            and m.object.omfs.made_by == pins.PART_TAG]


def subtract_from_cap(guide, cap):
    """The guide's sleeve cylinders and fixation holes that cross a pin of the cap, subtracted from the
    cap too (after its unions). Returns their names."""
    parts = cap_pins(cap)
    done = []
    for mod in guide.modifiers:
        tool = mod.object if mod.type == 'BOOLEAN' else None
        if not cylinders.is_cylinder(tool) or tool.omfs_cylinder.kind == 'OTHER':
            continue
        if any(overlaps(tool, part) for part in parts):
            if not any(m.type == 'BOOLEAN' and m.object == tool for m in cap.modifiers):
                add_boolean(cap, tool, 'DIFFERENCE')
            done.append(tool.name)
    return done


def apply_stack(context, obj):
    """Every modifier of obj applied in order (Booleans sorted first); the Boolean tools hidden."""
    guide_workshop.make_active(context, obj)
    for mod in obj.modifiers:
        if mod.type == 'BOOLEAN':
            mod.solver = guide_workshop.SOLVER   # older stacks: Exact left open edges, then a cube (case1)
    guide_workshop.order_booleans(obj)
    tools = [m.object for m in obj.modifiers if m.type == 'BOOLEAN' and m.object is not None]
    count = len(obj.modifiers)
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for tool in tools:
        tool.hide_set(True)
    return count


def mega_boolean(context):
    """Apply the guide's stack and then the cap's. Returns (report, None) or (None, error)."""
    guide = guide_workshop.target_guide(context)
    if guide is None:
        return None, "Pas encore de guide de coupe"
    cap = context.scene.omfs_pins.cap
    cap = cap if cap is not None and cap.name in context.scene.objects else None
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    guide.hide_set(False)
    crossed = subtract_from_cap(guide, cap) if cap is not None else []
    hollowed = hollow_for_males(context, guide)
    report = [f"{guide.name}: {apply_stack(context, guide)} opérations"]
    if hollowed:
        report.append("guide creusé pour le mâle de " + ", ".join(hollowed))
    if cap is not None and cap.modifiers:
        cap.hide_set(False)
        report.append(f"{cap.name}: {apply_stack(context, cap)} opérations")
    if crossed:
        report.append("soustraits aussi de la chape: " + ", ".join(crossed))
    return "; ".join(report), None


# --- Operators ------------------------------------------------------------------------------------

def mouse_ray(region, mouse):
    return (view3d_utils.region_2d_to_origin_3d(region, region.data, mouse),
            view3d_utils.region_2d_to_vector_3d(region, region.data, mouse))


class SleeveOperator:
    """Click near a screw, slide the cylinder along its axis with the mouse, click to fix it; again for
    the next one. Esc drops the one sliding; right-click or Enter: done."""
    bl_options = {'REGISTER', 'UNDO'}
    KIND = 'TUBE'

    def invoke(self, context, event):
        self.guide = guide_workshop.target_guide(context)
        if self.guide is None:
            self.report({'ERROR'}, "Pas encore de guide de coupe")
            return {'CANCELLED'}
        if not screws.screws(context.scene):
            self.report({'ERROR'}, "Placez d'abord les vis finales (Positionnement de vis bicorticale)")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        pins.take_over(self)
        self.sliding = None                             # (cylinder, screw) while it follows the mouse
        self.added = []
        self.area.header_text_set(self.header(None))
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def header(self, t):
        name = SLEEVES[self.KIND][0]
        if self.sliding is None:
            return (f"{name}: cliquez près de la vis voulue ({len(self.added)} placés)"
                    "   |   Clic droit / Entrée: terminer")
        where = f"{-t:.1f} mm devant l'entrée" if t < 0 else f"{t:.1f} mm après l'entrée"
        return (f"{self.sliding[1].name}: bougez la souris le long de la vis (centre à {where}), clic: fixer"
                "   |   Échap: retirer celui-ci")

    def drop(self):
        if self.sliding is not None:
            obj = self.sliding[0]
            mesh = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)
            self.sliding = None

    def finish(self, context):
        self.drop()
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()
        if self.added:                                  # the last one active: its arrows show
            guide_workshop.make_active(context, self.added[-1])
            guide_workshop.props(context).cutter = self.added[-1]
        return {'FINISHED'}

    def modal(self, context, event):
        if pins.superseded(self):
            self.finish(context)
            return {'PASS_THROUGH'}
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.finish(context)
        if event.value == 'PRESS' and event.type == 'ESC':
            if self.sliding is None:
                return self.finish(context)
            self.drop()
            self.area.header_text_set(self.header(None))
            return {'RUNNING_MODAL'}
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        origin, direction = mouse_ray(*under_mouse)
        if event.type == 'MOUSEMOVE' and self.sliding is not None:
            obj, screw = self.sliding
            s = screw.omfs_screw
            t, _distance = along_axis(s.entry, s.exit, origin, direction)
            obj.matrix_world = on_screw(screw, t)
            self.area.header_text_set(self.header(t))
            return {'RUNNING_MODAL'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        if self.sliding is None:
            screw, t = nearest_screw(context.scene, origin, direction)
            if screw is None:
                self.report({'WARNING'}, "Cliquez près d'une vis")
                return {'RUNNING_MODAL'}
            self.sliding = (add_sleeve(context, self.guide, self.KIND, screw, t), screw)
            self.area.header_text_set(self.header(t))
            return {'RUNNING_MODAL'}
        obj = self.sliding[0]
        add_boolean(self.guide, obj, SLEEVES[self.KIND][3])
        self.added.append(obj)
        self.sliding = None
        self.area.header_text_set(self.header(None))
        return {'RUNNING_MODAL'}


class OMFS_OT_add_sleeve_tube(SleeveOperator, bpy.types.Operator):
    """Sleeve tube on a final screw: click near the screw, slide along its axis, click. United to the guide"""
    bl_idname = "omfs.add_sleeve_tube"
    bl_label = "Tube de manchon (sur une vis)"
    KIND = 'TUBE'

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS['ADD_TUBE'])


class OMFS_OT_add_sleeve_bore(SleeveOperator, bpy.types.Operator):
    """Sleeve bore on a final screw: click near the screw, slide along its axis, click. Subtracted from the guide"""
    bl_idname = "omfs.add_sleeve_bore"
    bl_label = "Trou de manchon (sur une vis)"
    KIND = 'BORE'

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS['ADD_BORE'])


class OMFS_OT_add_fixation_hole(pins.ClickOperator, bpy.types.Operator):
    """Click on the guide where a fixation screw holds it to the bone: a hole through the guide, along
    the surface normal, subtracted last. Again for the next; right-click or Enter: done"""
    bl_idname = "omfs.add_fixation_hole"
    bl_label = "Trou de fixation du guide (clic)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS['FIXATION_HOLE'])

    def invoke(self, context, event):
        guide = guide_workshop.target_guide(context)
        if guide is None:
            self.report({'ERROR'}, "Pas encore de guide de coupe")
            return {'CANCELLED'}
        guide.hide_set(False)
        self.count = 0
        return self.start(context, guide)

    def header(self, context):
        return (f"Cliquez sur {self.target.name} où le visser à l'os ({self.count} trous)"
                "   |   Clic droit / Entrée: terminer")

    def clicked(self, context, point, normal):
        add_fixation_hole(context, self.target, point, normal)
        self.count += 1
        return False


class OMFS_OT_mega_boolean(bpy.types.Operator):
    """Apply every Boolean of the guide in order (unions, differences, cut edge, fixation holes), then
    the cap's; sleeves crossing a pin are subtracted from the cap too"""
    bl_idname = "omfs.mega_boolean"
    bl_label = "Méga Boolean (guide puis chape)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def description(cls, context, properties):
        return guide_workshop.tool_description(guide_workshop.TOOLS['MEGA_BOOLEAN'])

    def execute(self, context):
        report, error = mega_boolean(context)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        self.report({'INFO'}, report)
        return {'FINISHED'}


classes = (OMFS_OT_add_sleeve_tube, OMFS_OT_add_sleeve_bore, OMFS_OT_add_fixation_hole, OMFS_OT_mega_boolean)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
