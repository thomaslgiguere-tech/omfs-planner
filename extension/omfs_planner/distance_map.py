"""Section & Analysis: distance map. The source surface coloured by its distance to a target.

Generic on purpose (segment to bone, guide to bone, later screw to root with a ~2 mm margin).

Non-destructive: a Geometry Nodes modifier on the source. For every source vertex it finds the
nearest point of the target surface (Geometry Proximity), signs the distance with the target's
normal there (negative = inside the target), and stores the distance and a colour as attributes
of the *evaluated* mesh only. The source mesh data is never written; removing the modifier
removes the map. It updates live when the source or the target moves.

The minimum is taken over the source vertices (exact on the target side): on a coarse source
mesh the true minimum can be slightly smaller.
"""

import blf
import bpy
import gpu
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

from . import analysis, params, scene_setup

GROUP = "OMFS Distance Map"
MODIFIER = "Distance Map"
DISTANCE = "OMFS distance"         # float per vertex, mm, negative inside the target
NEAREST = "OMFS nearest"           # nearest target point per vertex, source local coordinates
COLOUR = "OMFS distance colour"
RED, YELLOW, GREEN = (0.90, 0.10, 0.10, 1.0), (0.95, 0.85, 0.10, 1.0), (0.20, 0.75, 0.25, 1.0)
MARKER_COLOUR = (1.0, 0.2, 1.0, 1.0)

_closest = {}   # source name -> (min distance, source point, target point), world coordinates


def legend(props):
    red, yellow = props.dist_red, props.dist_yellow
    return [('STRIP_COLOR_01', f"< {red:g} mm (or inside the target)"),
            ('STRIP_COLOR_03', f"{red:g} – {yellow:g} mm"),
            ('STRIP_COLOR_04', f"≥ {yellow:g} mm")]


# --- The node group ------------------------------------------------------------------

def build_group():
    """Geometry Nodes: signed distance to the target, colour bands, stored on the evaluated mesh."""
    tree = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
    face = tree.interface
    face.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    face.new_socket("Target", in_out='INPUT', socket_type='NodeSocketObject')
    face.new_socket("Red below", in_out='INPUT', socket_type='NodeSocketFloat').default_value = params.DIST_RED
    face.new_socket("Yellow below", in_out='INPUT', socket_type='NodeSocketFloat').default_value = params.DIST_YELLOW
    face.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, link = tree.nodes, tree.links.new

    def node(kind, **settings):
        n = nodes.new(kind)
        for key, value in settings.items():
            setattr(n, key, value)
        return n

    group_in, group_out = node('NodeGroupInput'), node('NodeGroupOutput')
    target = node('GeometryNodeObjectInfo', transform_space='RELATIVE')   # target in the source's space
    link(group_in.outputs["Target"], target.inputs["Object"])
    nearest = node('GeometryNodeProximity', target_element='FACES')
    link(target.outputs["Geometry"], nearest.inputs["Target"])
    normal = node('GeometryNodeSampleNearestSurface', data_type='FLOAT_VECTOR')   # target normal there
    link(target.outputs["Geometry"], normal.inputs["Mesh"])
    link(node('GeometryNodeInputNormal').outputs[0], normal.inputs["Value"])
    away = node('ShaderNodeVectorMath', operation='SUBTRACT')   # nearest point -> vertex
    link(node('GeometryNodeInputPosition').outputs[0], away.inputs[0])
    link(nearest.outputs["Position"], away.inputs[1])
    side = node('ShaderNodeVectorMath', operation='DOT_PRODUCT')
    link(away.outputs["Vector"], side.inputs[0])
    link(normal.outputs["Value"], side.inputs[1])
    sign = node('ShaderNodeMath', operation='SIGN')   # + outside the target, - inside
    link(side.outputs["Value"], sign.inputs[0])
    signed = node('ShaderNodeMath', operation='MULTIPLY')
    link(nearest.outputs["Distance"], signed.inputs[0])
    link(sign.outputs[0], signed.inputs[1])

    def below(threshold):
        compare = node('FunctionNodeCompare', data_type='FLOAT', operation='LESS_THAN')
        link(signed.outputs[0], compare.inputs["A"])
        link(group_in.outputs[threshold], compare.inputs["B"])
        return compare.outputs["Result"]

    yellow_or_green = node('GeometryNodeSwitch', input_type='RGBA')
    link(below("Yellow below"), yellow_or_green.inputs["Switch"])
    yellow_or_green.inputs["False"].default_value = GREEN
    yellow_or_green.inputs["True"].default_value = YELLOW
    colour = node('GeometryNodeSwitch', input_type='RGBA')
    link(below("Red below"), colour.inputs["Switch"])
    link(yellow_or_green.outputs[0], colour.inputs["False"])
    colour.inputs["True"].default_value = RED

    geometry = group_in.outputs["Geometry"]
    for name, data_type, value in ((DISTANCE, 'FLOAT', signed.outputs[0]),
                                   (NEAREST, 'FLOAT_VECTOR', nearest.outputs["Position"]),
                                   (COLOUR, 'FLOAT_COLOR', colour.outputs[0])):
        store = node('GeometryNodeStoreNamedAttribute', data_type=data_type, domain='POINT')
        store.inputs["Name"].default_value = name
        link(geometry, store.inputs["Geometry"])
        link(value, store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    link(geometry, group_out.inputs["Geometry"])
    return tree


def group():
    return bpy.data.node_groups.get(GROUP) or build_group()


def socket_id(name):
    return next(s.identifier for s in group().interface.items_tree
                if s.item_type == 'SOCKET' and s.in_out == 'INPUT' and s.name == name)


def set_input(mod, name, value):
    """A modifier input (Blender 5: modifier.properties.inputs.<socket identifier>.value)."""
    getattr(mod.properties.inputs, socket_id(name)).value = value


# --- The modifier ----------------------------------------------------------------------

def find_map(scene):
    """(source, modifier) of the distance map, or (None, None)."""
    for obj in scene.objects:
        mod = obj.modifiers.get(MODIFIER) if obj.type == 'MESH' else None
        if mod is not None and mod.type == 'NODES':
            return obj, mod
    return None, None


def set_thresholds(scene):
    source, mod = find_map(scene)
    if mod is not None:
        props = scene.omfs_section
        set_input(mod, "Red below", props.dist_red)
        set_input(mod, "Yellow below", props.dist_yellow)
        source.update_tag()   # modifier inputs set from Python: ask for a re-evaluation


def add_map(scene, source, target):
    remove_map(scene)
    mod = source.modifiers.new(MODIFIER, 'NODES')
    mod.node_group = group()
    set_input(mod, "Target", target)
    # Other colour attributes of the source (e.g. the occlusion map) would be shown instead.
    for attribute in source.data.color_attributes:
        if mod.node_group.nodes.get(f"Hide {attribute.name}") is None:
            hide_attribute(mod.node_group, attribute.name)
    set_thresholds(scene)
    return mod


def hide_attribute(tree, name):
    """Remove a colour attribute from the evaluated mesh only, so ours is the one shown."""
    out = next(n for n in tree.nodes if n.bl_idname == 'NodeGroupOutput')
    last = out.inputs["Geometry"].links[0].from_socket
    remove = tree.nodes.new('GeometryNodeRemoveAttribute')
    remove.name = f"Hide {name}"
    remove.inputs["Name"].default_value = name
    tree.links.new(last, remove.inputs["Geometry"])
    tree.links.new(remove.outputs["Geometry"], out.inputs["Geometry"])


def remove_map(scene):
    source, mod = find_map(scene)
    if source is not None:
        _closest.pop(source.name, None)
        source.modifiers.remove(mod)


def closest(source, depsgraph):
    """(min distance, source point, target point) in world coordinates, from the evaluated mesh."""
    evaluated = source.evaluated_get(depsgraph)
    mesh = evaluated.data
    attribute = mesh.attributes.get(DISTANCE)
    if attribute is None or not len(mesh.vertices):
        return None
    d = np.empty(len(mesh.vertices), np.float32)
    attribute.data.foreach_get("value", d)
    i = int(np.argmin(d))
    m = evaluated.matrix_world
    return float(d[i]), m @ mesh.vertices[i].co, m @ mesh.attributes[NEAREST].data[i].vector


def cached_closest(source):
    return _closest.get(source.name)


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """The map is re-evaluated when the source or the target moves: find the new minimum."""
    source, _mod = find_map(scene)
    if source is None:
        return
    if source.name not in _closest or any(
            u.is_updated_geometry and isinstance(u.id, bpy.types.Object) and u.id.original == source
            for u in depsgraph.updates):
        _closest[source.name] = closest(source, depsgraph)


def set_shading(context, on):
    """Attribute colours in Solid mode show the map; off: back to the role colours."""
    for _window, area, _region, space in scene_setup.view3d_areas(context):
        space.shading.color_type = 'VERTEX' if on else 'OBJECT'
        area.tag_redraw()


# --- Closest point marker ----------------------------------------------------------------

def _marker():
    scene = bpy.context.scene
    source, _mod = find_map(scene)
    found = cached_closest(source) if source else None
    if not found or not scene.omfs_section.dist_marker:
        return None
    return found


def _draw_marker_3d():
    found = _marker()
    if found is None:
        return
    _d, p, q = found
    gpu.state.depth_test_set('NONE')
    line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    line.uniform_float("viewportSize", (region.width, region.height))
    line.uniform_float("lineWidth", 2.0)
    line.uniform_float("color", MARKER_COLOUR)
    batch_for_shader(line, 'LINES', {"pos": [p, q]}).draw(line)
    dots = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    gpu.state.point_size_set(10.0)
    dots.uniform_float("color", MARKER_COLOUR)
    batch_for_shader(dots, 'POINTS', {"pos": [p, q]}).draw(dots)
    gpu.state.point_size_set(1.0)


def _draw_marker_label():
    found = _marker()
    if found is None:
        return
    d, p, q = found
    xy = location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, (p + q) / 2)
    if xy is None:
        return
    font = 0
    blf.size(font, 16 * bpy.context.preferences.system.ui_scale)
    blf.color(font, *MARKER_COLOUR)
    blf.position(font, xy.x + 12, xy.y + 8, 0)
    blf.draw(font, f"{d:.2f} mm" if d >= 0 else f"inside {-d:.2f} mm")


_draw_handles = []


# --- Operator --------------------------------------------------------------------------

class OMFS_OT_distance_map(bpy.types.Operator):
    """Colour the source by its distance to the target (live), with the minimum distance and a marker
    at the closest point. Click again to remove it. The meshes are never modified"""
    bl_idname = "omfs.distance_map"
    bl_label = "Show Distance Map"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if find_map(scene)[0] is not None:
            remove_map(scene)
            set_shading(context, False)
            return {'FINISHED'}
        props = scene.omfs_section
        source, target = props.dist_source, props.dist_target
        if not (analysis.is_analysable(source) and analysis.is_analysable(target)) or source == target:
            self.report({'ERROR'}, "Pick two different objects")
            return {'CANCELLED'}
        add_map(scene, source, target)
        _closest[source.name] = closest(source, context.evaluated_depsgraph_get())
        set_shading(context, True)
        return {'FINISHED'}


classes = (OMFS_OT_distance_map,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    if not bpy.app.background:
        _draw_handles.append((bpy.types.SpaceView3D.draw_handler_add(_draw_marker_3d, (), 'WINDOW', 'POST_VIEW')))
        _draw_handles.append((bpy.types.SpaceView3D.draw_handler_add(_draw_marker_label, (), 'WINDOW', 'POST_PIXEL')))


def unregister():
    for handle in _draw_handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _draw_handles.clear()
    bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
