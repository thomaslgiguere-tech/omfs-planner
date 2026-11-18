"""Clicked points drawn as solid, shaded balls (surgeon, 2026-11-18): landmarks (porion, nasion, foramina...),
the splint's coverage points and the osteotomy lines' points. Only the drawing changes: the points are still the
same Empties (Blender's "Empty": a position with no mesh), placed, moved and saved exactly as before; the ball is
drawn over Blender's thin wire sphere by a viewport draw handler. Meshmixer: none (its points are flat dots).
"""

import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import params

_sphere = None
_handle = None


def unit_sphere(rings=10, segments=16):
    """(vertices, triangles) of a unit UV sphere."""
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            verts.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)))
    verts.append((0.0, 0.0, -1.0))
    tris = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            tris += [(a + j, b + j, b + k), (a + j, b + k, a + k)]
    last = len(verts) - 1
    base = 1 + (rings - 2) * segments
    tris += [(base + j, last, base + (j + 1) % segments) for j in range(segments)]
    return np.array(verts, np.float32), np.array(tris, np.int32)


def ball_colour(obj):
    """Green for landmarks and splint points; an osteotomy line keeps its front / back colours."""
    from . import ortho_osteotomy
    if obj.omfs.made_by == ortho_osteotomy.MADE_BY:
        return tuple(obj.color)
    return params.POINT_BALL_COLOUR


def points(scene):
    """Visible point Empties that get a ball."""
    from . import ortho_osteotomy, splint
    found = []
    for obj in scene.objects:
        if obj.type != 'EMPTY' or not obj.visible_get():
            continue
        if obj.omfs.role == 'LANDMARK' or obj.omfs.made_by in (splint.POINT_TAG, ortho_osteotomy.MADE_BY):
            found.append(obj)
    return found


def _draw():
    global _sphere
    context = bpy.context
    found = points(context.scene)
    rv3d = context.region_data
    if not found or rv3d is None:
        return
    if _sphere is None:
        _sphere = unit_sphere()
    verts, tris = _sphere
    view = np.array(rv3d.view_rotation @ Vector((0.0, 0.0, 1.0)))       # towards the viewer
    light = view * 0.8 + np.array((0.0, 0.0, 0.6))
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(verts @ light, 0.0, 1.0)                # lit from the viewer, a bit from above
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    # a point set to show through the bone (as landmarks are) gets its ball in front too; the others are hidden by it
    for in_front in (False, True):
        group = [o for o in found if o.show_in_front == in_front]
        if not group:
            continue
        positions, colours, indices = [], [], []
        for n, obj in enumerate(group):
            radius = max(obj.empty_display_size * max(obj.matrix_world.to_scale()) * 1.05, params.POINT_BALL_MIN_RADIUS)
            centre = np.array(obj.matrix_world.translation)
            r, g, b, a = ball_colour(obj)
            positions.append(verts * radius + centre)
            colours.append(np.column_stack((shade * r, shade * g, shade * b, np.full(len(verts), 1.0))))
            indices.append(tris + n * len(verts))
        batch = batch_for_shader(shader, 'TRIS', {"pos": np.vstack(positions).astype(np.float32),
                                                  "color": np.vstack(colours).astype(np.float32)},
                                 indices=np.vstack(indices).astype(np.int32))
        if in_front:
            gpu.state.depth_test_set('NONE')
        else:
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
        shader.bind()
        batch.draw(shader)
        gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set('NONE')


def register():
    global _handle
    if not bpy.app.background:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
