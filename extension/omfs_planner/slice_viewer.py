"""Coupes: a DICOM-style slice viewer in the 3D view. The 3D viewport is split into Blender's Quad View: three
locked orthographic quarters show the CBCT in grayscale (axial, sagittal, coronal, like 3D Slicer's Four-Up), the
fourth stays the ordinary 3D view, where the three slice planes are drawn among the meshes.
Every visible mesh of the scene (bone, teeth, nerve, scans, segments, guides...) is drawn on each slice as a
contour in its own colour: the check that the AI segmentation follows the gray image.

DRAFT (2026-11-21), see docs/slice_viewer.md. Strictly non-destructive and memory-only, like volume_scene.py:
the volume is patient data, it is never written into the .blend; slices are drawn by draw handlers, no objects.

Mouse in a slice quarter (one modal operator, events elsewhere pass through):
  wheel                    next / previous slice (trackpad: two-finger swipe)
  left + right held, drag  scroll along the view's slices (up = next)
  left click / drag        move the crosshair (the two other slices follow)
  right drag               window (horizontal) / level (vertical)
  Ctrl + wheel             zoom          middle drag   pan

Blender concepts: Quad View (View > Area > Toggle Quad View, Ctrl Alt Q) splits one 3D viewport into four
regions; a draw handler is Python code that draws in a viewport each time it redraws (nothing is added to the
scene); a modal operator is a tool that keeps receiving the mouse until it ends. No Meshmixer equivalent
(Meshmixer has no CT viewer).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import blf
import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import params, roles, slice_core as sc
from .ui import draw_text

# Reading order of the three locked quarters (Slicer's Four-Up: red axial top left, yellow sagittal bottom
# left, green coronal bottom right, 3D top right).
QUARTERS = ('AXIAL', 'SAGITTAL', 'CORONAL')
BLENDER_AXIS = {'AXIAL': 'BOTTOM', 'CORONAL': 'FRONT', 'SAGITTAL': 'RIGHT'}   # same screen axes as slice_core.VIEWS
LABELS = {'AXIAL': "Axiale", 'CORONAL': "Coronale", 'SAGITTAL': "Sagittale"}
DEPTH_LETTER = {'AXIAL': "S", 'CORONAL': "A", 'SAGITTAL': "L"}
COLORS = {'AXIAL': (0.95, 0.33, 0.30, 1.0), 'SAGITTAL': (0.95, 0.85, 0.30, 1.0), 'CORONAL': (0.45, 0.80, 0.40, 1.0)}
AXIS_LETTERS = ("R", "L", "A", "P", "I", "S")   # -x, +x, -y, +y, -z, +z in LPS


class _State:
    volume = None          # slice_core.Volume: memory only
    area = 0               # as_pointer() of the 3D viewport in Quad View
    regions = {}           # RegionView3D.as_pointer() -> view, for the three locked quarters
    wl_unit = 1.0          # gray values per mouse pixel for window / level
    modal = False
    hover = None           # (rv3d pointer, LPS point, gray value) under the mouse
    images = {}            # view -> (key, raw slice image, corners)
    textures = {}          # view -> (key, GPUTexture)
    triangles = {}         # object name -> (key, triangles (N, 3, 3) float32, low (N, 3), high (N, 3))
    contours = {}          # view -> (key, [(colour, segments (M, 2, 3))])
    versions = {}          # object name -> geometry version (depsgraph updates)


S = _State
_handles = []


def _reset():
    S.volume, S.area, S.regions, S.hover = None, 0, {}, None
    S.images, S.textures, S.triangles, S.contours = {}, {}, {}, {}


# --- Settings ---------------------------------------------------------------------------------------------------

def _changed(self, context):
    _sync_depths()
    _redraw()


class OMFS_SliceSettings(bpy.types.PropertyGroup):
    center: bpy.props.FloatVectorProperty(
        name="Croisement", size=3, subtype='TRANSLATION', unit='LENGTH', update=_changed,
        description="Point où se croisent les trois coupes (mm, coordonnées du patient LPS)")
    window: bpy.props.FloatProperty(name="Fenêtre", min=1.0, update=_changed,
                                    description="Largeur de l'échelle de gris (valeurs du CBCT)")
    level: bpy.props.FloatProperty(name="Niveau", update=_changed, description="Centre de l'échelle de gris")
    show_contours: bpy.props.BoolProperty(name="Contours des maillages", default=True, update=_changed,
                                          description="Chaque maillage visible dessiné sur les coupes, dans sa couleur")
    show_planes: bpy.props.BoolProperty(name="Coupes dans la vue 3D", default=True, update=_changed,
                                        description="Les trois coupes dessinées parmi les maillages dans la vue 3D")
    contour_width: bpy.props.FloatProperty(name="Épaisseur", default=params.SLICE_CONTOUR_WIDTH, min=0.5, max=8.0,
                                           update=_changed, description="Épaisseur des contours, en pixels")


def settings(context=None):
    return (context or bpy.context).scene.omfs_slices


# --- Viewer area and regions ------------------------------------------------------------------------------------

def _viewer():
    """(window, area) of the viewer's 3D viewport, or None."""
    if not S.area:
        return None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.as_pointer() == S.area:
                return window, area
    return None


def _quarters(area):
    """[(region, view)] of the viewer's locked quarters."""
    found = []
    for region in area.regions:
        if region.type == 'WINDOW' and region.data and region.data.as_pointer() in S.regions:
            found.append((region, S.regions[region.data.as_pointer()]))
    return found


def _redraw():
    hit = _viewer()
    if hit:
        hit[1].tag_redraw()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sync_depths():
    """Keep each quarter's view centre on its slice (depth only: the user's pan is kept)."""
    hit = _viewer()
    if not hit or S.volume is None:
        return
    center = Vector(settings().center)
    for region, view in _quarters(hit[1]):
        scroll = Vector(sc.axes(view)[3])
        rv3d = region.data
        loc = Vector(rv3d.view_location)
        rv3d.view_location = loc + scroll * (scroll.dot(center) - scroll.dot(loc))


# --- Slices as textures -----------------------------------------------------------------------------------------

def _depth(view, center):
    return round(float(np.dot(center, sc.axes(view)[2])), 4)


def _slice(view, center):
    """(raw image, corners) of a view's slice, resampled only when its depth changes."""
    key = (id(S.volume), _depth(view, center))
    hit = S.images.get(view)
    if hit and hit[0] == key:
        return hit[1], hit[2]
    image, corners = sc.reslice(S.volume, view, center, params.SLICE_PIXEL)
    S.images[view] = (key, image, corners)
    return image, corners


def _texture(view, prefs):
    center = np.array(prefs.center)
    image, corners = _slice(view, center)
    key = (id(S.volume), _depth(view, center), round(prefs.window, 3), round(prefs.level, 3))
    hit = S.textures.get(view)
    if hit and hit[0] == key:
        return hit[1], corners
    gray = sc.window_level(image, prefs.window, prefs.level)
    rgba = np.empty(gray.shape + (4,), np.float32)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = 1.0
    try:
        buffer = gpu.types.Buffer('FLOAT', rgba.size, rgba.ravel())        # numpy, fast (buffer protocol)
    except TypeError:
        buffer = gpu.types.Buffer('FLOAT', rgba.size, rgba.ravel().tolist())
    texture = gpu.types.GPUTexture((gray.shape[1], gray.shape[0]), format='RGBA16F', data=buffer)
    S.textures[view] = (key, texture)
    return texture, corners


def _draw_quad(texture, corners, on_top):
    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(shader, 'TRI_FAN', {
        "pos": [tuple(map(float, c)) for c in corners], "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))})
    gpu.state.depth_test_set('NONE' if on_top else 'LESS_EQUAL')
    gpu.state.blend_set('NONE')
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)


def _draw_lines(points, colour, width, region):
    if len(points) == 0:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": [tuple(map(float, p)) for p in points]})
    shader.bind()
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", colour)
    batch.draw(shader)


# --- Meshes on the slices ---------------------------------------------------------------------------------------

def _surfaces(context):
    for obj in context.visible_objects:
        if obj.type != 'MESH':
            continue
        role = getattr(getattr(obj, "omfs", None), "role", 'OTHER')
        if role in roles.NOT_SURFACES:
            continue
        yield obj


def _triangles(obj, depsgraph):
    """World triangles of an object's evaluated mesh (modifiers included), with per-axis bounds per triangle."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.data
    key = (S.versions.get(obj.name, 0), tuple(tuple(r) for r in evaluated.matrix_world),
           len(mesh.vertices), len(mesh.loop_triangles))
    hit = S.triangles.get(obj.name)
    if hit and hit[0] == key:
        return hit[1:]
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    index = np.empty(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", index)
    matrix = np.array(evaluated.matrix_world, np.float32)
    world = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    tri = world[index.reshape(-1, 3)]
    low, high = tri.min(axis=1), tri.max(axis=1)
    S.triangles[obj.name] = (key, tri, low, high)
    return tri, low, high


def _contours(context, view, center):
    """[(colour, segments)] of the visible meshes on a view's slice."""
    _right, _up, normal, _scroll = sc.axes(view)
    axis = int(np.argmax(np.abs(normal)))       # the views are axis-aligned: the plane is coordinate = value
    value = float(center[axis])
    depsgraph = context.evaluated_depsgraph_get()
    objects = list(_surfaces(context))
    key = (round(value, 4), tuple((o.name, S.versions.get(o.name, 0), tuple(tuple(r) for r in o.matrix_world),
                                   tuple(o.color)) for o in objects))
    hit = S.contours.get(view)
    if hit and hit[0] == key:
        return hit[1]
    found = []
    for obj in objects:
        tri, low, high = _triangles(obj, depsgraph)
        near = (low[:, axis] <= value) & (high[:, axis] >= value)
        if not near.any():
            continue
        segments = sc.plane_segments(tri[near], center, normal)
        if len(segments):
            colour = tuple(obj.color[:3]) + (1.0,)
            found.append((colour, segments.reshape(-1, 3)))
    S.contours[view] = (key, found)
    return found


# --- Drawing ----------------------------------------------------------------------------------------------------

def _draw_view():
    context = bpy.context
    if S.volume is None or context.region_data is None:
        return
    prefs = settings(context)
    region = context.region
    view = S.regions.get(context.region_data.as_pointer())
    center = np.array(prefs.center)
    try:
        if view:                       # a slice quarter: the slice on top of everything, contours, crosshair
            texture, corners = _texture(view, prefs)
            _draw_quad(texture, corners, on_top=True)
            if prefs.show_contours:
                for colour, points in _contours(context, view, center):
                    _draw_lines(points, colour, prefs.contour_width, region)
            normal = sc.axes(view)[2]
            reach = float(np.linalg.norm(np.ptp(S.volume.corners(), axis=0)))
            for other in QUARTERS:
                if other == view:
                    continue
                direction = np.cross(normal, sc.axes(other)[2])
                _draw_lines([center - reach * direction, center + reach * direction], COLORS[other], 1.0, region)
        elif prefs.show_planes:        # the 3D view: the three slices among the meshes
            for other in QUARTERS:
                texture, corners = _texture(other, prefs)
                _draw_quad(texture, corners, on_top=False)
                outline = [corners[i] for i in (0, 1, 1, 2, 2, 3, 3, 0)]
                _draw_lines(outline, COLORS[other], 2.0, region)
    finally:
        gpu.state.depth_test_set('NONE')


def _screen_letter(rv3d, screen_axis):
    """LPS letter of a screen direction (e.g. the left edge of an axial slice shows 'R')."""
    world = rv3d.view_rotation @ Vector(screen_axis)
    i = max(range(3), key=lambda k: abs(world[k]))
    return AXIS_LETTERS[2 * i + (1 if world[i] > 0 else 0)]


def _draw_text():
    context = bpy.context
    if S.volume is None or context.region_data is None:
        return
    view = S.regions.get(context.region_data.as_pointer())
    if not view:
        return
    prefs, region, rv3d = settings(context), context.region, context.region_data
    scale = context.preferences.system.ui_scale
    size, pad = 13 * scale, 10 * scale
    center = np.array(prefs.center)
    index, count = sc.slice_index(S.volume, view, center)
    depth = float(np.dot(center, sc.axes(view)[3]))
    lines = [f"{LABELS[view]}   coupe {index + 1} / {count}   {DEPTH_LETTER[view]} {depth:+.1f} mm",
             f"F {prefs.window:.0f}   N {prefs.level:.0f}"]
    if S.hover and S.hover[0] == rv3d.as_pointer():
        p, value = S.hover[1], S.hover[2]
        lines.append(f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm   valeur {value:.0f}")
    blf.size(0, size)
    blf.color(0, *COLORS[view])
    y = region.height - pad - size
    for line in lines:
        blf.position(0, pad, y, 0)
        blf.draw(0, line)
        blf.color(0, 0.9, 0.9, 0.9, 1.0)
        y -= size * 1.4
    blf.color(0, 0.9, 0.9, 0.9, 1.0)
    w, h = region.width, region.height
    for axis, (x, y) in (((-1, 0, 0), (pad, h / 2)), ((1, 0, 0), (w - pad - size, h / 2)),
                         ((0, 1, 0), (w / 2, h - pad - size * 4.5)), ((0, -1, 0), (w / 2, pad))):
        blf.position(0, x, y, 0)
        blf.draw(0, _screen_letter(rv3d, axis))


# --- Mouse --------------------------------------------------------------------------------------------------------

def _under_mouse(window, event):
    """(region, view) of the slice quarter under the mouse, or None."""
    hit = _viewer()
    if not hit or hit[0] != window:
        return None
    for region, view in _quarters(hit[1]):
        if region.x <= event.mouse_x < region.x + region.width and region.y <= event.mouse_y < region.y + region.height:
            return region, view
    return None


def _mouse_point(region, event, center):
    coord = (event.mouse_x - region.x, event.mouse_y - region.y)
    return np.array(view3d_utils.region_2d_to_location_3d(region, region.data, coord, Vector(center)))


class OMFS_OT_slices_mouse(bpy.types.Operator):
    """Souris dans les coupes (molette, gauche + droit, fenêtre / niveau, réticule); ailleurs, rien ne change"""
    bl_idname = "omfs.slices_mouse"
    bl_label = "Coupes : souris"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        if S.modal:
            return {'CANCELLED'}
        S.modal = True
        self.left = self.right = False
        self.mode, self.target, self.start, self.acc = None, None, (0, 0, 0.0, 0.0), 0.0
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self):
        S.modal = False
        return {'FINISHED'}

    def modal(self, context, event):
        if S.volume is None or _viewer() is None:
            return self.finish()
        area = _viewer()[1]
        if not area.spaces.active.region_quadviews:          # Quad View turned off by hand: the viewer ends
            _reset()
            return self.finish()
        hit = self.target if self.mode else _under_mouse(context.window, event)
        if hit is None:
            if S.hover:
                S.hover = None
                area.tag_redraw()
            return {'PASS_THROUGH'}
        region, view = hit
        prefs = settings(context)
        center = np.array(prefs.center)
        kind, value = event.type, event.value

        if kind in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            sign = 1 if kind == 'WHEELUPMOUSE' else -1
            if event.ctrl:
                region.data.view_distance *= params.SLICE_ZOOM_STEP ** -sign
                area.tag_redraw()
            else:
                prefs.center = sc.step_center(S.volume, view, center, sign)
            return {'RUNNING_MODAL'}
        if kind == 'TRACKPADPAN':
            self.acc += event.mouse_y - event.mouse_prev_y
            self._scroll(prefs, view, center, params.SLICE_TRACKPAD_PIXELS)
            return {'RUNNING_MODAL'}

        if kind == 'LEFTMOUSE':
            if value == 'PRESS':
                self.left = True
                self._press(region, view, event, prefs)
            elif value == 'RELEASE':
                self.left = False
                if self.mode == 'CROSS':
                    self._crosshair(region, event, prefs)
                self.mode, self.target = None, None
            return {'RUNNING_MODAL'}
        if kind == 'RIGHTMOUSE':
            if value == 'PRESS':
                self.right = True
                self._press(region, view, event, prefs)
            elif value == 'RELEASE':
                self.right = False
                self.mode, self.target = None, None
            return {'RUNNING_MODAL'}
        if kind == 'MIDDLEMOUSE':
            if value == 'PRESS':
                self.mode, self.target = 'PAN', hit
                self.start = (event.mouse_x, event.mouse_y, 0.0, 0.0)
            elif value == 'RELEASE':
                self.mode, self.target = None, None
            return {'RUNNING_MODAL'}

        if kind in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if self.mode == 'CROSS':
                moved = abs(event.mouse_x - self.start[0]) + abs(event.mouse_y - self.start[1])
                if moved >= params.SLICE_DRAG_THRESHOLD:
                    self._crosshair(region, event, prefs)
            elif self.mode == 'SCROLL':
                self.acc += event.mouse_y - event.mouse_prev_y
                self._scroll(prefs, view, center, params.SLICE_SCROLL_PIXELS)
            elif self.mode == 'WL':
                x0, y0, w0, l0 = self.start
                k = params.SLICE_WL_SPEED * S.wl_unit
                prefs.window = max(1.0, w0 + (event.mouse_x - x0) * k)
                prefs.level = l0 + (event.mouse_y - y0) * k
            elif self.mode == 'PAN':
                before = _mouse_point(region, type("E", (), {"mouse_x": event.mouse_prev_x,
                                                             "mouse_y": event.mouse_prev_y}), center)
                now = _mouse_point(region, event, center)
                region.data.view_location = Vector(region.data.view_location) + Vector(before - now)
                area.tag_redraw()
            else:
                point = _mouse_point(region, event, center)
                S.hover = (region.data.as_pointer(), point, float(S.volume.sample(point)))
                area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}       # keyboard shortcuts keep working over the slices

    def _press(self, region, view, event, prefs):
        self.target = (region, view)
        self.acc = 0.0
        if self.left and self.right:
            self.mode = 'SCROLL'          # both buttons: scroll along the view's slices
        elif self.left:
            self.mode = 'CROSS'           # set on release, or while dragging past the threshold
        else:
            self.mode = 'WL'
        self.start = (event.mouse_x, event.mouse_y, prefs.window, prefs.level)

    def _scroll(self, prefs, view, center, pixels):
        steps = int(self.acc / pixels)
        if steps:
            self.acc -= steps * pixels
            prefs.center = sc.step_center(S.volume, view, center, steps)

    def _crosshair(self, region, event, prefs):
        point = _mouse_point(region, event, prefs.center)
        prefs.center = sc.clamp_center(S.volume, point)


# --- Open / close -------------------------------------------------------------------------------------------------

def load_volume(path):
    """A .nrrd read directly; DICOM (folder or .dcm), .nii(.gz), .mha converted first by
    segmentation/dicom_to_nrrd.py with SimpleITK (Blender has no DICOM reader), in a temporary folder deleted
    right after reading."""
    path = Path(path)
    if path.is_file() and path.name.lower().endswith(".nrrd"):
        return sc.read_nrrd(path)
    from . import new_case
    python, folder = new_case.segmentation_python(), new_case.segmentation_folder()
    script = folder / "dicom_to_nrrd.py"
    if not python.exists():
        raise RuntimeError(f"Python avec SimpleITK introuvable : {python} (Préférences de l'extension)")
    if not script.exists():
        raise RuntimeError(f"Dossier segmentation introuvable : {folder} (Préférences de l'extension)")
    work = Path(tempfile.mkdtemp(prefix="omfs_coupes_"))
    try:
        out = work / "volume.nrrd"
        done = subprocess.run([str(python), str(script), str(path), "--sortie", str(out)],
                              capture_output=True, text=True, timeout=params.DICOM_TIMEOUT)
        if done.returncode != 0 or not out.exists():
            raise RuntimeError("Lecture DICOM impossible :\n" + (done.stderr or done.stdout)[-1500:])
        volume = sc.read_nrrd(out)
        volume.path = str(path)
        return volume
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _main_region(area):
    return max((r for r in area.regions if r.type == 'WINDOW'), key=lambda r: r.width * r.height)


def _setup_quarters(tries=[0]):
    """Timer, once Quad View exists: give each locked quarter its view (Blender's Bottom / Front / Right are the
    radiological axial / coronal / sagittal), centre it on the volume, start the mouse operator."""
    hit = _viewer()
    if hit is None or S.volume is None:
        return None
    window, area = hit
    space = area.spaces.active
    quads = {rv3d.as_pointer() for rv3d in space.region_quadviews}
    locked = [r for r in area.regions if r.type == 'WINDOW' and r.data and r.data.as_pointer() in quads]
    if len(locked) < 3:
        tries[0] += 1
        return 0.1 if tries[0] < 30 else None
    tries[0] = 0
    locked.sort(key=lambda r: (-(r.y + r.height), r.x))
    prefs = settings()
    center = Vector(prefs.center)
    extent = float(np.ptp(S.volume.corners(), axis=0).max())
    view_prefs = bpy.context.preferences.view
    smooth, view_prefs.smooth_view = view_prefs.smooth_view, 0     # no animation: the location is set right after
    try:
        S.regions = {}
        for region, view in zip(locked, QUARTERS):
            rv3d = region.data
            rv3d.lock_rotation = False          # Quad View locks its side views; unlock to choose the axis
            with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
                bpy.ops.view3d.view_axis(type=BLENDER_AXIS[view])
            rv3d.lock_rotation = True           # then no orbiting by mistake in a slice
            rv3d.view_perspective = 'ORTHO'
            rv3d.view_location = center
            rv3d.view_distance = extent * params.SLICE_FIT
            S.regions[rv3d.as_pointer()] = view
    finally:
        view_prefs.smooth_view = smooth
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=_main_region(area)):
        bpy.ops.omfs.slices_mouse('INVOKE_DEFAULT')
    _redraw()
    return None


class OMFS_OT_slices_open(bpy.types.Operator):
    """Ouvrir un CBCT : un dossier DICOM (entrer dans le dossier, puis Accepter sans choisir de fichier), un .dcm
    de la série, ou un volume .nrrd / .nii / .mha. La vue 3D passe en 4 vues : axiale, sagittale, coronale et 3D"""
    bl_idname = "omfs.slices_open"
    bl_label = "Ouvrir un CBCT (coupes)…"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_glob: bpy.props.StringProperty(default="*.nrrd;*.nii;*.nii.gz;*.mha;*.mhd;*.dcm", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        path = self.filepath if self.filepath and not self.filepath.endswith(os.sep) else self.directory
        if not path or not Path(path).exists():
            self.report({'ERROR'}, "Choisir un dossier DICOM ou un fichier de volume")
            return {'CANCELLED'}
        try:
            volume = load_volume(path)
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, show_volume(context, context.window, context.area, volume))
        return {'FINISHED'}


def show_volume(context, window, area, volume):
    """Put a loaded volume in the viewer: Quad View in this 3D viewport, window / level, crosshair at the centre.
    Main thread only (load_volume, which has no bpy, may run in a worker thread: Nouveau cas does)."""
    _reset()
    S.volume = volume
    S.area = area.as_pointer()
    prefs = settings(context)
    w, l = sc.auto_window(volume, params.SLICE_WINDOW_PERCENTILES)
    S.wl_unit = w / 1000.0
    prefs.window, prefs.level = w, l
    prefs.center = volume.center()
    if not area.spaces.active.region_quadviews:
        with context.temp_override(window=window, screen=window.screen, area=area, region=_main_region(area)):
            bpy.ops.screen.region_quadview()
    bpy.app.timers.register(_setup_quarters, first_interval=0.05)
    size = " x ".join(str(s) for s in volume.shape)
    spacing = " x ".join(f"{s:.3g}" for s in volume.spacing)
    return f"CBCT {size} voxels de {spacing} mm (mémoire seulement, jamais dans le .blend)"


class OMFS_OT_slices_close(bpy.types.Operator):
    """Fermer les coupes : la vue 3D redevient une seule vue, le volume est retiré de la mémoire"""
    bl_idname = "omfs.slices_close"
    bl_label = "Fermer les coupes"

    @classmethod
    def poll(cls, context):
        return S.volume is not None

    def execute(self, context):
        hit = _viewer()
        _reset()
        if hit:
            window, area = hit
            if area.spaces.active.region_quadviews:
                with context.temp_override(window=window, screen=window.screen, area=area,
                                           region=_main_region(area)):
                    bpy.ops.screen.region_quadview()
        _redraw()
        return {'FINISHED'}


class OMFS_OT_slices_preset(bpy.types.Operator):
    """Préréglage de l'échelle de gris (les valeurs d'un CBCT ne sont pas de vrais HU : point de départ)"""
    bl_idname = "omfs.slices_preset"
    bl_label = "Préréglage"
    bl_options = {'INTERNAL'}

    preset: bpy.props.StringProperty()

    def execute(self, context):
        prefs = settings(context)
        if self.preset == "AUTO" and S.volume is not None:
            prefs.window, prefs.level = sc.auto_window(S.volume, params.SLICE_WINDOW_PERCENTILES)
        else:
            for name, window, level in params.SLICE_PRESETS:
                if name == self.preset:
                    prefs.window, prefs.level = window, level
        return {'FINISHED'}


class OMFS_PT_slices(bpy.types.Panel):
    bl_label = "Coupes (CBCT)"
    bl_idname = "OMFS_PT_slices"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        prefs = settings(context)
        if S.volume is None:
            draw_text(layout, context, "Voir le CBCT en coupes (axiale, sagittale, coronale) avec les maillages "
                                       "en couleur par-dessus. Le volume reste en mémoire seulement.")
            row = layout.row()
            row.scale_y = 1.3
            row.operator("omfs.slices_open", icon='FILE_VOLUME')
            return
        layout.label(text=Path(S.volume.path).name, icon='FILE_VOLUME')
        col = layout.column(align=True)
        col.prop(prefs, "window")
        col.prop(prefs, "level")
        row = layout.row(align=True)
        row.operator("omfs.slices_preset", text="Auto").preset = "AUTO"
        for name, _w, _l in params.SLICE_PRESETS:
            row.operator("omfs.slices_preset", text=name).preset = name
        row = layout.row(align=True)
        row.prop(prefs, "show_contours")
        row.prop(prefs, "contour_width", text="")
        layout.prop(prefs, "show_planes")
        layout.prop(prefs, "center")
        box = layout.box()
        for line in ("Molette : coupe suivante / précédente", "Gauche + droit, glisser : défiler les coupes",
                     "Clic gauche : déplacer le réticule", "Clic droit, glisser : fenêtre / niveau",
                     "Ctrl + molette : zoom   ·   clic milieu : déplacer"):
            box.label(text=line)
        layout.operator("omfs.slices_close", icon='X')


# --- Handlers -----------------------------------------------------------------------------------------------------

@bpy.app.handlers.persistent
def _on_depsgraph(scene, depsgraph):
    for update in depsgraph.updates:
        if update.is_updated_geometry and isinstance(update.id, bpy.types.Object):
            name = update.id.original.name
            S.versions[name] = S.versions.get(name, 0) + 1


@bpy.app.handlers.persistent
def _on_load(*_args):
    _reset()       # another file, maybe another patient: never keep the volume


classes = (OMFS_SliceSettings, OMFS_OT_slices_mouse, OMFS_OT_slices_open, OMFS_OT_slices_close,
           OMFS_OT_slices_preset, OMFS_PT_slices)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_slices = bpy.props.PointerProperty(type=OMFS_SliceSettings)
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), 'WINDOW', 'POST_VIEW'))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_text, (), 'WINDOW', 'POST_PIXEL'))
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.load_pre.append(_on_load)


def unregister():
    _reset()
    for handle in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _handles.clear()
    for handlers, func in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph),
                           (bpy.app.handlers.load_pre, _on_load)):
        if func in handlers:
            handlers.remove(func)
    if bpy.app.timers.is_registered(_setup_quarters):
        bpy.app.timers.unregister(_setup_quarters)
    del bpy.types.Scene.omfs_slices
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
