"""OMFS Planner application template: a clean, surgery-only Blender interface.

Blender runs this module when you start "OMFS Planner" from File > New, or launch
`Blender --app-template omfs_planner`. It only changes the interface; the surgical tools
live in the extension (extension/omfs_planner).

Clean mode (default):
  - a "Planning" workspace with one 3D viewport, the OMFS sidebar tab and a small object list
  - no workspace tabs, no Render menu, no scene / view layer selectors
  - 3D viewport header reduced to View / Select menus, shading and overlays
  - no default cube, camera or light; units in mm
Developer mode (checkbox at the top right) restores the full Blender interface; the
other workspaces (Layout, Sculpting, Shading, ...) are kept, only hidden.
"""

import addon_utils
import bpy
from bpy.app.handlers import persistent

EXTENSION_ID = "omfs_planner"
WORKSPACE = "Planning"
FACTORY_OBJECTS = ("Cube", "Camera", "Light")


# --- Replacement draw functions (clean mode) ---------------------------------------

def draw_topbar_menus(self, context):
    layout = self.layout
    layout.menu("TOPBAR_MT_blender", text="", icon='BLENDER')
    layout.menu("TOPBAR_MT_file")
    layout.menu("TOPBAR_MT_edit")
    layout.menu("TOPBAR_MT_window")
    layout.menu("TOPBAR_MT_help")


def draw_upper_bar_left(self, context):
    # Menus only: no workspace tabs.
    bpy.types.TOPBAR_MT_editor_menus.draw_collapsible(context, self.layout)


def draw_upper_bar_right(self, context):
    pass  # no scene / view layer selectors


def draw_developer_toggle(self, context):
    # Appended to the top bar in both modes; the top bar draws its left and right parts separately.
    if context.region.alignment == 'RIGHT':
        self.layout.prop(context.window_manager, "omfs_developer_mode")


def draw_view3d_header(self, context):
    layout = self.layout
    view = context.space_data
    layout.row(align=True).template_header()   # editor type button
    bpy.types.VIEW3D_MT_editor_menus.draw_collapsible(context, layout)
    layout.separator_spacer()

    row = layout.row(align=True)
    row.prop(view.overlay, "show_overlays", icon='OVERLAY', text="")
    row.popover(panel="VIEW3D_PT_overlay", text="")

    row = layout.row(align=True)
    row.prop(view.shading, "show_xray", icon='XRAY', text="")
    row.prop(view.shading, "type", text="", expand=True)
    row.popover(panel="VIEW3D_PT_shading", text="")


def draw_view3d_menus(self, context):
    self.layout.menu("VIEW3D_MT_view")
    self.layout.menu("VIEW3D_MT_select_object")


# (class name, method name, replacement) -> swapped in clean mode, restored in developer mode.
OVERRIDES = (
    ("TOPBAR_MT_editor_menus", "draw", draw_topbar_menus),
    ("TOPBAR_HT_upper_bar", "draw_left", draw_upper_bar_left),
    ("TOPBAR_HT_upper_bar", "draw_right", draw_upper_bar_right),
    ("VIEW3D_HT_header", "draw", draw_view3d_header),
    ("VIEW3D_MT_editor_menus", "draw", draw_view3d_menus),
)
_originals = {}


def set_overrides(enabled):
    for class_name, method, replacement in OVERRIDES:
        cls = getattr(bpy.types, class_name)
        key = (class_name, method)
        if enabled:
            _originals.setdefault(key, getattr(cls, method))
            setattr(cls, method, replacement)
        elif key in _originals:
            setattr(cls, method, _originals.pop(key))
    redraw_all()


def redraw_all():
    # The top bar isn't in screen.areas; clicking the toggle redraws it anyway.
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


# --- Planning workspace ------------------------------------------------------------

def create_planning_workspace(window):
    """Copy the "Layout" workspace as "Planning" and show it in the window."""
    before = set(bpy.data.workspaces)
    with bpy.context.temp_override(window=window, workspace=bpy.data.workspaces["Layout"]):
        bpy.ops.workspace.duplicate()
    workspace = (set(bpy.data.workspaces) - before).pop()
    workspace.name = WORKSPACE
    window.workspace = workspace   # takes effect on the next event-loop tick


def configure_planning_screen(window):
    """Keep only the 3D viewport and the object list. The screen must be on display:
    changing areas Blender hasn't drawn yet crashes it."""
    screen = window.screen
    for area in list(screen.areas):
        if area.type not in {'VIEW_3D', 'OUTLINER'}:
            with bpy.context.temp_override(window=window, screen=screen, area=area):
                bpy.ops.screen.area_close()
    for area in screen.areas:
        space = area.spaces.active
        if area.type == 'OUTLINER':
            setup_object_list(space)
        elif area.type == 'VIEW_3D':
            # Region visibility rebuilds the area, which needs a full context.
            with bpy.context.temp_override(window=window, screen=screen, area=area):
                space.show_region_toolbar = False
                space.show_region_ui = True   # sidebar (N panel)


def setup_object_list(outliner):
    outliner.display_mode = 'VIEW_LAYER'
    outliner.use_filter_collection = False       # flat list, no collection folders
    outliner.use_filter_object_content = False   # no mesh data / modifiers under objects
    outliner.use_filter_children = False
    outliner.use_filter_object_camera = False
    outliner.use_filter_object_light = False
    outliner.use_filter_object_empty = False
    outliner.use_filter_object_others = False
    outliner.show_restrict_column_select = True  # the arrow icon = selectable / locked


def open_omfs_tab(screen):
    """Return False while the sidebar hasn't been drawn yet (its tabs are unknown until then)."""
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'UI':
                    if region.active_panel_category == 'UNSUPPORTED':
                        return False
                    if hasattr(bpy.types, "OMFS_PT_case"):  # extension loaded
                        region.active_panel_category = "OMFS"
    return True


# --- Scene -------------------------------------------------------------------------

def remove_factory_objects():
    """Delete the default cube, camera and light, but only in a new, unsaved file."""
    if bpy.data.filepath:
        return
    for name in FACTORY_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj)


def ensure_extension_enabled():
    for module in addon_utils.modules():
        name = module.__name__
        if name.startswith("bl_ext.") and name.endswith("." + EXTENSION_ID):
            if name not in bpy.context.preferences.addons:
                addon_utils.enable(name, default_set=True)
            return
    print("OMFS Planner: extension not found; run tools/dev_link.sh")


# --- Apply clean / developer mode --------------------------------------------------

def apply_mode(developer):
    window = bpy.context.window_manager.windows[0]
    if developer:
        set_overrides(False)
        return
    workspace = bpy.data.workspaces.get(WORKSPACE)
    if workspace is None:
        create_planning_workspace(window)
        bpy.app.timers.register(_finish_new_planning_workspace, first_interval=0.05)
    else:
        window.workspace = workspace
        bpy.app.timers.register(_finish_clean_mode, first_interval=0.05)
    set_overrides(True)


# Timer callbacks: return None to stop, or a delay in seconds to be called again.
RETRY = 0.05


def _finish_new_planning_workspace():
    window = bpy.context.window_manager.windows[0]
    if window.workspace.name != WORKSPACE:
        return RETRY   # the workspace switch hasn't happened yet
    configure_planning_screen(window)
    return _finish_clean_mode()


def _finish_clean_mode():
    window = bpy.context.window_manager.windows[0]
    if window.workspace.name != WORKSPACE:
        return RETRY
    if hasattr(bpy.types, "OMFS_OT_setup_scene"):  # extension loaded
        with bpy.context.temp_override(window=window):
            bpy.ops.omfs.setup_scene()
    bpy.app.timers.register(_open_tab_when_ready, first_interval=RETRY)
    return None


def _open_tab_when_ready():
    window = bpy.context.window_manager.windows[0]
    return None if open_omfs_tab(window.screen) else RETRY


def _on_developer_mode_changed(self, context):
    apply_mode(self.omfs_developer_mode)


def _setup_after_load():
    ensure_extension_enabled()
    remove_factory_objects()
    apply_mode(bpy.context.window_manager.omfs_developer_mode)
    return None  # run once


@persistent
def on_load_post(*_args):
    # The window and screens are only ready after the file has finished loading.
    bpy.app.timers.register(_setup_after_load, first_interval=0.05)


def register():
    bpy.types.WindowManager.omfs_developer_mode = bpy.props.BoolProperty(
        name="Developer mode",
        description="Show the full Blender interface",
        default=False,
        update=_on_developer_mode_changed,
    )
    bpy.types.TOPBAR_HT_upper_bar.append(draw_developer_toggle)
    bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    set_overrides(False)
    bpy.types.TOPBAR_HT_upper_bar.remove(draw_developer_toggle)
    del bpy.types.WindowManager.omfs_developer_mode
