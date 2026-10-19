# Application Template: what is hidden, and what can't be

Start it with File > New > OMFS Planner, or `Blender --app-template omfs_planner`.

## Hidden in clean mode (pure Python, no recompiling)

- Workspace tabs (Layout, Sculpting, Shading, Animation, Compositing, Geometry Nodes, ...): the
  workspaces still exist, only the tabs are hidden.
- Render menu, scene and view-layer selectors in the top bar.
- Properties editor, Timeline (animation): closed in the Planning workspace.
- 3D viewport header: mode selector (Sculpt / Edit / Paint...), Add and Object menus, transform
  orientation, snapping and proportional editing. Kept: View, Select, overlays, X-ray, shading.
- Toolbar (T) in the Planning workspace.
- Default cube, camera and light (new files only).
- Outliner reduced to a flat object list (no cameras, lights, empties, collections, mesh data).

Developer mode puts all of it back; nothing is deleted.

## Can't be hidden without compiling Blender

- **Individual Properties editor tabs** (Render, Output, World, Physics, Particles...): the tab
  list is built in C. We can only close the whole editor.
- Built-in **C-defined panels**, such as the Item > Transform panel in the sidebar, and the
  built-in sidebar tabs (Item, Tool, View, Animation).
- **Editor types**: the editor-type dropdown always offers every editor (Compositor, Video
  Sequencer, Node editors...). A user can still switch an area to one of them.
- **Keyboard shortcuts** still reach hidden features (e.g. Ctrl+Tab mode pie, F12 render); they
  could be removed from the keymap later if needed.
- **F3 operator search** still finds every Blender command.
- The **status bar** and the **splash screen** content (the splash image itself can be replaced
  with a `splash.png` in the template folder).
