# Glossary: Blender for Meshmixer users

Blender term → what it actually does → closest Meshmixer equivalent. Grows with every feature.

| Blender term | What it does | Meshmixer equivalent |
|---|---|---|
| **Object** | A container placed in the scene: name, position/rotation/scale, colour, and a link to its data (e.g. a mesh). | An object in the Object Browser. |
| **Mesh (data-block)** | The actual vertices, edges and faces. Separate from the object, so two objects can share one mesh. We name both after the file. | None: in Meshmixer an object *is* its mesh. |
| **Data-block** | Any reusable thing stored in a .blend file (objects, meshes, scenes, workspaces...). | None. |
| **Scene** | Everything in one planning case: objects, units, settings. | The whole Meshmixer session. |
| **Blender unit (BU)** | Blender's internal length unit. We set **1 BU = 1 mm**. Mesh coordinates are stored as raw numbers; the unit only changes how they are displayed. | Preferences > Units/Scale (mm). |
| **Unit scale / length unit** | Scene > Units. `scale_length = 0.001` + `MILLIMETERS` makes Blender show 1 BU as "1 mm". The vertex coordinates are not changed. | Same as above. |
| **Transform (location / rotation / scale)** | Where the object sits. Changing it moves the whole object **without** changing the vertex coordinates stored in the mesh. | Edit > Transform. |
| **Origin** | The object's pivot point. Imported meshes keep their origin at the file's 0,0,0, so all meshes from one case stay registered. | Transform pivot. |
| **Transform locks** | Per-axis padlocks on location/rotation/scale: the object can't be moved by hand. | None. |
| **Selectable (`hide_select`)** | When off, clicks go through the object in the viewport and outliner. It stays visible, and the Measure tool can still snap to it. Together with transform locks this is our **Lock**. | None (Meshmixer can only hide objects). |
| **Import axes (forward / up)** | Importers can rotate a file while reading it. We force forward = Y, up = Z, scale = 1, which means "keep the file's coordinates as-is". Blender's OBJ importer rotates by default! | Meshmixer imports raw coordinates. |
| **Empty** | An object with no mesh: just a position and three axes. We use one as the clinical reference frame, so the anatomy never has to be moved or rotated. | None. |
| **Rigid transform** | Rotation + translation only (no scale), stored in the object's transform. The vertices, so the shape, are untouched. Used to register the intra-oral scan. | The result of Align: the object moves, its shape doesn't. |
| **Point-pair alignment (Kabsch)** | Least-squares rotation + translation that best maps ≥3 clicked points on one mesh onto their partners on another. Coarse: gets within ~1 mm. | Align > manual point pairs. |
| **ICP (iterative closest point)** | Repeats: find the closest surface point for each scan crown point, move the scan to reduce those distances (point-to-plane), until it stops moving. Rigid only here. | Align > surface align. |
| **Vertex group** | A named set of vertices (with a 0-1 weight each) stored on a mesh. Our "Crowns" group says which scan vertices ICP uses. Doesn't move any vertex. | A saved selection. |
| **Weight Paint mode** | Paint vertex-group weights with a brush: red = 1 (used), blue = 0 (ignored). Used to correct the crown selection. | Select brush (but saved, and with weights). |
| **Color attribute** | A colour per vertex stored on the mesh. We write the registration distance map into one; Viewport Shading > Color > Attribute shows it. | Vertex colours / Analysis > Deviation colours. |
| **Raycast** | Shoots an invisible ray from the mouse into the scene and returns the first surface point it hits. How landmarks are snapped onto bone. Reads the mesh, never changes it. | What happens when you click a point on a surface (e.g. Drop Pivot). |
| **Modal operator** | A command that stays active and listens to the mouse/keyboard until it finishes (e.g. "click to place a landmark", Esc to cancel). Orbit and zoom still work meanwhile. | An active tool waiting for clicks (e.g. Measure). |
| **Collection** | A folder of objects in the Outliner. Landmarks go in the "Landmarks" collection. | None (Meshmixer has no folders). |
| **Landmark** *(OMFS Planner)* | A named Empty placed on a surface by raycast (Porion R, Nasion, Pogonion...). Locked after placing; place it again to move it. | None (closest: Drop Pivot, but unnamed and not saved). |
| **Parent / child** | A child object follows its parent's transform. The three reference planes are children of the frame Empty, so updating the frame moves them together. | None (closest: grouping objects). |
| **Reference frame** *(OMFS Planner)* | Empty built from porions, orbitales, nasion and basion: Frankfort horizontal, midsagittal and coronal planes, origin = nasion projected on FH. Movements are measured in its axes (left +, anterior +, up +). The anatomy is never moved. | None (Meshmixer only has the world axes). |
| **Boolean modifier** | Combines two closed meshes: Intersect (keep the overlap), Difference (subtract), Union. We cut the bone with a prism under the osteotomy plane: Intersect = segment, Difference = proximal part. New objects are made; the source mesh is untouched. | Edit > Boolean. |
| **Boolean solver (Exact / Manifold / Float)** | Exact: robust, works on meshes with small holes (our CBCT bone), ~1 s on a mandible. Manifold: much faster but refuses any mesh that isn't watertight. Float: older, fragile. | None (Meshmixer picks for you). |
| **Closest point on mesh** | Finds the nearest surface point to a given point. Used to snap the yellow stars onto the buccal bone. Reads the mesh only. | None. |
| **Osteotomy plane** *(OMFS Planner)* | Yellow plane placed below the midpoint of the mental foramina, with height, profile angle, frontal angle and posterior extents measured in the reference frame. Apply Cut splits the bone along it. | Plane Cut, but parametric and re-doable. |
| **Gizmo** | An interactive handle drawn in the 3D view (arrow, ring...) bound to a property: dragging it changes the value, typing the value moves it. The osteotomy has a height arrow, profile and frontal rings, and posterior-extent arrows. | Transform tool arrows and rings. |
| **Sculpt Mode** | Mode where brushes act on the surface. We only use the Mask brush, on a working copy: it never moves vertices. | Sculpt (but here only for masking). |
| **Mask (sculpt)** | A 0-1 value painted per vertex (dark grey = masked). Can grow, shrink, smooth, invert. | Select brush (the selection). |
| **Mask Extract** | Sculpt > Mask > Extract: new object from the masked area, projected on the surface, with a Solidify modifier. | Edit > Extract. |
| **Modifier stack** | The list of modifiers on an object, applied top to bottom (e.g. Displace for the clearance, then Solidify for the thickness). Non-destructive until applied. Shown in OMFS > Confection de guide > 3. | None (Meshmixer operations are immediate). |
| **Solidify / Displace / Remesh modifiers** | Solidify: gives a surface a thickness. Displace: pushes it along its normals (clearance). Remesh (voxel): rebuilds a closed, even mesh. | Extract/Offset, Offset, Make Solid. |
| **3D cursor** | The red-white circle: where new objects appear, oriented by the surface when placed with the Cursor tool (or Shift + right-click). | Drop Pivot. |
| **Snapping** | Shift+Tab: moved objects stick to the surface under the mouse and can align to its normal. | Transform > Snap / Align. |
| **Working copy** *(OMFS Planner)* | Copy of the bone or teeth made by Confection de guide; masking and extraction happen on it, the anatomy is only hidden. | Duplicate. |
| **Cut edge** *(OMFS Planner)* | Confection de guide > 5: a Boolean Difference with a hidden box filling one side of the osteotomy plane, so the guide's edge lies exactly on the cut and guides the saw. Re-click after changing the cut. | Edit > Plane Cut. |
| **3D Viewport** | The editor showing the 3D scene. | The main Meshmixer view. |
| **Editor / Area** | Blender's window is split into areas, and each area shows one editor (3D Viewport, Outliner, ...). | None (Meshmixer has a single fixed layout). |
| **Region** | A part of an area: header, toolbar (T), sidebar (N), main view. | None. |
| **Header** | The strip of menus at the top of each area. | The top menu bar. |
| **Top bar** | The very top of the window: File/Edit menus, workspace tabs. | The top menu bar. |
| **Sidebar (N panel)** | Right-hand side of the viewport, organised in **tabs**. Ours is the **OMFS** tab. Toggle with **N**. | The tool panel that opens on the left when you pick a tool. |
| **Panel** | A collapsible box of buttons inside a region. | A tool's property panel. |
| **Toolbar (T)** | Left-hand side of the viewport: interactive tools (select, move, measure...). Hidden in the OMFS layout. | The left vertical menu (Select, Sculpt, Edit, Analysis...). |
| **Outliner** | Tree list of everything in the scene. In the OMFS layout it is a flat list of objects. The arrow icon shows whether an object is selectable (greyed = locked). | Object Browser. |
| **Properties editor** | Tabs of settings for render, scene, object, modifiers, materials... Hidden in the OMFS layout. | None (spread across Meshmixer's menus). |
| **Workspace** | A saved arrangement of editors for one task (Layout, Sculpting...). We add **Planning**. | None. |
| **Viewport shading** | How the viewport draws surfaces: Wireframe, **Solid** (ours), Material, Rendered. Display only, never changes the mesh. | View > Shaders / Toggle Wireframe. |
| **Studio light** | Solid-mode lighting that is fixed relative to the camera, so anatomy is always lit from the front. | The default Meshmixer lighting. |
| **Cavity** | Solid-mode effect that darkens grooves and brightens ridges. Makes bone surface detail readable. | None. |
| **Object colour** | A per-object colour used when shading colour is set to "Object". We colour by role. | Roughly: per-object colour in the Object Browser. |
| **X-ray** | See-through display (Alt+Z). Useful to look at teeth inside a scan. | Transparent shader. |
| **Backface culling** | Hides the back side of faces. Off for us, so cut or open meshes can be seen from inside. | None (Meshmixer tints back faces). |
| **Clip start / end** | Nearest and farthest distance the viewport draws. Set to 0.1 mm to 10 m so anatomy isn't sliced near the camera. | None. |
| **Frame All (Home)** | Zooms the view so all objects are visible. | View > Recenter View. |
| **Centre view (C)** *(OMFS shortcut)* | Blender's `view3d.view_center_pick` (normally Alt + middle-click) mapped to **C**: the point under the mouse becomes the centre of orbit and zoom. Replaces Blender's circle select on C. | C (centre on point under cursor). |
| **Trackpad navigation** | Two-finger swipe = orbit, Shift + two fingers = pan, pinch = zoom. The navigation gizmo (top right) can be dragged or clicked too. | Alt + left/middle/right drag. |
| **Auto Depth / Zoom to Mouse Position** | Preferences > Navigation: orbit and zoom around the surface under the mouse instead of the view centre. **Off** in OMFS, so orbit and zoom turn around the point set with C. Display only. | None (Meshmixer turns around the C point). |
| **Emulate Numpad** | Preferences > Input: the number row acts as the numpad (1 front, 3 right, 7 top, Ctrl = opposite side). | None. |
| **Keymap** | The table of shortcuts per editor (Preferences > Keymap). Extensions add their own entries, e.g. our C. | Hotkeys list. |
| **Active tool / WorkSpaceTool** | The tool chosen in the toolbar decides what a left click or drag does, and its shortcuts win over the editor's. Our **Navigate** tool makes drag = orbit, Shift + drag = pan, click = select. Activated on Import Case. | Choosing a tool in the left menu. |
| **Object mode / Edit mode** | Object mode moves whole objects; Edit mode edits vertices/faces. OMFS works in Object mode. | Object level vs the Select tool. |
| **Operator** | One Blender command (e.g. `omfs.import_case`), undoable, callable from a button, menu, search (F3) or Python. | A Meshmixer command/tool (e.g. Edit > Make Solid). |
| **Undo (Ctrl+Z)** | Operators marked UNDO can be undone. | Ctrl+Z. |
| **Custom property / PropertyGroup** | Extra data we attach to objects and save in the .blend, e.g. `object.omfs.role`. | None. |
| **Role** *(OMFS Planner)* | Our label on each object: Bone, Teeth, Scan, Nerve, Skin, Segment, Guide, Other. Sets colour and default lock (everything from the CT/CBCT is locked). | None. |
| **Extension** | Modern Blender add-on package, described by `blender_manifest.toml`. Ours adds the OMFS tab and operators. | None (Meshmixer has no plug-ins). |
| **Add-on** | Python code that extends Blender; extensions are the current packaging of add-ons. | None. |
| **Application Template** | A startup configuration that replaces Blender's default UI and scene. Ours is the clean OMFS Planner interface. | None. |
| **Preferences** | User-level settings (enabled extensions, keymap, themes). Saved separately from .blend files. | Meshmixer Preferences. |
| **bpy** | Blender's Python API: everything in this project is written with it. | None. |
