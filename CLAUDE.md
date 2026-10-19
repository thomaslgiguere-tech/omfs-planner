# OMFS Planner

Free, Blender-based tool for virtual surgical planning (VSP) and 3D-printed surgical
guide design in oral and maxillofacial surgery. Built by an OMFS resident.

## Goal

Reproducible, parameterised genioplasty planning and guide design (cutting +
repositioning guide) in Blender. The architecture must later extend to full
orthognathic surgery (Le Fort I, BSSO, multi-segment). **Keep it modular**: segments,
reference planes, movements and guides are generic concepts, never hard-coded to the chin.

## Repo layout

```
extension/omfs_planner/      Blender extension (blender_manifest.toml): operators + panels
  roles.py                   object roles (bone, teeth, scan, segment, guide, ...) + colours
  scene_setup.py             millimetre units, viewport shading, framing
  mesh_io.py                 STL / OBJ / PLY import with raw coordinates
  locking.py                 lock / unlock anatomy
  ops_import.py              "Import Case" operator
  ui.py                      sidebar tab "OMFS" in the 3D viewport
app_template/omfs_planner/   Application Template: clean surgical UI + Developer mode
tools/dev_link.sh            symlinks both into the Blender user dirs (development)
tools/make_test_case.py      writes synthetic test meshes (never patient data)
tests/run_tests.py           headless tests: blender -b --factory-startup --python tests/run_tests.py
docs/GLOSSARY.md             Blender term -> what it does -> Meshmixer equivalent
docs/APP_TEMPLATE.md         what the template hides, and what can't be hidden without compiling
```

## Environment

- Blender 5.2.2 LTS (Python 3.13), macOS, Mac mini M5 Pro, 64 GB RAM.
- Extension installed as a symlink in the `user_default` repo -> module `bl_ext.user_default.omfs_planner`.
- App template symlinked in `scripts/startup/bl_app_templates_user/omfs_planner`.
- Claude controls the running Blender through the blender-mcp add-on (socket 127.0.0.1:9876).

## Development notes (learned the hard way)

- Run tests: `/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/run_tests.py`
- Test the template in a *separate* instance (`Blender --app-template omfs_planner --python <script>`),
  never by replacing the user's open file without asking.
- Never change region visibility (`show_region_toolbar`, ...) on a screen that isn't displayed yet,
  or from a timer without `temp_override(window, screen, area)`: Blender crashes.
- `workspace.duplicate` switches the window only on the next tick: diff `bpy.data.workspaces` to find the copy.
- `region.active_panel_category` is 'UNSUPPORTED' until the sidebar has been drawn once: retry with a timer.
- `hasattr(bpy.ops.x, "y")` is always True; check `bpy.types.X_OT_y` instead.
- Script-driven UI changes don't redraw the top bar (it isn't in `screen.areas`); real clicks do.
- Reload the extension live without `script.reload` (it would restart the MCP add-on): disable,
  drop `bl_ext.user_default.omfs_planner*` from `sys.modules`, enable.

## Inputs

- Bone and teeth meshes exported from 3D Slicer (CT/CBCT segmentations), typically STL; OBJ and PLY also supported.
- Intraoral scan STLs (upper/lower jaw).
- Import keeps raw file coordinates (forward Y, up Z, scale 1) so all meshes from one case stay registered.

## Units

Millimetres everywhere: **1 Blender unit = 1 mm** (scene unit scale 0.001, length unit mm).

## Guide hardware (all values must be parameters, never constants buried in code)

- Synthes 1.85 mm screws.
- MatrixMANDIBLE drill sleeve 03.503.045: outer diameter just under 4 mm, **exact value to be measured**.
- 1.4 mm drill 03.511.340.

## Planned features (in order)

1. Import + scene setup  *(done: Import Case)*
2. Precise alignment of CT teeth with the intraoral scan STL
3. Osteotomy plane(s) with numeric advancement, vertical and rotation, measured relative to a
   user-defined reference plane (e.g. from landmarks the user places)
4. Segment movement with live measurements
5. Guide body from the bone contact surface
6. Drill sleeves: diameter control, optional physical sleeve cylinders protruding from the guide surface
7. Print-ready export with manifold checks

## Interaction rules

- Anatomy that must not move is **locked**: visible and measurable, but not selectable or movable
  (`hide_select` + transform locks). Bone is locked on import.
- Segments move only through numeric controls in the panel; an optional unlock allows manual adjustment.

## Rules

- **Never commit patient data** (.gitignore blocks meshes, DICOM, .blend, data/, cases/).
- Keep code simple and readable. One feature per commit.
- Every explanation to the user doubles as a Blender tutorial: name the Blender concept, what it
  does to the mesh data, and the closest Meshmixer equivalent (or say there is none).
  Add new terms to docs/GLOSSARY.md.
- Ask before anything destructive.
