# OMFS Planner

Free, Blender-based tool for virtual surgical planning (VSP) and 3D-printed surgical
guide design in oral and maxillofacial surgery. Built by an OMFS resident.

## Goal

Reproducible, parameterised genioplasty planning and guide design (cutting +
repositioning guide) in Blender. The architecture must later extend to full
orthognathic surgery (Le Fort I, BSSO, multi-segment). **Keep it modular**: segments,
reference planes, movements and guides are generic concepts, never hard-coded to the chin.
Genioplasty specifics (pogonion, chin techniques) are presets on top of generic code.

## Which document wins

- **Read [docs/workflow.md](docs/workflow.md) before working on any feature.** It is the clinical spec.
- **docs/workflow.md wins** on clinical content: steps, gestures, what the surgeon must see.
- **CLAUDE.md wins** on architecture, code, hardware, units and patient-data safety.

## Repo layout

```
extension/omfs_planner/      Blender extension (blender_manifest.toml): operators + panels
  roles.py                   object roles (bone, teeth, scan, segment, guide, ...) + colours
  scene_setup.py             millimetre units, viewport shading, framing
  mesh_io.py                 STL / OBJ / PLY import with raw coordinates
  locking.py                 lock / unlock anatomy
  crown_fusion.py            CBCT teeth crowns brought onto the scan crowns, then the scan's own crown triangles
                             transplanted (occlusal / incisal anatomy identical), seam to the CBCT tooth;
                             roots untouched, new mesh, CBCT mesh kept; splints / chape accept these teeth
  ops_import.py              "Import Case" operator
  new_case.py                "Nouveau cas": CBCT (DICOM) + upper / lower scan -> scans imported, AI segmentation in
                             the background (or reused), CBCT in the slices (worker thread), Segmentation workspace
                             when ready, registration partners set on import; next steps ticked in the Case panel
  movement.py                segment movement (Step 5): 6 DOF on the osteotomy plane or frame axes; presets per
                             segment type (genioplasty, Le Fort, BSSO), followers carried by a segment
  movement_gizmos.py         drag arrows / rings for the segment movement, on the movement axes
  occlusion.py               occlusion seating: arch scans, rigid mobile blocks, pre-op ghost
  occlusion_prepare.py       seating's first step: hide everything but the two arch scans, clean their open
                             borders (band, spikes, crumbs; still open, original mesh kept)
  occlusion_seating.py       springs between the arches, seating solver (upper onto the fixed lower), overlap map
  occlusion_segments.py      segmental set-up, step 1: copy of the upper arch, palatal midline cut, candidate cuts at
                             the contacts (proximal walls), planes centred between the roots (CBCT); hover / click;
                             split into segments with zero thickness
  occlusion_segment_seat.py  segmental seating: each segment on the lower arch (seating solver), angulation hold
                             (rotational spring), "Selon la courbe" (contact copies of the neighbours)
  splint.py                  Object creator: Splint (coverage points -> two surfaces, automatic round outline) /
                             Chape (arch curve); blank, imprints, solid, text
  splint_holes.py            splint option: wire holes in the buccal flange (click, slide, click; one Boolean)
  cylinders.py               adjustable workshop cylinders (diameter, length) + their drag handles
  screws.py                  bicortical screws: two clicks on the bone (entry, exit), exact lengths,
                             ends slid on the bone surface
  sleeves.py                 drill-sleeve tube / bore on a final screw's axis (click, slide, click), guide
                             fixation holes, Mega Boolean (guide then cap, in order)
  trim.py                    Plane cut (line on screen -> live half-space Boolean) and Erase & Fill (smooth patch, rim kept)
  pins.py                    tubes and pins: conical females in the guide (stop on the osteotomy
                             plane), males and curved arms on the cap; same pins between the models
  text_labels.py             universal text tool: engraved / embossed text on any printed part (two
                             clicks, Boolean cutter), print check and STL export
  model_base.py              Création de modèle: open arch scan closed into a printable model (wall,
                             flat base, bar on a horseshoe; teeth zone never trimmed)
  tidy.py                    used helper objects (Boolean operands no longer needed) moved to
                             a hidden "Outils utilisés" collection, back if needed again
  section.py                 Section & Analysis: section plane, slab (clipping), radiograph view
  section_contours.py        live section contours, caps, overlap on the section
  analysis.py                exact intersection volume (Boolean overlay, live or snapshot)
  distance_map.py            distance map (Geometry Nodes modifier), closest-point marker
  snapshot.py                section caption + scale bar, PNG snapshot of the view, section on or off (offscreen render)
  measure.py                 point-to-point measurements (click-drag), drawn in the view and the snapshot
  print_scene.py             Impression workspace: print scene (copies on a virtual build plate), bearing
                             faces -> no-support slabs, live supports, raft, checks, export .goo, test piece
  print_chain.py             the external tools behind it: 3MF, PrusaSlicer (supports, slicing), layers ->
                             support preview, UVtools (.goo, islands); no Blender UI, runs in worker threads
  print_profiles.py          printer and resin profiles as JSON (shipped in print_profiles/, the user's own
                             in Blender's config folder)
  print_ui.py                "Imprimer" panel (OMFS tab) and the "Impression" tab of the print scene
  ortho_osteotomy.py         Le Fort I and mandibular (BSSO) cuts: Le Fort, clicked points pushed into a cutting sheet
                             along the view up to two pterygomaxillary stops, then down at 90°; BSSO, a buccal vertical cut
                             (2 clicks) down to the nerve canal, then a curve (clicks) to the lingual cortex; then the bone (a copy) split into
                             distal / proximal segments; separate from the genioplasty plane (osteotomy.py)
  ortho_movement.py          Le Fort I movement: maxilla + upper teeth as one block (followers), tracked points
                             (ANS, A, upper incisor, PNS), occlusal plane vs Frankfort
  ortho_ui.py                "Ostéotomies orthognathiques" and "Mouvement du Le Fort" panels (OMFS tab)
  volume_scene.py            Segmentation workspace: a DentalSeg .nrrd (drag and drop) -> one live surface per
                             structure (threshold, pieces, smoothing), pieces given to another structure or
                             deleted (voxel flood fill), then imported into the plan with their roles
  volume_ui.py               "Segmentation (.nrrd)" entry (OMFS tab) and the "Segmentation" tab
  slice_core.py              slice viewer core, numpy only: grayscale NRRD (LPS), trilinear reslicing, window / level,
                             mesh section by a plane (tests: python3 tests/test_slice_core.py)
  slice_viewer.py            Coupes: CBCT in Quad View (axial / sagittal / coronal + 3D), mesh contours in colour,
                             mouse modal (wheel, left + right scroll, crosshair, window / level); draft, docs/slice_viewer.md
  point_balls.py             clicked points (landmarks, splint points, osteotomy lines) drawn as solid shaded balls
  ui.py                      sidebar tab "OMFS" in the 3D viewport
app_template/omfs_planner/   Application Template: clean surgical UI + Developer mode
tools/dev_link.sh            symlinks both into the Blender user dirs (development)
tools/make_test_case.py      writes synthetic test meshes (never patient data)
tools/print_spike/           Phase 0 spike of the print chain (scripts; report in docs/print_spike_report.md)
tests/run_tests.py           headless tests: blender -b --factory-startup --python tests/run_tests.py
docs/workflow.md             clinical workflow, step by step (the feature spec)
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
- Never monkeypatch methods of a registered class (e.g. a GizmoGroup's draw_prepare) or call
  `wm.redraw_timer` from an MCP script: Blender crashed. Debug gizmos in a separate instance.
- A live reload doesn't always rebuild gizmo groups; restart Blender to check gizmo changes.
- Clipping planes: `rv3d.use_clip_planes = True` alone clips nothing; run `view3d.clip_border` once
  (in a region override), then overwrite `rv3d.clip_planes`. A point is kept when n . p + d >= 0.
- `screen.screenshot_area` returns the last drawn frame, which can be stale (window not redrawn,
  e.g. unfocused): for images of the view use `GPUOffScreen.draw_view3d` (Python draw handlers are
  not included: draw them into the offscreen yourself). `look` (MCP) doesn't show draw handlers either.
- Blender 5 Geometry Nodes modifier inputs: `mod.properties.inputs.<socket identifier>.value`
  (`mod["Socket_1"]` no longer works).
- GPU index buffers need 4-byte ints: numpy arithmetic silently upcasts int32 to int64.
- `mathutils.geometry.tessellate_polygon` mis-fills complex CT section loops (bridges separate
  regions): fill sections row by row with the even-odd rule instead (section_contours.fill_rows).
- Sculpt face sets are the face attribute `.sculpt_face_set` (readable live in Sculpt Mode); the mask is
  the point attribute `.sculpt_mask`: write it in Object Mode, then go back to Sculpt Mode.
- Solidify `use_even_offset` spikes (tens of mm) at sharp folds of an open border: keep it off for guides.
- An object deleted in the viewport stays in `bpy.data` while a PointerProperty refers to it, outside
  every collection: check `obj.name in scene.objects` before using a stored pointer.
- Reload the extension live without `script.reload` (it would restart the MCP add-on): disable,
  drop `bl_ext.user_default.omfs_planner*` from `sys.modules`, enable.

## Inputs

All meshes are typically STL; OBJ and PLY are also supported. Role names in code are in brackets.

- From the CBCT (3D Slicer segmentations, all in the same coordinates):
  bone [BONE], whole teeth with roots [TEETH], skin surface [SKIN], inferior alveolar nerve canal [NERVE].
- Intraoral scan STLs, upper/lower jaw, crowns + gingiva, open meshes [SCAN].
- Clinical photos (later, for natural head position). Patient data: never committed.
- Import keeps raw file coordinates (forward Y, up Z, scale 1) so all meshes from one case stay registered.
- **CBCT mesh data is never modified.** In planning, CBCT meshes may move only as a rigid block
  (parented to a block Empty, see occlusion.py), and the pre-op position is always recoverable.
  The CBCT teeth dictate the teeth **position**; the scan dictates the true crown **shape**.
  The scan is registered onto the CBCT teeth by a **rigid transform** only
  (rotation + translation, no scale, vertices untouched). "Canner la position dentaire initiale" then bakes that
  rigid pose into the scan's vertices (shape unchanged, transform reset to identity): it is the scan's
  position zero for every later step (registration.freeze_scan; file coordinates kept as frozen_matrix).
  Exception asked by the surgeon (2026-11-18): "Modifier la géométrie des dents CBCT" gives the CBCT teeth
  object a **new** mesh whose crowns lie on the scan crowns (crown_fusion.py); the CBCT mesh data-block itself
  is kept untouched on the object and can be put back. Roots never move.
- **Anatomy is never reoriented.** The clinical reference frame is a separate object (Empty) built from
  landmarks; movements are expressed in that frame. Export stays in the original patient coordinates.

## Units

Millimetres everywhere: **1 Blender unit = 1 mm** (scene unit scale 0.001, length unit mm).

## Guide hardware (all values must be parameters, never constants buried in code)

- Synthes 1.85 mm screws: used for the bicortical fixation and to hold the guide to the bone
  (guide fixation holes are sized for them).
- MatrixMANDIBLE drill sleeve 03.503.045: outer diameter just under 4 mm, **exact value to be measured**.
  Drill-sleeve tube inner diameter = measured sleeve outer diameter + fit clearance (both parameters).
  The "4 mm" in docs/workflow.md is a provisional default only.
- 1.4 mm drill 03.511.340.
- All defaults (clearances, thicknesses, thresholds) live in one parameter module, not scattered in code.

## Planned features (in order; details in docs/workflow.md)

1. Import + scene setup  *(done: Nouveau cas = CBCT + two scans; Import Case = folder of STLs)*
2. Register the intraoral scan onto the CBCT teeth (point matching, then rigid ICP on crowns only)
3. Reference frame: three perpendicular planes from landmarks (Frankfort horizontal by default)
4. Osteotomy plane(s): technique picker, then place the cut planes
5. Segment movement in 6 degrees of freedom, in the reference frame, with live measurements
   *(done: Mouvement du segment)*
6. Soft tissue check (not in version 1)
7. Bicortical fixation screws / drill-sleeve channels with collision checks (roots, nerve)
8. Guides: surface-to-shell, cutting guide + drill-sleeve tubes, guide fixation holes, text labels,
   tooth-borne positioning guide
9. Print-ready export with manifold checks

Orthognathic extension: Le Fort I / BSSO osteotomy design and segment separation, then occlusion
seating (upper and lower arch scans set into the final occlusion; done: arches, mobile blocks,
shape check, pre-op ghost, springs + magnet + overlap map), then a "Mouvement chx orthognathique"
tab: what moves with each arch (associations, code kept in occlusion.py) and movement of the
maxillomandibular complex.

Cross-cutting tool (not a numbered step, usable at every step): **Section & Analysis** panel.
Section plane + slab, section contours, overlap highlight and intersection volume, distance map,
PNG snapshot. Strictly non-destructive: clipping, draw handlers, modifiers or overlay objects,
never the original mesh data. Details in docs/workflow.md.

## Interaction rules

- Anatomy that must not move is **locked**: visible and measurable, but not selectable or movable
  (`hide_select` + transform locks). Bone is locked on import.
- Segments move two ways, always in sync: dragging in the viewport (constrained to the reference frame
  axes) and typing values in the movement panel. An optional "Lock segment" toggle disables dragging.

## Rules

- **Never commit patient data** (.gitignore blocks meshes, DICOM, .blend, data/, cases/).
- Keep code simple and readable. One feature per commit.
- Every explanation to the user doubles as a Blender tutorial: name the Blender concept, what it
  does to the mesh data, and the closest Meshmixer equivalent (or say there is none).
  Add new terms to docs/GLOSSARY.md.
- Ask before anything destructive.
