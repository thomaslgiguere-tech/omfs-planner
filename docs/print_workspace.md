# OMFS Planner: print workspace (spec, not built yet)

Status (2026-11-16): Phase 0 done (docs/print_spike_report.md); first version built in the extension
(print_scene.py, print_chain.py, print_profiles.py, print_ui.py), see "What is built" at the end.
Not yet validated by a print: one test print of the synthetic test piece is the remaining step.
This document extends Step 9 (Export) of docs/workflow.md. CLAUDE.md still wins on architecture,
units, parameters and patient-data safety.

## Goal

A separate "Impression" workspace that takes the printable parts of a plan (guides, splints, caps,
models) straight to a print-ready Elegoo .goo file, without leaving Blender. It should feel like
Lychee Slicer or Chitubox: place the parts on a virtual build plate, see the supports update by
themselves, check the printer and resin settings on the right, export.

Behind the scenes, existing tools do the hard work:
- **PrusaSlicer** (CLI): SLA support generation and slicing.
- **UVtools** (UVtoolsCmd): conversion to .goo and printer / resin properties.
OMFS Planner is the interface and the glue. We do not write our own support algorithm.

## User flow

1. **Choose the parts.** From any planning step, an "Imprimer" button opens a list of printable
   objects in the scene (roles: guide, splint, cap, model, and any object the user ticks). One click
   per object to select, then "Préparer l'impression".
2. **Print scene.** The selected parts appear in a dedicated Blender scene ("Impression"), larger and
   uncluttered, centred on a virtual build plate. A transparent box shows the printer's build volume.
3. **Place the parts.** The user moves and rotates parts with the usual gizmos (and typed values).
   Helpers: drop to plate, lift height, auto-arrange, quick tilt presets (e.g. 30 / 45 degrees).
   A part that leaves the build volume turns red.
4. **Supports are automatic.** With "Supports" ticked, every time the user finishes a move or a
   rotation (mouse released, typed value confirmed), the supports of that part are regenerated in the
   background from its current orientation. No button to press. Moving again cancels the job in progress.
5. **Right panel (always visible).** Printer, resin profile, layer settings, estimated time and resin
   volume (details below).
6. **Export.** Only at the very end, "Exporter .goo" slices the plate and writes the .goo file.
   Slicing never runs during placement.

## Non-destructive rules (most important)

- The planning objects and their patient coordinates are **never** modified. The print scene works on
  copies: evaluated meshes (all modifiers and Booleans applied) copied into the "Impression" scene.
- Print transforms (position, rotation on the plate) live only in the print scene.
- Going back to planning and editing a guide must offer "Mettre à jour la pièce dans l'impression",
  keeping the part's print orientation.
- Save the print setup with the case (orientation, supports settings, resin profile, export date) for
  traceability (useful for the master's thesis methods section).

## Clinical rule: protect the bearing surfaces

- Supports must **never** touch a surface that seats on bone or teeth (guide intaglio, splint tooth
  imprints, cap seat) or a precision feature (drill-sleeve tube bores, pin females).
- These surfaces are known from the role and the tool that built them: mark them automatically as
  "no-support" faces (face attribute), shown in a distinct colour in the print scene. The user can paint
  more no-support areas.
- If PrusaSlicer cannot avoid them in the current orientation: warn ("Supports sur une surface d'appui")
  and suggest flipping the part. Never export silently.
- Default orientation preset per role: bearing surface facing up (away from the plate).

## Right panel

- **Printer**: picker with profiles stored as data (JSON), first profile Elegoo Mars 4 Ultra:
  LCD 8520 x 4320 px, pixel 18 um, build volume 153.36 x 77.76 x 165 mm (verify against Elegoo's
  official values before use). Output format: .goo.
- **Resin profile**: the user's own saved resins (name, brand, colour), each with layer height, bottom
  layer count, bottom exposure (s), normal exposure (s), transition layers, lift distance and speed,
  retract speed, light-off delay. One resin marked as favourite and selected by default. Values come from
  the resin maker's sheet, never hard-coded.
- **Supports**: on / off, density, tip diameter, support base (pad) on / off, lift height.
- **Estimates** (refresh after each support update): number of layers, print time, resin volume (ml),
  bounding box on the plate.
- **Checks** (green / red list, see below) and the "Exporter .goo" button.

Printer and resin profiles live in user config (JSON), not in code. Defaults follow the CLAUDE.md
rule: one parameter module, no scattered constants.

## Pre-export checks (run automatically, block export when red)

- Every part closed and manifold.
- Every part and its supports inside the build volume.
- No support on a no-support face.
- No unsupported island (from PrusaSlicer's support points / slice preview).
- Minimum wall thickness for the resin (parameter).
- File name: case ID + part names + resin, no patient names.

## External tools

- PrusaSlicer and UVtools are separate free installs (both on the Mac mini, in /Applications).
- Paths set in the extension preferences, auto-detected when possible. If a tool is missing, a clear
  message with the download link; the rest of the add-on keeps working.
- Calls run in a subprocess off the UI thread (never freeze Blender), with a timeout and a cancel.
- Work files go to a temporary folder per case, deleted after export.
- Expected chain at export: parts (with supports) -> PrusaSlicer CLI with a custom SLA printer profile
  matching the printer -> .sl1 -> UVtoolsCmd convert -> .goo -> resin exposure settings applied.

## Phase 0: spike (do this first, report before any UI work)

Answer these with the real installed versions, on a synthetic test mesh (never patient data):
1. PrusaSlicer CLI: exact command to generate SLA supports and export a .sl1 for a custom printer
   profile (resolution, pixel size, build volume of the Mars 4 Ultra). Version used.
2. How to get the **support geometry back into Blender** for the live preview. Candidates, in order:
   a) CLI export of the object with its supports as a mesh (STL / 3MF), if the CLI can;
   b) read the support points PrusaSlicer writes in a 3MF and draw a light preview in Blender;
   c) otherwise report what is possible.
3. How to pass no-support areas to PrusaSlicer (support blockers, painted areas in a 3MF, or other).
4. UVtoolsCmd: exact command to convert .sl1 to .goo for the Mars 4 Ultra, and how to set exposure,
   layer and lift values (CLI options, or a profile file).
5. Time for one support update on a typical guide (target: a few seconds; if longer, show a progress
   indicator on the part and keep the previous supports visible meanwhile).
6. Open the resulting .goo in the UVtools GUI and check the layers by eye.
If a step is impossible, propose the closest alternative instead of working around it silently.

## Later phases

1. Print scene + part selection + build volume box + placement helpers (no supports yet).
2. Live supports + no-support faces + warnings.
3. Right panel: printer and resin profiles, estimates.
4. Export .goo with pre-export checks.
5. Later: send the file to the printer over the network (Mars 4 Ultra, once the hospital network
   allows it), several printers, saved plate layouts.

## Tests

- Headless tests with synthetic meshes only (tests/run_tests.py): copy into print scene leaves the
  source untouched, build-volume check, no-support face tagging per role, profile loading.
- External tool calls mocked in tests; one optional integration test runs only when the tools are installed.

## Glossary

Add to docs/GLOSSARY.md: Scene (Blender) vs Meshmixer's single workspace; evaluated mesh; build volume;
SLA support tree, island, pad; exposure / bottom layers; .sl1 and .goo formats.

## What is built (2026-11-16)

- **Choose the parts**: "Imprimer" panel in the OMFS tab; ticked parts are copied (evaluated meshes) into
  the scene "Impression". The planning objects are never modified. "Mettre à jour la pièce" re-copies one
  part and keeps its place and orientation.
- **Plate**: build plate, build volume (wire box), "Avant" arrow. Parts are moved with the usual handles
  (arrows and rings shown on the selected part), G / R, or typed values; the height is automatic (locked).
  Tilt presets (0 / 30 / 45 / 90 degrees, sideways, flip), "Disposer". A part out of the volume turns red.
- **Bearing surfaces**: faces within PRINT_BEARING_DISTANCE of the bone / segment / teeth / scan the part fits
  on, and facing it. They arrive turned up and tilted 30 degrees, and a thin slab over them is given to
  PrusaSlicer as a support blocker (drawn translucent on the part).
- **Live supports**: 0.4 s after a part stops moving, PrusaSlicer runs in the background at preview
  resolution and the supports come back as a simplified mesh (about 2 s). Moving again cancels the run.
- **Base**: "Radeau" (surgeon, 2026-11-16): one rigid plate under every part, cured against the build plate,
  its top edge overhanging its foot (bevel) so a scraper slides under it; thickness, margin and bevel are
  settings. Or PrusaSlicer's pad per part, or none.
- **Resin**: profiles as JSON; the two the surgeon prints with are shipped (read from their Lychee .goo
  headers), the favourite selected by default; the panel lists the values; "Modifier / nouvelle résine".
- **Checks and export**: closed parts, inside the volume, supports up to date, tools found. Export slices at
  the printer's resolution, converts to .goo v3, then checks the file: every setting of the profiles is in it
  (39 settings identical to the surgeon's Lychee files), and nothing floats (UVtools islands over 1 mm2; the
  Lychee files have none over 0.2 mm2). The file is kept only if all is green; a .json record is written next
  to it (date, resin, supports, each part's position and rotation).
- **Test piece** ("Pièce test"): one synthetic part for the one validating print: cut corner + raised "R"
  (which way round it prints), 30 x 20 mm plate and 1.0 / 1.5 / 2.0 / 2.5 mm steps (scale), drill-sleeve bore,
  fixation hole, underside as bearing surface (no support on it).

Learned while building:
- A flat face parallel to the plate that starts on support tips is flagged as floating; tilting clears it.
  Hence the default tilt and the message.
- PrusaSlicer's per-material overrides (material_ow_...) win over the plain settings: set both.
- The raft is a part of the plate with supports off (per-object setting in the 3MF), or it gets lifted too.

Not built yet: painting extra no-support areas; precision features (sleeve bores, pin females) as blockers;
minimum wall thickness check; sending to the printer over the network; saved plate layouts.
