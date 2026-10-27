# OMFS Planner: clinical workflow

Living document. Claude Code should read this before working on any feature.
Add clinical nuances under the step they belong to.
This document wins on clinical content; CLAUDE.md wins on architecture, code, hardware, units and
patient-data safety.

Naming used below (matches the object roles in code):
- **CBCT teeth** [TEETH]: whole teeth with roots, segmented from the CBCT. Dictate the teeth **position**. Mesh data never modified; in orthognathic planning they move only as a rigid block, pre-op position always recoverable.
- **Scan teeth** [SCAN]: open STL from the intra-oral scan (digital impression), crowns only. Dictate the true crown **shape**.

## Context

- Goal: plan maxillofacial surgery and design surgical guides in Blender, with free tools only.
- First target: genioplasty. Later: full orthognathic surgery.
- Author is a maxillofacial surgery resident. Knows Meshmixer, new to Blender.
- When explaining Blender, give the Meshmixer equivalent where one exists.
- Blender version: 5.2.2 LTS.
- Decision: Blender does not read DICOM. Imaging and segmentation happen outside Blender (3D Slicer or the DICOM-to-STL tool). The add-on starts from meshes (STL; OBJ and PLY also accepted).
- Code stays generic (segments, planes, movements, guides); genioplasty specifics such as pogonion or the chin techniques are presets, so the same tools later serve Le Fort I and BSSO.

## Pipeline overview

1. Import STL meshes
2. Superimpose the teeth (intra-oral scan) onto the bone (CBCT reconstruction)
3. Orient to a reference plane
4. Define the osteotomy
5. Simulate the movement
6. Check soft tissue
7. Plan the bicortical fixation screws
8. Design the guides
9. Export

## Step 1. Import STL meshes

- Inputs: bone reconstruction STL (from DICOM), skin surface STL (from the same CBCT, for Step 6), scan teeth STL (from the digital impression / intra-oral scan, crowns only), CBCT teeth STL (separate segmentation from the DICOM, whole teeth with roots), inferior alveolar nerve canal STL (segmented from the same CBCT, same coordinates, for Step 7), clinical photos
- Coordinate system: STL must leave the segmentation tool with correct orientation and scale, so everything stays faithful to the patient. Verify on import. The add-on keeps the raw file coordinates and never moves the CBCT meshes.
- What I need to see:
- Nuances: (add here)

## Step 2. Superimpose teeth onto bone

- First step after import.
- Goal: bring the accurate crown shape of the scan teeth onto the CBCT, by registering the scan teeth onto the CBCT teeth. Nothing is deleted: the CBCT teeth (with roots) stay for Step 7.
- What moves: only the scan teeth. Bone and CBCT teeth stay fixed.
- Rigid transform only (rotation + translation, no scale): the scan's shape is never changed, only its position.
- Method (two stages, standard approach in the literature):
  1. Coarse alignment by point matching: click 3+ matching points on both meshes (e.g. incisor tips, molar cusps). Gets close, not precise.
  2. Refinement by surface matching (ICP, iterative closest point), using ONLY the tooth crowns: never the gingiva, palate or other soft tissue captured by the intra-oral scan.
- Selecting the region: the user paints or selects the crowns to use on each mesh. Allow excluding teeth with metal restorations or crowns, since CBCT scatter artifacts distort them on the bone model.
- Blender implementation: point-to-plane ICP in numpy with mathutils.kdtree (built into Blender), no external library needed. Meshmixer equivalent: Align tool (manual points, then surface align).
- How I judge that the alignment is good: show a colour distance map on the crowns plus the mean / RMS error in mm. Working target: mean / RMS error on crowns of about 0.3 mm, flag above 0.5 mm (literature reports ~0.3 mm for IOS-to-CBCT registration; 0.5 mm is a self-chosen safety margin, not a published standard). Overall planning tolerance in the literature: under 2 mm and under 4° between plan and outcome, so registration error must stay a small fraction of that.
- Restorations: more restorations means lower registration accuracy, whatever the method. Show a warning when many teeth are excluded.
- Possible later upgrade: ICP followed by a non-rigid refinement (Thin Shell Demons, open source in ITK), reported to tighten the fit compared to ICP alone. Note: non-rigid would change the scan's shape, which goes against the rigid-only rule above; only consider it as an explicit, optional step.
- Nuances: (add here)

## Step 3. Orient to a reference frame

- Goal: build a clinical reference frame around the anatomy, so every later movement is expressed in clinical terms, e.g. "5 mm straight advancement" or "6 mm forward, 2 mm down, 1 mm right".
- The anatomy is not moved or rotated: the frame is a separate object (an Empty in Blender) placed from the landmarks. Movements and measurements are expressed in that frame; meshes keep their original patient coordinates (needed for Step 9).
- A movement in 3D needs a full reference frame of three perpendicular planes, not one plane:
  - Horizontal: Frankfort horizontal (porion R + L, orbitale), default.
  - Midsagittal: perpendicular to FH, through nasion and a second midline point. Defines left / right.
  - Coronal: perpendicular to both, through nasion. Defines forward / back.
- Axes: X = left/right, Y = forward/back, Z = up/down. Positive directions to be fixed once and shown in the UI.
- Literature: FH is the most widely used reference, but it can deviate several degrees from the patient's natural head position. Some authors recommend natural head orientation (NHO / NHP) as the true horizontal.
- Plan for the add-on: start with FH as default, then allow an optional correction of pitch / roll / yaw to match natural head position (e.g. from a clinical profile photo).
- What I need to see: the landmarks I click, the three planes drawn on the model, and the axis directions.
- Nuances: (add here)

## Step 4. Define the osteotomy

- Version 1 target: standard horizontal sliding genioplasty, one planar cut.
- Osteotomy types to support later (each = one or more cutting planes):
  - Standard horizontal sliding: one plane below the mental foramina and canine roots.
  - Extended / long sliding (chin wing type): same plane extended posteriorly along the lower border.
  - Stepped / double-step: two stacked planes, for large advancements.
  - Reduction (vertical shortening): two parallel planes removing a horizontal wedge.
  - Lengthening: one plane + interpositional graft gap.
  - Narrowing / widening (T-shaped): horizontal plane + midline vertical cut(s), central wedge removed or segments spread.
  - Asymmetric / rotational: horizontal plane, segment rotated or shifted to correct asymmetry.
- Parameters the user sets for each cut: height of the cut below the mental foramina, height below canine root apices, posterior extent (left and right), cut angle in profile relative to FH.
- Key point: the cut angle defines the movement vector. A steeper cut gives more vertical change for the same advancement.
- Safety checks the add-on should display: distance from the cut to each mental foramen and to the canine / incisor root apices; warn under a threshold (threshold TO CONFIRM with supervisor).
- UI, part 1: choose the technique first.
  - A technique picker shows each technique as a small schematic mandible sketch (frontal or oblique view) with its osteotomy plane(s) drawn as yellow lines.
  - One click selects the technique. The add-on then knows how many cut planes are coming and what each one is (e.g. stepped = 2 horizontal planes; T-shaped = 1 horizontal + 1 vertical midline).
- UI, part 2: place the cuts interactively.
  - After the technique is chosen, the matching number of cut planes appear on the patient's bone, and the user moves / tilts each one to its final location.
- Visual landmark: a yellow star 5 mm inferior to each mental foramen (left and right), on the buccal surface of the mandible. Visualization only, not a constraint.
  - Requires the user to mark both mental foramina first (or detect them later).
  - "Inferior" = along the Z axis of the Step 3 reference frame. Star is snapped onto the buccal bone surface.
- What I need to see: mental foramina and root apices marked, the yellow stars, the cut plane(s) over the bone, the distances above.
- Nuances: (add here)

## Step 5. Simulate the movement

- Movements: advancement / setback, up / down, left / right, plus rotations (pitch, roll, yaw), all in the Step 3 reference frame.
- Reference point for the numbers: a landmark on each mobile segment; for genioplasty, pogonion.
- Two ways to move the segment, always in sync:
  - Drag it directly in the viewport, for visualization, constrained to the Step 3 axes. An optional "Lock segment" toggle disables dragging.
  - A movement panel that stays visible at all times, showing the current values (mm and degrees). Clicking a value lets the user type a number, e.g. "6" for 6 mm advancement, and the segment moves to match.
- Dragging updates the numbers live; typing a number updates the segment.
- What I need to see: the segment moving, the bony gap / overlap at the cut, the live numbers.
- Soft tissue: not in version 1, but the design should leave room to show it later (see Step 6).
- Nuances: (add here)

## Step 6. Check soft tissue

- Source: skin surface reconstructed from the CBCT (exported as STL with the bone, same coordinates). No separate 3D face scan.
- Not in version 1.
- What I need to see:
- Nuances: (add here)

## Step 7. Plan the bicortical fixation screws

- Done in the post-movement state (segment at its planned position from Step 5).
- Hardware: Synthes 1.85 mm screws, 1.4 mm drill 03.511.340, MatrixMANDIBLE drill sleeve 03.503.045.
- Each screw is simulated as a cylinder, about 4 mm in diameter: it enters the buccal cortex of the genial (chin) segment, crosses the osteotomy, and engages the lingual cortex of the mandible (proximal segment).
- The cylinder is the drill-sleeve channel, not the screw itself: it becomes the inner diameter of a tube on the cutting guide (see Step 8). Its diameter is a parameter = measured sleeve outer diameter (just under 4 mm, TO MEASURE) + fit clearance. 4 mm is a provisional default.
- User places each cylinder (entry point + direction), adjustable like the cut planes.
- Collision checks, shown live:
  - Teeth: distance from each cylinder to the tooth roots, using the separate CBCT teeth STL (whole teeth with roots). The intra-oral scan only has crowns.
  - Nerve: distance from each cylinder to the nerve canal STL and to the mental foramina.
  - Warn under a threshold (TO CONFIRM).
- Also display: whether each cylinder actually engages both cortices (buccal of segment, lingual of mandible), and the resulting screw length from entry to lingual exit.
- What I need to see: the cylinders through the bone, the nerve canal and roots, the distances and warnings.
- Nuances: (add here)

## Step 8. Design the guides

- Two separate devices, each its own object in Blender: a cutting guide and a positioning guide.
- Designs will vary a lot between cases, so version 1 focuses on generic building tools, not fixed templates.

### Tool 1: surface-to-shell

- User selects (paints) the bone surface to be covered.
- Add-on creates a closed 3D object from it: offset x mm from the surface (clearance gap), then thickened, with smooth boundaries.
- Offset and thickness are adjustable. Default clearance offset: 0.10 mm.

### Cutting guide

- Bone-borne, built with Tool 1.
- Thickness about 10 mm along the saw's path, i.e. in the same plane as the osteotomy (Step 4), to guide the saw blade.
- About 3 mm thick elsewhere.
- The transition between the two thicknesses should be smooth.
- Drill-sleeve tubes: for each bicortical screw from Step 7, a tube is extruded from the cutting guide along the screw axis. Its inner diameter is the Step 7 cylinder (sleeve outer diameter + clearance), so it holds a drill sleeve. Tube length and wall thickness are adjustable per tube. Defaults: length 5 mm, wall thickness 3 mm.
- Because the cutting guide moves with the chin segment, the tubes end up aligned with the planned screw paths once the positioning guide sets the segment in place.

### Tool 2: screw holes (to fix the guide itself)

- Separate from the bicortical screws of Step 7. These only hold the guide to the bone.
- Click a point on the guide to place a fixation screw hole.
- Holes accommodate the Synthes 1.85 mm screws. Hole diameter adjustable (default sized for 1.85 mm screws, exact clearance after test prints). Hole axis: TO DEFINE (perpendicular to the bone surface by default?).

### Tool 3: text labels

- User draws a rectangle on the guide surface to define where a label goes.
- User types the text; the add-on scales it automatically to fit inside the rectangle (keeping proportions).
- Text follows the curved surface of the guide (projected / wrapped onto it), then is merged into the guide.
- Engraved (cut into the guide). Default depth 1 mm, adjustable.
- Safety check: engraving depth must leave enough wall underneath (e.g. 1 mm in a 3 mm area leaves 2 mm). Warn if the text would come too close to the inner (bone-facing) surface.
- Minimum letter height: warn if the rectangle forces very small letters (rule of thumb about 3 mm or more for legible prints; adjust after test prints).
- Blender building blocks: text object, converted to mesh, fitted to the rectangle's bounds, wrapped onto the surface (shrinkwrap), merged with a boolean.

### Positioning guide

- Separate object, tooth-borne (sits on the teeth from the intra-oral scan, Step 2).
- Has a "smart connection" that fits into the cutting guide.
- Role: positions the chin segment. The cutting guide stays screwed to the mobile chin segment and moves with it after the cut. The tooth-borne positioning guide then interlocks with the cutting guide and holds it, and therefore the segment, in the planned new position (from Step 5) until fixation.
- Design consequences:
  - The connection is designed with the cutting guide in its moved (post-movement) position, while the positioning guide sits on the teeth in their unchanged position.
  - The cutting guide needs fixation screws in the mobile segment (below the cut), so it travels with the chin.
  - Connection geometry should fit only one way (no ambiguity in position).

- Nuances: (add here)

## Step 9. Export

- Output: STL, one file per guide (cutting guide, positioning guide).
- Units: millimetres, in the patient's original coordinates (so the STL lines up with the source data in any other software).
- Pre-export checks, run automatically:
  - Mesh is closed and manifold (watertight), no holes or flipped faces, so it slices cleanly for printing.
  - All features are merged into each guide (drill-sleeve tubes, screw holes, engraved labels).
- File names: case ID + guide type (e.g. CASE01_cutting_guide.stl). Avoid patient names in file names.
- Nuances: (add here)

## Orthognathic extension: occlusion seating

- Goal: set the upper and lower arch scans (precise intra-oral STLs) against each other in the final
  post-operative occlusion, the most important goal of orthognathic surgery.
- Order: comes after the osteotomy design and separation tools for orthognathic surgery (Le Fort I,
  mandibular osteotomy such as BSSO). Without them, "bone that moves with the arch" can only be whole
  bones: the whole skull would follow the upper arch, and the condyles would follow the lower arch.
- Absolute constraint: the scans (and any CBCT mesh moving with them) are moved, never modified.
  Rigid displacement only; a shape check proves the mesh data is unchanged.
- Part 1 (done):
  - Top of the panel: choose which scan is the upper arch and which is the lower arch.
  - Each arch has a mobile block. Lists "moves with the upper arch" / "moves with the lower arch"
    associate other objects (CBCT teeth, bone segment, nerve). Optional; one block per object.
  - Removing an object from a block puts it back at its pre-op position. "Back to pre-op" resets a block.
  - Pre-op ghost: semi-transparent linked duplicates (same mesh data) frozen at the pre-op position.
  - Warnings: an arch with associated objects but no teeth registration; a moved block when trying
    to register its scan.
- Part 2 (to specify with the surgeon): how the final occlusion is obtained and judged.
  - Source: scan of hand-articulated models in occlusion (register each arch onto it), virtual
    seating in Blender, or both?
  - Which arch moves (lower onto a fixed upper, the reverse, or either)?
  - Help for virtual seating: manual 6-DOF moves with a contact / penetration colour map, cusp-to-fossa
    point pairs, automatic "settle" (drop to contact, then maximise contacts without penetration).
  - Criteria shown live, with thresholds: contacts (number, left/right, anterior/posterior), max
    penetration, overjet, overbite, midlines, molar and canine class, transverse coordination, cant.
  - Then: movement of the maxillomandibular complex relative to the skull (Step 5), intermediate and
    final splints?
- Troubleshooting to keep in mind: open scan meshes make penetration detection unreliable near their
  borders; orthodontic brackets and wires create false contacts; gingiva overlap.
- Nuances: (add here)

## Open questions

- Superimposition: confirm the 0.5 mm flag threshold with a supervisor or the program's usual practice.
- Screw holes: default hole axis (perpendicular to the bone surface?).
- Drill sleeve 03.503.045: measure the exact outer diameter, then set the fit clearance after test prints.
- Osteotomy: safety margins below the mental foramen and root apices.
- Bicortical screws: how many (2? 3?), minimum distance to nerve canal and roots.
