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
- Done: "Mouvement du segment" panel (movement.py, movement_gizmos.py).
  - Six values on the segment: advance / up / left (mm, net translation of the pivot) and pitch
    (+ front up), roll (+ left side up), yaw (+ front to the left).
  - Axes (surgeon's choice, 2026-10-31): by default the **osteotomy plane**, as in a sliding
    genioplasty: "up" = the cut's normal (+ opens the cut, - closes it), "advance" = along the cut
    where it meets the facial midsagittal plane (forward as on the face, never sideways),
    "left" completes them; the midline is perpendicular to the cut, through the pivot (pogonion or
    menton). Alternative: the reference frame (Frankfort / midsagittal / coronal). Changing the
    axes never moves the segment: the numbers are rewritten for the same pose.
  - With the osteotomy axes, the panel also gives each tracked point's movement in the reference
    frame (e.g. 6 mm along a 12° cut = 5.9 mm forward, 1.2 mm up on the face).
  - Pivot: pogonion (default), menton or segment centre, taken before the move. Changing the pivot
    never moves the segment; the translation is rewritten as the new pivot's displacement.
  - Drag: three arrows (left red, anterior green, up blue) and three rings at the pivot, on the
    movement axes (not the object's or the world's). Typing and dragging write the same values.
    "Lock segment" hides the handles. The segment's own transform is locked, so G / R can't desync.
  - Live boxes next to the pogonion and menton where the segment takes them, with their net
    translation in words (e.g. "6.0 mm avance, 2.0 mm bas, 0.0 mm gauche/droite") and the total;
    a magenta line from the pre-op point. Same numbers in the panel.
  - Pre-op pose stored when the cut is applied ("Retour préop" is exact); re-applying the cut keeps
    the planned movement.
  - Rotations are Euler angles applied X (pitch), then Y (roll), then Z (yaw): with several
    rotations at once, each number is that component, not an independent geometric angle.
  - Limitation: the numbers use the current axes; rebuilding the frame or changing the cut angles
    after moving the segment changes what the same numbers mean (the segment stays where it is until
    a value changes). Re-applying the cut keeps the six numbers, so the segment follows the new cut.
  - "Appliquer le même mouvement au guide de coupe" (surgeon, 2026-11-01): the cutting guide stays
    screwed to the chin segment after the cut (Step 8), so it moves by the segment's exact movement,
    rigidly, with the tools of its Boolean modifiers (sleeve cylinders, text, cut-edge box) so its
    holes go with it. Each keeps its pre-op pose; re-clicking after a new movement starts from it,
    and the back arrow puts the guide exactly back. One click, not a live link: the guide stays where
    it is until re-clicked.
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
- First tool (surgeon, 2026-11-04), guide workshop section 4, "Positionnement de vis bicorticale"
  (screws.py): the screw itself as a cylinder (Synthes 1.85 mm, SCREW_DIAMETER), not yet the sleeve
  channel. Click the entry on the bone (usually the buccal cortex of the genial segment), the screw
  then follows the mouse with its live length, click the exit (usually the lingual cortex of the
  proximal segment); the two clicks may land on two different objects (roles bone / segment;
  guide, teeth, scans in front are passed through). Several in a row. The panel lists "Vis 1:
  14.32 mm" with the objects of entry and exit. Each end can be moved: it slides on the bone surface
  under the mouse (the surface is the absolute reference), a click fixes it, the length follows.
  Still to come: collision checks (roots, nerve), both cortices engaged. The sleeve cylinders now take
  their axis from these screws (Step 8, cutting guide).

## Step 8. Design the guides

- Two separate devices, each its own object in Blender: a cutting guide and a positioning guide.
- Designs will vary a lot between cases, so version 1 focuses on generic building tools, not fixed templates.

### Trimming tools (surgeon, 2026-11-05)

- **Plane cut** (as Meshmixer's Edit > Plane Cut): drag a line across the part; the plane goes
  through it, edge-on to the view, and the smaller side is removed (live preview). Move / turn the plane
  with the gizmo, flip the side, apply or cancel. The part stays closed (Boolean Difference with a
  half-space box). The cut waits before the other Booleans: applying it applies the modifiers above it
  (thickness) and leaves tubes, pins, holes live. A trim is never redone like a hole.
- **Erase & Fill** (as Meshmixer's Erase & Fill): the faces selected in Edit Mode are deleted and each
  hole is filled with a smooth patch (membrane on the hole's rim, refined to the rim's edge length).
  The rest of the mesh, rim included, is not modified; the part stays closed.

### Order of operations (surgeon, 2026-11-04)

1. Simulate the osteotomy (Step 4).
2. Create the cutting guide: shape and extent (zones painted on the bone, extracted; cut edge).
3. Hide the guide and simulate the segment's movement (Step 5).
4. Place the final bicortical screws (Step 7), segment in its planned position.
5. Back to the guide: place the pins' females (usually a little away from the screws so the Booleans
   miss them, but crossing is acceptable).
6. Move the guide with the genioplasty ("Appliquer le même mouvement au guide de coupe"): the pins
   follow it (their Empties move, every part is rebuilt there).
7. Create the arms from the cap to the pins, where the pins are now.
8. Create and adjust the drill-sleeve cylinders (tube and bore): their axis is the final screw's,
   their position along it is chosen by the surgeon (click near the screw, slide, click); arrows and
   diameter stay adjustable until the Booleans are applied.
9. Mega Boolean, in order: unions (bosses, bridges, tubes), then differences (pin bores, sleeve
   bores), the cut edge, and last the small holes that fix the guide to the bone. A sleeve tube or
   hole crossing a male pin or arm of the cap is subtracted from the cap too. Then the cap's own stack.
   Pins may be placed right over the guide's body: where a male overlaps the guide itself (its own
   volume, before any Boolean), the guide is hollowed for it (shoulder diameter + clearance, from the
   osteotomy plane out along the pin axis, PIN_MALE_CHANNEL long, so the male can go in), with the
   other differences.
   Holes always win (surgeon, 2026-11-05): a difference applied earlier (e.g. by "Appliquer les
   modificateurs" before polishing) is remembered on the part and dug again after any later union that
   reaches its cutter (a pin boss over a sleeve bore must never close the sleeve's passage).

### Tool 1: surface-to-shell

- User selects (paints) the bone surface to be covered.
- Add-on creates a closed 3D object from it: offset x mm from the surface (clearance gap), then thickened, with smooth boundaries.
- Offset and thickness are adjustable. Default clearance offset: 0.10 mm.
- Painting the zones (Confection de guide, section 1, surgeon's choice 2026-10-31): Face Sets, not the
  mask. Several zones at once (guide contact, cut window, sleeves, areas to avoid), sharp borders,
  kept with the working copy; pick a zone by clicking it, grow / shrink / isolate it, then extract it
  as a shell. The mask stays as an option (1b) to smooth a zone's border before extracting.
- Which object the tools act on (surgeon's feedback 2026-10-31: a global "Départ" picker was
  counter-productive, modifiers landed on the working copy): the object to copy is asked only when
  creating the working copy; a "Guide à modifier" picker (default: the last extracted guide) is what
  sections 2 to 6 modify. With the working copy or locked anatomy active, the tools go to the guide;
  another free mesh that is active (e.g. a sleeve cylinder) is used as is.
- Polishing the outside of a guide by hand: "Polir l'extérieur au pinceau" applies the modifiers,
  masks the intrados (guide vertices closer than 1 mm to the bone: params.GUIDE_PROTECT_DISTANCE) and
  opens Sculpt Mode with the Smooth brush; any brush is then safe for the fit. "Lisser tout
  l'extérieur" smooths the whole unmasked surface one pass per click.
- Sculpt section, workshop section 7 (surgeon, 2026-11-02: sculpt almost any object, mainly the
  guide, to soften it, with the intrados masked; mask from a sharp-angle control and from a precise
  selection): any free mesh (guide, cap, arm; never anatomy), modifiers applied first. Brushes
  Smooth / Flatten / Clay Strips / Grab, "smooth all unmasked" filter, the usual mask tools. Three ways
  to protect the intrados:
  - distance to the bone (former "Polir l'extérieur", moved here);
  - "Découper en zones par angle aigu": face sets split wherever neighbouring faces fold more than the
    angle (default 20 degrees, params.SCULPT_SHARP_ANGLE; on case1's cap 20 splits intrados / outside,
    30 merges them through a rounded rim), then click zones to mask (Ctrl: unmask). Blender's own
    "Initialize Face Sets > Sharp Edges" only reads edges marked sharp by hand, so the angle is computed;
  - "Masque = sélection": the Edit Mode selection becomes the mask exactly.
- Sleeve cylinders are adjustable (surgeon, 2026-11-01: moving and turning worked well, size should
  be as quick): each cylinder keeps its diameter and length (cylinders.py) and its mesh is rebuilt
  from them, so a typed 4.10 mm is exact. Drag handles on the active cylinder: blue arrows at both
  ends change the length while the other end stays put (the end in the bone does not move), an
  orange side arrow changes the diameter; typing in section 4 changes the length about the centre.
  A new cylinder becomes the Boolean tool ("Avec") automatically.
- Boolean order (found on case1, 2026-11-01: holes dug before the tubes were unioned, so the tubes
  filled them): the Booleans are kept sorted, tubes (Union) first, then holes (Difference), then the
  cut edge last, in the slots they occupy (Displace / Solidify stay first). "Appliquer les
  modificateurs" sorts once more, applies, then hides (never deletes) the cylinders and boxes used.
- Tubes and pins, workshop section 6 (surgeon, 2026-11-01; replaces a hand-made 544k-vertex pin
  model with generated parts): the cutting guide and the cap plug into each other, usually 3
  connections. Surgeon's choices: pin axis perpendicular to the osteotomy plane (warned: if that
  plane is not perpendicular to the cap's insertion axis, cap-on-teeth and pins-in-females cannot
  seat in one gesture), conical pins like the surgeon's model, curved arm.
  - Placing a female, guide and models alike (surgeon, 2026-11-03): click, slide with the mouse,
    click to confirm. On the guide it first appears next to the clicked surface (below), then slides
    in the osteotomy plane: its stop stays exactly in the plane. Slid away from the guide, a bridge
    joins them: the boss's whole guide-side half (as wide as the boss, as deep), straight out to the
    guide (1.5 mm into it), its top flush with the plane, so it also adds cutting-plane surface for
    the blade (surgeon's correction: a bridge, not a rod).
  - Female: click on the guide; the axis runs `embed` mm inside the boss's edge from the clicked
    surface, perpendicular to the plane (osteotomy plane moved with the guide if it followed the
    segment). Boss (Union) and conical bore (Difference) below the plane; the boss's top is exactly
    in the plane: the male's shoulder rests on it, so the axial stop is the osteotomy plane (and the
    guide's cut edge, flush with the plane, trims nothing of it). The tip stays 0.3 mm short of the
    bore's bottom: one stop only.
  - Male: "Ajouter les pins mâles" puts a pin (bore minus 0.25 mm radial clearance) in every female,
    shoulder and back above the plane, unioned to the chosen cap (Manifold solver). 0.10 mm printed
    too tight on the models (surgeon, 2026-11-07); 0.25 mm is to be confirmed by a print.
  - Arm: one click on the cap per pin; a tube leaves the cap at 90 degrees to its surface and
    arrives on the pin's back along its axis, joined by a cubic Bezier (handles a third of the
    distance), "like plugging in a cable". Warns when the bend is tighter than the tube's radius.
  - Every part is a generated closed mesh under a Boolean modifier: settings rebuild all parts;
    nothing is cut until Appliquer. Each connection is an Empty "Pin n" at its stop, trash to remove.
    Dimensions in params.py (PIN_*), all TO TEST by printing. Case1: build 0.14 s, cap Boolean 0.6 s.
- "Tourner autour de l'axe du curseur" (surgeon, 2026-11-01): rotates the active object about the 3D
  cursor's Z axis (the bone normal where the cursor was placed), through the cursor's point; X / Y
  switch to the cursor's other axes. E.g. turn a sleeve cylinder about its entry point on the bone.
- A working copy deleted from the viewport counts as gone (it used to crash the zone tools: the
  panel's pointer kept it in the file, outside the scene; surgeon, 2026-11-01).

### Cutting guide

- Bone-borne, built with Tool 1.
- Thickness about 10 mm along the saw's path, i.e. in the same plane as the osteotomy (Step 4), to guide the saw blade.
- About 3 mm thick elsewhere.
- The transition between the two thicknesses should be smooth.
- Drill-sleeve tubes: for each bicortical screw from Step 7, a tube is extruded from the cutting guide along the screw axis. Its inner diameter is the Step 7 cylinder (sleeve outer diameter + clearance), so it holds a drill sleeve. Tube length and outer diameter are adjustable per tube. Defaults: length 5 mm, outer diameter 7.5 mm (surgeon, 2026-11-04).
- Because the cutting guide moves with the chin segment, the tubes end up aligned with the planned screw paths once the positioning guide sets the segment in place.
- Done (2026-11-04, sleeves.py): the sleeve cylinders take the axis of a final screw (Step 7); click near
  the screw, the cylinder slides along its axis with the mouse, a click fixes it; the tube is united to
  the guide and the bore subtracted at once (live modifiers: arrows and diameter still adjust them).

### Tool 2: screw holes (to fix the guide itself)

- Separate from the bicortical screws of Step 7. These only hold the guide to the bone.
- Click a point on the guide to place a fixation screw hole.
- Holes accommodate the Synthes 1.85 mm screws. Hole diameter adjustable (default sized for 1.85 mm screws, exact clearance after test prints). Hole axis: TO DEFINE (perpendicular to the bone surface by default?).
- First version (2026-11-04): a click on the guide gives a hole of FIXATION_HOLE_DIAMETER (2.0 mm, TO TEST)
  through the guide along the surface normal there, subtracted last in the Mega Boolean.

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
  - Each arch has a mobile block. Associating other objects with an arch (CBCT teeth, bone segment,
    nerve: "moves with the upper / lower arch") is kept in the code but removed from this panel: it
    belongs to a future tab, "Mouvement chx orthognathique".
  - Removing an object from a block puts it back at its pre-op position. "Back to pre-op" resets a block.
  - Pre-op ghost: semi-transparent linked duplicates (same mesh data) frozen at the pre-op position.
  - Warnings: an arch with associated objects but no teeth registration; a moved block when trying
    to register its scan.
- Pre-positioning the upper arch (done), before seating: "Flip and place above" turns the upper
  scan teeth down (about the front-back axis, incisors stay anterior) and places it 5 mm above the
  lower arch, centred; "Adjust by hand" unlocks it for G / R. This sets the scan's starting (pre-op)
  pose, not a surgical movement: "Back to pre-op" keeps it. Computed in the lower scan's frame, so it
  also works once the lower arch is registered onto the CBCT.
- Part 2 (done): virtual seating, modelled on IPS CaseDesigner (KLS Martin) virtual occlusion.
  - The final occlusion comes entirely from virtual seating in Blender (no scan of articulated models).
  - The lower arch (and its block) is seated onto the fixed upper arch.
  - Gesture: click corresponding points on both arches, upper first (e.g. cusp tip), then lower (the
    fossa it sits in). 3 pairs or more, spread out (incisors + left and right molars). Each pair is a
    spring, drawn as a zig-zag (green < 0.6 mm, yellow < 2 mm, orange beyond) with its length listed.
  - "Seat": animated. Phase 1, the lower block glides to the best fit of the springs (works from any
    start, even a scan exported upside down). Phase 2, springs and magnet keep pulling and the teeth
    stop at the overlap limit of their zone. Last iterations on every vertex near the contacts, so a
    cusp tip is held too (within about 0.05 mm). Esc cancels and goes back. A warning says if it did
    not settle.
  - Overlap limits, two zones, set by the surgeon before seating (grinding is the surgeon's decision):
    - Anterior, canine to canine: 0 mm by default (no anterior grinding).
    - Posterior, premolars and molars: 0.2 mm by default. Less is better; usual grinding is around
      0.3-0.4 mm at most, rarely beyond 0.75 mm, on both arches.
    - The limits are real stops, not preferences: the guidance does not push past them. If springs ask
      for more, the spring stays longer instead. This includes the canine springs: the anterior limit
      wins over the Class I canine relation.
    - The number shown ignores lone spikes under 0.1 mm² (scan artefacts): the map can show a little more.
    - Zone: the two canine springs define it. Limit on the upper arch 3.5 mm behind the line of the
      canine tips (distal slope of the canine), on the lower arch the line of the canine / premolar
      embrasures. A contact is anterior if either of its two sides is. A white line shows the limit.
      Without both canine springs there is no anterior zone: the posterior limit applies to the whole
      arch, and the panel says so.
  - After any change to the occlusion, the final position stays exactly as it is, unless the surgeon
    clicks "Resume from here": the same seating without phase 1, from the current position, with
    sliding blocked (no translation across the arch or antero-posteriorly, no rotation about the
    occlusal axis: the midline and the Class stay where the surgeon put them, the arch only closes and
    tilts). Across the arch = the line of the lower canine points, else the block's X axis. Needs no
    minimum number of springs: fewer springs leave more to the occlusion and to the surgeon. Refused,
    with a message, when nothing can guide it (no active spring and the arches beyond the magnet's
    range).
  - Canine springs (right and left): the Class I canine relation matters more than the posterior
    relation. Upper canine cusp tip, then the lower canine / first premolar embrasure where it should
    rest. Each canine spring pulls "canine weight" times harder than another spring (3 by default),
    so the anterior contact is right antero-posteriorly, even if the posterior springs are then less
    well satisfied, within the anterior overlap limit (the limit wins). Drawn thicker; always listed first; redoing one replaces it.
  - Midline spring: one named spring, upper dental midline to lower dental midline, always listed after
    the canine springs. Ordinary weight (not the canine weight); redoing it replaces it.
  - "Springs" button at the top of the seating panel: one click hides (or shows) the points, the springs
    and the white lines that close the anterior zone, to see the teeth and the overlap map unobstructed.
    Display only: the springs stay active for the seating. Showing the overlap map (by hand or at the
    end of an animated seating) hides them automatically; hiding the map shows them again.
  - Magnet ("Seat the occlusion"): like pushing the two models together, it closes the gaps between
    the arches (within 3 mm) so stable contacts form, up to the overlap limits. Springs and magnet share the guidance: springs 70 % by default ("Springs' share"),
    magnet 30 %. The springs bring the key positions roughly; the occlusion itself finishes the seating.
  - Segmental surgery of the upper arch (optional, e.g. transverse expansion of the posterior segments
    for a better intercuspation), before the springs:
    - "Draw a line": click-drag an osteotomy line across the upper arch (occlusal view), like the guide's
      plane cut; several in a row. A line cuts the full height (along the arch's occlusal axis) but only
      along its own length + 1 mm, so an interdental line does not cross the other side: draw past the
      border of the scan, and make the lines that close a segment meet (e.g. two interdental lines and a
      palatal line joining them). The cut shows live as orange dots on the scan.
    - "Adjust the lines": press near an end and drag it; it slides on the scan (or past its border), and
      the cut follows live, to put it exactly in the embrasure.
    - "Separate the segments": each closed region becomes a segment (pieces under 5 mm² ignored), named
      anterior / right / left posterior from the midline, each with its own block. The scan is never
      modified: segments hold copies of its faces; it is hidden. Upper spring points move to the segment
      they sit on.
    - Seating (surgeon's choice): the lower arch seats first on the reference segment (the anterior one
      by default: Class and midline), with that segment's springs only (3 or more). Then the lower is
      fixed and each other segment seats on it, independently (springs on that segment, magnet, overlap
      limits), free to slide: an expansion is a slide across the arch. Point fit only for a segment with
      3 spread springs or more. The overlap map covers every segment.
    - Panel: each segment with its expansion (mm its centre moved away from the midline), select (G / R)
      and back to start. "Join the arch" deletes the segments and their movements (confirmation if
      moved), shows the scan again, and keeps the lines to adjust and separate again.
    - Not yet: overlap between neighbouring segments is not checked; the CBCT maxilla is not cut (later,
      with the Le Fort I tools).
  - Contacts may be as large as the dental anatomy allows: more contact area is better, no upper limit.
  - Stability: "stable tripod" when there is contact on the right, anterior and left (>= 1 mm² each).
    Anterior = the canine to canine zone above, so a premolar touching the distal slope of the upper
    canine counts as anterior; without both canine springs, a band of 15 mm each side of the midline.
  - Then fine-tune by hand (adjust the lower arch, G / R); the map refreshes when the movement stops.
  - What I need to see: overlap map on both arches, in bands: magenta > 1.0 mm, red 0.75-1.0, orange
    0.4-0.75, yellow 0-0.4 overlap; green contact (gap < 0.3 mm); light blue gap 0.3-1.0; grey > 1.0.
    Legend in the panel. Deepest overlap anterior and posterior in mm, each against its limit (warn
    > 0.4 mm "beyond usual grinding", alert > 0.75 mm "major grinding"), and contact area right /
    anterior / left (mm²).
- Next (to specify): movement of the maxillomandibular complex relative to the skull (Step 5), and
  intermediate / final splints.
- Future tab "Mouvement chx orthognathique", idea noted by the surgeon: control of the rotation point
  (pivot) of the maxillomandibular segment.
  - The pivot decides what stays put during a rotation: e.g. correct the pitch from the posterior or
    from the anterior (and vice versa).
  - A movable pivot with a gizmo in the viewport, repositioned at any time while working, before the
    transformation is locked. Example sequence: shift the maxilla laterally; put the pivot on the
    midline and work the yaw there; once the lateral centring is right, move the pivot posteriorly
    to control the pitch (superior / inferior) of the posterior maxilla.
  - Moving the pivot never moves the segment; the numbers stay expressed in the reference frame.
  - Ideas to confirm: snap the pivot to landmarks (upper incisor midpoint, PNS, molars, condyles),
    show the pivot's position in clinical terms.
- Troubleshooting:
  - Open scan meshes: faces on the open border are ignored by the contact test (the inside/outside
    sign means nothing there). Case1 scans checked: normals point outward (99.8 %).
  - Upper and lower scans may each come in the scanner's own frame (Case1: both sit on z = 0, one
    inside the other). The point-fit phase handles any start; register the upper arch first anyway,
    otherwise the lower arch (and its bone) is seated in scanner coordinates (warning in the panel).
  - Points nearly on one line are refused (rotation about that line undetermined).
  - Overlap > 3 mm usually means inverted normals or a scan border, not real overlap (hint shown).
  - Lone spiky vertices: max overlap is the depth reached over at least 0.1 mm², not a single vertex.
  - Points whose closest point is on an edge of the other mesh count as outside (the face normal
    says nothing about the sign there). Case1 had one such vertex reading 3.8 mm.
  - Orthodontic brackets and wires create false contacts; gingiva can overlap.
- Nuances: (add here)

## Object creator: Splint / Chape

- Goal: a thin occlusal splint (intermediate or final) between the two arch scans in their planned
  positions, or a cap (chape) on one arch. Example the surgeon showed: "Teeth impression: up to
  brackets", "Buccal contour: thin".
- Holes where the teeth touch are accepted (surgeon, 2026-10-31): no need to open the bite.
- Method (chosen over extracting tooth zones, which follows undercuts and makes one shell per arch):
  a blank swept along an arch curve, then the tooth imprints by projection along the insertion axis
  (no undercuts by construction, open scans are fine), 0.10 mm clearance.
- Steps:
  1. Arch curve *(done)*: type (Intermédiaire / Finale / Chape) and arches (default: those of the
     occlusion seating). "Cliquer les points": 5 to 7 clicks on the occlusal surfaces, from the last
     molar on one side to the other through the incisors; Backspace removes the last one. For a
     splint, each click is moved halfway between the lower and upper occlusal surfaces on the vertical
     (reference frame up), so the curve runs through the middle of the bite. Bezier curve with
     automatic handles: passes through every point. Drawn thick in orange, points numbered, on top.
     "Modifier": Edit Mode on the curve with snapping onto the surfaces (G, X, Tab).
  2. Parametric blank *(done)*: a closed slab along the curve, ending with it (no distal extension:
     surgeon, 2026-10-31; where the curve stops, the splint stops). Width measured on the scans every
     1 mm of curve (both arches together), only in the crown band the splint covers (no palate, no
     gingiva): the most lingual and most buccal crown points, smoothed over 3 mm, plus the margins
     (surgeon's defaults: 1.5 mm lingual, 1.5 mm buccal, separate so a segmental splint can get more
     buccal only). Height: depth above and below the middle of the bite (splint: 3 mm each, the "up
     to brackets" setting); a cap sits on the occlusal curve, 3 mm above and 0.8 mm down onto the
     teeth by default. Top and bottom are fine grids (0.1 mm by default, down to 0.05 mm: the scans have ~0.15 mm
     between vertices, so finer copies nothing more; case1 cap: 0.3 mm 0.05 s, 0.1 mm 0.2 s, 0.05 mm 0.8 s) in vertex groups Top / Bottom, for the
     projection. Margins update live; a changed curve or arch shows "mettre à jour". Case1 lower
     arch: 0.04 s.
  3. Tooth imprints by projection *(done)*: "Empreinte des dents" presses the top face onto the
     upper arch and the bottom face onto the lower arch (a cap: only the face on its arch; a cap on
     the upper arch hangs under it) along the insertion axis (reference frame up). Each vertex goes
     to the first tooth surface met coming from the other arch, so where a crown bulges the splint
     stops at its widest point: undercuts blocked out, the splint goes on and off. Where no tooth is
     met within the depth (margins), the face stays flat. Clearance 0.10 mm in every direction: a
     ball rolled over the tooth height field (dilation), sideways by max(clearance, resolution) so
     the step off a vertical wall lands outside the tooth. (A first version lifted vertices on
     steep walls by clearance / cos(slope): spikes, and faces cutting the incisors' labial corner;
     surgeon spotted it on case1, 2026-11-01.) Where the arches leave no
     room, top and bottom meet (zero thickness, shown as "Contacts (trous)" in mm2): real holes cut
     at step 4. Panel shows the thinnest point outside the contacts. Live with the other settings;
     case1 lower cap: 0.08 s, thinnest 0.73 mm over the highest cusps.
  4. Solid object *(done)*: "Solidifier en un objet" freezes the blank into its own object
     ("Chape (solide)", "Finale (solide)", ...): the settings no longer change it, so it can be
     sculpted, polished, engraved, exported. Where top and bottom met (contacts) the faces are removed
     and the top / bottom vertices welded: real holes, the mesh stays closed. Text already put on the
     blank comes along (its Boolean modifiers are copied). The blank stays, hidden ("Montrer le bloc
     paramétrique"); each solidify makes a new object, older ones are never deleted. Panel lists the
     solids, closed or with how many open edges. Case1 cap at 0.05 mm: 0.4 s, closed.
  5. Text and checks *(done)*, on the blank or on a solid (the active one, else the latest solid).
     Placement chosen over painting a region or drawing a curve (proposed to the surgeon,
     2026-11-01): two clicks on the surface, where the text starts and where it ends, give place,
     reading direction and length in one gesture; the letters fill the space between the clicks up
     to a maximum height (3 mm) and read from the outside, left to right from the first click.
     Engraved (default, 0.5 mm) or embossed. The letters are laid on the surface along the line of
     sight (they follow its curve) and made into a closed cutter shown as a wireframe; a Boolean
     modifier (Manifold solver) subtracts or adds it, so the splint mesh is untouched until export.
     Select a text's wireframe to change its words, mode, height or depth live; trash to remove.
     Text on the blank follows every rebuild and comes along when solidifying. Warnings: text
     running off the surface, letters under 2 mm. "Vérifier": closed mesh with the text applied,
     volume, thinnest point, material left under each engraving (>= 0.5 mm). "Exporter STL": the
     text applied, original patient coordinates, 1 unit = 1 mm. Case1 cap: placing 1.2 s,
     Boolean 0.6 s.
- Nuances: (add here)

## Model base: "Création de modèle (base)"

Surgeon's request (2026-11-03): reproduce, automated, what was done by hand on case1: open arch scans
closed into printable dental models. Most important: the teeth (the dental segment) stay untouched.

- What it does (model_base.py), on a copy, the scan itself is only read:
  - no vertex of the scan ever moves: faces are only removed (loose scraps, the ragged fringe along
    the open border, whatever lies beyond the base plane) and added (small hole patches, wall, base);
  - teeth zone, never trimmed: every vertex less than 8 mm (MODEL_TEETH_BAND) below the cusp tips
    within 6 mm around it (seen from above); only points within 6 mm of the highest one count as
    cusp tips, so the palate or the vestibule far from the teeth is not protected by itself;
  - if trimming the fringe would cut a protected part off (it would be thrown away as a scrap), the
    fringe joining it is put back; at a bow-tie (two runs of border touching at one vertex) the border
    is followed by turning around the vertex, nothing is removed or split;
  - a wall drops from the open border along the occlusal axis to a flat base; only the base outline
    is smoothed (its own new vertices).
- Occlusal axis: the reference frame's if there is one, else the direction in which the arch is
  thinnest, pointing away from the open border; "Inverser l'axe" if wrong. (case1's scans: Y.)
- Two kinds, detected (is the middle of the teeth zone covered by the scan, seen from above?):
  - full palate (upper arch): the whole palate is kept, the base goes 3 mm under its deepest point
    (MODEL_UNDER_PALATE) whatever the height asked; full flat plate;
  - horseshoe (lower arch): base at "Hauteur du modèle" under the cusp tips (default 20 mm; case1 by
    hand: 15.9) and a bar across the opening (Ø 4 mm, at the base, Boolean union Manifold) so the
    model does not flex. It runs parallel to the opening (the longest edge of the outline's convex
    hull), just inside it, and each end sits in the middle of one branch, where that line crosses it
    (moved further in until it crosses both): sunk into the solid on both sides (surgeon, 2026-11-03:
    "la barre doit relier les deux côtés"; the first version only touched one side on case1). Exact
    Boolean solver (Manifold cut off a part pinched at one vertex). The report checks both ends are
    inside the model and the result is one piece; case1: heights 15.9 / 20 / 25 mm x bars 3 / 4 / 6 mm,
    all anchored, one piece.
- Report in the panel: palate / horseshoe, scan vertices kept (identical), teeth zone and how many of
  its vertices were removed (must be 0), plus those at the very edge of the open border or in scraps
  already loose in the scan, holes patched, base outline crossings, closed (watertight).
- Case1: lower 0.5 s (66,415 vertices kept, 0 removed in the teeth zone, 13 in a scrap already loose
  in the scan), upper 0.9 s (117,398 kept, palate kept whole, base at 28.2 mm), both closed.
- Clean solids for Booleans (found adding pins, 2026-11-03): case1's upper model vanished under an
  Exact Boolean (only the pin parts were left) because its base outline crossed itself once, and a
  sliver of gingiva hanging by one vertex (bow-tie) was cut off by Manifold. Now the base outline is
  smoothed further until it no longer crosses itself (case1: +10 and +140 passes), and parts hanging
  by one vertex smaller than 300 vertices are removed (case1: 45 and 20 vertices, reported with the
  border slivers, never counted as teeth). Both models are then fully manifold, and the Manifold solver
  is used for the bar and the model pins (Exact left 252 non-manifold edges on the lower model).
- Repositioning pins between the models (surgeon, 2026-11-03): the same pins as the cutting guide
  (pins.py, kind 'MODEL'), so the two printed casts find the planned occlusion by plugging together.
  - Female: click on the lower model, then move the mouse: the pin slides on the lower model's base
    plane (the header shows its distance to the model), click to confirm. Its boss stands flat on
    the base plane (prints on the build plate), its axis along the occlusal axis, opening up.
  - Stem: a cylinder (Ø 5 mm, PIN_STEM_DIAMETER) flat on the base plane, from the boss to the closest
    point of the model, 1.5 mm into it.
  - Arm: one click on the upper model per pin, exactly as from the cap: leaves at 90 degrees, arrives
    along the pin axis; the male is unioned to the upper model.
  - Starting a pin tool ends the one still running (case1: the females tool, still waiting for a
    right-click, kept asking for the lower model when the surgeon pressed the arms button).
  - Case1 (separate Blender, copy of the case, 3 pins): 0.4 s; both models one piece, manifold; arms
    never touch the lower model; the only contact is the male's shoulder on the boss top (the stop):
    lifted 0.02 mm, nothing touches.
- Bite check (surgeon, 2026-11-03): the third STL ("bite" in its name) is a picture of the side of
  the intercuspation: a strip of upper teeth and a strip of lower teeth captured together. Its
  shape, not where the object sits, says whether the arches are in the right occlusion. The arches
  are never moved; the bite object is not moved either, its pose is only computed:
  1. global search: start poses at 60 points spread over both arches, 8 orientations each (bite's
     long axis along the arch, its height or thickness along the occlusal axis, both ways), short
     trimmed ICP, best 6 refined: the pose where the bite fits both arches at once;
  2. its upper part (points on the upper arch at that pose) fitted alone on the upper arch, its lower
     part alone on the lower arch;
  3. in the right occlusion both poses are the same: their gap (mm at the bite, degrees) is the error.
  Green check "L'occlusion correspond au STL Bite" when the common fit holds 60 % of the bite within
  0.3 mm, each arch holds 10 %, and the gap is under 0.5 mm and 1.5 degrees; else a red X. A wireframe
  ghost "(recalé, contrôle)" shows where the bite belongs (same mesh data). About 7 s.
  - Fitting the bite on one arch alone is ambiguous (only half of it matches: 23 % at best, poses
    170 degrees apart); on both arches the right pose stands out (case1: 78 %, next best 29 %).
  - "One rigid bite on both arches" residuals alone were too insensitive (3 mm shift: +0.08 mm).
  - Case1 as is: 78 %, gap 0.20 mm / 0.6 degrees: green. Lower arch moved on purpose (separate
    Blender, copy of the case): 0.2 mm -> 0.28 mm, 0.5 deg -> 0.39 mm / 1.3 deg, 1 mm front-back ->
    0.96 mm, 2 deg -> 0.94 mm / 2.8 deg, 3 mm -> 2.9 mm (46 %); 0.5-1 mm in the other axes: no common
    fit (30 %). Detection limit about 0.5 mm / 1 degree; thresholds TO CONFIRM on more cases.

### Occlusal axis without a reference frame

Surgeon, 2026-11-03: a final splint built without a reference frame "did not work at all": up was
assumed to be the file's Z, and case1's scans have their occlusal axis along Y, so the blank stood on
its side. Now, without a frame, up comes from the arches (splint.arch_up): the direction across which
each arch is thinnest, averaged over both, pointing from the lower arch to the upper; for a cap, the
arch's own axis pointing at its teeth (away from a scan's open border, or from a closed model's flat
base). The reference frame still wins when there is one. Case1: up found 0.99 along Y; blank and both
imprints right (1.2 s).

## Cross-cutting tool: Text (engraved / embossed)

Surgeon, 2026-11-03: one text tool for every printed part (model, splint, cap, guide, bone
reconstruction...), instead of a text step inside the splint panel only. Panel "Texte (gravure /
relief)" (text_labels.py), also drawn as step 5 of the splint panel:
- works on the active object (or the active text's part, or a part chosen in the panel); two clicks
  on its surface, letters fit between them; engraved or embossed, height and depth live;
- never on raw anatomy (roles bone, teeth, nerve, skin, scan, segment: CBCT and scanner data are
  never modified), nor on a locked object: write on a model, a guide or a working copy instead;
- the texts on the part are listed (click to edit, trash to remove); a text follows its part when it
  is rebuilt (splint blank, dental model made again);
- generic print check (closed mesh with every modifier applied: text, pins, tubes, bar...; material
  left under each engraving >= 0.5 mm; letters >= 2 mm) and STL export with every modifier applied;
- case1 (separate Blender): "SC BAS" engraved under the lower model's base (6.6 mm left under the
  letters, closed), "SC HAUT" embossed on the upper model; a text running off the surface is warned.

## Cross-cutting: used tools put away

Surgeon, 2026-11-03: the helper objects (pin parts, texts, cylinders, bar, cut-edge box...) are of no
use once their modifiers are applied; they should gather in one folder by themselves. tidy.py: any
object that has been a Boolean operand is marked; when no modifier uses it any more (applied with
the OMFS buttons or Blender's own Ctrl+A, or removed), it moves to a hidden "Outils utilisés"
collection. Never deleted: if a modifier uses it again (Ctrl+Z, a new Boolean) it goes back to the
collection it came from. Anatomy, models, guides, splints and planning objects never move, nor an
object that has not been used yet. A timer checks once a second, in Object Mode only.
Planning aids too (surgeon, same day): the reference planes while hidden (back when shown; the
frame itself never moves), the arch curve once a splint is solidified and its blank hidden (back when
the blank is shown again to change it).

## Cross-cutting tool: Section & Analysis

Not a step: a panel usable at every planning step to inspect how objects (bone, teeth, scan,
moved segments, guide, screws, sleeves) interact, like scrolling through CT slices but for
meshes. Everything is non-destructive (no original mesh is ever modified) and must stay
responsive with CT meshes of several hundred thousand triangles.

1. Section plane *(done)*: axial / coronal / sagittal, in the reference frame when built (scanner
   axes otherwise), or the local XY plane of any object. Position slider along the normal
   (+ = up / anterior / left), thickness 0.2 to 10 mm for a slab centred on the plane, "Slab Only"
   hides everything outside the slab in every 3D viewport (Blender's clipping region, GPU-side,
   instant on large meshes), "Align View" looks perpendicular to the plane in orthographic, like a
   radiograph (axial from the feet, coronal from the front, sagittal from the patient's right;
   "Other Side" flips). Hook: a new plane (e.g. a user-defined occlusal plane) is a new orientation
   in `section.base_matrix`.
2. Section contours *(done)*: outline of every visible object on the plane, in its colour, drawn
   on top (always readable), live with the slider (about 2 ms per slider step for a
   900 000-triangle mandible: vertices and triangles are read once and cached). "Caps" fills the
   cut of closed objects, row by row every 0.05 mm with the even-odd rule (holes such as sinuses,
   canals and self-intersecting segmentations are handled); visible with Slab Only, since outside
   the slab the object covers its own cap. Open meshes (scans) give an outline only. A master
   checkbox in the panel header turns the whole tool on / off.
3. Overlap highlight *(done)*: red where two visible closed objects interpenetrate on the section
   (the two caps intersected row by row), with the area per pair in the panel (below 0.05 mm²
   ignored: sampling noise). Separate 3D tool: the exact common volume of two picked objects, in
   red, seen through the bone, with its volume in mm³. It is an overlay object (role Analysis) with
   a Boolean Intersect modifier; A and B are never modified, "Remove Intersection" deletes it.
   - Both meshes closed: Manifold solver, live (0.1 s), follows A and B when they move.
   - A few open edges (common in segmentations; Case1 mandible: 3 of 1.4 million): the Manifold
     solver refuses them, the Exact solver takes about 5 s on a CT mandible, too slow to follow a
     drag. So it is computed once (snapshot); "Moved: Refresh" appears when A or B has moved.
     The result is then not closed either: volume shown as approximate.
   - Open meshes (scans): no inside, refused.
   - Segmentations overlap where they touch (Case1: teeth roots inside the mandible, 1325 mm³):
     read the volume relative to what the two objects are.
4. Distance map *(done)*: source surface coloured by distance to a target (defaults red < 1 mm,
   yellow < 2 mm, green beyond; editable live), legend, minimum distance and a marker (line +
   label) between the closest points. Signed: negative = the source enters the target ("inside").
   A Geometry Nodes modifier on the source (attributes on the evaluated mesh only; one click
   removes it), live when either object moves (0.14 s for 100 000 vertices, 0.4 s for a whole CT
   mandible). The minimum is over the source vertices: on a coarse source it can be slightly
   larger than the true minimum. Display uses Solid shading "Attribute", so the other objects
   show white while the map is on (a viewport-wide setting). Generic: later used for
   screw-to-root clearance (target margin about 2 mm). Case1: a lower root at 0.86 mm from the
   canal segmentation.
5. Snapshot *(done)*: PNG of the section view with plane name, position, thickness and a scale
   bar, to a folder outside the repo (documentation, thesis). The caption and scale bar are also
   drawn live in the viewport corner ("Caption"); the scale bar needs an orthographic view
   (Align View), a perspective view has no single scale. The PNG is the viewport redrawn
   offscreen at its on-screen size, without Blender's interface, grid or 3D cursor. A folder
   inside the git repository is refused. File name:
   `section_<plane>_<position>mm_<thickness>mm_<date-time>.png`.

## Open questions

- Superimposition: confirm the 0.5 mm flag threshold with a supervisor or the program's usual practice.
- Screw holes: default hole axis (perpendicular to the bone surface?).
- Drill sleeve 03.503.045: measure the exact outer diameter, then set the fit clearance after test prints.
- Osteotomy: safety margins below the mental foramen and root apices.
- Bicortical screws: how many (2? 3?), minimum distance to nerve canal and roots.
