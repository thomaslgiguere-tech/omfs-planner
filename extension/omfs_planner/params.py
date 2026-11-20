"""Clinical and hardware defaults, in one place. Lengths in mm, angles in degrees.

Each default can be changed in the panel for a given case. Values marked TO CONFIRM must be
checked with a supervisor before clinical use.
"""

# Registration of the intraoral scan onto the CBCT teeth
REG_CROWN_BONE_CLEARANCE = 1.5  # mm: CBCT teeth surface this far outside the bone counts as crown
REG_SUGGEST_DISTANCE = 1.0      # mm: scan vertices this close to the CBCT crowns are suggested
REG_TARGET_ERROR = 0.3          # mm, working target for mean / RMS on crowns (literature ~0.3 mm)
REG_WARN_ERROR = 0.5            # mm, warn above (self-chosen safety margin), TO CONFIRM
ICP_SAMPLES = 5000              # crown vertices used per ICP iteration
ICP_MAX_ITERATIONS = 60
ICP_MAX_PAIR_DISTANCE = 2.0     # mm: farther closest points are ignored (gingiva, artefacts)

# Osteotomy (genioplasty, version 1)
CUT_BELOW_FORAMINA = 5.0      # mm, cut below the midpoint of the mental foramina
CUT_PROFILE_ANGLE = 0.0       # deg relative to Frankfort, + = cut rises anteriorly
CUT_FRONTAL_ANGLE = 0.0       # deg, + = patient's left side higher
CUT_POSTERIOR_EXTENT = 30.0   # mm behind each mental foramen (surgeon 2026-11-18: far enough to cover the whole chin)

# Safety distances from the cut (warn below)
NERVE_SAFETY_MARGIN = 4.0     # mm to the inferior alveolar nerve canal, TO CONFIRM
ROOT_SAFETY_MARGIN = 3.0      # mm to the tooth roots, TO CONFIRM

# Visual landmark only, never a constraint
STAR_BELOW_FORAMEN = 5.0      # mm, yellow star below each mental foramen

# Guide design (workshop defaults)
GUIDE_THICKNESS = 3.0           # mm, guide body away from the saw path
GUIDE_CLEARANCE = 0.10          # mm gap between bone and guide (fit)
SELECT_FLAT_ANGLE = 40.0        # deg, "linked flat faces": stops at the rim of the guide
DENSIFY_CUTS = 1                # cuts per edge of "Densifier la sélection" (1 = each triangle into 4)
GUIDE_REMESH_VOXEL = 0.2        # mm, voxel size when remeshing a guide watertight
TRIM_PLANE_SIZE = 60.0          # mm, side of the plane cut's handle (the cut itself is unbounded)
TRIM_BOX_SIZE = 400.0           # mm, half-space box behind the plane cut (bigger than any jaw)
ERASE_FILL_EDGE_FACTOR = 1.5    # Erase & Fill: patch edges longer than this x the rim's mean edge are split
ERASE_FILL_REFINE_STEPS = 6     # Erase & Fill: rounds of splitting (each halves the long edges)
ERASE_FILL_SMOOTH_ITERATIONS = 200   # Erase & Fill: membrane smoothing of the patch (its rim never moves)
GUIDE_PROTECT_DISTANCE = 1.0    # mm: guide vertices closer than this to the bone are the intrados (protected when polishing)
SCULPT_SHARP_ANGLE = 20.0       # deg, "zones by sharp angle": faces folded more than this start a new zone (case1 cap: 20 splits intrados / outside, 30 merges them)
SLEEVE_OUTER_DIAMETER = 4.0     # mm, MatrixMANDIBLE sleeve 03.503.045: just under 4 mm, TO MEASURE
SLEEVE_FIT_CLEARANCE = 0.10     # mm added to the sleeve diameter for the tube bore, TO TEST (print)
SLEEVE_TUBE_LENGTH = 5.0        # mm
SLEEVE_TUBE_DIAMETER = 7.5      # mm, outer diameter of the drill-sleeve tube (wall = (7.5 - bore) / 2)
SLEEVE_BORE_OVERSHOOT = 1.0     # mm the sleeve bore reaches past each end of its tube (opens it cleanly)
FIXATION_HOLE_DIAMETER = 2.0    # mm, hole holding the guide to the bone: Synthes 1.85 mm screw + clearance, TO TEST (print)
FIXATION_HOLE_OVERSHOOT = 1.0   # mm the fixation hole reaches past the guide on both sides
IRRIGATION_DIAMETER = 2.0       # mm, irrigation hole: water reaches the drill through the guide, TO TEST (print)
IRRIGATION_OVERSHOOT = 1.0      # mm the irrigation hole reaches past each clicked end (opens cleanly)
SCREW_DIAMETER = 1.85           # mm, Synthes bicortical screw (the screw itself, entry to lingual exit)
SCREW_COLOUR = (0.85, 0.65, 0.15, 1.0)   # gold, like the titanium screws in the planning images
CYLINDER_MIN_SIZE = 0.5         # mm: smallest diameter or length a workshop cylinder can be dragged to

# Tubes and pins (cutting guide <-> cap): conical pins, axial stop on the osteotomy plane. TO TEST (print)
PIN_BASE_DIAMETER = 4.0         # mm: cone diameter at the osteotomy plane
PIN_TIP_DIAMETER = 3.0          # mm: cone diameter at the tip
PIN_CONE_LENGTH = 5.0           # mm the cone goes into the female, below the plane
PIN_CLEARANCE = 0.25            # mm radial between pin and bore (0.10 printed too tight, surgeon 2026-11-07), TO TEST (print)
PIN_TIP_GAP = 0.3               # mm left under the tip: the shoulder on the plane is the only stop
PIN_WALL = 2.0                  # mm of boss around the bore
PIN_FLOOR = 1.5                 # mm of boss under the bore
PIN_EMBED = 1.0                 # mm the boss sinks into the guide surface where clicked
PIN_SHOULDER_DIAMETER = 6.0     # mm: the pin above the plane, resting on the boss top
PIN_BACK_LENGTH = 2.0           # mm of pin above the plane, up to the arm
PIN_MALE_CHANNEL = 15.0         # mm: where a male sits in the guide's body, the guide is hollowed this far
                                # above the plane along the pin axis (the male goes in from there)
PIN_ARM_DIAMETER = 4.0          # mm: the tube from the cap to the pin's back
BONE_REST_THICKNESS = 4.0       # mm, bone rest (appui osseux): block length along the arm (its width = the arm's Ø)
BONE_REST_GAP = 0.0             # mm left between the bone rest and the bone (0 = exact contact), TO TEST (print)
BONE_REST_CUT_MARGIN = 1.0      # mm: the rest's whole footprint stays this far above the osteotomy plane (saw kerf)
BONE_REST_GRID = 0.25           # mm between the rays that fit the rest's foot to the bone surface

# Occlusion seating (virtual, lower arch onto the fixed upper arch)
OCC_TRIM_BAND = 1.0              # mm removed along the open border of each arch scan (first step of the seating)
OCC_TRIM_SPIKE_ANGLE = 60.0      # degrees: a border tip sharper than this is a spike, removed
OCC_TRIM_MIN_ISLAND = 2.0        # %: loose pieces smaller than this share of the arch area are removed
OCC_TOLERATED_ANTERIOR = 0.0     # mm: overlap limit canine to canine (grinding there is avoided)
OCC_TOLERATED_POSTERIOR = 0.2    # mm: overlap limit on the premolars and molars (less is better)
OCC_CANINE_ZONE_MARGIN = 3.5     # mm behind the upper canine tip: the distal slope of the canine stays anterior
OCC_LIMIT_BAND = 0.5             # mm: points this close to their overlap limit are held by the solver
OCC_LIMIT_STIFFNESS = 1000.0     # how hard a held point resists, vs springs + magnet (1.0)
OCC_LIMIT_MARGIN = 0.05          # mm: overlap this close to its limit still reads as within it (sampling)
OCC_LIMIT_ROUNDS = 30            # rounds per iteration to catch every point that would pass its limit
OCC_CONTACT = 0.3               # mm: closer than this counts as an occlusal contact (green)
OCC_OVERLAP_STEPS = (0.4, 0.75, 1.0)  # mm: yellow / orange / red bands, magenta beyond the last
OCC_OVERLAP_RARE = 0.4          # mm: usual grinding stays around 0.3 - 0.4 mm: warn above
OCC_OVERLAP_MAX = 0.75          # mm: grinding rarely goes beyond this: alert above
OCC_GAP_FAR = 1.0               # mm: gaps beyond this are grey (no contact)
OCC_OVERLAP_MIN_AREA = 0.1      # mm²: max overlap is the depth reached over at least this area (ignores lone spikes)
OCC_SEARCH_DISTANCE = 4.0       # mm: farther surfaces are ignored by the contact search (= depth of the distance grid)
OCC_GRID_CELL = 0.75            # mm: cell of the distance grid built once per arch (read in one go, not point by point)
OCC_EXACT_BAND = 0.75           # mm: closer than this to the other arch, a distance is recomputed exactly (contacts, limits)
OCC_SAMPLES = 4000              # vertices per arch used by the spring solver
OCC_CANINE_WEIGHT = 3.0         # a canine spring pulls this many times harder than another spring
OCC_SPRING_SHARE = 70.0         # %: springs' share of the guidance; the rest is the magnet (closing the gaps)
OCC_SPRING_RELEASE = 3.0        # mm: a spring shorter than this pulls less and less, its share going to the magnet (= magnet range)
                                # (springs bring the arches "macro", the intercuspation finishes it; surgeon 2026-11-09)
OCC_SPRING_RESIDUAL = 0.1       # share of its pull a released spring keeps at length 0 (so nothing floats away)
OCC_MAGNET_RANGE = 3.0          # mm: gaps wider than this are not pulled by the magnet
OCC_TRIPOD_MIN_AREA = 1.0       # mm²: contact area right, anterior and left for a stable "tripod"
OCC_MAX_ITERATIONS = 100        # full steps (each aims at the equilibrium, shortened only if it does not improve)
OCC_MAX_STEP = 2.0              # mm and degrees: longest step, so the linearised contacts stay valid
OCC_LINE_SEARCH = 4             # times a step may be halved before the seating is considered settled
OCC_SETTLED = 0.1               # mm and degrees per iteration: below this the position is stable
OCC_FINISH_ITERATIONS = 30      # last iterations, on every vertex near the contacts
OCC_FINISH_BAND = 1.0           # mm: at the end, every vertex this close to the other arch is also held at its overlap limit
OCC_FINISH_SPACING = 0.2        # mm: one vertex per cube this size is enough for the overlap limits
OCC_PLAYBACK_FPS = 60           # the seating is computed first, then played back at this frame rate
OCC_PLAYBACK_MM = 0.1           # mm per frame (and OCC_PLAYBACK_DEG degrees): even speed of the playback
OCC_PLAYBACK_DEG = 0.2
OCC_ANTERIOR_HALF_WIDTH = 15.0  # mm each side of the midline: anterior contacts, beyond = left / right
OCC_PREPOSITION_GAP = 5.0       # mm between the arches after "flip and place above the lower arch"

# Object creator: Splint / Chape
SPLINT_BUCCAL_MARGIN = 1.5      # mm beyond the most buccal crown surface (more for a segmental splint)
SPLINT_LINGUAL_MARGIN = 1.5     # mm beyond the most lingual crown surface
SPLINT_DEPTH = 3.0              # mm the splint reaches onto the teeth, above and below the middle of the bite
SPLINT_CAP_DEPTH = 0.8          # mm a cap (chape) reaches down onto the teeth from the occlusal curve
SPLINT_RESOLUTION = 0.1         # mm between the blank's vertices (the imprint is only as fine as this; scans: ~0.15)
SPLINT_MEASURE_STEP = 1.0       # mm between the curve stations where the arch width is measured
SPLINT_SMOOTH = 3.0             # mm: the outline is smoothed over this length along the arch
SPLINT_BUCCAL_CLEARANCE = 0.9   # mm, splint (not the cap): beyond the most buccal crown surface (surgeon 2026-11-13; more for a segmental splint)
SPLINT_LINGUAL_CLEARANCE = 3.7  # mm, splint: beyond the most lingual crown surface (surgeon 2026-11-13)
SPLINT_OUTLINE_SMOOTH = 9.0     # mm, splint: the outline is an envelope smoothed over this length (round like the companies')
SPLINT_SURFACE_SMOOTH = 6.0     # mm, splint: the top / bottom surfaces through the coverage points are smoothed over this length
SPLINT_ARCH_BIN = 2.0           # deg: the automatic arch centre line is measured in sectors this wide, seen from the arch's middle
SPLINT_ARCH_SMOOTH = 12.0       # deg: and smoothed over this angle
SPLINT_CLEARANCE = 0.10         # mm between the imprint and the teeth, in every direction (walls too)
SPLINT_END_BAND = 1.0           # mm: the splint ends at the last crown point this close to the bite (not the gingiva behind the molars)
SPLINT_END_MARGIN = 0.0         # mm the splint's end goes past the last tooth of the shorter arch (flush)
SPLINT_RIM_MIN = 0.1            # mm kept on the outline where the teeth meet: a contact hole never cuts the wall (closed mesh)
SPLINT_EDGE_ROUND = 1.0         # mm radius of the cap's rounded rim (top and bottom edges of the outline); 0 = sharp
SPLINT_SPLINT_EDGE_ROUND = 0.4  # mm, the splint's rounded rim (surgeon 2026-11-13)
SPLINT_WIRE_HOLE_DIAMETER = 1.5 # mm: holes for the osteosynthesis wire, through the buccal flange
SPLINT_WIRE_HOLE_TEETH_GAP = 0.5  # mm of material at least between a wire hole and the teeth (either arch)
SPLINT_WIRE_HOLE_WALL = 0.5     # mm of material at least between a wire hole and the splint's outer edge
SPLINT_WIRE_HOLE_OVERSHOOT = 1.0  # mm the hole cylinder sticks out above and below the splint

# Section & Analysis (cross-cutting viewing tool)
SECTION_THICKNESS = 2.0         # mm, default slab thickness
SECTION_THICKNESS_MIN = 0.2     # mm
SECTION_THICKNESS_MAX = 10.0    # mm
SECTION_PLANE_SIZE = 140.0      # mm, smallest square drawn for the plane
SECTION_LINE_WIDTH = 3.0        # px, contour lines
SECTION_FILL_STEP = 0.05        # mm between the rows that fill caps and measure overlap areas
SECTION_OVERLAP_MIN_AREA = 0.05 # mm²: smaller overlaps on the section are noise (row sampling)
ANALYSIS_MAX_OPEN_EDGES = 0.001 # fraction of edges: up to this many open edges, a mesh is "nearly closed"
DIST_RED = 1.0                  # mm: distance map red below this (and inside the target)
DIST_YELLOW = 2.0               # mm: yellow below this, green beyond (screw to root: target margin ~2 mm)

# Text labels (splint, later guides) and print checks
TEXT_HEIGHT = 3.0               # mm: maximum letter height (letters fit between the two clicks)
TEXT_DEPTH = 0.5                # mm engraved into the part, or embossed above it
TEXT_OVERLAP = 0.3              # mm the cutter reaches past the surface, for a clean Boolean
TEXT_MIN_HEIGHT = 2.0           # mm: smaller letters may not read once printed
PRINT_MIN_WALL = 0.5            # mm of material that must stay under an engraving

# Model base ("Création de modèle", model_base.py): an open arch scan closed into a printable model
MODEL_HEIGHT = 20.0             # mm from the highest cusp tip to the flat base (case1 by hand: lower 15.9, upper 34.4 with the palate)
MODEL_MIN_WALL = 1.0            # mm: the scan is cut at least this far above the base, so a wall always remains
MODEL_FRINGE = 1.0              # mm of ragged border removed along the scan's open edge (never in the teeth zone)
MODEL_TEETH_BAND = 8.0          # mm under the neighbouring cusp tips that is never trimmed (crowns + marginal gingiva)
MODEL_TEETH_RADIUS = 6.0        # mm, seen from above: how far to look for the neighbouring cusp tips
MODEL_OCCLUSAL_SPREAD = 6.0     # mm: points at most this far below the highest one count as cusp tips (curve of Spee, tilt)
MODEL_OUTLINE_SMOOTH = 30       # smoothing passes of the base outline (moves only the base's own vertices)
MODEL_OUTLINE_SMOOTH_MAX = 500  # extra passes allowed until the outline no longer crosses itself
MODEL_PINCH_MAX = 300           # vertices: a part hanging by one vertex and smaller than this is removed (a loose sliver)
MODEL_SCRAP_FRACTION = 0.01     # detached pieces smaller than this share of the scan are removed
MODEL_UNDER_PALATE = 3.0        # mm: with a full palate, the base goes this far under its deepest point (nothing of the palate is cut)
MODEL_BAR_DIAMETER = 8.0        # mm, the bar across a horseshoe's opening (stops it flexing; 4 was too thin, surgeon 2026-11-09)
MODEL_PALATE_MARGIN = 3.0       # mm of palatal gingiva kept beyond the teeth zone (seen from above) when the palate is left out
MODEL_BAR_INSET = 1.0           # mm the bar's line starts inside the opening (beyond its radius)
MODEL_BAR_MAX_INSET = 20.0      # mm: farther in than this, give up (the panel says there is no bar)
MODEL_REATTACH_RINGS = 30       # most face rings of fringe put back to keep a protected part attached

# Bite check (model_base.py): does the bite's shape say the arches are in the right occlusion?
# Case1 (2026-11-03, lower arch moved on purpose): none 0.20 mm / 0.6 deg, 0.2 mm -> 0.28 mm,
# 0.5 deg -> 0.39 mm / 1.3 deg, 2 deg -> 0.94 mm / 2.8 deg, 3 mm -> 2.9 mm; 0.5-1 mm in the other
# axes: no common fit (30 %). Detection limit about 0.5 mm / 1 deg. TO CONFIRM on more cases.
BITE_TOLERANCE = 0.5            # mm: gap between the bite fitted on the upper arch and on the lower arch
BITE_MAX_ANGLE = 1.5            # deg: same, in rotation
BITE_MIN_FIT = 0.6              # share of the bite within BITE_INLIER of the arches in the common fit (case1: 0.78)
BITE_MIN_SHARE = 0.10           # each arch must hold at least this share of the bite (else: not a bite of both)
BITE_INLIER = 0.3               # mm: a bite point this close to an arch fits it
BITE_SEARCH_SPOTS = 60          # start points of the global search, spread over the arches
BITE_NEIGHBOURHOOD = 8.0        # mm around a start point used for the arch's direction there
PIN_STEM_DIAMETER = 5.0         # mm, models: the stem from a female to the lower model, flat on the base plane
PIN_STEM_SINK = 1.5             # mm the stem goes into the lower model past its surface

# Print workspace ("Impression", print_scene.py / print_chain.py; measured in docs/print_spike_report.md)
PRINT_TOOL_TIMEOUT = 300.0      # s: an external tool (PrusaSlicer, UVtools) silent this long is stopped
PRINT_PREVIEW_PIXEL = 0.144     # mm: pixel of the fast slicing that brings the supports back (8 x the printer's 18 um)
PRINT_PREVIEW_ADAPTIVITY = 0.3  # 0 - 1: flat areas of the support preview are meshed with fewer faces
PRINT_UPDATE_DELAY = 0.4        # s without moving before the supports of a part are regenerated
PRINT_SUPPORT_LIFT = 5.0        # mm between the pad and the lowest point of a supported part
PRINT_SUPPORT_DENSITY = 100     # %: PrusaSlicer's relative density of support points
PRINT_SUPPORT_TIP = 0.4         # mm: diameter of a support tip where it touches the part, TO TEST (print)
PRINT_SUPPORT_PENETRATION = 0.2 # mm a support tip goes into the part (PrusaSlicer's default)
PRINT_SUPPORT_PAD = True        # a pad (raft) under the supports
PRINT_BEARING_DISTANCE = 0.5    # mm: faces of a part this close to the bone, teeth or scan it fits are bearing surfaces
PRINT_BLOCKER_HALF = 0.3        # mm each side of a bearing surface: the slab where no support may start
PRINT_ISLAND_MIN_AREA = 1.0     # mm2: an unsupported island smaller than this on one layer is a rasterisation sliver
PRINT_ISLAND_MIN_LAYERS = 5     # an island group counts when its volume exceeds that area over this many layers
PRINT_PART_GAP = 3.0            # mm left between parts (and to the plate's edge) by "Disposer"
PRINT_TILTS = (0, 30, 45, 90)   # degrees: the quick tilt presets
PRINT_RAFT_THICKNESS = 1.2      # mm: the raft, one rigid plate under every part, cured against the build plate
PRINT_RAFT_MARGIN = 2.0         # mm the raft extends past the parts and their supports
PRINT_RAFT_LIP = 1.0            # mm: the raft's top edge overhangs its foot by this much (bevel: the scraper slides under)
PRINT_TILT_DEFAULT = 30.0       # degrees: a part arrives bearing surface up and tilted (no flat face parallel to the plate)
PRINT_TEST_PLATE = (30.0, 20.0, 2.5)   # mm: the synthetic test piece's plate (asymmetric: one corner cut)

# Segmentation workspace ("Segmentation", volume_scene.py): a .nrrd from DentalSeg turned into surfaces, checked and
# corrected in Blender before planning
VOL_THRESHOLD = 50.0            # %: the surface passes where the structure's probability has this value
VOL_MIN_PIECE = 60.0            # mm3: smaller loose pieces are hidden (same as DentalSegmentator in 3D Slicer)
VOL_SMOOTH_LABELS = 8           # smoothing passes by default for a label file (staircase surface); 0 for probabilities
VOL_SMOOTH_FACTOR = 0.5         # strength of one smoothing pass
VOL_PICK_RADIUS = 4             # voxels: a click this close to a piece picks it
VOL_UPDATE_DELAY = 0.3          # s without touching a setting before the surface is recomputed
VOL_EXPECTED_PIECES = {"Maxillaire": 1, "Mandibule": 1, "Canal mandibulaire": 2}   # more than this: a warning

# Orthognathic osteotomies (ortho_osteotomy.py): Le Fort I and mandibular (BSSO) cut lines, then separation
ORTHO_LEFORT_SPACING = 1.5      # mm between the grid lines of the Le Fort I cut surface
ORTHO_PTM_DROP = 40.0           # mm the Le Fort I's posterior stop goes down from the cut plane (past the tuberosities)
ORTHO_MARGIN = 15.0             # mm the Le Fort I surface starts in front of the clicked line (the nasal spine sticks out)
ORTHO_END_EXTENSION_LEFORT = 30.0   # mm, Le Fort I: wide laterally (the buttress / zygoma runs past the last point)
ORTHO_BSSO_OUTSIDE = 10.0       # mm the BSSO's buccal vertical cut starts outside the bone (towards the surgeon)
ORTHO_BSSO_NERVE_BAND = 3.0     # mm: nerve canal vertices this close to the buccal cut's plane set its depth
ORTHO_BSSO_CANAL = 4.0          # mm: the canal's width (its vertices this far beyond the nearest one belong to it)
ORTHO_BSSO_DEFAULT_DEPTH = 6.0  # mm: depth of the buccal cut when no nerve canal is found there
ORTHO_BSSO_EXIT = 6.0           # mm the split runs past its last point (so it exits the lingual cortex)
ORTHO_BSSO_HEIGHT_RADIUS = 4.0  # mm: the bone this close to each point of the split sets its height there
ORTHO_BSSO_HEIGHT_MARGIN = 5.0  # mm the split reaches above and below the bone
ORTHO_BSSO_SPACING = 1.5        # mm between samples of the split's path
ORTHO_KERF = 0.5                # mm: thickness of the simulated saw cut
ORTHO_SMOOTH_STEPS = 6          # samples per segment of the smooth line through the clicked points
ORTHO_MIN_PIECE = 500.0         # mm3: smaller pieces left by the cut (chips) join the nearest large piece

# Points shown as solid balls (point_balls.py): landmarks, splint coverage points, osteotomy line points
POINT_BALL_COLOUR = (0.15, 0.85, 0.30, 1.0)   # green
POINT_BALL_MIN_RADIUS = 1.2     # mm: a ball is never smaller than this, whatever the point's own size

# Measurements (Section & Analysis, measure.py)
MEASURE_COLOUR = (1.0, 0.85, 0.1, 1.0)   # yellow line, end points and label
MEASURE_LINE_WIDTH = 2.5                 # px
MEASURE_POINT_SIZE = 9.0                 # px, end points
MEASURE_DRAG_PX = 6                      # px: shorter mouse travel = a click (then click the second point)
