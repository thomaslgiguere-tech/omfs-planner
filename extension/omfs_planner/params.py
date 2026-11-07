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
CUT_POSTERIOR_EXTENT = 15.0   # mm behind each mental foramen

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

# Occlusion seating (virtual, lower arch onto the fixed upper arch)
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
OCC_SEARCH_DISTANCE = 5.0       # mm: farther surfaces are ignored by the contact search
OCC_SAMPLES = 4000              # vertices per arch used by the spring solver
OCC_SPRING_STEP = 0.25          # fraction of the spring length closed per iteration (animation)
OCC_CANINE_WEIGHT = 3.0         # a canine spring pulls this many times harder than another spring
OCC_SPRING_SHARE = 70.0         # %: springs' share of the guidance; the rest is the magnet (closing the gaps)
OCC_MAGNET_RANGE = 3.0          # mm: gaps wider than this are not pulled by the magnet
OCC_MAGNET_STEP = 0.25          # fraction of each gap closed per iteration
OCC_TRIPOD_MIN_AREA = 1.0       # mm²: contact area right, anterior and left for a stable "tripod"
OCC_MAX_ITERATIONS = 300
OCC_SETTLED = 0.1               # mm and degrees per iteration: below this the position is stable
OCC_FINISH_ITERATIONS = 40      # last iterations, on every vertex near the contacts
OCC_FINISH_BAND = 1.0           # mm: vertices this close to the other arch are checked one by one at the end
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
SPLINT_CLEARANCE = 0.10         # mm between the imprint and the teeth, in every direction (walls too)

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
MODEL_BAR_DIAMETER = 4.0        # mm, the bar across a horseshoe's opening (stops it flexing)
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
