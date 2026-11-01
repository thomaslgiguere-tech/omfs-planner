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
GUIDE_REMESH_VOXEL = 0.2        # mm, voxel size when remeshing a guide watertight
GUIDE_PROTECT_DISTANCE = 1.0    # mm: guide vertices closer than this to the bone are the intrados (protected when polishing)
SLEEVE_OUTER_DIAMETER = 4.0     # mm, MatrixMANDIBLE sleeve 03.503.045: just under 4 mm, TO MEASURE
SLEEVE_FIT_CLEARANCE = 0.10     # mm added to the sleeve diameter for the tube bore, TO TEST (print)
SLEEVE_TUBE_LENGTH = 5.0        # mm
SLEEVE_TUBE_WALL = 3.0          # mm

# Occlusion seating (virtual, lower arch onto the fixed upper arch)
OCC_TOLERATED_PENETRATION = 0.2  # mm: the teeth resist only beyond this overlap (0 = stop at contact)
OCC_CONTACT = 0.3               # mm: closer than this counts as an occlusal contact (green)
OCC_OVERLAP_STEPS = (0.5, 1.0, 1.5)  # mm: yellow / orange / red bands, magenta beyond the last
OCC_OVERLAP_RARE = 1.0          # mm: overlap above this is rare, warn, TO CONFIRM
OCC_OVERLAP_MAX = 1.5           # mm: overlap above this is beyond what is tolerated, TO CONFIRM
OCC_GAP_FAR = 1.0               # mm: gaps beyond this are grey (no contact)
OCC_OVERLAP_MIN_AREA = 0.1      # mm²: max overlap is the depth reached over at least this area (ignores lone spikes)
OCC_SEARCH_DISTANCE = 5.0       # mm: farther surfaces are ignored by the contact search
OCC_SAMPLES = 4000              # vertices per arch used by the spring solver
OCC_SPRING_STEP = 0.25          # fraction of the spring length closed per iteration (animation)
OCC_CANINE_WEIGHT = 3.0         # a canine spring pulls this many times harder than another spring
OCC_SPRING_SHARE = 70.0         # %: springs' share of the guidance; the rest is the magnet (closing the gaps)
OCC_MAGNET_RANGE = 3.0          # mm: gaps wider than this are not pulled by the magnet
OCC_MAGNET_STEP = 0.25          # fraction of each gap closed per iteration
OCC_CONTACT_WEIGHT = 20.0       # resistance of the teeth vs springs + magnet (1.0): overlap stays near tolerance
OCC_TRIPOD_MIN_AREA = 1.0       # mm²: contact area right, anterior and left for a stable "tripod"
OCC_MAX_ITERATIONS = 300
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
