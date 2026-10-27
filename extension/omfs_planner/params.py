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
GUIDE_REMESH_VOXEL = 0.2        # mm, voxel size when remeshing a guide watertight
SLEEVE_OUTER_DIAMETER = 4.0     # mm, MatrixMANDIBLE sleeve 03.503.045: just under 4 mm, TO MEASURE
SLEEVE_FIT_CLEARANCE = 0.10     # mm added to the sleeve diameter for the tube bore, TO TEST (print)
SLEEVE_TUBE_LENGTH = 5.0        # mm
SLEEVE_TUBE_WALL = 3.0          # mm
