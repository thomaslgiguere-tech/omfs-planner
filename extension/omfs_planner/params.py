"""Clinical and hardware defaults, in one place. Lengths in mm, angles in degrees.

Each default can be changed in the panel for a given case. Values marked TO CONFIRM must be
checked with a supervisor before clinical use.
"""

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
