"""Object roles: what each object is in the surgical plan (bone, teeth, scan, segment, ...).

The role is stored on every object as `obj.omfs.role`, so it is saved in the .blend file.
Roles are generic on purpose: a genioplasty segment, a Le Fort I maxilla or a BSSO distal
segment are all just "SEGMENT" objects.
"""

import re
from pathlib import Path

import bpy

ROLES = (
    ('BONE', "Bone", "CT/CBCT bone segmentation (locked by default)"),
    ('TEETH', "Teeth", "Teeth segmented from the CT/CBCT"),
    ('SCAN', "Scan", "Intraoral scan"),
    ('SEGMENT', "Segment", "Osteotomy segment that will be moved"),
    ('GUIDE', "Guide", "Surgical guide"),
    ('OTHER', "Other", "Anything else"),
)

# Viewport colours (RGBA, 0-1), shown when viewport shading colour is set to "Object".
COLORS = {
    'BONE': (0.90, 0.84, 0.70, 1.0),     # ivory
    'TEETH': (0.55, 0.75, 0.95, 1.0),    # light blue
    'SCAN': (0.95, 0.60, 0.60, 1.0),     # gingiva pink
    'SEGMENT': (0.55, 0.85, 0.55, 1.0),  # green
    'GUIDE': (0.95, 0.70, 0.30, 1.0),    # orange
    'OTHER': (0.60, 0.60, 0.60, 1.0),    # grey
}

LOCKED_BY_DEFAULT = {'BONE'}

# Filename keywords, checked in this order (a file called "ios_teeth" is a scan, not teeth).
# Short keywords (< 4 letters) must be a whole word, so "os" does not match inside "ios".
KEYWORDS = (
    ('SCAN', ("scan", "ios", "intraoral", "upperjaw", "lowerjaw", "itero", "trios", "medit")),
    ('TEETH', ("teeth", "tooth", "dent", "dents", "dentition")),
    ('BONE', ("bone", "mandible", "mandibule", "maxilla", "maxillaire", "skull", "crane",
              "crâne", "os", "mand", "chin", "menton", "symphysis")),
)


def role_from_filename(filename):
    stem = Path(filename).stem.lower()
    words = [w for w in re.split(r"[^a-zâ0-9]+", stem) if w]
    for role, keywords in KEYWORDS:
        for keyword in keywords:
            if len(keyword) < 4:
                if keyword in words:
                    return role
            elif keyword in stem:
                return role
    return 'OTHER'


def apply_role_color(obj):
    obj.color = COLORS[obj.omfs.role]


def _on_role_changed(self, context):
    apply_role_color(self.id_data)


class OMFS_ObjectProperties(bpy.types.PropertyGroup):
    role: bpy.props.EnumProperty(name="Role", items=ROLES, default='OTHER', update=_on_role_changed)


def register():
    bpy.utils.register_class(OMFS_ObjectProperties)
    bpy.types.Object.omfs = bpy.props.PointerProperty(type=OMFS_ObjectProperties)


def unregister():
    del bpy.types.Object.omfs
    bpy.utils.unregister_class(OMFS_ObjectProperties)
