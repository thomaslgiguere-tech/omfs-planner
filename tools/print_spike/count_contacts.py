"""Spike check: support voxels touching the bearing surface, from .sl1 layers (synthetic files only).
python count_contacts.py with_supports.sl1 part_only.sl1 part_plus_bearing.sl1
Stacks aligned at their top layer (supports lift the part). Support = lit with supports but not in the part alone; bearing layer = lit in part + bearing slab but not in
the part alone (the 0.3 mm just off the bearing face). Prints the support voxels inside that layer."""
import io
import sys
import zipfile

import numpy as np
from PIL import Image


def layers(path):
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png") and not n.startswith("thumbnail"))
        return np.stack([np.array(Image.open(io.BytesIO(z.read(n))).convert("L")) > 127 for n in names])


full, part, plus = (layers(p) for p in sys.argv[1:4])
n = min(len(full), len(part), len(plus))   # aligned at the top: supports lift the part, its top stays the top
full, part, plus = full[-n:], part[-n:], plus[-n:]
support = full & ~part
bearing = plus & ~part
print(f"CONTACTS|{sys.argv[1]}|support voxels {int(support.sum())}|bearing layer voxels {int(bearing.sum())}|"
      f"support inside the bearing layer {int((support & bearing).sum())}")
