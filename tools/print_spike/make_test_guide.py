"""Synthetic test guide for the print spike (never patient data).

A curved shell standing in for a chin cutting guide (an arc of a thick cylinder: the concave inner face
is the bearing surface that must stay free of supports) plus a drill-sleeve tube on its outer face.
Writes binary STL files in millimetres:
  guide.stl          shell + tube, one closed mesh each, in one file
  bearing.stl        a thin closed slab across the concave face (+-0.3 mm): the "no-support" area as a volume
  bearing_band.stl   a slab 0.15 to 0.6 mm in front of that face: where a support reaching it must pass

Usage: python3 tools/print_spike/make_test_guide.py tools/print_spike/out
"""
import math
import struct
import sys
from pathlib import Path

R_IN, THICK, HEIGHT, ARC = 20.0, 2.5, 14.0, math.radians(110)   # mm, the shell
TUBE_R_OUT, TUBE_R_IN, TUBE_LEN = 3.0, 2.1, 8.0                # mm, the drill-sleeve tube


def shell(r_in, r_out, height, arc, n=60, z0=0.0):
    """Closed arc of a thick cylinder, axis Z, centred on +Y: (vertices, triangles)."""
    verts, tris = [], []
    for i in range(n + 1):
        a = -arc / 2 + arc * i / n + math.pi / 2
        for r in (r_in, r_out):
            for z in (z0, z0 + height):
                verts.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(n):   # per step: 4 vertices (in-low, in-high, out-low, out-high)
        a, b = 4 * i, 4 * (i + 1)
        quads = [(a + 0, b + 0, b + 1, a + 1),   # inner face (normal towards the axis)
                 (a + 2, a + 3, b + 3, b + 2),   # outer face
                 (a + 0, a + 2, b + 2, b + 0),   # bottom
                 (a + 1, b + 1, b + 3, a + 3)]   # top
        for q in quads:
            tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    for a, flip in ((0, False), (4 * n, True)):   # the two ends
        q = (a + 0, a + 1, a + 3, a + 2)
        t = [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
        tris += [tuple(reversed(x)) for x in t] if flip else t
    return verts, tris


def tube(r_out, r_in, length, centre, n=32):
    """Closed tube along +Y starting at centre: (vertices, triangles)."""
    cx, cy, cz = centre
    verts, tris = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        for r in (r_in, r_out):
            for y in (0.0, length):
                verts.append((cx + r * math.cos(a), cy + y, cz + r * math.sin(a)))
    for i in range(n):
        a, b = 4 * i, 4 * ((i + 1) % n)
        quads = [(a + 0, a + 1, b + 1, b + 0),   # bore (normal inwards)
                 (a + 2, b + 2, b + 3, a + 3),   # outside
                 (a + 0, b + 0, b + 2, a + 2),   # end y = 0
                 (a + 1, a + 3, b + 3, b + 1)]   # end y = length
        for q in quads:
            tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return verts, tris


def write_stl(path, parts):
    tris = []
    for verts, faces in parts:
        tris += [[verts[i] for i in reversed(f)] for f in faces]   # outward normals
    with open(path, "wb") as f:
        f.write(b"OMFS Planner print spike (synthetic)".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            u = [b[k] - a[k] for k in range(3)]
            v = [c[k] - a[k] for k in range(3)]
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            length = math.sqrt(sum(x * x for x in n)) or 1.0
            f.write(struct.pack("<12fH", *(x / length for x in n), *a, *b, *c, 0))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/print_spike/out")
    out.mkdir(parents=True, exist_ok=True)
    r_out = R_IN + THICK
    parts = [shell(R_IN, r_out, HEIGHT, ARC),
             tube(TUBE_R_OUT, TUBE_R_IN, TUBE_LEN, (0.0, r_out - 0.5, HEIGHT / 2))]
    write_stl(out / "guide.stl", parts)
    write_stl(out / "bearing.stl", [shell(R_IN - 0.3, R_IN + 0.3, HEIGHT + 1.0, ARC * 1.04, z0=-0.5)])   # past the edges
    # check band 0.15 - 0.6 mm in front of the bearing face (spike check of the blocker, count_contacts.py)
    write_stl(out / "bearing_band.stl", [shell(R_IN - 0.6, R_IN - 0.15, HEIGHT, ARC * 0.98, z0=0.0)])
    print(f"wrote {out / 'guide.stl'}, {out / 'bearing.stl'} and {out / 'bearing_band.stl'}")
