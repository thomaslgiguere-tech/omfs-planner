"""Write a synthetic test case (no patient data): bone STL, teeth OBJ, scan PLY.

Usage: python3 tools/make_test_case.py cases/synthetic
Shapes are crude but in realistic millimetre sizes and offset from the origin like a
3D Slicer export, so import, units, colours and framing can be tested.
"""

import math
import struct
import sys
from pathlib import Path

OFFSET = (10.0, -80.0, -60.0)  # Slicer exports are rarely centred on the origin


def arch_tube(arch_radius, tube_radius, z, segments=48, sides=16):
    """U-shaped tube (half torus) standing in for a mandible: returns (vertices, triangles)."""
    verts, tris = [], []
    for i in range(segments + 1):
        u = math.pi * i / segments
        cx, cy = arch_radius * math.cos(u), arch_radius * math.sin(u)
        for j in range(sides):
            v = 2 * math.pi * j / sides
            r = arch_radius + tube_radius * math.cos(v)
            verts.append((r * math.cos(u), r * math.sin(u), z + 1.6 * tube_radius * math.sin(v)))
    for i in range(segments):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            tris += [(a, c, b), (b, c, d)]
    return verts, tris


def box(center, size):
    (x, y, z), (sx, sy, sz) = center, size
    verts = [(x + dx * sx / 2, y + dy * sy / 2, z + dz * sz / 2)
             for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return verts, tris


def shift(verts):
    return [(x + OFFSET[0], y + OFFSET[1], z + OFFSET[2]) for x, y, z in verts]


def write_stl(path, verts, tris):
    with open(path, "wb") as f:
        f.write(b"synthetic test mesh".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            for i in tri:
                f.write(struct.pack("<3f", *verts[i]))
            f.write(b"\0\0")


def write_obj(path, verts, tris):
    with open(path, "w") as f:
        f.writelines(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in verts)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in tris)


def write_ply(path, verts, tris):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n")
        f.writelines(f"{x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in verts)
        f.writelines(f"3 {a} {b} {c}\n" for a, b, c in tris)


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    bone = arch_tube(arch_radius=45, tube_radius=8, z=0)
    write_stl(folder / "mandible_bone.stl", shift(bone[0]), bone[1])

    teeth_verts, teeth_tris = [], []
    for k in range(10):
        u = math.pi * (k + 0.5) / 10
        v, t = box((45 * math.cos(u), 45 * math.sin(u), 18), (7, 7, 9))
        teeth_tris += [(a + len(teeth_verts), b + len(teeth_verts), c + len(teeth_verts)) for a, b, c in t]
        teeth_verts += v
    write_obj(folder / "lower_teeth.obj", shift(teeth_verts), teeth_tris)

    scan = arch_tube(arch_radius=45, tube_radius=5, z=19)
    write_ply(folder / "lowerjaw_scan.ply", shift(scan[0]), scan[1])

    (folder / "notes.txt").write_text("Not a mesh: Import Case must skip this file.\n")
    print(f"Wrote synthetic test case to {folder}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cases/synthetic")
