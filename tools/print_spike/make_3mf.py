"""Spike: write a PrusaSlicer 3MF with one object made of a part and support blocker / enforcer volumes.

python3 tools/print_spike/make_3mf.py out.3mf part.stl [--blocker a.stl] [--enforcer b.stl] [--rotate-x DEG]
    [--at X,Y]
Volumes are listed in Metadata/Slic3r_PE_model.config by triangle range, with volume_type ModelPart,
SupportBlocker or SupportEnforcer, the way PrusaSlicer 2.9 writes them. The object is turned about X
(degrees) and placed on the bed at X,Y with its lowest point at z = 0 (PrusaSlicer adds the elevation).
"""
import argparse
import math
import struct
import zipfile


def read_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    return [struct.unpack("<9f", data[84 + 50 * i + 12:84 + 50 * i + 48]) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"), ap.add_argument("part")
    ap.add_argument("--blocker", action="append", default=[])
    ap.add_argument("--enforcer", action="append", default=[])
    ap.add_argument("--extra-part", action="append", default=[], help="another printed volume (checks)")
    ap.add_argument("--rotate-x", type=float, default=0.0)
    ap.add_argument("--at", default="76.68,38.88")
    a = ap.parse_args()
    volumes = [(a.part, "ModelPart")] + [(p, "ModelPart") for p in a.extra_part] + [(p, "SupportBlocker") for p in a.blocker] + [(p, "SupportEnforcer") for p in a.enforcer]
    c, s = math.cos(math.radians(a.rotate_x)), math.sin(math.radians(a.rotate_x))
    verts, tris, ranges = [], [], []
    for path, kind in volumes:
        first = len(tris)
        index = {}   # shared vertices per volume, as in PrusaSlicer's own 3MF (a triangle soup crashed it)
        for t in read_stl(path):
            ids = []
            for k in range(3):
                key = t[3 * k:3 * k + 3]
                if key not in index:
                    x, y, z = key
                    index[key] = len(verts)
                    verts.append((x, y * c - z * s, y * s + z * c))
                ids.append(index[key])
            tris.append(ids)
        ranges.append((path, kind, first, len(tris) - 1))
    part_end = ranges[0][3]
    part_verts = [verts[i] for t in tris[:part_end + 1] for i in t]
    lo = [min(v[k] for v in part_verts) for k in range(3)]
    hi = [max(v[k] for v in part_verts) for k in range(3)]
    centre = [(lo[k] + hi[k]) / 2 for k in range(3)]
    verts = [(x - centre[0], y - centre[1], z - centre[2]) for x, y, z in verts]   # object origin at its centre
    bx, by = (float(v) for v in a.at.split(","))
    bz = centre[2] - lo[2]
    model = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">',
             ' <metadata name="slic3rpe:Version3mf">1</metadata>', ' <resources>', '  <object id="1" type="model">',
             '   <mesh>', '    <vertices>']
    model += [f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in verts]
    model += ['    </vertices>', '    <triangles>']
    model += [f'     <triangle v1="{i}" v2="{j}" v3="{k}"/>' for i, j, k in tris]
    model += ['    </triangles>', '   </mesh>', '  </object>', ' </resources>', ' <build>',
              f'  <item objectid="1" transform="1 0 0 0 1 0 0 0 1 {bx} {by} {bz:.6f}" printable="1"/>',
              ' </build>', '</model>']
    config = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>', ' <object id="1" instances_count="1">',
              '  <metadata type="object" key="name" value="spike"/>']
    for path, kind, first, last in ranges:
        config += [f'  <volume firstid="{first}" lastid="{last}">',
                   f'   <metadata type="volume" key="name" value="{path.split("/")[-1]}"/>',
                   f'   <metadata type="volume" key="volume_type" value="{kind}"/>',
                   '   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>',
                   '  </volume>']
    config += [' </object>', '</config>']
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "\n".join(model))
        z.writestr("Metadata/Slic3r_PE_model.config", "\n".join(config))
    print(f"wrote {a.out}: {len(tris)} triangles, volumes {[(k, l - f + 1) for _p, k, f, l in ranges]}")


if __name__ == "__main__":
    main()
