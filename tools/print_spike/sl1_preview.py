"""Spike: a light preview mesh of the part + its supports, from the layers of a PrusaSlicer .sl1, inside
Blender only (OpenImageIO reads the PNG layers, openvdb stores the voxels, Volume to Mesh makes the mesh).

Blender -b --factory-startup --python tools/print_spike/sl1_preview.py -- <file.sl1> <out.blend|out.stl> [factor]
factor: pixels merged per voxel side (8 -> 8 x 18 um = 0.144 mm). Synthetic test files only.
"""
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
import openvdb

args = sys.argv[sys.argv.index("--") + 1:]
sl1, out = Path(args[0]), Path(args[1])
factor = int(args[2]) if len(args) > 2 else 8
t0 = time.time()

with zipfile.ZipFile(sl1) as z:
    ini = dict(line.split(" = ", 1) for line in z.read("prusaslicer.ini").decode().splitlines() if " = " in line)
    layers = sorted(n for n in z.namelist() if n.endswith(".png") and not n.startswith("thumbnail"))
    pixel = float(ini["display_width"]) / int(ini["display_pixels_x"])
    layer_h = float(ini["layer_height"])
    mirror_x = ini.get("display_mirror_x", "0") == "1"
    step = max(1, round(factor * pixel / layer_h))            # layers merged per voxel: about cubic voxels
    with tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp, layers)

        def read(name):
            """One layer, reduced at once (block maximum: a thin support stays visible)."""
            img = oiio.ImageBuf(str(Path(tmp) / name)).get_pixels(oiio.UINT8)[:, :, 0]
            h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
            return img[:h, :w].reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:
            small = np.stack(list(pool.map(read, layers)))     # (layer, row, col)
t1 = time.time()
n = small.shape[0] // step * step
small = np.concatenate((small[:n].reshape(-1, step, *small.shape[1:]).max(axis=1),
                        small[n:].max(axis=0, keepdims=True)) if n < small.shape[0] else
                       (small.reshape(-1, step, *small.shape[1:]).max(axis=1),))
on = np.argwhere(small.max(axis=0) > 127)
(r0, c0), (r1, c1) = on.min(axis=0), on.max(axis=0) + 1
small = small[:, r0:r1, c0:c1]
if mirror_x:
    small = small[:, :, ::-1]
dense = np.ascontiguousarray(np.transpose(small, (2, 1, 0)).astype(np.float32) / 255.0)   # (x, y, z)
t2 = time.time()

grid = openvdb.FloatGrid()
grid.copyFromArray(dense, tolerance=0.01)
grid.name = "density"
voxel = (pixel * factor, pixel * factor, layer_h * step)
grid.transform = openvdb.createLinearTransform(np.diag(voxel + (1.0,)).tolist())
vdb = out.with_suffix(".vdb")
openvdb.write(str(vdb), grids=[grid])
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.object.volume_import(filepath=str(vdb))
volume = bpy.context.active_object
holder = bpy.data.objects.new("Volume to Mesh", bpy.data.meshes.new("empty"))   # the modifier lives on a mesh
bpy.context.scene.collection.objects.link(holder)
mod = holder.modifiers.new("Mesh", 'VOLUME_TO_MESH')
mod.object = volume
mod.threshold = 0.5
mod.resolution_mode = 'GRID'
mod.adaptivity = 0.2   # fewer faces on flat areas
depsgraph = bpy.context.evaluated_depsgraph_get()
mesh = bpy.data.meshes.new_from_object(holder.evaluated_get(depsgraph))
t3 = time.time()
preview = bpy.data.objects.new("Aperçu supports", mesh)
bpy.context.scene.collection.objects.link(preview)
if out.suffix == ".stl":
    bpy.ops.object.select_all(action='DESELECT')
    preview.select_set(True)
    bpy.ops.wm.stl_export(filepath=str(out), export_selected_objects=True)
else:
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
dims = np.array(dense.shape) * np.array(voxel)
print(f"PREVIEW|{len(layers)} couches, voxel {voxel[0]:.3f} x {voxel[1]:.3f} x {voxel[2]:.3f} mm, grille {dense.shape}, "
      f"boîte {dims.round(1).tolist()} mm|{len(mesh.polygons)} faces|lecture PNG {t1 - t0:.1f} s, volume {t2 - t1:.1f} s, "
      f"maillage {t3 - t2:.1f} s, total {time.time() - t0:.1f} s")
