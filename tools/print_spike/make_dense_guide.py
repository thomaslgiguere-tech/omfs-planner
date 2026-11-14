"""Spike: a dense, slightly bumpy copy of the synthetic guide (about the triangle count of a real guide), so the
support timing is realistic. Synthetic only.
Blender -b --factory-startup --python tools/print_spike/make_dense_guide.py -- in.stl out.stl [levels] [bump mm]"""
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
levels = int(args[2]) if len(args) > 2 else 4
bump = float(args[3]) if len(args) > 3 else 0.1   # mm, 0 = smooth
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.wm.stl_import(filepath=args[0])
obj = bpy.context.selected_objects[0]
mod = obj.modifiers.new("Subdivide", 'SUBSURF')
mod.subdivision_type, mod.levels = 'SIMPLE', levels
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=mod.name)
mesh = obj.data
co = np.empty(len(mesh.vertices) * 3)
mesh.vertices.foreach_get("co", co)
nor = np.empty(len(mesh.vertices) * 3)
mesh.vertices.foreach_get("normal", nor)
co, nor = co.reshape(-1, 3), nor.reshape(-1, 3)
bumps = bump * np.sin(co[:, 0] * 1.7) * np.sin(co[:, 1] * 2.3) * np.sin(co[:, 2] * 1.9)   # smooth, 0.1 mm
mesh.vertices.foreach_set("co", (co + nor * bumps[:, None]).ravel())
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.ops.wm.stl_export(filepath=args[1], export_selected_objects=True)
print(f"DENSE|{len(mesh.polygons)} faces")
