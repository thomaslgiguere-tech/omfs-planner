"""Import STL / OBJ / PLY meshes with their raw file coordinates.

All importers are forced to forward = Y, up = Z, scale = 1, so the coordinates in the
file are kept as-is. Meshes exported from 3D Slicer share one coordinate system; any
rotation or scaling here would break the registration between bone, teeth and scans.
(Blender's OBJ importer rotates by default, assuming Y-up files.)
"""

from pathlib import Path

import bpy

RAW_AXES = dict(forward_axis='Y', up_axis='Z', global_scale=1.0)

IMPORTERS = {
    ".stl": lambda path: bpy.ops.wm.stl_import(filepath=path, use_scene_unit=False, **RAW_AXES),
    ".obj": lambda path: bpy.ops.wm.obj_import(filepath=path, **RAW_AXES),
    ".ply": lambda path: bpy.ops.wm.ply_import(filepath=path, use_scene_unit=False, **RAW_AXES),
}


def mesh_files(folder):
    """Supported mesh files in a folder (not sub-folders), sorted by name."""
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMPORTERS)


def import_mesh(path):
    """Import one file and return the new objects it created."""
    path = Path(path)
    before = set(bpy.data.objects)
    IMPORTERS[path.suffix.lower()](str(path))
    return [obj for obj in bpy.data.objects if obj not in before]
