"""Print chain: the external tools behind the Impression workspace. No Blender UI here.

OMFS Planner is the interface and the glue; the hard work is done by two free programs:
- PrusaSlicer (command line): SLA supports and slicing -> .sl1 (a zip of PNG layers);
- UVtools (UVtoolsCmd): .sl1 -> Elegoo .goo, and the check for unsupported islands.

    parts (+ support blockers) -> 3MF -> PrusaSlicer -> .sl1 -> layers
        live:   at preview resolution -> voxels -> the supports as a light mesh (about 2 s)
        export: at the printer's resolution -> UVtoolsCmd convert -> .goo -> checks

Everything here works on plain arrays and files, so it can run in a worker thread while Blender stays
responsive, and be tested without the UI. Measured in docs/print_spike_report.md.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import params

PRUSASLICER_PATHS = ("/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
                     "/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer")
UVTOOLS_PATHS = ("/Applications/UVtools.app/Contents/MacOS/UVtoolsCmd",)
DOWNLOADS = {"PrusaSlicer": "https://www.prusa3d.com/prusaslicer", "UVtools": "https://github.com/sn4k3/UVtools/releases"}

# One printed part: vertices (n, 3) and triangles (m, 3) already turned as on the plate (mm), its support
# blockers [(vertices, triangles)] in the same coordinates, and where its bounding-box centre goes on the
# bed (x, y in mm from the bed's corner).
# options: PrusaSlicer settings for this object only (the raft: no supports, flat on the plate).
Part = namedtuple("Part", "name vertices triangles blockers bed_xy options", defaults=({},))


class ToolMissing(RuntimeError):
    """PrusaSlicer or UVtools is not installed where expected."""


class Cancelled(RuntimeError):
    """The job was cancelled (the part moved again)."""


def find_tool(name, custom=""):
    """Path of PrusaSlicer or UVtoolsCmd: the one set in the preferences, else the usual places."""
    for path in ([custom] if custom else []) + list(PRUSASLICER_PATHS if name == "PrusaSlicer" else UVTOOLS_PATHS):
        if path and os.access(path, os.X_OK):
            return path
    found = shutil.which("prusa-slicer" if name == "PrusaSlicer" else "UVtoolsCmd")
    if found:
        return found
    raise ToolMissing(f"{name} introuvable. Installez-le ({DOWNLOADS[name]}) ou indiquez son chemin.")


def run(cmd, cancel=None, timeout=params.PRINT_TOOL_TIMEOUT):
    """Run an external command; returns its output. cancel: a threading.Event that stops it."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    waited = 0.0
    while True:
        try:
            out, _ = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            waited += 0.1
            if (cancel is not None and cancel.is_set()) or waited > timeout:
                process.kill()
                process.communicate()
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                raise RuntimeError(f"{Path(cmd[0]).name}: pas de réponse après {timeout:.0f} s")
    if process.returncode != 0:
        tail = [line for line in out.splitlines() if "Voronoi" not in line][-3:]
        raise RuntimeError(f"{Path(cmd[0]).name} a échoué: " + " | ".join(tail))
    return out


# --- 3MF --------------------------------------------------------------------------------------

def weld(vertices, triangles):
    """Identical vertices merged: PrusaSlicer crashed on a 3MF whose triangles did not share vertices."""
    unique, inverse = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)[triangles]


def write_3mf(path, parts):
    """A PrusaSlicer 3MF: one object per part, made of its mesh (ModelPart) and its support blockers
    (SupportBlocker volumes: no support may start or end inside them). Each object is centred on its
    part's bounding box and placed on the bed by its build transform; PrusaSlicer sets the height."""
    model = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">',
             ' <metadata name="slic3rpe:Version3mf">1</metadata>', ' <resources>']
    config = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>']
    items = []
    for number, part in enumerate(parts, 1):
        low, high = part.vertices.min(axis=0), part.vertices.max(axis=0)
        centre = (low + high) / 2
        vertices, triangles, ranges = [], [], []
        for kind, (v, t) in [("ModelPart", (part.vertices, part.triangles))] + [("SupportBlocker", b) for b in part.blockers]:
            v, t = weld(np.asarray(v, float) - centre, np.asarray(t))
            first = sum(len(x) for x in triangles)
            triangles.append(t + sum(len(x) for x in vertices))
            vertices.append(v)
            ranges.append((kind, first, first + len(t) - 1))
        vertices, triangles = np.vstack(vertices), np.vstack(triangles)
        model += [f'  <object id="{number}" type="model">', '   <mesh>', '    <vertices>']
        model += ['     <vertex x="%.5f" y="%.5f" z="%.5f"/>' % tuple(v) for v in vertices]
        model += ['    </vertices>', '    <triangles>']
        model += ['     <triangle v1="%d" v2="%d" v3="%d"/>' % tuple(t) for t in triangles]
        model += ['    </triangles>', '   </mesh>', '  </object>']
        items.append(f'  <item objectid="{number}" transform="1 0 0 0 1 0 0 0 1 {part.bed_xy[0]:.5f} '
                     f'{part.bed_xy[1]:.5f} {(high[2] - low[2]) / 2:.5f}" printable="1"/>')
        config += [f' <object id="{number}" instances_count="1">',
                   f'  <metadata type="object" key="name" value="part{number}"/>']
        config += [f'  <metadata type="object" key="{key}" value="{value}"/>' for key, value in part.options.items()]
        for kind, first, last in ranges:
            config += [f'  <volume firstid="{first}" lastid="{last}">',
                       f'   <metadata type="volume" key="volume_type" value="{kind}"/>',
                       '   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>',
                       '  </volume>']
        config.append(' </object>')
    model += [' </resources>', ' <build>'] + items + [' </build>', '</model>']
    config.append('</config>')
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "\n".join(model))
        z.writestr("Metadata/Slic3r_PE_model.config", "\n".join(config))


# --- PrusaSlicer configuration ------------------------------------------------------------------

def slicer_config(printer, resin, supports, preview=False):
    """The PrusaSlicer settings (a dict) for a printer profile, a resin profile and the support settings.
    preview: same plate and supports, but coarse pixels (PRINT_PREVIEW_PIXEL), to get the supports back fast.
    The Elegoo-only settings (lift, waits, speeds...) travel in the printer notes, which UVtools reads."""
    px, py = printer["pixels_x"], printer["pixels_y"]
    if preview:
        px = max(1, round(printer["width"] / params.PRINT_PREVIEW_PIXEL))
        py = max(1, round(printer["height"] / params.PRINT_PREVIEW_PIXEL))
    w, h = printer["width"], printer["height"]
    notes = ["FILEFORMAT_GOO", "", "START_CUSTOM_VALUES"] + [f"{k}_{v:g}" for k, v in resin["elegoo"].items()] + ["END_CUSTOM_VALUES"]
    return {
        "printer_technology": "SLA",
        "printer_settings_id": printer["machine_name"],
        "display_pixels_x": px, "display_pixels_y": py, "display_width": w, "display_height": h,
        "display_orientation": "landscape",
        "display_mirror_x": int(printer["mirror_x"]), "display_mirror_y": int(printer["mirror_y"]),
        "max_print_height": printer["z"], "bed_shape": f"0x0,{w}x0,{w}x{h},0x{h}",
        "printer_notes": "\\n".join(notes),
        "layer_height": resin["layer_height"], "exposure_time": resin["exposure_time"],
        "initial_exposure_time": resin["bottom_exposure_time"],
        "faded_layers": resin["bottom_layers"], "material_ow_faded_layers": resin["bottom_layers"],
        "supports_enable": int(supports["enable"]), "pad_enable": int(supports["enable"] and supports["pad"]),
        "support_object_elevation": supports["lift"],
        # each of these also has a per-material override (material_ow_...) that wins when left at its default
        "support_points_density_relative": supports["density"], "material_ow_support_points_density_relative": supports["density"],
        "support_head_front_diameter": supports["tip"], "material_ow_support_head_front_diameter": supports["tip"],
        "support_head_penetration": supports.get("penetration", params.PRINT_SUPPORT_PENETRATION),
        "material_ow_support_head_penetration": supports.get("penetration", params.PRINT_SUPPORT_PENETRATION),
    }


def expected_goo(printer, resin):
    """What the .goo must say (UVtools property -> value) for these profiles: checked after each export."""
    expected = {"Version": printer["goo_version"], "ResolutionX": printer["pixels_x"], "ResolutionY": printer["pixels_y"],
                "DisplayWidth": printer["width"], "DisplayHeight": printer["height"], "MachineZ": printer["z"],
                "MachineName": printer["machine_name"], "LayerHeight": resin["layer_height"],
                "BottomLayerCount": resin["bottom_layers"], "BottomExposureTime": resin["bottom_exposure_time"],
                "ExposureTime": resin["exposure_time"]}
    expected.update(resin["elegoo"])
    return expected


def slice_plate(parts, config, out_sl1, slicer, cancel=None):
    """Supports + slicing of the plate (parts where they are, never rearranged). Returns the PrusaSlicer log."""
    work = Path(out_sl1).parent
    stem = Path(out_sl1).stem
    ini, model = work / f"{stem}.ini", work / f"{stem}.3mf"
    ini.write_text("\n".join(f"{k} = {v}" for k, v in config.items()) + "\n")
    write_3mf(model, parts)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    return run([slicer, "--load", str(ini), "--dont-arrange", "--loglevel", "2", "--export-sla", "-o", str(out_sl1), str(model)], cancel)


# --- Layers -------------------------------------------------------------------------------------

def read_png(data):
    """A layer image (grey) as a 2D uint8 array. OpenImageIO inside Blender, Pillow elsewhere (tests)."""
    try:
        import OpenImageIO as oiio
    except ImportError:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(data)).convert("L"))
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(data)
        f.flush()
        return oiio.ImageBuf(f.name).get_pixels(oiio.UINT8)[:, :, 0]


def read_layers(sl1):
    """(layers (z, row, col) bool, info) of a .sl1: info has the pixel size, layer height, mirrors, the
    resin volume (ml) and the layer count."""
    with zipfile.ZipFile(sl1) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png") and "thumbnail" not in n)
        ini = dict(line.split(" = ", 1) for line in z.read("prusaslicer.ini").decode().splitlines() if " = " in line)
        job = dict(line.split(" = ", 1) for line in z.read("config.ini").decode().splitlines() if " = " in line)
        layers = np.stack([read_png(z.read(n)) > 127 for n in names]) if names else np.zeros((0, 1, 1), bool)
    info = {"pixel": float(ini["display_width"]) / int(ini["display_pixels_x"]), "layer": float(ini["layer_height"]),
            "mirror_x": ini.get("display_mirror_x", "0") == "1", "mirror_y": ini.get("display_mirror_y", "0") == "1",
            "width": float(ini["display_width"]), "height": float(ini["display_height"]),
            "resin_ml": float(job.get("usedMaterial", 0.0)), "layers": len(names)}
    return layers, info


def to_bed(layers, info):
    """Layers flipped so that index (z, y, x) grows with the bed's Y and X (undoes the screen mirrors: the
    image rows run against the bed's Y, and the columns against X when the screen is mirrored)."""
    out = layers if info["mirror_y"] else layers[:, ::-1]
    return out[:, :, ::-1] if info["mirror_x"] else out


def dilate(volume, steps=1):
    for _ in range(steps):
        grown = volume.copy()
        for axis in range(3):
            lead = (slice(None),) * axis + (slice(1, None),)
            lag = (slice(None),) * axis + (slice(None, -1),)
            grown[lead] |= volume[lag]
            grown[lag] |= volume[lead]
        volume = grown
    return volume


def reduce_z(volume, step):
    """Layers merged step by step (any layer lit), from the top down so the part's top stays aligned."""
    if step <= 1 or not len(volume):
        return volume
    pad = (-len(volume)) % step
    padded = np.concatenate((np.zeros((pad,) + volume.shape[1:], bool), volume))
    return padded.reshape(-1, step, *volume.shape[1:]).any(axis=1)


def supports_only(with_supports, without, info):
    """(voxels (z, y, x) bool of the supports and pad only, z step in mm): what is printed with supports but
    is not the part itself. The two stacks are aligned at their top layer (supports lift the part, its top
    stays the top); the part is grown by one voxel so no skin of it is left."""
    step = max(1, round(info["pixel"] / info["layer"]))          # about cubic voxels
    full, part = reduce_z(to_bed(with_supports, info), step), reduce_z(to_bed(without, info), step)
    lifted = np.zeros_like(full)
    n = min(len(full), len(part))
    lifted[len(full) - n:] = part[len(part) - n:]
    return full & ~dilate(lifted), info["layer"] * step, (len(full) - n) * info["layer"] * step


def voxels_to_mesh(voxels, pixel, z_step):
    """Surface of a voxel volume (z, y, x): (vertices in mm, bed coordinates, z up from the plate; triangles;
    quads). openvdb does the meshing (it ships with Blender)."""
    import openvdb
    if not voxels.any():
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 4), int)
    on = np.argwhere(voxels)
    low, high = on.min(axis=0), on.max(axis=0) + 1
    crop = np.pad(voxels[low[0]:high[0], low[1]:high[1], low[2]:high[2]], 1)
    grid = openvdb.FloatGrid()
    grid.copyFromArray(np.ascontiguousarray(np.transpose(crop, (2, 1, 0)).astype(np.float32)))   # (x, y, z)
    points, triangles, quads = grid.convertToPolygons(isovalue=0.5, adaptivity=params.PRINT_PREVIEW_ADAPTIVITY)
    offset = np.array((low[2] - 1, low[1] - 1, low[0] - 1), float)
    vertices = (points + offset + 0.5) * np.array((pixel, pixel, z_step))
    return vertices, triangles.reshape(-1, 3)[:, ::-1], quads.reshape(-1, 4)[:, ::-1]      # outward normals


SupportPreview = namedtuple("SupportPreview", "vertices triangles quads lift resin_ml layers warnings")


def support_preview(part, printer, resin, supports, slicer, work, cancel=None):
    """The supports of one part, alone on the bed centre, as a light mesh centred on the part in X and Y,
    with z = 0 on the plate. lift: how high the part's lowest point ends up above the plate (mm).
    Two slicings at preview resolution run side by side: with and without supports."""
    work = Path(work)
    centre = (printer["width"] / 2, printer["height"] / 2)
    alone = part._replace(bed_xy=centre)
    config = slicer_config(printer, resin, supports, preview=True)
    bare = dict(config, supports_enable=0, pad_enable=0)
    with ThreadPoolExecutor(2) as pool:      # the two slicings side by side
        first = pool.submit(slice_plate, [alone], config, work / "with.sl1", slicer, cancel)
        second = pool.submit(slice_plate, [alone._replace(blockers=[])], bare, work / "without.sl1", slicer, cancel)
        log = first.result()
        second.result()
    full, info = read_layers(work / "with.sl1")
    part_only, _ = read_layers(work / "without.sl1")
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    voxels, z_step, lift = supports_only(full, part_only, info)
    vertices, triangles, quads = voxels_to_mesh(voxels, info["pixel"], z_step)
    if len(vertices):
        vertices[:, :2] -= centre
    warnings = []
    if supports["enable"] and not voxels.any():
        warnings.append("Aucun support généré dans cette orientation")
    if "No pad can be generated" in log:
        warnings.append("Socle impossible dans cette orientation")
    return SupportPreview(vertices, triangles, quads, lift, info["resin_ml"], info["layers"], warnings)


# --- Export -------------------------------------------------------------------------------------

def goo_properties(path, uvtools):
    """{property: text value} of a .goo header (UVtoolsCmd print-properties): never the layers."""
    out = run([uvtools, "print-properties", str(path)])
    found = {}
    for line in out.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key.isidentifier() and key not in found:
            found[key] = value.strip()
    return found


def settings_differences(found, expected):
    """[(property, expected, found)] where the .goo does not say what the profiles ask."""
    wrong = []
    for key, value in expected.items():
        got = found.get(key)
        try:
            same = got is not None and abs(float(got) - float(value)) < 1e-6
        except ValueError:
            same = got == str(value)
        if not same:
            wrong.append((key, value, got))
    return wrong


def islands(goo, uvtools, pixel, layer, cancel=None):
    """Unsupported islands of a .goo large enough to matter: [(first layer, area mm2 or volume mm3 text)].
    UVtools lists every island, grouped with what rests on it; the 1 - 2 pixel ones are rasterisation
    slivers, so only those over PRINT_ISLAND_MIN_AREA (one layer) or its volume equivalent are kept."""
    out = run([uvtools, "print-issues", str(goo), "--islands", "--group-islands"], cancel)
    found = []
    for line in out.splitlines():
        if not line.startswith("Island"):
            continue
        fields = [f.strip() for f in line.split(",")]
        size = next((f for f in fields if f.endswith("px²") or f.endswith("px³")), "")
        value = float(size[:-3])
        area = value * pixel ** 2 if size.endswith("px²") else None
        volume = value * pixel ** 2 * layer if size.endswith("px³") else None
        if (area is not None and area >= params.PRINT_ISLAND_MIN_AREA) or \
           (volume is not None and volume >= params.PRINT_ISLAND_MIN_AREA * layer * params.PRINT_ISLAND_MIN_LAYERS):
            first = int(fields[1].split("-")[0].split()[0])
            found.append((first, f"{area:.2f} mm²" if area is not None else f"{volume:.2f} mm³"))
    return found


ExportResult = namedtuple("ExportResult", "path layers resin_ml print_time problems log")


def export_goo(parts, printer, resin, supports, out_goo, slicer, uvtools, work, cancel=None):
    """Slice the plate at the printer's resolution and write the .goo, then check it: the settings the
    profiles ask for are in the file, and nothing floats. problems: [text], empty when the file is good.
    The file is written even with problems (to look at it in UVtools); the caller decides what to keep."""
    work = Path(work)
    sl1 = work / "plate.sl1"
    log = slice_plate(parts, slicer_config(printer, resin, supports), sl1, slicer, cancel)
    out_goo = Path(out_goo)
    run([uvtools, "convert", str(sl1), "goo", str(out_goo), "-v", str(printer["goo_version"])], cancel)
    found = goo_properties(out_goo, uvtools)
    problems = [f"Réglage {key}: attendu {want}, écrit {got}" for key, want, got in settings_differences(found, expected_goo(printer, resin))]
    pixel = printer["width"] / printer["pixels_x"]
    floating = islands(out_goo, uvtools, pixel, resin["layer_height"], cancel)
    if floating:
        first, size = floating[0]
        problems.append(f"{len(floating)} zone(s) sans appui (la première à la couche {first}, {size}): "
                        "ajoutez des supports ou tournez la pièce")
    if "No pad can be generated" in log:
        problems.append("Socle impossible: une pièce n'a aucun support")
    return ExportResult(out_goo, int(found.get("LayerCount", 0)), float(found.get("Volume", 0)) / 1000 if found.get("Volume") else 0.0,
                        float(found.get("PrintTime", 0)), problems, log)


def print_time(layers, resin):
    """Rough print time (s) before export: exposures, waits and the lift travel of each layer."""
    e = resin["elegoo"]
    bottom = min(layers, resin["bottom_layers"])

    def travel(prefix):
        up = e.get(f"{prefix}LiftHeight", 0) / max(e.get(f"{prefix}LiftSpeed", 60), 1) + \
            e.get(f"{prefix}LiftHeight2", 0) / max(e.get(f"{prefix}LiftSpeed2", 60), 1)
        down = (e.get(f"{prefix}LiftHeight", 0) + e.get(f"{prefix}LiftHeight2", 0) - e.get(f"{prefix}RetractHeight2", 0)) / \
            max(e.get(f"{prefix}RetractSpeed", 60), 1) + e.get(f"{prefix}RetractHeight2", 0) / max(e.get(f"{prefix}RetractSpeed2", 60), 1)
        return (up + down) * 60      # speeds are in mm / min
    waits = e.get("WaitTimeBeforeCure", 0) + e.get("WaitTimeAfterCure", 0) + e.get("WaitTimeAfterLift", 0) + e.get("LightOffDelay", 0)
    bottom_waits = e.get("BottomWaitTimeBeforeCure", 0) + e.get("BottomWaitTimeAfterCure", 0) + e.get("BottomWaitTimeAfterLift", 0) + e.get("BottomLightOffDelay", 0)
    return bottom * (resin["bottom_exposure_time"] + bottom_waits + travel("Bottom")) + \
        (layers - bottom) * (resin["exposure_time"] + waits + travel(""))


def load_json(path):
    return json.loads(Path(path).read_text())
