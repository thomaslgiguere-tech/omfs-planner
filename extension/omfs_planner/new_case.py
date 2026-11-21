"""Nouveau cas: one CBCT (DICOM) and the two intraoral scans are enough to start; everything else follows.

What "Démarrer" does, in order (surgeon, 2026-11-21: a streamlined start instead of three separate entries):
1. The two scan STLs are imported (raw coordinates, role Scan), named "Scan du haut" / "Scan du bas".
2. The AI segmentation (segmentation/segment.py, DentalSegmentator) starts in the background on the CBCT: Blender
   stays usable, the panel shows the elapsed time and the last line of its log. A segmentation already made for
   this CBCT (<folder>_probabilites.nrrd next to it) is reused instead (20-30 min saved in precise mode).
3. Meanwhile, optionally, the CBCT opens in the slice viewer (Coupes: axial / sagittal / coronal), read in a
   worker thread so the interface never freezes.
4. When the segmentation is ready, the Segmentation workspace opens on it (check, correct a piece, "Valider et
   importer"). The slices then show the AI surfaces as coloured contours on the gray image: the check.
5. On import, each scan gets its registration partners (upper scan -> "Dents du haut" + "Maxillaire", lower ->
   "Dents du bas" + "Mandibule"), and the Case panel points to the next step (Teeth Registration, then the
   reference frame...).

Patient data stays where it is: the segmentation is written next to the CBCT (as the DentalSeg app does), the
volume shown in the slices lives in memory only. The external tools (Python environment with SimpleITK /
nnU-Net, the segmentation folder with the model weights) are set in the extension's preferences; empty = this
machine's defaults (params.DICOM_PYTHON, the repo's segmentation folder).

Blender concepts: a timer (bpy.app.timers) runs a function every second without blocking the interface, which
is how the background job is watched; a worker thread reads the CBCT while you keep working. No Meshmixer
equivalent (Meshmixer imports meshes only, it has no CT or segmentation step).
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import bpy

from . import locking, mesh_io, params, registration, roles, scene_setup, view

SCAN_NAMES = {'UPPER': "Scan du haut", 'LOWER': "Scan du bas"}
# Structures of the Segmentation workspace (volume_scene.STRUCTURES) each scan registers onto: (teeth, bone).
PARTNERS = {'UPPER': ("Dents du haut", "Maxillaire"), 'LOWER': ("Dents du bas", "Mandibule")}
VOLUME_EXTS = (".nrrd", ".nii", ".nii.gz", ".mha", ".mhd")


# --- External tools -------------------------------------------------------------------------------

def _preferences():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def segmentation_python():
    """The preferences' choice, else the environment made by "Installer ce qui manque" (segmentation/.venv),
    else params.DICOM_PYTHON (a conda environment made by hand)."""
    prefs = _preferences()
    chosen = prefs.segmentation_python if prefs is not None else ""
    if chosen:
        return Path(os.path.expanduser(chosen))
    installed = segmentation_folder() / ".venv" / "bin" / "python"
    return installed if installed.exists() else Path(os.path.expanduser(params.DICOM_PYTHON))


def segmentation_folder():
    prefs = _preferences()
    chosen = prefs.segmentation_folder if prefs is not None else ""
    if chosen:
        return Path(os.path.expanduser(chosen))
    return Path(__file__).resolve().parents[2] / "segmentation"    # repo/extension/omfs_planner -> repo/segmentation


def tool_problem():
    """Why the segmentation can't run on this machine, or None."""
    python, folder = segmentation_python(), segmentation_folder()
    if not python.exists():
        return "Segmentation non installée : panneau Installation et mises à jour > Installer ce qui manque"
    if not (folder / "segment.py").exists():
        return f"segment.py introuvable dans {folder} (Préférences de l'extension)"
    if not (folder / "model" / "Dataset111_453CT").exists():
        return "Poids du modèle absents : panneau Installation et mises à jour > Installer ce qui manque"
    return None


# --- Inputs ---------------------------------------------------------------------------------------

def clean(path):
    return Path(bpy.path.abspath(path)).expanduser() if path else None


def segmentation_result(cbct):
    """Where segment.py writes the probabilities for this CBCT (same naming as segment.read_input)."""
    cbct = Path(cbct)
    name = cbct.name.lower()
    if cbct.is_file() and name.endswith(VOLUME_EXTS):
        base = cbct.name[:-7] if name.endswith(".nii.gz") else cbct.stem
        return cbct.parent / f"{base}_probabilites.nrrd"
    folder = cbct if cbct.is_dir() else cbct.parent
    return folder.parent / f"{folder.name}_probabilites.nrrd"


def input_problems(case):
    problems = []
    cbct = clean(case.cbct)
    if cbct is None or not cbct.exists():
        problems.append("Choisissez le dossier DICOM du CBCT")
    for label, path in (("du haut", case.upper_file), ("du bas", case.lower_file)):
        path = clean(path)
        if path is None or not path.is_file():
            problems.append(f"Choisissez le scan {label} (STL, OBJ ou PLY)")
        elif path.suffix.lower() not in mesh_io.IMPORTERS:
            problems.append(f"Scan {label} : format non pris en charge ({path.suffix})")
    return problems


def import_scan(scene, path, arch_id):
    """One intraoral scan, raw coordinates, role Scan, clear name. Replaces the previous one of this new case."""
    old = scene.omfs_case.upper if arch_id == 'UPPER' else scene.omfs_case.lower
    if old is not None and old.name in scene.objects and not old.omfs_reg.frozen:
        mesh = old.data
        bpy.data.objects.remove(old)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    objects = mesh_io.import_mesh(path)
    if not objects:
        raise RuntimeError(f"Rien d'importé depuis {Path(path).name}")
    obj = objects[0]
    for extra in objects[1:]:            # a scan file holds one mesh; anything else is not the arch
        bpy.data.objects.remove(extra)
    obj.name = SCAN_NAMES[arch_id]
    obj.data.name = obj.name
    obj.omfs.role = 'SCAN'
    roles.apply_role_color(obj)
    locking.set_locked(obj, False)
    obj.select_set(False)
    setattr(scene.omfs_case, "upper" if arch_id == 'UPPER' else "lower", obj)
    return obj


# --- Background segmentation ----------------------------------------------------------------------

class _Job:
    process = None        # subprocess.Popen of segment.py
    log = None            # its log file (temporary folder)
    scene = ""            # name of the planning scene that started it
    result = None         # expected probabilities .nrrd


class _Slices:
    thread = None
    volume = None
    error = ""
    area = 0              # as_pointer() of the 3D viewport the slices go in
    scene = ""


def running():
    return _Job.process is not None and _Job.process.poll() is None


def last_log_line():
    if _Job.log is None or not Path(_Job.log).exists():
        return ""
    try:
        lines = [line.strip() for line in Path(_Job.log).read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    if not lines:
        return ""
    line = lines[-1]
    return line.split("] ", 1)[-1] if line.startswith("[") else line


def start_segmentation(scene, cbct, precise):
    """Launch segment.py in the background (Blender stays usable). Its own process: it survives if Blender closes,
    and the result is then found next to the CBCT on the next "Démarrer"."""
    result = segmentation_result(cbct)
    command = [str(segmentation_python()), str(segmentation_folder() / "segment.py"), "--sans-ouvrir",
               "--sortie", str(result.parent), str(cbct)]
    if not precise:
        command.insert(2, "--rapide")
    log = Path(tempfile.gettempdir()) / f"omfs_segmentation_{os.getpid()}.log"
    with open(log, "w") as out:
        _Job.process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        cwd=str(segmentation_folder()), start_new_session=True)
    _Job.log, _Job.scene, _Job.result = str(log), scene.name, result
    case = scene.omfs_case
    case.status, case.started, case.result, case.message = 'RUNNING', time.time(), str(result), ""
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=1.0, persistent=True)


def cancel_segmentation():
    if running():
        _Job.process.terminate()


def _notify(message):
    if sys.platform == "darwin" and not bpy.app.background:
        script = f'display notification "{message}" with title "OMFS Planner" sound name "Glass"'
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _redraw_panels():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def finish(scene):
    """The job ended: ready (open the Segmentation workspace) or failed (keep the log's last line)."""
    case = scene.omfs_case
    code = _Job.process.returncode if _Job.process is not None else None
    if code == 0 and _Job.result is not None and Path(_Job.result).exists():
        case.status, case.message = 'READY', ""
        _notify("Segmentation prête : vérifiez-la puis importez")
        open_segmentation(scene)
    else:
        case.status = 'FAILED'
        case.message = last_log_line() or f"segment.py s'est arrêté (code {code})"
        _notify("Segmentation interrompue")
    _Job.process = None


def poll():
    """Timer, every second while the segmentation runs: progress in the panel, then finish."""
    scene = bpy.data.scenes.get(_Job.scene)
    if _Job.process is None or scene is None:
        return None
    if _Job.process.poll() is None:
        scene.omfs_case.message = last_log_line()
        _redraw_panels()
        return params.CASE_POLL_INTERVAL
    finish(scene)
    _redraw_panels()
    return None


def _view3d():
    """(window, area, region) of a 3D viewport, for operators run from a timer."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next(r for r in area.regions if r.type == 'WINDOW')
                return window, area, region
    return None


def open_segmentation(scene, context=None):
    """Open the result in the Segmentation workspace (volume_scene). From a timer too (no button context)."""
    path = scene.omfs_case.result
    if not path or not Path(path).exists():
        return False
    hit = _view3d()
    if hit is None:               # background (tests): no window, the operator still builds the scene
        bpy.ops.omfs.volume_open('EXEC_DEFAULT', filepath=path)
        return True
    window, area, region = hit
    window.scene = scene              # the plan becomes the Segmentation scene's source
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
        bpy.ops.omfs.volume_open('EXEC_DEFAULT', filepath=path)
    return True


# --- Slices, read in a worker thread --------------------------------------------------------------

def start_slices(context, cbct):
    from . import slice_viewer
    if context.area is None or context.area.type != 'VIEW_3D' or bpy.app.background:
        return False
    _Slices.volume, _Slices.error = None, ""
    _Slices.area, _Slices.scene = context.area.as_pointer(), context.scene.name

    def read():
        try:
            _Slices.volume = slice_viewer.load_volume(cbct)
        except Exception as error:          # shown in the panel; the rest of the case goes on
            _Slices.error = str(error)

    _Slices.thread = threading.Thread(target=read, daemon=True)
    _Slices.thread.start()
    bpy.app.timers.register(_slices_ready, first_interval=0.5)
    return True


def slices_loading():
    return _Slices.thread is not None and _Slices.thread.is_alive()


def _slices_ready():
    from . import slice_viewer
    if slices_loading():
        return 0.5
    _Slices.thread = None
    scene = bpy.data.scenes.get(_Slices.scene)
    if _Slices.error or _Slices.volume is None:
        if scene is not None:
            scene.omfs_case.slices_message = (_Slices.error or "lecture impossible")[:300]
        _redraw_panels()
        return None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.as_pointer() == _Slices.area:
                with bpy.context.temp_override(window=window, screen=window.screen, area=area):
                    slice_viewer.show_volume(bpy.context, window, area, _Slices.volume)
    _Slices.volume = None              # the viewer holds it now (memory only)
    _redraw_panels()
    return None


# --- After the Segmentation workspace's import ----------------------------------------------------

def structures_imported(scene, made):
    """Called by "Valider et importer": each scan of the new case gets its CBCT teeth and bone."""
    case = scene.omfs_case
    by_name = {}
    for obj in made:
        by_name.setdefault(obj.name.split(".")[0], obj)
    for arch_id, scan in (('UPPER', case.upper), ('LOWER', case.lower)):
        if scan is None or scan.name not in scene.objects:
            continue
        teeth, bone = (by_name.get(name) for name in PARTNERS[arch_id])
        props = scan.omfs_reg
        if teeth is not None:
            props.target = teeth
        if bone is not None:
            props.bone = bone
        # The click came from the Segmentation scene: the plan must be the context scene for the role checks.
        with bpy.context.temp_override(scene=scene):
            registration.default_partners(scene, scan)   # fills whatever is still empty
    if case.status in {'READY', 'RUNNING', 'NONE'}:
        case.status = 'IMPORTED'


# --- Steps shown in the Case panel ----------------------------------------------------------------

def next_steps(scene):
    """[(done, text)] of the first planning steps after a new case."""
    from . import reference_frame
    case = scene.omfs_case
    scans = [s for s in (case.upper, case.lower) if s is not None and s.name in scene.objects]
    has_bone = any(o.type == 'MESH' and o.omfs.role == 'BONE' for o in scene.objects)
    return [
        (len(scans) == 2, "Scans importés"),
        (has_bone, "Segmentation vérifiée et importée (os, dents, canal)"),
        (bool(scans) and all(s.omfs_reg.frozen for s in scans), "Recalage des scans : panneau Teeth Registration"),
        (reference_frame.find_frame(scene) is not None, "Repères et référentiel : panneaux Landmarks, Reference Frame"),
    ]


# --- Operators and properties ---------------------------------------------------------------------

class OMFS_OT_new_case(bpy.types.Operator):
    """Démarrer le cas : importe les deux scans, lance la segmentation du CBCT en arrière-plan (ou reprend celle
    déjà faite à côté du CBCT) et ouvre le CBCT en coupes. Blender reste utilisable pendant la segmentation"""
    bl_idname = "omfs.new_case"
    bl_label = "Démarrer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        case = scene.omfs_case
        problems = input_problems(case)
        if problems:
            self.report({'ERROR'}, " ; ".join(problems))
            return {'CANCELLED'}
        if running():
            self.report({'ERROR'}, "Une segmentation est déjà en cours (Annuler d'abord)")
            return {'CANCELLED'}
        cbct = clean(case.cbct)
        result = segmentation_result(cbct)
        reuse = case.reuse and result.exists()
        problem = None if reuse else tool_problem()
        if problem:
            self.report({'ERROR'}, problem)
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        upper = import_scan(scene, clean(case.upper_file), 'UPPER')
        lower = import_scan(scene, clean(case.lower_file), 'LOWER')
        scene_setup.setup_scene(context)
        scene_setup.frame_all(context)
        if not bpy.app.background:
            view.activate_navigate_tool(context)
        case.slices_message = ""
        if case.open_slices:
            start_slices(context, cbct)
        if reuse:
            case.status, case.result, case.message = 'READY', str(result), ""
            open_segmentation(scene, context)
            what = "segmentation existante ouverte"
        else:
            start_segmentation(scene, cbct, case.mode == 'PRECISE')
            what = f"segmentation lancée ({'précis' if case.mode == 'PRECISE' else 'rapide'})"
        self.report({'INFO'}, f"{upper.name} et {lower.name} importés ; {what}")
        return {'FINISHED'}


class OMFS_OT_case_cancel(bpy.types.Operator):
    """Arrêter la segmentation en cours"""
    bl_idname = "omfs.case_cancel"
    bl_label = "Annuler la segmentation"

    def execute(self, context):
        cancel_segmentation()
        return {'FINISHED'}


class OMFS_OT_case_open_segmentation(bpy.types.Operator):
    """Ouvrir la segmentation de ce cas pour la vérifier, la corriger et l'importer"""
    bl_idname = "omfs.case_open_segmentation"
    bl_label = "Vérifier et importer la segmentation"

    def execute(self, context):
        if not open_segmentation(context.scene, context):
            self.report({'ERROR'}, "Fichier de segmentation introuvable")
            return {'CANCELLED'}
        return {'FINISHED'}


class OMFS_OT_case_arches(bpy.types.Operator):
    """Prendre les deux scans du nouveau cas comme arcades du haut et du bas"""
    bl_idname = "omfs.case_arches"
    bl_label = "Arcades du cas (Scan du haut / Scan du bas)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        case = context.scene.omfs_case
        return case.upper is not None and case.lower is not None

    def execute(self, context):
        case, props = context.scene.omfs_case, context.scene.omfs_occlusion
        props.upper, props.lower = case.upper, case.lower
        return {'FINISHED'}


def _poll_scan(props, obj):
    return obj.type == 'MESH' and obj.omfs.role == 'SCAN'


class OMFS_CaseProperties(bpy.types.PropertyGroup):
    cbct: bpy.props.StringProperty(name="CBCT", subtype='DIR_PATH',
                                   description="Dossier DICOM du CBCT (ou un .dcm de la série, ou un .nrrd / .nii)")
    upper_file: bpy.props.StringProperty(name="Scan du haut", subtype='FILE_PATH',
                                         description="Empreinte optique de l'arcade du haut (STL, OBJ, PLY)")
    lower_file: bpy.props.StringProperty(name="Scan du bas", subtype='FILE_PATH',
                                         description="Empreinte optique de l'arcade du bas (STL, OBJ, PLY)")
    mode: bpy.props.EnumProperty(
        name="Segmentation", default=params.CASE_SEGMENTATION_MODE,
        items=(('PRECISE', "Précis", "Le volume vu 8 fois en miroir : ~20-30 min (M5 Pro). Le plus sûr sur un "
                                     "CBCT grossier ou à coupes épaisses"),
               ('FAST', "Rapide", "Un seul passage : ~3-5 min (M5 Pro). Quelques centièmes de mm de différence "
                                  "sur un CBCT fin et isotrope")))
    reuse: bpy.props.BoolProperty(name="Réutiliser la segmentation déjà faite", default=True,
                                  description="Une segmentation de ce CBCT existe à côté de lui : l'ouvrir au lieu "
                                              "de la refaire")
    open_slices: bpy.props.BoolProperty(name="Ouvrir le CBCT en coupes", default=True,
                                        description="Axiale, sagittale, coronale et 3D (lecture en arrière-plan). "
                                                    "Les surfaces apparaissent en contours sur les coupes")
    upper: bpy.props.PointerProperty(name="Scan du haut", type=bpy.types.Object, poll=_poll_scan)
    lower: bpy.props.PointerProperty(name="Scan du bas", type=bpy.types.Object, poll=_poll_scan)
    status: bpy.props.EnumProperty(items=(('NONE', "", ""), ('RUNNING', "En cours", ""), ('READY', "Prête", ""),
                                          ('FAILED', "Échec", ""), ('IMPORTED', "Importée", "")))
    started: bpy.props.FloatProperty()
    result: bpy.props.StringProperty()
    message: bpy.props.StringProperty()
    slices_message: bpy.props.StringProperty()


classes = (OMFS_CaseProperties, OMFS_OT_new_case, OMFS_OT_case_cancel, OMFS_OT_case_open_segmentation,
           OMFS_OT_case_arches)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_case = bpy.props.PointerProperty(type=OMFS_CaseProperties)


def unregister():
    for timer in (poll, _slices_ready):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    del bpy.types.Scene.omfs_case
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
