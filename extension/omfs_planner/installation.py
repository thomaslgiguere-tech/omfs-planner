"""Installation et mises à jour: what this computer has, one button to install what is missing, one to update.

The planning itself needs nothing but Blender (numpy, openvdb, OpenImageIO ship with it). The extras:
- segmentation and DICOM reading (Nouveau cas, Coupes): a Python 3.12 environment with PyTorch / nnU-Net /
  SimpleITK and the DentalSegmentator weights. "Installer ce qui manque" runs segmentation/install.py in the
  background (uv, segmentation/.venv, exact versions, weights checked by SHA-256); progress shown here;
- printing (Impression tab): PrusaSlicer and UVtools, Mac applications: a download link, found once installed;
- updates: the project folder is a git clone; "Rechercher" fetches, "Mettre à jour" pulls (fast-forward only:
  never overwrites a local change), then Blender is restarted.

Shown in the OMFS tab (panel "Installation et mises à jour") and in the extension's preferences.
Blender concept: like Nouveau cas, a background process watched by a timer (the interface never freezes). No
Meshmixer equivalent (Meshmixer updates itself and has no optional components).
"""

import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import bpy

from . import print_chain

CHECKPOINT = Path("model", "Dataset111_453CT", "nnUNetTrainer__nnUNetPlans__3d_fullres", "fold_0",
                  "checkpoint_final.pth")


def repo():
    return Path(__file__).resolve().parents[2]


def _prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


# --- What this computer has -------------------------------------------------------------------------

def _command_line_tools():
    try:
        return subprocess.run(["xcode-select", "-p"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_MACHINE = {}


def machine():
    """Cached once: (Apple silicon, macOS major version, git available)."""
    if not _MACHINE:
        version = platform.mac_ver()[0] if sys.platform == "darwin" else ""
        _MACHINE.update(apple=sys.platform == "darwin" and platform.machine() == "arm64",
                        macos=int(version.split(".")[0]) if version else 0, version=version,
                        git=sys.platform != "darwin" or _command_line_tools())
    return _MACHINE


def _tool(name, custom):
    try:
        return print_chain.find_tool(name, custom)
    except print_chain.ToolMissing:
        return None


def components():
    """[(key, label, ok, detail, how to get it: 'install' | url | text | None)] for the checklist."""
    from . import new_case
    m = machine()
    folder = new_case.segmentation_folder()
    python = new_case.segmentation_python()
    prefs = _prefs()
    slicer = _tool("PrusaSlicer", prefs.prusaslicer_path if prefs else "")
    uvtools = _tool("UVtools", prefs.uvtools_path if prefs else "")
    computer_ok = m["apple"] and m["macos"] >= 14
    return [
        ("blender", "Blender", True, bpy.app.version_string, None),
        ("computer", "Puce Apple, macOS 14+", computer_ok,
         f"macOS {m['version'] or '?'}" + ("" if m["apple"] else ", pas de puce Apple"),
         None if computer_ok else "Segmentation impossible sur cet ordinateur (importer des STL segmentés)"),
        ("python", "Environnement de segmentation", python.exists(), str(python), 'install'),
        ("model", "Poids DentalSegmentator", (folder / CHECKPOINT).exists(), str(folder / "model"), 'install'),
        ("git", "git (mises à jour)", m["git"] and (repo() / ".git").exists(),
         "dossier cloné avec git" if (repo() / ".git").exists() else "pas un clone git",
         "Terminal : xcode-select --install" if not m["git"] else None),
        ("prusaslicer", "PrusaSlicer (impression)", slicer is not None, slicer or "", print_chain.DOWNLOADS["PrusaSlicer"]),
        ("uvtools", "UVtools (impression)", uvtools is not None, uvtools or "", print_chain.DOWNLOADS["UVtools"]),
    ]


def segmentation_missing():
    return any(not ok for key, _l, ok, _d, _h in components() if key in {"python", "model"})


# --- Install job (segmentation/install.py in the background) ----------------------------------------

class _Install:
    process = None
    log = None
    started = 0.0
    result = ""          # "" / "TERMINÉ" / "ÉCHEC : ..."


def installing():
    return _Install.process is not None and _Install.process.poll() is None


def _log_lines():
    if _Install.log is None or not Path(_Install.log).exists():
        return []
    try:
        return [line.strip() for line in Path(_Install.log).read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return []


def install_progress():
    """(step text, percent of the current download or None, last line)."""
    lines = _log_lines()
    steps = [line for line in lines if line.startswith("ÉTAPE")]
    advance = [line for line in lines if line.startswith("AVANCE")]
    last = next((line for line in reversed(lines) if not line.startswith("AVANCE")), "")
    percent = int(advance[-1].split()[1]) if advance and lines[-1].startswith("AVANCE") else None
    return (steps[-1] if steps else "démarrage…"), percent, last


def start_install():
    from . import new_case
    script = new_case.segmentation_folder() / "install.py"
    if not script.exists():
        raise RuntimeError(f"install.py introuvable dans {script.parent} (Préférences de l'extension)")
    log = Path(tempfile.gettempdir()) / "omfs_installation.log"
    with open(log, "w") as out:
        _Install.process = subprocess.Popen([sys.executable, "-I", str(script)], stdout=out, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
    _Install.log, _Install.started, _Install.result = str(log), time.time(), ""
    if not bpy.app.timers.is_registered(_watch):
        bpy.app.timers.register(_watch, first_interval=1.0, persistent=True)


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _watch():
    if _Install.process is None:
        return None
    if _Install.process.poll() is None:
        _redraw()
        return 1.0
    lines = _log_lines()
    _Install.result = lines[-1] if lines and (lines[-1] == "TERMINÉ" or lines[-1].startswith("ÉCHEC")) \
        else f"ÉCHEC : l'installation s'est arrêtée (code {_Install.process.returncode})"
    _Install.process = None
    _redraw()
    return None


# --- Updates (git) ----------------------------------------------------------------------------------

class _Update:
    thread = None
    behind = None         # commits available, None = not checked
    message = ""
    pulled = False


def _git(*args, timeout=60):
    return subprocess.run(["git", "-C", str(repo()), *args], capture_output=True, text=True, timeout=timeout)


def _check():
    try:
        fetched = _git("fetch", "--quiet")
        if fetched.returncode != 0:
            _Update.message = "Pas de connexion au dépôt : " + (fetched.stderr.strip().splitlines() or ["?"])[-1]
            return
        count = _git("rev-list", "--count", "HEAD..@{upstream}")
        if count.returncode != 0:
            _Update.message = "Ce dossier ne suit aucune branche distante"
            return
        _Update.behind = int(count.stdout.strip() or 0)
        _Update.message = ""
    except (OSError, subprocess.TimeoutExpired) as error:
        _Update.message = f"git indisponible : {error}"


def _pull():
    try:
        done = _git("pull", "--ff-only", timeout=300)
        if done.returncode == 0:
            _Update.behind, _Update.pulled, _Update.message = 0, True, ""
        else:
            _Update.message = ("Mise à jour refusée (des fichiers ont été modifiés dans ce dossier) : "
                               + (done.stderr.strip().splitlines() or ["?"])[-1])
    except (OSError, subprocess.TimeoutExpired) as error:
        _Update.message = f"git indisponible : {error}"


def _run_update(target):
    _Update.thread = threading.Thread(target=target, daemon=True)
    _Update.thread.start()

    def wait():
        if _Update.thread is not None and _Update.thread.is_alive():
            return 0.5
        _Update.thread = None
        _redraw()
        return None
    bpy.app.timers.register(wait, first_interval=0.5)


def updating():
    return _Update.thread is not None and _Update.thread.is_alive()


# --- Drawing (OMFS tab and preferences) -------------------------------------------------------------

def draw(layout, context):
    from .ui import draw_text
    col = layout.column(align=True)
    for key, label, ok, detail, how in components():
        row = col.row(align=True)
        row.label(text=label, icon='CHECKMARK' if ok else 'ERROR')
        if ok:
            row.label(text=Path(detail).name if "/" in detail else detail)
        elif how and how.startswith("http"):
            row.operator("wm.url_open", text="Télécharger", icon='URL').url = how
        elif how and how != 'install':
            draw_text(col, context, how)

    box = layout.box()
    if installing():
        text, percent, last = install_progress()
        minutes, seconds = divmod(int(time.time() - _Install.started), 60)
        box.label(text=f"{text}  ·  {minutes} min {seconds:02d} s", icon='SORTTIME')
        if percent is not None:
            box.progress(factor=percent / 100.0, text=f"{percent} %")
        if last and last != text:
            draw_text(box, context, last[-160:])
    elif segmentation_missing():
        draw_text(box, context, "Segmentation et lecture DICOM : environnement Python (~2 Go) et poids du modèle "
                                "(230 Mo), téléchargés une seule fois. Blender reste utilisable pendant ce temps.")
        row = box.row()
        row.scale_y = 1.4
        row.enabled = machine()["apple"] and machine()["macos"] >= 14
        row.operator("omfs.install_missing", icon='IMPORT')
    else:
        box.label(text="Segmentation prête", icon='CHECKMARK')
    if _Install.result:
        row = box.row()
        row.alert = _Install.result.startswith("ÉCHEC")
        draw_text(row, context, "Installation terminée" if _Install.result == "TERMINÉ" else _Install.result)

    if (repo() / ".git").exists() and machine()["git"]:
        box = layout.box()
        box.label(text="Mises à jour", icon='FILE_REFRESH')
        if updating():
            box.label(text="git en cours…", icon='SORTTIME')
        elif _Update.pulled:
            alert = box.row()
            alert.alert = True
            alert.label(text="Mis à jour : redémarrez Blender", icon='INFO')
        else:
            if _Update.behind is not None:
                box.label(text=f"{_Update.behind} mise(s) à jour disponible(s)" if _Update.behind else
                          "À jour", icon='INFO' if _Update.behind else 'CHECKMARK')
            row = box.row(align=True)
            row.operator("omfs.check_updates", icon='VIEWZOOM')
            if _Update.behind:
                row.operator("omfs.pull_updates", icon='IMPORT')
        if _Update.message:
            row = box.row()
            row.alert = True
            draw_text(row, context, _Update.message)


# --- Operators and panel ----------------------------------------------------------------------------

class OMFS_OT_install_missing(bpy.types.Operator):
    """Installer l'environnement de segmentation (Python 3.12, PyTorch, nnU-Net, SimpleITK) et les poids du modèle,
    en arrière-plan. Une seule fois par ordinateur ; ce qui est déjà là est sauté"""
    bl_idname = "omfs.install_missing"
    bl_label = "Installer ce qui manque"

    def execute(self, context):
        if installing():
            return {'CANCELLED'}
        try:
            start_install()
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class OMFS_OT_check_updates(bpy.types.Operator):
    """Chercher une nouvelle version d'OMFS Planner sur GitHub"""
    bl_idname = "omfs.check_updates"
    bl_label = "Rechercher"

    def execute(self, context):
        if not updating():
            _run_update(_check)
        return {'FINISHED'}


class OMFS_OT_pull_updates(bpy.types.Operator):
    """Télécharger la nouvelle version (git pull, jamais par-dessus une modification locale), puis redémarrer
    Blender. Les cas des patients ne sont pas dans ce dossier : jamais touchés"""
    bl_idname = "omfs.pull_updates"
    bl_label = "Mettre à jour"

    def execute(self, context):
        if not updating():
            _run_update(_pull)
        return {'FINISHED'}


class OMFS_PT_installation(bpy.types.Panel):
    bl_label = "Installation et mises à jour"
    bl_idname = "OMFS_PT_installation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_order = 100
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        if segmentation_missing() and not installing():
            self.layout.label(icon='ERROR')

    def draw(self, context):
        draw(self.layout, context)


classes = (OMFS_OT_install_missing, OMFS_OT_check_updates, OMFS_OT_pull_updates, OMFS_PT_installation)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    if bpy.app.timers.is_registered(_watch):
        bpy.app.timers.unregister(_watch)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
