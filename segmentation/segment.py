"""
DentalSeg : segmente un CBCT avec le modèle DentalSegmentator (nnUNet, Dataset111_453CT)
sur le GPU de la puce Apple (MPS), puis écrit un .seg.nrrd importable dans 3D Slicer.

Mode précis (par défaut, plus long) : augmentation au moment de l'inférence (le volume vu 8 fois en
miroir, probabilités moyennées), fenêtres glissantes plus chevauchées (step 0,35 au lieu de 0,5), et
les probabilités de chaque structure gardées (<nom>_probabilites.nrrd) pour des surfaces sous le voxel
(surfaces.py). Mode --rapide : l'ancien réglage (sans augmentation, step 0,5, sans probabilités).

Usage : python segment.py [--rapide] [--step 0.35] [--sortie DOSSIER] <dossier DICOM | .dcm | .nrrd | .nii(.gz) | .mha> [...]
"""
import argparse
import os
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import nrrd
from scipy import ndimage

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
DATASET = "Dataset111_453CT"
MIN_ISLAND_MM3 = 60  # même seuil que DentalSegmentator dans Slicer
PRECISE_STEP = 0.35  # chevauchement des fenêtres en mode précis (0,5 = réglage nnU-Net par défaut)

# Mêmes noms, couleurs et étiquettes que DentalSegmentator dans Slicer
SEGMENTS = [
    (1, "Maxilla & Upper Skull", "#E3DD90"),
    (2, "Mandible", "#D4A1E6"),
    (3, "Upper Teeth", "#DC9565"),
    (4, "Lower Teeth", "#EBDFB4"),
    (5, "Mandibular canal", "#D8654F"),
]
REMOVE_ISLANDS = {1, 2, 3, 4}  # pas le canal mandibulaire, comme dans Slicer

VOLUME_EXTS = (".nrrd", ".nii", ".nii.gz", ".mha", ".mhd")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_input(path: Path):
    """Retourne (image SimpleITK, nom de base, dossier de sortie)."""
    name = path.name
    if path.is_file() and name.lower().endswith(VOLUME_EXTS):
        base = name[:-7] if name.lower().endswith(".nii.gz") else path.stem
        return sitk.ReadImage(str(path)), base, path.parent

    folder = path if path.is_dir() else path.parent
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder)) or []
    if not series_ids:
        raise RuntimeError(f"Aucune série DICOM trouvée dans {folder}")

    files_by_series = {sid: reader.GetGDCMSeriesFileNames(str(folder), sid) for sid in series_ids}
    if path.is_file():
        # Fichier .dcm déposé : on prend la série qui le contient
        chosen = next((sid for sid, files in files_by_series.items()
                       if str(path) in files), None)
    else:
        chosen = None
    if chosen is None:
        chosen = max(files_by_series, key=lambda sid: len(files_by_series[sid]))
    files = files_by_series[chosen]
    log(f"Série DICOM : {len(files)} coupes ({len(series_ids)} série(s) dans le dossier)")
    reader.SetFileNames(files)
    image = reader.Execute()
    return image, folder.name, folder.parent


def run_nnunet(in_file: Path, out_dir: Path, tta: bool, step: float, probabilities: bool):
    env = dict(os.environ,
               nnUNet_results=str(MODEL_DIR), nnUNet_raw=str(MODEL_DIR), nnUNet_preprocessed=str(MODEL_DIR),
               PYTORCH_ENABLE_MPS_FALLBACK="1")
    predict = Path(sys.executable).parent / "nnUNetv2_predict"
    cmd = [str(predict), "-i", str(in_file.parent), "-o", str(out_dir),
           "-d", DATASET, "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "-c", "3d_fullres",
           "-f", "0", "-chk", "checkpoint_final.pth", "-step_size", str(step),
           "-npp", "1", "-nps", "1", "-device", "mps"]
    if not tta:
        cmd.append("--disable_tta")
    if probabilities:
        cmd.append("--save_probabilities")
    subprocess.run(cmd, env=env, check=True)


def write_probabilities(npz: Path, reference: sitk.Image, out_path: Path):
    """Probabilités des 5 structures (sans le fond) dans un .nrrd 4D, 0-255, même géométrie que le scan :
    la surface d'une structure passe là où sa probabilité vaut 50 %, entre les voxels (surfaces.py)."""
    probs = np.load(npz)["probabilities"]                     # (classes, z, y, x), classe 0 = fond
    data = np.clip(np.rint(probs[1:].astype(np.float32) * 255), 0, 255).astype(np.uint8)
    data = np.transpose(data, (0, 3, 2, 1))                   # (structure, x, y, z) pour NRRD
    direction = np.array(reference.GetDirection()).reshape(3, 3)
    spacing = np.array(reference.GetSpacing())
    header = {
        "type": "uint8", "dimension": 4, "space": "left-posterior-superior",
        "space directions": np.vstack(([np.nan] * 3, (direction * spacing).T)),   # NaN : l'axe des structures
        "space origin": np.array(reference.GetOrigin()),
        "kinds": ["list", "domain", "domain", "domain"], "encoding": "gzip",
        "labels": ['"structure"', '""', '""', '""'],
    }
    for value, name, _hex in SEGMENTS:
        header[f"structure_{value - 1}"] = name
    nrrd.write(str(out_path), data, header)


def remove_small_islands(labels: np.ndarray, voxel_mm3: float):
    min_voxels = int(np.ceil(MIN_ISLAND_MM3 / voxel_mm3))
    for value in REMOVE_ISLANDS:
        mask = labels == value
        if not mask.any():
            continue
        comps, n = ndimage.label(mask)
        if n <= 1:
            continue
        sizes = np.bincount(comps.ravel())
        small = np.where(sizes < min_voxels)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(comps, small)] = 0


def write_seg_nrrd(seg: sitk.Image, out_path: Path):
    """Écrit un .seg.nrrd (une couche, une valeur par segment) lisible directement par Slicer."""
    arr = sitk.GetArrayFromImage(seg).astype(np.uint8)  # (z, y, x)
    remove_small_islands(arr, float(np.prod(seg.GetSpacing())))
    data = np.transpose(arr, (2, 1, 0))  # (x, y, z) pour NRRD

    direction = np.array(seg.GetDirection()).reshape(3, 3)
    spacing = np.array(seg.GetSpacing())
    header = {
        "type": "uint8",
        "dimension": 3,
        "space": "left-posterior-superior",  # SimpleITK travaille en LPS, comme les fichiers de Slicer
        "space directions": (direction * spacing).T,  # une ligne par axe i, j, k
        "space origin": np.array(seg.GetOrigin()),
        "kinds": ["domain", "domain", "domain"],
        "encoding": "gzip",
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ContainedRepresentationNames": "Binary labelmap|",
        "Segmentation_ReferenceImageExtentOffset": "0 0 0",
    }

    i = 0
    for value, name, hex_color in SEGMENTS:
        idx = np.argwhere(data == value)
        if idx.size == 0:
            continue
        lo, hi = idx.min(axis=0), idx.max(axis=0)
        rgb = [int(hex_color[k:k + 2], 16) / 255 for k in (1, 3, 5)]
        header.update({
            f"Segment{i}_ID": f"Segment_{value}",
            f"Segment{i}_Name": name,
            f"Segment{i}_NameAutoGenerated": "0",
            f"Segment{i}_Color": " ".join(f"{c:.6f}" for c in rgb),
            f"Segment{i}_ColorAutoGenerated": "0",
            f"Segment{i}_LabelValue": str(value),
            f"Segment{i}_Layer": "0",
            f"Segment{i}_Extent": f"{lo[0]} {hi[0]} {lo[1]} {hi[1]} {lo[2]} {hi[2]}",
            f"Segment{i}_Tags": "|",
        })
        i += 1
    nrrd.write(str(out_path), data, header)
    return i


def process(path: Path, fast: bool = False, step: float = PRECISE_STEP, out_dir: Path = None):
    start = time.time()
    tta, probabilities = not fast, not fast
    step = 0.5 if fast else step
    log(f"Lecture de {path}")
    image, base, out_folder = read_input(path)
    out_folder = out_dir or out_folder
    log(f"Volume {image.GetSize()} voxels, espacement {tuple(round(s, 3) for s in image.GetSpacing())} mm")

    work = Path(tempfile.mkdtemp(prefix="dentalseg_"))
    try:
        (work / "in").mkdir()
        in_file = work / "in" / "volume_0000.nii.gz"
        sitk.WriteImage(image, str(in_file))

        mode = "rapide" if fast else f"précis (augmentation x8, step {step})"
        log(f"Segmentation nnUNet sur le GPU (MPS), mode {mode}…")
        run_nnunet(in_file, work / "out", tta, step, probabilities)

        seg = sitk.ReadImage(str(work / "out" / "volume.nii.gz"))
        out_path = out_folder / f"{base}_segmentation.seg.nrrd"
        log("Nettoyage des petits îlots et écriture du .seg.nrrd…")
        n = write_seg_nrrd(seg, out_path)
        if probabilities:
            log("Écriture des probabilités (surfaces sous le voxel)…")
            try:
                write_probabilities(work / "out" / "volume.npz", seg, out_folder / f"{base}_probabilites.nrrd")
            except Exception:
                kept = out_folder / f"{base}_probabilites_brutes.npz"   # le calcul est long : ne jamais le perdre
                shutil.copy(work / "out" / "volume.npz", kept)
                log(f"Probabilités gardées telles quelles : {kept}")
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)

    minutes, seconds = divmod(int(time.time() - start), 60)
    log(f"Terminé en {minutes} min {seconds} s : {n} segments → {out_path}")
    return out_path


def notify(title, message):
    script = f'display notification {message!r} with title {title!r} sound name "Glass"'
    subprocess.run(["osascript", "-e", script.replace("'", '"')], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--rapide", action="store_true", help="ancien réglage : sans augmentation, step 0,5")
    parser.add_argument("--step", type=float, default=PRECISE_STEP, help="chevauchement des fenêtres (mode précis)")
    parser.add_argument("--sortie", type=Path, default=None, help="dossier de sortie (par défaut : à côté du scan)")
    parser.add_argument("--sans-ouvrir", action="store_true", help="ne pas montrer le résultat dans le Finder")
    args = parser.parse_args()
    if not args.inputs:
        print(__doc__)
        sys.exit(1)
    failures = 0
    for arg in args.inputs:
        try:
            out = process(Path(arg).expanduser().resolve(), args.rapide, args.step,
                          args.sortie.expanduser().resolve() if args.sortie else None)
            if not args.sans_ouvrir:
                notify("DentalSeg", f"Segmentation prête : {out.name}")
                subprocess.run(["open", "-R", str(out)], check=False)
        except Exception as e:
            failures += 1
            log(f"ERREUR pour {arg} : {e}")
            notify("DentalSeg", f"Échec : {Path(arg).name}")
    sys.exit(1 if failures else 0)
