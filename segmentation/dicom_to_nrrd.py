"""
DICOM → .nrrd en niveaux de gris, pour le visualiseur de coupes d'OMFS Planner (et pour tout le reste : c'est le
format pivot de l'étape 1 du « 3 en 1 »). Espacement, origine et orientation gardés, en LPS comme 3D Slicer.

Même lecture de série que segment.py (SimpleITK / GDCM : la plus grande série du dossier, ou celle du .dcm
déposé ; pente et ordonnée de la remise à l'échelle DICOM appliquées). Le volume n'est pas modifié.

Usage (environnement dentalseg) :
    ~/miniconda3/envs/dentalseg/bin/python segmentation/dicom_to_nrrd.py <dossier DICOM | .dcm | volume> [--sortie fichier.nrrd]

La dernière ligne écrite est « NRRD: <chemin> » (lue par Blender).
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment import log, read_input  # noqa: E402


def convert(path: Path, out: Path = None) -> Path:
    image, base, out_folder = read_input(path)
    out = out or out_folder / f"{base}_volume.nrrd"
    log(f"Volume {image.GetSize()} voxels, espacement {tuple(round(s, 4) for s in image.GetSpacing())} mm, "
        f"type {image.GetPixelIDTypeAsString()}")
    sitk.WriteImage(image, str(out), useCompression=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entree", type=Path)
    parser.add_argument("--sortie", type=Path, default=None, help="fichier .nrrd (par défaut : à côté du scan)")
    args = parser.parse_args()
    out = convert(args.entree, args.sortie)
    print(f"NRRD: {out}", flush=True)


if __name__ == "__main__":
    main()
