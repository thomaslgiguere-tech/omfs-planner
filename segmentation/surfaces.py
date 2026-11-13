"""
Surfaces : étape 3 du "3 en 1". Chaque structure segmentée devient une surface fermée (STL binaire, en mm,
coordonnées du patient LPS comme les exports de 3D Slicer), prête pour OMFS Planner.

- Depuis un .seg.nrrd (étiquettes) : la surface suit les voxels (marches d'escalier d'un demi-voxel).
- Depuis un _probabilites.nrrd (segment.py, mode précis) : la surface passe là où la probabilité vaut 50 %,
  entre les voxels (sous le voxel), sans lissage artificiel.

Usage :
  python surfaces.py <fichier .seg.nrrd | _probabilites.nrrd> --sortie DOSSIER
  python surfaces.py --comparer A.stl B.stl      (écart en mm entre deux surfaces)
"""
import argparse
import struct
from pathlib import Path

import nrrd
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

NAMES = ["Maxilla & Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]
FILE_NAMES = ["Maxillaire", "Mandibule", "Dents du haut", "Dents du bas", "Canal mandibulaire"]
ISLAND_MARGIN = 2   # voxels autour de la structure nettoyée (segment.py) où la probabilité est gardée


def geometry(header, spatial_axes):
    """(directions 3 x 3 : une colonne par axe i, j, k, avec l'espacement ; origine), en LPS."""
    directions = [np.array(d, float) for d in header["space directions"] if d is not None and not
                  (isinstance(d, float) and np.isnan(d))]
    matrix = np.stack(directions[-spatial_axes:], axis=1)
    return matrix, np.array(header["space origin"], float)


def to_patient(verts_ijk, matrix, origin):
    return verts_ijk @ matrix.T + origin


def surface(volume, level, matrix, origin):
    """Isosurface fermée d'un volume (x, y, z) au niveau donné, en coordonnées patient, normales vers
    l'extérieur. Recadrée sur la structure pour la vitesse ; bordée de zéros pour rester fermée."""
    inside = np.argwhere(volume > level)
    if not len(inside):
        return None, None
    lo = np.maximum(inside.min(0) - 2, 0)
    hi = np.minimum(inside.max(0) + 3, volume.shape)
    crop = np.pad(volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32), 1)
    verts, faces, _n, _v = measure.marching_cubes(crop, level, allow_degenerate=False)
    verts = verts - 1 + lo                                  # indices dans le volume complet
    if np.linalg.det(matrix) < 0:                           # repère indirect : garder les normales dehors
        faces = faces[:, ::-1]
    return to_patient(verts, matrix, origin), faces


def write_stl(path, verts, faces):
    tri = verts[faces].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    with open(path, "wb") as f:
        f.write(b"OMFS Planner surfaces.py".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        records = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        records["n"], records["v"] = normals, tri
        f.write(records.tobytes())


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    records = np.frombuffer(data[84:84 + 50 * count], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return records["v"].reshape(-1, 3).astype(np.float64)


def compare(a, b):
    """Écart entre deux surfaces (mm), dans les deux sens : moyenne, 95e centile, maximum. Approché par
    les sommets les plus proches (maillages denses : erreur bien sous le demi-voxel)."""
    pa, pb = np.unique(read_stl(a), axis=0), np.unique(read_stl(b), axis=0)
    d = np.concatenate((cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0]))
    return float(d.mean()), float(np.percentile(d, 95)), float(d.max())


def extract(path, out_dir):
    data, header = nrrd.read(str(path))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if data.ndim == 4:                                      # probabilités : (structure, x, y, z), 0-255
        matrix, origin = geometry(header, 3)
        labels = Path(str(path).replace("_probabilites.nrrd", "_segmentation.seg.nrrd"))
        if labels.exists():   # seulement autour de la structure nettoyée : pas les petits îlots qu'elle a perdus
            seg, _h = nrrd.read(str(labels))
            volumes = [(k, np.where(ndimage.binary_dilation(seg == k + 1, iterations=ISLAND_MARGIN), data[k], 0), 127.5)
                       for k in range(data.shape[0])]
        else:
            volumes = [(k, data[k], 127.5) for k in range(data.shape[0])]
        suffix = "probabilites"
    else:                                                   # étiquettes 1-5
        matrix, origin = geometry(header, 3)
        volumes = [(k, (data == k + 1).astype(np.uint8), 0.5) for k in range(5)]
        suffix = "etiquettes"
    for k, volume, level in volumes:
        verts, faces = surface(volume, level, matrix, origin)
        if verts is None:
            continue
        out = out_dir / f"{FILE_NAMES[k]} ({suffix}).stl"
        write_stl(out, verts, faces)
        written.append((out, len(faces)))
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--sortie", type=Path)
    parser.add_argument("--comparer", nargs=2, type=Path, metavar=("A", "B"))
    args = parser.parse_args()
    if args.comparer:
        mean, p95, worst = compare(*args.comparer)
        print(f"Écart moyen {mean:.3f} mm, 95 % sous {p95:.3f} mm, max {worst:.3f} mm")
    elif args.source:
        for out, n in extract(args.source, args.sortie or args.source.parent):
            print(f"{out.name}: {n} triangles")
    else:
        print(__doc__)
