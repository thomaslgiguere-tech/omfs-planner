"""
Compare deux segmentations DentalSeg du même scan (ex. mode précis classique vs accéléré).

Usage : python comparer_segmentations.py A_segmentation.seg.nrrd B_segmentation.seg.nrrd
        [--proba A_probabilites.nrrd B_probabilites.nrrd]

Donne : % de voxels identiques, Dice par structure, voxels passés du maxillaire à la mandibule (ou l'inverse),
et, avec --proba, l'écart des probabilités et les voxels qui changent de côté du seuil de 50 % (la surface).
"""
import argparse
import sys

import nrrd
import numpy as np

NAMES = {1: "Maxillaire", 2: "Mandibule", 3: "Dents sup.", 4: "Dents inf.", 5: "Canal mand."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--proba", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()

    a, ha = nrrd.read(args.a)
    b, hb = nrrd.read(args.b)
    if a.shape != b.shape or not np.allclose(ha["space directions"], hb["space directions"]):
        sys.exit("Les deux segmentations n'ont pas la même géométrie : pas le même scan ?")

    print(f"Voxels identiques : {100 * np.mean(a == b):.3f} %")
    print(f"{'Structure':<13} {'Dice':>7} {'Vox. A':>10} {'Vox. B':>10}")
    for v, name in NAMES.items():
        ma, mb = a == v, b == v
        sa, sb = int(ma.sum()), int(mb.sum())
        dice = 2 * np.logical_and(ma, mb).sum() / (sa + sb) if sa + sb else 1.0
        print(f"{name:<13} {dice:7.4f} {sa:10d} {sb:10d}")
    swap = int(np.sum((a == 1) & (b == 2)) + np.sum((a == 2) & (b == 1)))
    print(f"Voxels échangés maxillaire <-> mandibule : {swap}")

    if args.proba:
        pa, _ = nrrd.read(args.proba[0])
        pb, _ = nrrd.read(args.proba[1])
        diff = np.abs(pa.astype(np.int16) - pb.astype(np.int16))   # 0-255 = 0-100 %
        print("\nProbabilités (écart en points de %)")
        print(f"{'Structure':<13} {'moyen':>8} {'max':>7} {'changent de côté du 50 %':>26}")
        for i, (v, name) in enumerate(NAMES.items()):
            if i >= pa.shape[0]:
                break
            flips = int(np.sum((pa[i] >= 128) != (pb[i] >= 128)))
            print(f"{name:<13} {diff[i].mean() / 2.55:8.4f} {diff[i].max() / 2.55:7.2f} {flips:26d}")


if __name__ == "__main__":
    main()
