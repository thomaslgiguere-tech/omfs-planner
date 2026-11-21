"""Tests of the slice viewer's core (extension/omfs_planner/slice_core.py), without Blender:

    python3 tests/test_slice_core.py

Synthetic phantoms only. The accuracy test: spheres at known LPS positions (mm) in an oblique-free volume with
anisotropic voxels and an offset origin must appear at those positions on the axial, coronal and sagittal slices,
and survive a write / read of the .nrrd.
"""

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extension" / "omfs_planner"))
import slice_core as sc  # noqa: E402

SPACING = np.array([0.3, 0.4, 0.5])
ORIGIN = np.array([-40.0, 25.0, -60.0])
SHAPE = (200, 150, 120)
SPHERES = [(np.array([-10.0, 40.0, -30.0]), 4.0, 1000), (np.array([5.0, 50.0, -20.0]), 3.0, 2000)]


def phantom():
    i, j, k = np.indices(SHAPE)
    xyz = ORIGIN + np.stack([i, j, k], -1) * SPACING
    data = np.full(SHAPE, -1000, np.int16)
    for c, r, v in SPHERES:
        data[np.linalg.norm(xyz - c, axis=-1) <= r] = v
    return sc.Volume(data, np.diag(SPACING), ORIGIN)


def centroid_on_slice(volume, view, center, value):
    """LPS centroid of the pixels of a slice whose gray value is above half of `value`."""
    image, corners = sc.reslice(volume, view, center)
    right, up, normal, _ = sc.axes(view)
    a0, b0, pixel, width, height, depth = sc.slice_grid(volume, view, center)
    rows, cols = np.nonzero(image > value / 2)
    a = a0 + (cols + 0.5) * pixel
    b = b0 + (rows + 0.5) * pixel
    return a.mean() * right + b.mean() * up + depth * normal


def test_world_voxel_round_trip():
    v = phantom()
    p = np.array([1.23, 45.6, -7.8])
    assert np.allclose(v.to_world(v.to_voxel(p)), p)
    assert np.allclose(v.to_world([0, 0, 0]), ORIGIN)


def test_spheres_where_they_are_on_every_view():
    v = phantom()
    for c, _r, value in SPHERES:
        for view in sc.VIEWS:
            got = centroid_on_slice(v, view, c, value)
            err = np.linalg.norm(got - c)
            assert err < 0.5 * SPACING.min() + 0.05, f"{view}: sphere at {c} seen at {got} ({err:.3f} mm)"


def test_nrrd_round_trip_keeps_geometry():
    v = phantom()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "phantom.nrrd"
        sc.write_nrrd(path, v)
        back = sc.read_nrrd(path)
    assert back.data.dtype == np.int16 and back.shape == SHAPE
    assert np.array_equal(back.data, v.data)
    assert np.allclose(back.directions, v.directions) and np.allclose(back.origin, v.origin)


def test_ras_header_is_turned_into_lps():
    v = phantom()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ras.nrrd"
        sc.write_nrrd(path, v)
        text = path.read_bytes()
        head, body = text.split(b"\n\n", 1)
        ras_dirs = np.diag(SPACING) * [-1, -1, 1]
        ras_origin = ORIGIN * [-1, -1, 1]
        lines = []
        for line in head.decode().splitlines():
            if line.startswith("space:"):
                line = "space: right-anterior-superior"
            elif line.startswith("space directions:"):
                line = "space directions: " + " ".join("(%r,%r,%r)" % tuple(float(x) for x in r) for r in ras_dirs)
            elif line.startswith("space origin:"):
                line = "space origin: (%r,%r,%r)" % tuple(float(x) for x in ras_origin)
            lines.append(line)
        path.write_bytes(("\n".join(lines)).encode() + b"\n\n" + body)
        back = sc.read_nrrd(path)
    assert np.allclose(back.directions, v.directions) and np.allclose(back.origin, v.origin)


def test_views_follow_the_radiological_convention():
    right, up, normal, scroll = sc.axes('AXIAL')
    assert right[0] == 1 and up[1] == -1 and scroll[2] == 1      # patient's left on the right, anterior up
    right, up, normal, scroll = sc.axes('CORONAL')
    assert normal[1] == -1                                       # seen from the front
    right, up, normal, scroll = sc.axes('SAGITTAL')
    assert right[1] == 1 and normal[0] == 1                      # seen from the left, anterior on the left


def test_stepping_stays_on_the_voxel_grid_and_in_the_volume():
    v = phantom()
    c = v.center()
    i0, n = sc.slice_index(v, 'AXIAL', c)
    assert n == SHAPE[2]
    c1 = sc.step_center(v, 'AXIAL', c, 3)
    assert np.isclose(c1[2] - c[2], 3 * SPACING[2], atol=SPACING[2] / 2 + 1e-9)
    assert sc.slice_index(v, 'AXIAL', c1)[0] == i0 + 3
    top = sc.step_center(v, 'AXIAL', c, 10_000)
    assert sc.slice_index(v, 'AXIAL', top)[0] == n - 1
    k = v.to_voxel(top)[2]
    assert np.isclose(k, round(k))


def test_window_level():
    img = np.array([-1000, 0, 400, 1000], np.float32)
    g = sc.window_level(img, 2000, 0)
    assert np.allclose(g, [0.0, 0.5, 0.7, 1.0])


def test_plane_segments_of_a_cube():
    # A closed 10 mm cube (12 triangles) cut by z = 2: a square of 4 sides, 40 mm long.
    p = np.array([[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)], float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    tri = p[np.array(faces)]
    seg = sc.plane_segments(tri, (0, 0, 2), (0, 0, 1))
    assert np.allclose(seg[:, :, 2], 2)
    assert np.isclose(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1).sum(), 40.0)
    assert len(sc.plane_segments(tri, (0, 0, 20), (0, 0, 1))) == 0


def main():
    failed = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, func in tests:
        try:
            func()
            print(f"ok    {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
