"""Slice viewer core, without Blender: a grayscale CT / CBCT volume in the patient's coordinates, resliced on the
axial, coronal and sagittal planes, window / level, and the section of the plan's meshes by a slice plane.

Coordinates: Blender's world = LPS millimetres, like the STL files of a 3D Slicer export and of
segmentation/surfaces.py (x towards the patient's Left, y Posterior, z Superior). Voxel (i, j, k) is at
origin + i * d0 + j * d1 + k * d2, where d0, d1, d2 are the rows of `directions` (NRRD "space directions").
The volume is never modified: every slice is a new resampled image (trilinear, like 3D Slicer's default).

Pure numpy, so it is tested outside Blender too (tests/test_slice_core.py).
"""

import gzip
from pathlib import Path

import numpy as np

# Screen axes of each view, in LPS (radiological convention, as in 3D Slicer):
#   axial    seen from the feet:  patient's right on the left of the screen, anterior up
#   coronal  seen from the front: patient's right on the left, superior up
#   sagittal seen from the left:  anterior on the left, superior up
# `scroll` is the direction of "next slice" (wheel up): superior, anterior, left.
VIEWS = {
    'AXIAL': {"right": (1.0, 0.0, 0.0), "up": (0.0, -1.0, 0.0), "scroll": (0.0, 0.0, 1.0)},
    'CORONAL': {"right": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0), "scroll": (0.0, -1.0, 0.0)},
    'SAGITTAL': {"right": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0), "scroll": (1.0, 0.0, 0.0)},
}

NRRD_TYPES = {
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "short": "i2", "short int": "i2", "signed short": "i2", "signed short int": "i2", "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2", "uint16": "u2", "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "float": "f4", "double": "f8",
}


def axes(view):
    """(right, up, normal towards the viewer, scroll) of a view, as unit numpy vectors."""
    v = VIEWS[view]
    right, up, scroll = (np.array(v[k], float) for k in ("right", "up", "scroll"))
    return right, up, np.cross(right, up), scroll


class Volume:
    """A grayscale volume in memory. data: (i, j, k) array in its own type (never modified); directions: 3 x 3,
    one row per voxel axis (mm, LPS); origin: LPS position of voxel (0, 0, 0)."""

    def __init__(self, data, directions, origin, path=""):
        self.data = data
        self.directions = np.asarray(directions, float)
        self.origin = np.asarray(origin, float)
        self.path = str(path)
        self._inverse = np.linalg.inv(self.directions)
        self.background = float(data.min())

    @property
    def shape(self):
        return self.data.shape

    @property
    def spacing(self):
        return np.linalg.norm(self.directions, axis=1)

    def to_world(self, ijk):
        return np.asarray(ijk, float) @ self.directions + self.origin

    def to_voxel(self, xyz):
        return (np.asarray(xyz, float) - self.origin) @ self._inverse

    def corners(self):
        """The 8 corner voxel centres, in LPS mm."""
        n = np.array(self.shape) - 1
        ijk = np.array([[i, j, k] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])], float)
        return self.to_world(ijk)

    def center(self):
        return self.corners().mean(axis=0)

    def sample(self, xyz):
        """Trilinear values at LPS points (..., 3); points outside the volume get the background value."""
        xyz = np.asarray(xyz, float)
        flat = xyz.reshape(-1, 3)
        ijk = self.to_voxel(flat)
        shape = np.array(self.shape)
        inside = np.all((ijk >= -0.5) & (ijk <= shape - 0.5), axis=1)
        ijk = np.clip(ijk, 0, shape - 1)
        base = np.clip(np.floor(ijk).astype(np.int64), 0, np.maximum(shape - 2, 0))
        frac = ijk - base
        data = self.data
        out = np.zeros(len(flat), np.float32)
        for di in (0, 1):
            wi = frac[:, 0] if di else 1 - frac[:, 0]
            ii = np.minimum(base[:, 0] + di, shape[0] - 1)
            for dj in (0, 1):
                wj = frac[:, 1] if dj else 1 - frac[:, 1]
                jj = np.minimum(base[:, 1] + dj, shape[1] - 1)
                for dk in (0, 1):
                    wk = frac[:, 2] if dk else 1 - frac[:, 2]
                    kk = np.minimum(base[:, 2] + dk, shape[2] - 1)
                    out += (wi * wj * wk).astype(np.float32) * data[ii, jj, kk].astype(np.float32)
        out[~inside] = self.background
        return out.reshape(xyz.shape[:-1])


# --- NRRD -------------------------------------------------------------------------------------------------------

def read_nrrd(path):
    """A 3D grayscale .nrrd (raw or gzip, any integer or float type) as a Volume, in LPS. Written by
    segmentation/dicom_to_nrrd.py (SimpleITK), or by 3D Slicer (File > Save, volume as .nrrd)."""
    raw = Path(path).read_bytes()
    end = raw.find(b"\n\n")
    if not raw.startswith(b"NRRD") or end < 0:
        raise ValueError(f"{Path(path).name}: pas un fichier NRRD avec ses données (.nhdr détaché non pris en charge)")
    header = {}
    for line in raw[:end].decode("utf-8", "replace").splitlines()[1:]:
        if line.startswith("#") or ":" not in line:
            continue
        key, _sep, value = line.partition(":")
        header[key.strip().lower()] = value.lstrip("=").strip()
    kind = NRRD_TYPES.get(header.get("type", "").lower())
    if kind is None:
        raise ValueError(f"Type NRRD non pris en charge : {header.get('type')}")
    sizes = [int(v) for v in header["sizes"].split()]
    if len(sizes) != 3:
        raise ValueError(f"NRRD à {len(sizes)} dimensions : attendu un volume 3D en niveaux de gris")
    body = raw[end + 2:]
    encoding = header.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif encoding != "raw":
        raise ValueError(f"Encodage NRRD non pris en charge : {encoding}")
    endian = "<" if header.get("endian", "little").lower() == "little" else ">"
    dtype = np.dtype(kind if kind.endswith("1") else endian + kind)
    data = np.frombuffer(body, dtype=dtype, count=int(np.prod(sizes))).reshape(sizes, order='F')
    data = np.ascontiguousarray(data.astype(dtype.newbyteorder("="), copy=False))
    vectors = [v for v in header["space directions"].replace("none", "").split(")") if "(" in v]
    directions = np.array([[float(x) for x in v.split("(")[1].split(",")] for v in vectors])
    origin = np.array([float(x) for x in header.get("space origin", "(0,0,0)").strip("()").split(",")])
    space = header.get("space", "left-posterior-superior").lower()
    if space in ("right-anterior-superior", "ras"):
        flip = np.array([-1.0, -1.0, 1.0])
        directions, origin = directions * flip, origin * flip
    elif space not in ("left-posterior-superior", "lps"):
        raise ValueError(f"Espace NRRD non pris en charge : {space}")
    return Volume(data, directions, origin, path)


def write_nrrd(path, volume):
    """Write a Volume as a gzip NRRD, in LPS (tests)."""
    data = volume.data
    types = {"u1": "uint8", "i1": "int8", "i2": "int16", "u2": "uint16", "i4": "int32", "u4": "uint32",
             "f4": "float", "f8": "double"}
    kind = types[data.dtype.kind + str(data.dtype.itemsize)]
    spatial = " ".join("(%r,%r,%r)" % tuple(float(v) for v in row) for row in volume.directions)
    lines = ["NRRD0005", f"type: {kind}", "dimension: 3", "space: left-posterior-superior",
             "sizes: " + " ".join(str(s) for s in data.shape), "space directions: " + spatial,
             "kinds: domain domain domain", "endian: little", "encoding: gzip",
             "space origin: (%r,%r,%r)" % tuple(float(v) for v in volume.origin)]
    body = np.asarray(data, data.dtype.newbyteorder("<")).tobytes(order='F')
    Path(path).write_bytes(("\n".join(lines) + "\n\n").encode() + gzip.compress(body, 1))


# --- Slices -----------------------------------------------------------------------------------------------------

def slice_step(volume, view):
    """mm between two slices of a view: the spacing of the voxel axis closest to the view's normal."""
    _r, _u, _n, scroll = axes(view)
    units = volume.directions / volume.spacing[:, None]
    return float(volume.spacing[np.argmax(np.abs(units @ scroll))])


def depth_range(volume, view):
    """(first, last) depth of the volume along the view's scroll axis, in mm (voxel centres)."""
    d = volume.corners() @ axes(view)[3]
    return float(d.min()), float(d.max())


def slice_index(volume, view, center):
    """(index, count) of the slice through `center` (0-based), for the "coupe 12 / 400" readout."""
    first, last = depth_range(volume, view)
    step = slice_step(volume, view)
    count = int(round((last - first) / step)) + 1
    index = int(round((np.dot(center, axes(view)[3]) - first) / step))
    return min(max(index, 0), count - 1), count


def step_center(volume, view, center, steps):
    """`center` moved by a number of slices along the view's scroll axis, on the voxel grid, kept in the volume."""
    scroll = axes(view)[3]
    first, last = depth_range(volume, view)
    step = slice_step(volume, view)
    depth = float(np.dot(center, scroll))
    index = round((depth - first) / step) + steps
    target = min(max(first + index * step, first), last)
    return np.asarray(center, float) + (target - depth) * scroll


def clamp_center(volume, center):
    """`center` brought inside the volume (in voxel space), so the three slices always cross it."""
    ijk = np.clip(volume.to_voxel(center), 0, np.array(volume.shape) - 1)
    return volume.to_world(ijk)


def slice_grid(volume, view, center, pixel=0.0):
    """The pixel grid of a view's slice through `center`. The grid is anchored on the volume's bounds, not on the
    centre, so moving the crosshair in the plane doesn't resample the image (no shimmer).
    Returns (a0, b0, pixel, width, height, depth): pixel (i, j) is centred at
    (a0 + (i + 0.5) * pixel) * right + (b0 + (j + 0.5) * pixel) * up + depth * normal."""
    right, up, normal, _scroll = axes(view)
    pixel = float(pixel) if pixel and pixel > 0 else float(volume.spacing.min())
    c = volume.corners()
    a, b = c @ right, c @ up
    half = 0.5 * volume.spacing.max()
    a0, b0 = a.min() - half, b.min() - half
    width = int(np.ceil((a.max() + half - a0) / pixel))
    height = int(np.ceil((b.max() + half - b0) / pixel))
    return a0, b0, pixel, width, height, float(np.dot(center, normal))


def reslice(volume, view, center, pixel=0.0):
    """The slice of a view through `center`: (image (height, width) float32, quad corners (4, 3) LPS).
    Row 0 is the bottom of the screen and column 0 its left (OpenGL texture order). Corners: bottom-left,
    bottom-right, top-right, top-left."""
    right, up, normal, _scroll = axes(view)
    a0, b0, pixel, width, height, depth = slice_grid(volume, view, center, pixel)
    a = a0 + (np.arange(width) + 0.5) * pixel
    b = b0 + (np.arange(height) + 0.5) * pixel
    points = (b[:, None, None] * up + a[None, :, None] * right) + depth * normal
    image = volume.sample(points)
    a1, b1 = a0 + width * pixel, b0 + height * pixel
    corners = np.array([a0 * right + b0 * up, a1 * right + b0 * up, a1 * right + b1 * up, a0 * right + b1 * up])
    return image, corners + depth * normal


def auto_window(volume, percentiles=(1.0, 99.5), samples=2_000_000):
    """(window, level) from the volume's gray value percentiles (CBCT values aren't calibrated HU)."""
    flat = volume.data.ravel()
    if flat.size > samples:
        flat = flat[:: flat.size // samples]
    lo, hi = np.percentile(flat, percentiles)
    if hi <= lo:
        hi = lo + 1.0
    return float(hi - lo), float((hi + lo) / 2)


def window_level(image, window, level):
    """Gray values -> 0..1 (float32): level - window / 2 is black, level + window / 2 is white."""
    window = max(float(window), 1e-6)
    return np.clip((image - (level - window / 2)) / window, 0.0, 1.0).astype(np.float32)


# --- Meshes on a slice ------------------------------------------------------------------------------------------

def plane_segments(triangles, point, normal):
    """Section of a triangle mesh by a plane: (M, 2, 3) segments (LPS). triangles: (N, 3, 3) world vertices.
    A vertex exactly on the plane counts as above it, so no triangle gives a zero-length or double segment."""
    tri = np.asarray(triangles, float)
    if tri.size == 0:
        return np.zeros((0, 2, 3))
    normal = np.asarray(normal, float)
    dist = (tri - np.asarray(point, float)) @ normal          # (N, 3)
    above = dist >= 0
    crossing = above.any(axis=1) & ~above.all(axis=1)
    tri, dist, above = tri[crossing], dist[crossing], above[crossing]
    if len(tri) == 0:
        return np.zeros((0, 2, 3))
    starts, ends = (0, 1, 2), (1, 2, 0)
    points = np.empty((len(tri), 3, 3))
    cuts = np.empty((len(tri), 3), bool)
    for e, (s, t) in enumerate(zip(starts, ends)):
        ds, dt = dist[:, s], dist[:, t]
        cuts[:, e] = above[:, s] != above[:, t]
        denom = np.where(cuts[:, e], ds - dt, 1.0)
        f = np.where(cuts[:, e], ds / denom, 0.0)[:, None]
        points[:, e] = tri[:, s] + (tri[:, t] - tri[:, s]) * f
    order = np.argsort(~cuts, axis=1, kind="stable")[:, :2]   # the two edges that cross
    rows = np.arange(len(tri))[:, None]
    return points[rows, order]


def near_plane(triangles, point, normal, reach):
    """Mask of the triangles with a vertex within `reach` mm of the plane (cheap pre-filter)."""
    dist = (np.asarray(triangles, float) - np.asarray(point, float)) @ np.asarray(normal, float)
    return (dist.min(axis=1) <= reach) & (dist.max(axis=1) >= -reach)
