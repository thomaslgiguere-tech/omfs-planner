# Coupes (CBCT): slice viewer, draft 2026-11-21

Goal: see the CBCT the way 3D Slicer shows it (axial, sagittal, coronal, grayscale) inside OMFS Planner, with every
mesh of the plan drawn on the slices in its own colour. It is the user's check that the AI segmentation (and later
the segments, screws, guides) follow the real image. No volume rendering: the STL surfaces already show the 3D.

## Use

OMFS tab > **Coupes (CBCT)** > *Ouvrir un CBCT (coupes)…*: a DICOM folder (go into it, Accept without picking a
file), one .dcm of the series, or a .nrrd / .nii / .mha. The 3D viewport switches to Quad View (Ctrl Alt Q):
axial top left, sagittal bottom left, coronal bottom right, 3D top right (Slicer's Four-Up).

| In a slice | Does |
|---|---|
| Wheel (trackpad: two-finger swipe) | next / previous slice |
| Left + right held, drag up / down | scroll along the slices |
| Left click / drag | crosshair: the two other slices go through that point |
| Right drag | window (horizontal) / level (vertical) |
| Ctrl + wheel, middle drag | zoom, pan |

Corner text: slice number, depth (mm), window / level, position and gray value under the mouse. Edge letters
(R / L / A / P / S / I) come from the real view orientation: they are the check of the display convention.

## How it works

- `segmentation/dicom_to_nrrd.py` (dentalseg env, SimpleITK, same series reading as segment.py) writes the volume
  as .nrrd in a temporary folder, deleted right after Blender reads it. The volume lives in memory only.
- `slice_core.py` (numpy, no Blender): NRRD reader (LPS; RAS headers flipped), trilinear reslicing on a pixel grid
  anchored to the volume bounds, window / level, mesh / plane section. Tested by `python3 tests/test_slice_core.py`:
  spheres at known LPS positions found on all three views within half a voxel, NRRD round trip, RAS header,
  slice stepping on the voxel grid, cube section length.
- `slice_viewer.py` (Blender): Quad View set up with Blender's Bottom / Front / Right views (= radiological axial,
  coronal, sagittal), POST_VIEW draw handler (slice texture on top in the quarters, the three planes among the
  meshes in the 3D quarter, coloured contours, crosshair), POST_PIXEL text, one modal operator for the mouse
  (events outside the slice quarters pass through). Parameters at the end of params.py (`SLICE_*`, `DICOM_*`).

## Not checked yet (first run in Blender)

Written and unit-tested outside Blender's UI (bpy 5.3 headless: registration, loading, contours, stepping). Not yet
seen on screen: Quad View set-up, GPU drawing, the modal mouse. Things to watch first:

1. Quad View quarter order and Bottom / Front / Right orientation (edge letters must read R on the left of the axial).
2. `gpu.types.Buffer` from numpy (fallback to a list is slower), RGBA16F texture.
3. Left + right together on the Mac mouse; Ctrl + wheel; Blender's own shortcuts still working elsewhere.
4. Accuracy against 3D Slicer on a real case: same landmark clicked in both, same LPS mm (Slicer shows RAS:
   x and y signs flipped).

## Next

- Speed: reslicing is CPU (~40-80 ms for a 500³ CBCT): fine for the wheel, a little heavy for a crosshair drag.
  Next step: the volume as a 3D GPU texture, sampled by a shader (instant).
- Oblique slice on the Section & Analysis plane (the osteotomy plane), contour fill, slab (MIP) for the canal.
