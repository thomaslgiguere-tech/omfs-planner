# Segmentation (DentalSeg), the first brick of the "3 in 1"

Moved here from `~/DentalSegApp` on 2026-11-13, so the tool lives in one place with OMFS Planner.

## What is here

| File | Role |
|---|---|
| `segment.py` | CBCT / CT (DICOM folder, `.dcm`, `.nrrd`, `.nii(.gz)`, `.mha`) → DentalSegmentator (nnU-Net, `Dataset111_453CT`, fold 0) on the Apple GPU (MPS) → `<name>_segmentation.seg.nrrd` (next to the input, or `--sortie`), readable by 3D Slicer, plus the probabilities (both modes). Removes islands < 60 mm³ (except the canal), like the Slicer extension. |
| `comparer_segmentations.py` | Compares two segmentations of the same scan: identical voxels, Dice per structure, maxilla/mandible swaps, probability gap and voxels crossing the 50 % threshold. |
| `surfaces.py` | Step 3: labels or probabilities → closed STL surfaces per structure (patient coordinates, LPS); `--comparer A.stl B.stl` gives the distance between two surfaces. |
| `DentalSeg.applescript` | Drop-on-icon launcher: asks Précis or Rapide, then runs `segment.py` with the `dentalseg` conda env in Terminal. |
| `DentalSeg.app` | Compiled launcher (not in git; rebuild with `osacompile -o segmentation/DentalSeg.app segmentation/DentalSeg.applescript`). |
| `model/` | DentalSegmentator weights (236 MB; not in git: large, third-party licence, see the Zenodo record). |

Python environment: `~/miniconda3/envs/dentalseg` (nnU-Net v2, PyTorch with MPS, SimpleITK, pynrrd, SciPy).
It stays outside the repo (1.5 GB, machine-specific).

Labels: 1 maxilla & upper skull, 2 mandible, 3 upper teeth, 4 lower teeth, 5 mandibular canal. These map onto
OMFS Planner's roles BONE, BONE, TEETH, TEETH, NERVE.

## Intention (surgeon, 2026-11-13)

One tool, three steps, no 3D Slicer:

1. **DICOM → volume**: read the CBCT series, write it as `.nrrd` (spacing, orientation and origin kept, LPS), the
   pivot format for everything after.
2. **Segmentation, modular**: a pipeline of steps the user can choose and order (AI model, thresholding,
   island removal, hole filling, smoothing, manual corrections...). Each step is a small, replaceable module
   with parameters; the AI model is one module among others, so a better model can replace DentalSegmentator
   without touching the rest.
3. **Surfaces → OMFS Planner**: labels to closed meshes (marching cubes, smoothing, decimation), in the
   patient's coordinates, with their roles, straight into Blender: what 3D Slicer's export does today.

Rules carried over from OMFS Planner: patient data never committed (`*.nrrd`, `*.nii*`, `*.dcm` are ignored),
all defaults in one parameter module, the original volume never modified.

## AI segmentation options (research, 2026-11-13)

- **DentalSegmentator** (Dot et al., J Dent 2024; nnU-Net, 470 training CT/CBCT): 5 structures. External test
  (123 CBCT, 5 centres): Dice 94.2 %, NSD 98.4 %. Open model + Slicer extension. Teeth are one label per arch
  (no individual teeth, no pulp).
- **ToothFairy2** (MICCAI 2024, 480 CBCT, 42 classes: jaws, sinuses, canals, each tooth in FDI notation,
  implants...): winner "Scaling nnU-Net" (nnU-Net ResEnc-L), mean Dice 0.925. Dataset and benchmark public;
  no official weights found (training recipe is in the nnU-Net repo). **ToothFairy3** (MICCAI 2025, 532
  volumes, 77 classes incl. pulp, interactive canal task).
- **TotalSegmentator**: body CT; has `craniofacial_structures` and a teeth subtask, not built for CBCT.
- **nnInteractive** (DKFZ 2025): promptable 3D segmentation (points, scribbles, boxes, lasso), any modality:
  the best tool to correct an AI result (e.g. a canal or a condyle).
- Commercial (Relu Creator, Diagnocat): same order of accuracy. Ex vivo pilot on extracted teeth (n = 10):
  mean absolute deviation Relu 0.10 mm, Diagnocat 0.10 mm, DentalSegmentator ~0.13 mm. On impacted teeth,
  DentalSegmentator and Diagnocat beat Relu (Relu left 6 % of teeth unsegmented, DentalSegmentator 0.1 %).

## Precise mode and sub-voxel surfaces (done, 2026-11-13)

`segment.py` now runs in **precise mode by default**: test-time augmentation (8 mirrored passes, probabilities
averaged), sliding windows overlapping more (step 0.35 instead of 0.5), and the probabilities kept
(`<name>_probabilites.nrrd`, 4D uint8). `--rapide` gives the former setting (no augmentation, step 0.5), and
keeps the probabilities too. The DentalSeg app asks for the mode at launch (Précis by default, or Rapide). `surfaces.py` turns either the
labels (staircase) or the probabilities (iso-surface at 50 %, between voxels; only around the cleaned structure,
so no lost islands come back) into closed STL surfaces in patient coordinates (LPS, like 3D Slicer).

Measured on one real CBCT (512 x 613 x 280, 0.377 x 0.377 x 0.6 mm), M5 Pro 64 GB:

| | Fast | Precise |
|---|---|---|
| Time | 2 min 34 s | 32 min 35 s |
| Peak memory (nnU-Net process) | 9.4 GB | 9.4 GB |
| Reproducible (2 runs) | | identical voxels |

- Fast vs precise labels: Dice mandible 0.946, maxilla 0.974, upper teeth 0.949, lower teeth 0.980, canal 0.918.
  Surfaces: mandible mean 0.27 mm, 95 % under 0.92 mm. The largest disagreement: one bone split between
  "mandible" and "maxilla" in fast mode, labelled as one structure in precise mode (behind the last molars).
- Probability surfaces vs staircase (precise): mean 0.06 to 0.09 mm, 95 % under 0.22 mm (the half voxel); no
  non-manifold edge (staircase: 9 on the mandible, 21 on the maxilla, bone voxels touching only by an edge).
  Surfaces: about 4 s (labels) to 10 s (probabilities) for the 5 structures.
- Not measured: accuracy against a reference (manual segmentation or physical specimen). Precise mode is more
  consistent; whether it is closer to the truth needs a reference.

Second CBCT (2026-11-14; 601 x 601 x 534, 0.3 mm isotropic), same machine:

| | Fast | Precise |
|---|---|---|
| Time | 4 min 46 s | 25 min 51 s |
| Peak memory | 14.7 GB | 14.5 GB |

- Fast vs precise: 99.91 % identical voxels, Dice maxilla 0.982, mandible 0.993, upper teeth 0.994, lower teeth
  0.994, canal 0.959; no voxel swapped between maxilla and mandible. Probability surfaces: mandible mean 0.030 mm
  (95 % under 0.10 mm), lower teeth 0.025 mm, canal 0.070 mm (95 % under 0.20 mm).
- Reading of the two cases: on a fine isotropic scan the two modes agree to a few hundredths of a mm; on the
  coarser, anisotropic first scan (0.6 mm slices) they differ by 0.27 mm on average and fast mode split one bone.
  Precise mode matters most when the scan is coarse. Still not measured against a reference.

Still possible: an nnU-Net trained on ToothFairy2/3 for individual teeth and pulp; nnInteractive for corrections.

## Apple-chip speed-up (2026-11-16, measured)

`segment.py` no longer calls `nnUNetv2_predict`; it runs nnU-Net in-process with a subclass of its predictor:

1. **fp16 on MPS**: nnU-Net only enables autocast on CUDA; on the Mac everything ran in fp32.
2. **Mirrors batched**: the 8 mirrored copies of a window go through the network as one batch of 8 instead of
   one after the other (`--lot-miroirs N` to batch fewer if memory is short).
3. Window accumulation kept on the GPU (nnU-Net sends it back to the CPU on anything other than CUDA).

Same weights, windows, Gaussian weighting and mirror average. Checked on CPU with the real plans and random
weights: identical to nnU-Net's own code up to float rounding (3e-8 on a window, 100 % identical labels).
fp16 itself can only be checked on the Mac. `--classique` runs the old `nnUNetv2_predict` path, `--fp32` keeps
the batching without fp16.

Measured on the two CBCTs above (M5 Pro 64 GB), same session, classic = `--classique`:

| | Scan 1 (0.377 x 0.6 mm) | Scan 2 (0.3 mm isotropic) |
|---|---|---|
| Precise, classic | 33 min 18 s | 25 min 23 s |
| **Precise, accelerated** | **22 min 39 s (-32 %)** | **17 min 46 s (-30 %)** |
| Fast, accelerated (fast before) | 3 min 11 s (2 min 34 s) | 4 min 14 s (4 min 46 s) |
| Peak memory | 9.3 GB | 15.3 GB |

- Same result: precise accelerated vs classic, 99.998 % and 99.999 % identical voxels, Dice 0.9998 to 1.0000 per
  structure; probabilities differ by 0.002 percentage points on average (max 1.6); the voxels that change are
  the ones sitting at the 50 % threshold (fp16 rounding). 21 voxels swapped maxilla/mandible on scan 1, 0 on scan 2.
- Fast mode gains little or nothing: no mirrors to batch, so only fp16 is left.
- Memory: no gain and no cost; the run is limited by the GPU, not by RAM.
