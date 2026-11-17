# Print workspace: Phase 0 spike report

Date: 2026-11-14. Spec: [print_workspace.md](print_workspace.md). Synthetic meshes only, never patient data.
Scripts are in `tools/print_spike/`. Generated files go to `tools/print_spike/out/`, which git ignores.
No UI code was written.

## Installed versions

| Tool | Version | Path |
|---|---|---|
| PrusaSlicer | 2.9.7 | `/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer` |
| UVtools | core 7.1.2 (`UVtoolsCmd --version`: 2.2.0) | `/Applications/UVtools.app/Contents/MacOS/UVtoolsCmd` |
| Blender | 5.2.2 (bundles `OpenImageIO` and `openvdb`) | `/Applications/Blender.app` |

Hardware: Mac mini M5 Pro, 64 GB.

## Test parts

`make_test_guide.py` writes three files:
- **`guide.stl`**: a shell 2.5 mm thick, an arc of radius 20 mm over 110°, 14 mm high. Its concave face is the
  bearing surface. A drill-sleeve tube (outer diameter 6 mm, bore 4.2 mm, 8 mm long) sits on the outer face.
  Closed and manifold, 740 triangles, 1542 mm³.
- **`bearing.stl`**: a slab ±0.3 mm across the concave face. It extends past the edges and is used as the
  support blocker.
- **`bearing_band.stl`**: a slab 0.15 to 0.6 mm in front of that face, used only to check for contacts.

`make_dense_guide.py` (headless Blender) subdivides and slightly bumps the guide: 284,160 triangles, about the
triangle count of a real guide.

## Mars 4 Ultra printer profile

UVtools ships a PrusaSlicer printer profile for this printer:
`/Applications/UVtools.app/Contents/MacOS/Assets/PrusaSlicer/printer/Elegoo Mars 4 Ultra.ini`. Its values are
8520 x 4320 px, 153.36 x 77.76 mm, landscape, X mirrored, Z 165 mm. They match UVtools' own machine list
(`UVtoolsCmd print-machines`: "Elegoo Mars 4 Ultra | Resolution=8520x4320(px) Display=153.36x77.76(mm)
PixelSize=18x18(µm) Z=165mm") and the values in the spec.

What makes UVtools write a .goo is the profile's `printer_notes`: the keyword `FILEFORMAT_GOO`, plus a block
`START_CUSTOM_VALUES ... END_CUSTOM_VALUES` with the Elegoo-only settings (lift, waits, speeds).

`tools/print_spike/elegoo_abs_like_v3_translucent.ini` was built from that base, with the surgeon's resin settings
as Lychee Slicer 8.0.0 writes them (see "Reference files" below). It also sets `printer_settings_id = Elegoo Mars 4
Ultra`, which UVtools writes as the .goo's machine name.

```sh
P=/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer
A="/Applications/UVtools.app/Contents/MacOS/Assets/PrusaSlicer/printer/Elegoo Mars 4 Ultra.ini"
NOTES=$'PRINTER_VENDOR_ELEGOO\nPRINTER_MODEL_MARS4_ULTRA\nFILEFORMAT_GOO\n\nSTART_CUSTOM_VALUES\nTransitionLayerCount_3\nBottomWaitTimeBeforeCure_3\nWaitTimeBeforeCure_2.5\nBottomWaitTimeAfterCure_3\nWaitTimeAfterCure_1\nBottomLiftHeight_7\nLiftHeight_5\nBottomLiftSpeed_80\nLiftSpeed_100\nRetractSpeed_150\nBottomLightPWM_255\nLightPWM_255\nEND_CUSTOM_VALUES'
$P --load "$A" --printer-technology SLA --printer-notes "$NOTES" \
   --layer-height 0.05 --exposure-time 2.2 --initial-exposure-time 25 --faded-layers 4 --material-ow-faded-layers 4 \
   --supports-enable --pad-enable --support-object-elevation 5 --save elegoo_abs_like_v3_translucent.ini
```

The notes need real line breaks (`$'...'`). A literal `\n` breaks the keyword, and the conversion fails with
"Unable to find a valid target type from 'goo\'".

## 1. Supports and .sl1 from the CLI

```sh
$P --load elegoo_abs_like_v3_translucent.ini --export-sla -o out/part.sl1 out/part.3mf
# or, from an STL with the orientation given on the command line:
$P --load elegoo_abs_like_v3_translucent.ini --rotate-x -90 --center 76.68,38.88 --export-sla -o out/guide.sl1 out/guide.stl
```

- One command generates the supports and the pad, slices, and writes the .sl1. On a 284k-triangle guide this
  takes 2.4 to 2.5 s.
- The .sl1 is a zip: one PNG per layer, `config.ini` (layers, exposures, `usedMaterial` in ml) and
  `prusaslicer.ini` (all settings).
- The `printTime` in `config.ini` is PrusaSlicer's estimate for a Prusa printer, not for the Mars. UVtools
  computes its own estimate in the .goo.
- Orientation: either `--rotate-x/--rotate-y/--rotate` on the CLI, or a transform written into the 3MF. The
  extension will write the 3MF itself with the part's orientation on the plate, so the 3MF route is the one
  to keep.

## 2. Getting the supports back into Blender

| Candidate | Result |
|---|---|
| a) `--export-stl / --export-obj / --export-3mf` with supports on | **Not possible.** The model only: 740 facets, same volume as the input; the CLI never writes the support mesh. |
| b) Support points from a 3MF | **Not possible from the CLI.** The CLI's 3MF has no support-point file; only the GUI writes one. |
| UVtools `run <file.sl1> OperationLayerExportMesh` | Works, but heavy: 79 MB STL with `-p Quality=3`, 132 MB with `Quality=2`, 294 MB by default; 5 to 6 s. Use `--dummy`, or `run` rewrites the input file. |
| **c) Layers rebuilt in Blender (retained)** | `sl1_preview.py` |

The retained method, c), `sl1_preview.py`:
1. PrusaSlicer slices at a **preview resolution**: `--display-pixels-x 1065 --display-pixels-y 540`, 0.144 mm
   pixels, same build plate.
2. Blender reads the PNG layers with OpenImageIO.
3. It stacks them into an openvdb grid (voxels 0.144 x 0.144 x 0.15 mm).
4. A Volume to Mesh modifier makes the mesh: part plus supports, about 236k faces.

Timings and checks:
- **Supports unchanged by the preview resolution.** Material 4.24 ml and 498 layers at 1065 x 540 px, exactly
  the same as at 8520 x 4320.
- **Preview resolution, on the 284k-triangle guide**: slicing 0.8 s, preview in Blender 0.5 s.
- **Full resolution read in Python**: decoding 505 PNG layers of 8520 x 4320 takes about 20 s, and threads
  don't help. Rejected.

Blender notes: Volume to Mesh lives on a **mesh** object that points to the volume object, not on the volume
itself. Reading the layers needs no extra library: OpenImageIO and openvdb are in Blender 5.2.

## 3. No-support areas

**Support blockers** are honoured in SLA. They are volumes of type `SupportBlocker` in the 3MF
(`Metadata/Slic3r_PE_model.config`, one `<volume firstid lastid>` per volume, triangle ranges);
`make_3mf.py` writes them. Support enforcers (`SupportEnforcer`) work the same way, and the option
`--support-enforcers-only` exists.

Test: guide placed with its bearing face **down** (worst case), blocker over the whole face plus margin.
Support pixels counted in a band 0.15 to 0.6 mm in front of the face, which any support reaching it must
cross (`count_contacts.py`):

| | Support pixels in the band | Resin |
|---|---|---|
| No blocker | 217 (dozens of support tips on the face) | 4.06 ml |
| Blocker over the whole face | **0** | 2.44 ml |

- A blocker covering only 98 % of the arc left one support at the uncovered edge. The blocker must **overlap
  past the edges** of the bearing surface.
- In the extension, the no-support faces (face attribute) are turned into a thin closed slab, offset ±0.3 mm,
  written as a SupportBlocker volume in the 3MF.

**Important: removing supports can leave the part floating, and PrusaSlicer does not always say so.**
- Same orientation, everything blocked, simple guide: PrusaSlicer writes a .sl1 with **exit code 0**. The part
  hangs 5 mm above a lone pad.
- On the dense guide, it stops instead with "No pad can be generated for this model with the current
  configuration".
- The log line "Support mesh is empty" shows up in both normal and blocked runs: it means nothing for this.

So the "no unsupported island" check must be our own. UVtools does it (section 4).

**3MF pitfall:** a 3MF whose triangles don't share vertices (a "triangle soup") made PrusaSlicer **crash**
(code -11) or run for more than 10 minutes on the dense guide. With shared vertices, the same part takes
0.8 s. `make_3mf.py` merges identical vertices; the extension must do the same (or use Blender's 3MF export).

## 4. .sl1 to .goo with UVtoolsCmd

```sh
U=/Applications/UVtools.app/Contents/MacOS/UVtoolsCmd
$U convert out/part.sl1 goo out/part.goo         # 0.5 to 1.8 s, Goo v3
$U print-properties out/part.goo                 # check what was written
```

How each setting gets into the .goo:
- **Exposure** comes from the PrusaSlicer profile: `exposure_time`, `initial_exposure_time`.
- **Bottom layers** = `faded_layers`. Set `--material-ow-faded-layers` too, otherwise the material override
  (10) wins.
- **Transition layers, waits, lift heights and speeds, retract, PWM** come from the `CUSTOM_VALUES` block of
  the printer notes.
- All of these were checked one by one with `print-properties`: bottom layers 4, transition 3, bottom exposure
  25 s, exposure 2.2 s, lift 7/5 mm, lift speeds 80/100, retract 150, waits 2.5/1.
- In the extension, each **resin profile** writes these values into the notes and the profile at export, so
  nothing is hard-coded. Two-stage lift and retract (`LiftHeight2`, `LiftSpeed2`, `RetractHeight2`,
  `RetractSpeed2` and their `Bottom` versions) and `LightOffDelay` are accepted too.

Pitfalls found:
- **Use the explicit target `goo`, not `auto`:** `UVtoolsCmd convert part.sl1 goo part.goo` writes Goo **v3**
  (version 30, what Lychee writes for this printer) with every custom value applied, under the requested name.
  `auto` writes Goo v5 (51) and ignores the output name (it writes `<input name>.goo`).
- **`set-properties` cannot save in this version.** Asked to change values afterwards on a .goo or a .sl1
  (with or without `-o`), it fails: "File was partial decoded, a full encode is not possible". The settings
  must therefore be fixed in PrusaSlicer before conversion.
- **.goo version and mirror: settled by the reference files** (below): v3, mirrored horizontally.

Pre-export check: `$U print-issues out/part.goo --islands --group-islands`, 1.5 s.
- It finds the floating part: one island of 2048 x 778 px, layers 135 to 319.
- On a well supported part it also lists hundreds of 1 to 2 px "islands", which are rasterisation slivers:
  the check needs a minimum area (parameter).
- On the dense guide it flags the drill-sleeve tube's end (6 mm): it rests on a few very thin support tips
  (layer images). Not a floating part, but a weak point worth showing.

## 5. Time for one support update

Dense synthetic guide, 284,160 triangles, with blocker:

| Step | Time |
|---|---|
| Write the 3MF (Python) | 0.8 s |
| Supports + layers at preview resolution | 0.8 to 1.1 s |
| Preview mesh in Blender | 0.5 s |
| **Support update, total** | **about 2 s** |
| Export: full-resolution .sl1 | 2.5 s |
| Export: .sl1 to .goo | 1.8 s |
| Export: island check | 1.5 s |
| **Export, total** | **about 6 s** |

- Within the spec's target ("a few seconds"). A progress indicator is still useful, since one support update
  goes through three steps.
- The CLI run happens in a subprocess. The Blender step (PNG to volume to mesh) is short.
- A real guide with a bone imprint may need more support points: measure again on a real case export once the
  workspace exists.

## 6. The .goo checked by eye

- **Layers.** Seven layers of the dense guide's .sl1 were extracted to images and checked: pad, support
  pillars, the tube ring starting at layer 136, then the shell walls. Mirroring as expected (X mirrored, as in
  the profile). No missing or empty layer.
- **The .goo.** `out/dense.goo` was opened in the UVtools GUI on the Mac (`open -a UVtools out/dense.goo`) for
  the surgeon to scroll through. I cannot see that window: the GUI check by eye is for the user to do.
  Points to look at: the tube end (thin supports) and the Goo version / mirroring on the printer itself.

## Reference files (2026-11-16)

Two .goo files that the surgeon printed, exported by **Lychee Slicer 8.0.0** for the Mars 4 Ultra. Only their headers
were read (`UVtoolsCmd print-properties`), never their layers: they hold real cases and stay out of the repo.
- **Format and geometry**: Goo **v3**, 8520 x 4320 px, 153.36 x 77.76 mm, Z 165 mm, mirrored horizontally,
  anti-aliasing 8.
- **Resin profile**: "Elegoo ABS-Like V3 Translucent". Layer 0.03 mm; 6 bottom layers at 40 s; exposure 2.5 s;
  5 transition layers; light-off delay 1 s; wait 1 s before curing the bottom layers.
- **Lift and retract**: two stages, 3 mm at 60 then 4 mm at 180; retract 180, then 3 mm at 60.

`tools/print_spike/compare_goo.py REFERENCE.goo OURS.goo` compares 39 print settings. Our chain (profile above,
`convert … goo`) gives **39/39 identical** against both files.

What a header comparison cannot prove (it needs one test print, see below): that the layer images face the right
way (a mirrored guide would be a dangerous error, since guides are not symmetric) and that PrusaSlicer's supports
hold as well as Lychee's.

## Recommended chain (for Phase 1+)

```text
Blender (evaluated copies, no-support faces -> blocker slab, shared vertices)
  -> 3MF (part + SupportBlocker volumes, orientation in the build transform)
  -> live:   PrusaSlicer --load elegoo_abs_like_v3_translucent.ini --display-pixels-x 1065 --display-pixels-y 540 --export-sla
             -> layers -> openvdb -> Volume to Mesh preview                                  (~2 s)
  -> export: PrusaSlicer --load elegoo_abs_like_v3_translucent.ini --export-sla -> .sl1
             -> UVtoolsCmd convert .sl1 goo -> .goo v3 -> UVtoolsCmd print-issues --islands (area threshold)
             -> block export if a large island or a support in a blocker band                (~6 s)
```

Changes this suggests to the spec:
- "Support geometry back into Blender" is done from the preview-resolution layers, not from a mesh export.
- Resin settings go into the PrusaSlicer profile and notes, not into UVtools afterwards.
- The island check uses UVtools (`print-issues`), with a minimum area.

## Open points (to decide or check before Phase 1)

1. ~~Goo version and mirroring~~: v3, mirrored horizontally (reference files). Still to prove **with one test
   print of an asymmetric synthetic part** (e.g. a block with a raised "R"): the letter must read the right way.
2. ~~Real resin values~~: taken from the reference files (Elegoo ABS-Like V3 Translucent, 0.03 mm).
3. **Island threshold** in mm² for `print-issues`, and what to do about thin-tipped supports on precision
   features (tube ends).
4. **Support settings for guides**: Lychee's support settings are not in the .goo. Tip diameter (PrusaSlicer
   default 0.4 mm) and density to match what works in Lychee, then confirm by print.
