"""Headless tests, run inside Blender:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/run_tests.py

Uses synthetic meshes only (tools/make_test_case.py), written to a temporary folder.
"""

import os
import sys
from math import radians
import tempfile
import traceback
from pathlib import Path

import bpy
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "extension"), str(REPO / "tools")]

import make_test_case  # noqa: E402
import omfs_planner  # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
import bmesh  # noqa: E402
from omfs_planner import locking  # noqa: E402
from omfs_planner import occlusion_prepare  # noqa: E402
from omfs_planner import analysis, distance_map, guide_workshop, landmarks, movement, occlusion, occlusion_seating, osteotomy, params, reference_frame, registration, roles, section, section_contours, snapshot, splint, splint_holes, text_labels, cylinders, pins, model_base, tidy, screws, sleeves, trim, print_chain, print_profiles, print_scene, volume_scene, ortho_osteotomy  # noqa: E402

omfs_planner.register()


def test_role_from_filename():
    cases = {
        "mandible_bone.stl": 'BONE', "Segmentation_Mandible.stl": 'BONE', "os.stl": 'BONE',
        "lower_teeth.obj": 'TEETH', "MandibleTeeth.stl": 'TEETH',
        "lowerjaw_scan.ply": 'SCAN', "IOS_lower.stl": 'SCAN', "ios_teeth.stl": 'SCAN',
        "Patient-LowerJaw.stl": 'SCAN', "LowerScanSTL.stl": 'SCAN',
        "Mandibular canal.stl": 'NERVE', "IAN_left.stl": 'NERVE', "Skin.stl": 'SKIN',
        "Maxilla.stl": 'BONE', "random.stl": 'OTHER', "cosmos.stl": 'OTHER',
    }
    for filename, expected in cases.items():
        got = roles.role_from_filename(filename)
        assert got == expected, f"{filename}: expected {expected}, got {got}"


def test_import_case():
    with tempfile.TemporaryDirectory() as folder:
        make_test_case.main(folder)
        result = bpy.ops.omfs.import_case(directory=folder)
    assert result == {'FINISHED'}, result

    units = bpy.context.scene.unit_settings
    assert units.length_unit == 'MILLIMETERS' and abs(units.scale_length - 0.001) < 1e-9
    slots = bpy.context.scene.transform_orientation_slots
    assert slots[0].type == 'LOCAL' and all(s.type == 'DEFAULT' for s in slots[1:]), "orientation should be local"

    expected = {"mandible_bone": 'BONE', "lower_teeth": 'TEETH', "lowerjaw_scan": 'SCAN'}
    for name, role in expected.items():
        obj = bpy.data.objects.get(name)
        assert obj is not None, f"missing object {name}"
        assert obj.data.name == name, f"mesh data of {name} is called {obj.data.name}"
        assert obj.omfs.role == role, f"{name}: role {obj.omfs.role}"
        assert tuple(round(c, 2) for c in obj.color) == roles.COLORS[role], f"{name}: colour"
        locked = role in roles.LOCKED_BY_DEFAULT
        assert obj.hide_select == locked and all(obj.lock_location) == locked, f"{name}: lock"

    # Raw coordinates kept (no axis conversion): teeth boxes are centred at z = -60 + 18.
    teeth = bpy.data.objects["lower_teeth"]
    zs = [v.co.z for v in teeth.data.vertices]
    assert abs(min(zs) - (-42 - 4.5)) < 1e-3 and abs(max(zs) - (-42 + 4.5)) < 1e-3, (min(zs), max(zs))
    assert tuple(teeth.rotation_euler) == (0, 0, 0) and tuple(teeth.scale) == (1, 1, 1)


def test_toggle_lock():
    bone = bpy.data.objects["mandible_bone"]
    bpy.ops.omfs.toggle_lock(object_name=bone.name)
    assert not bone.hide_select and not any(bone.lock_location)
    bpy.ops.omfs.toggle_lock(object_name=bone.name)
    assert bone.hide_select and all(bone.lock_rotation)


def test_place_landmark():
    scene = bpy.context.scene
    bone = bpy.data.objects["mandible_bone"]
    vertices_before = [tuple(v.co) for v in bone.data.vertices]

    pog = landmarks.place(scene, 'POG', (0, -30, -60), bone)
    assert pog.type == 'EMPTY' and pog.name == "Pogonion" and pog.omfs.role == 'LANDMARK'
    assert pog.omfs.surface == bone.name and pog.hide_select, "landmark should be locked"
    assert pog.users_collection[0].name == landmarks.COLLECTION

    again = landmarks.place(scene, 'POG', (1, -31, -61), bone)   # placing again moves it
    assert again == pog and tuple(pog.location) == (1, -31, -61)
    assert sum(o.omfs.landmark == 'POG' for o in scene.objects) == 1
    assert [tuple(v.co) for v in bone.data.vertices] == vertices_before, "bone mesh changed"

    bpy.ops.omfs.toggle_landmarks()
    assert pog.hide_get(), "toggle should hide the landmarks"
    landmarks.place(scene, 'POG', (1, -31, -61), bone)
    assert not pog.hide_get(), "placing a landmark shows it again"

    bpy.ops.omfs.delete_landmark(landmark_id='POG')
    assert landmarks.find(scene, 'POG') is None


# Landmarks in LPS millimetres, with FH exactly on z = 0 and the midline on x = 0.
FRAME_POINTS = {
    'PO_R': Vector((-55, 80, 0)), 'PO_L': Vector((55, 80, 0)),
    'OR_R': Vector((-30, 5, 2)), 'OR_L': Vector((30, 5, -2)),   # midpoint on FH, orbits uneven
    'N': Vector((0, -5, 25)), 'BA': Vector((0, 70, -20)),
}


def assert_close(a, b, label, tol=1e-4):
    assert (Vector(a) - Vector(b)).length < tol, f"{label}: {tuple(a)} != {tuple(b)}"


def test_frame_axes():
    m = reference_frame.frame_matrix(FRAME_POINTS)
    assert_close(m.col[0].xyz, (1, 0, 0), "x = patient left")
    assert_close(m.col[1].xyz, (0, 1, 0), "y = posterior")
    assert_close(m.col[2].xyz, (0, 0, 1), "z = superior")
    assert_close(m.translation, (0, -5, 0), "origin = nasion projected on FH")


def test_frame_follows_head_rotation():
    # Same skull, tilted and turned: the frame must turn with it.
    turn = Matrix.Translation((90, 100, 80)) @ Euler((0.3, -0.1, 0.2)).to_matrix().to_4x4()
    m = reference_frame.frame_matrix({k: turn @ v for k, v in FRAME_POINTS.items()})
    expected = turn @ reference_frame.frame_matrix(FRAME_POINTS)
    for i in range(4):
        assert_close(m.col[i], expected.col[i], f"column {i}")


def test_build_frame_and_clinical_values():
    scene = bpy.context.scene
    frame = reference_frame.build(scene, FRAME_POINTS)
    assert frame.omfs.role == 'REFERENCE' and frame.hide_select
    assert len(frame.children) == 3 and all(c.omfs.role == 'REFERENCE' for c in frame.children)
    assert reference_frame.build(scene, FRAME_POINTS) == frame, "building again must update, not duplicate"
    left, anterior, up = reference_frame.to_clinical(frame, (2, -6, -1))   # LPS displacement
    assert_close((left, anterior, up), (2, 6, -1), "6 mm forward, 1 mm down, 2 mm left")


def box_object(name, role, center, size):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.omfs.role = role
    return obj


def volume(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    v = bm.calc_volume()
    bm.free()
    return v


def test_osteotomy():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # frame axes = world axes
    bone = box_object("Chin box", 'BONE', (0, 0, -50), (60, 40, 30))   # z from -65 to -35
    landmarks.place(scene, 'MF_R', (-20, 0, -40), bone)
    landmarks.place(scene, 'MF_L', (20, 0, -40), bone)
    landmarks.place(scene, 'POG', (0, -20, -55), bone)
    canal = box_object("Canal", 'NERVE', (-20, 5, -43), (2, 2, 0.001))  # 2 mm above the cut, right

    cut = osteotomy.add_cut(scene)
    props = cut.omfs_cut
    assert props.source == bone, "source should default to the bone under the pogonion"
    assert abs(osteotomy.height_on_plane(props, 0, 0) - (-5)) < 1e-6   # 5 mm below foramina

    rows = {label: d for label, d, _m in osteotomy.safety_rows(scene, cut)}
    assert abs(rows["Mental foramen R"] - 5) < 1e-4 and abs(rows["Mental foramen L"] - 5) < 1e-4, rows
    assert abs(rows["Nerve canal R"] - 2) < 1e-3 and rows["Nerve canal L"] is None, rows

    props.profile_angle = radians(10)    # rises anteriorly
    assert osteotomy.height_on_plane(props, 0, -20) > osteotomy.height_on_plane(props, 0, 20)
    props.profile_angle = 0
    props.frontal_angle = radians(10)    # left side higher
    assert osteotomy.height_on_plane(props, 20, 0) > osteotomy.height_on_plane(props, -20, 0)
    props.frontal_angle = 0

    assert props.extent_r == props.extent_l == params.CUT_POSTERIOR_EXTENT == 30.0, "default: far behind the foramina"
    props.extent_r = props.extent_l = 15.0
    vertices_before = len(bone.data.vertices)
    segment, proximal = osteotomy.apply_cut(bpy.context, cut)
    # Segment: x -30..30, y -20..15 (posterior extent), z -65..-45 -> 60 * 35 * 20 mm3.
    assert abs(volume(segment) - 42000) < 1, volume(segment)
    assert abs(volume(proximal) - (72000 - 42000)) < 1, volume(proximal)
    assert segment.omfs.role == 'SEGMENT' and proximal.omfs.role == 'BONE' and proximal.hide_select
    assert len(bone.data.vertices) == vertices_before and bone.hide_get(), "source must stay intact, hidden"
    assert props.applied_state == repr(osteotomy.state(props))

    helpers = osteotomy.cut_helpers(scene)
    assert {o.name for o in helpers} == {"Osteotomy plane", "Star R", "Star L"}
    assert all(o.hide_get() for o in helpers), "applying the cut hides the plane and stars"
    bpy.ops.omfs.toggle_cut_helpers()
    assert not any(o.hide_get() for o in helpers)

    osteotomy.apply_cut(bpy.context, cut)   # re-applying replaces, never duplicates
    assert sum(o.omfs.made_by == cut.name for o in scene.objects) == 2
    assert {o.name for o in scene.objects if o.omfs.role == 'MARKER'} == {"Star R", "Star L"}
    for obj in (canal,):
        bpy.data.objects.remove(obj)


def test_registration():
    scene = bpy.context.scene
    bone = box_object("Reg bone", 'BONE', (300, 0, -10), (60, 60, 20))   # top surface at z = 0
    bpy.ops.mesh.primitive_monkey_add(size=30, location=(300, 0, 5))   # asymmetric: ICP can lock on
    teeth = bpy.context.active_object
    teeth.name, teeth.omfs.role = "Reg teeth", 'TEETH'
    bpy.ops.object.transform_apply(location=True, scale=True)
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.modifier_apply(modifier=teeth.modifiers[0].name)
    scan = bpy.data.objects.new("Reg scan", teeth.data.copy())
    scene.collection.objects.link(scan)
    scan.omfs.role = 'SCAN'
    scan.omfs_reg.target, scan.omfs_reg.bone = teeth, bone
    true_pose = scan.matrix_world.copy()
    scan.matrix_world = Matrix.Translation((40, -30, 20)) @ Euler((0.4, -0.2, 0.9)).to_matrix().to_4x4()

    # Point pairs on the same vertices of both meshes, then a rigid fit.
    for n, vi in enumerate((0, 150, 300, 450), start=1):
        registration.add_point(scene, scan, f"REG_S{n}", f"Scan pt {n}",
                               scan.matrix_world @ scan.data.vertices[vi].co, parent=scan)
        registration.add_point(scene, scan, f"REG_T{n}", f"CBCT pt {n}", teeth.data.vertices[vi].co)
    assert len(registration.point_pairs(scene, scan)) == 4
    registration.set_points_hidden(scene, scan, True)
    assert not registration.points_visible(scene, scan)
    registration.set_points_hidden(scene, scan, False)
    rms = registration.align_points(scene, scan)
    assert rms < 1e-3, rms

    # Nudge it off by 1 degree and 0.6 mm, then ICP on the crowns must bring it back.
    registration.move_scan(scan, Matrix.Translation((0.4, -0.3, 0.3)) @ Euler((0.0175, 0, 0)).to_matrix().to_4x4()
                           @ scan.matrix_world)
    bvh = registration.crown_region_bvh(teeth, bone)
    count = registration.suggest_crowns(scan, bvh)
    indices = registration.crown_indices(scan)
    assert count == len(indices) > 100, count
    assert all(teeth.data.vertices[i].co.z > 0 for i in indices[:200]), "crowns must be above the bone"
    registration.icp(scan, bvh, indices)
    d = registration.crown_distances(scan, registration.teeth_bvh(teeth), indices)
    assert d.mean() < 0.001, d.mean()
    error = max((scan.matrix_world @ v.co - true_pose @ v.co).length for v in scan.data.vertices)
    assert error < 0.01, error
    assert all(abs(s - 1) < 1e-6 for s in scan.matrix_world.to_scale()), "registration must stay rigid"
    assert [tuple(v.co) for v in scan.data.vertices] == [tuple(v.co) for v in teeth.data.vertices], \
        "scan vertices must never change"


def test_freeze_teeth_position():
    """Figer la position dentaire: the registered pose becomes the scan's position zero (identity),
    nothing moves in the world, and later tools come back to it."""
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_monkey_add(size=20, location=(0, 0, 0))
    scan = bpy.context.active_object
    scan.name, scan.omfs.role = "Freeze scan", 'SCAN'
    registered = Matrix.Translation((-260, 40, 15)) @ Euler((0.3, -0.1, 0.7)).to_matrix().to_4x4()
    scan.matrix_world = registered
    file_co = [v.co.copy() for v in scan.data.vertices]
    point = registration.add_point(scene, scan, "REG_S1", "Freeze pt", registered @ file_co[5], parent=scan)
    ghost = bpy.data.objects.new("Freeze ghost", scan.data)    # e.g. a pre-op ghost sharing the mesh
    scene.collection.objects.link(ghost)
    ghost.matrix_world = Matrix.Translation((0, 0, 30)) @ registered
    bpy.context.view_layer.update()
    world = [scan.matrix_world @ v.co for v in scan.data.vertices]
    ghost_world = ghost.matrix_world @ scan.data.vertices[5].co
    point_world = point.matrix_world.translation.copy()

    scan.omfs_reg.points_rms = 0.2
    assert bpy.ops.omfs.freeze_scan(scan_name=scan.name) == {'FINISHED'}
    bpy.context.view_layer.update()
    assert scan.omfs_reg.frozen
    assert scan.matrix_basis == Matrix(), "position zero = identity transform"
    assert max((scan.matrix_world @ v.co - w).length for v, w in zip(scan.data.vertices, world)) < 1e-4
    assert (ghost.matrix_world @ scan.data.vertices[5].co - ghost_world).length < 1e-4
    assert (point.matrix_world.translation - point_world).length < 1e-4
    back = registration.file_matrix(scan)
    assert max((back @ v.co - c).length for v, c in zip(scan.data.vertices, file_co)) < 1e-4, \
        "file coordinates stay recoverable"
    try:
        bpy.ops.omfs.align_points(scan_name=scan.name)
        raise AssertionError("frozen: no re-registration")
    except RuntimeError as error:
        assert "cannée" in str(error)

    scan.matrix_world = Matrix.Translation((3, 0, 0))   # moved by hand, then back to its position zero
    assert bpy.ops.omfs.reset_scan(scan_name=scan.name) == {'FINISHED'}
    assert scan.matrix_basis == Matrix() and abs(scan.omfs_reg.points_rms - 0.2) < 1e-6, "results kept"

    # The upper arch's pre-positioning reads a frozen lower arch in its scanner frame, as before.
    upper = bpy.data.objects.new("Freeze upper", scan.data.copy())
    scene.collection.objects.link(upper)
    upper.data.transform(back)                      # same file coordinates as the lower scan
    lower_unfrozen = bpy.data.objects.new("Freeze lower ref", upper.data)
    scene.collection.objects.link(lower_unfrozen)
    lower_unfrozen.matrix_world = registered
    frozen_flip, ref_flip = occlusion.flip_above(upper, scan, 3.0), occlusion.flip_above(upper, lower_unfrozen, 3.0)
    assert all(abs(a - b) < 1e-4 for ra, rb in zip(frozen_flip, ref_flip) for a, b in zip(ra, rb))

    assert bpy.ops.omfs.unfreeze_scan(scan_name=scan.name) == {'FINISHED'}
    assert not scan.omfs_reg.frozen
    icons = {i.identifier for i in bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items}
    assert 'FREEZE' in icons
    for obj in (point, ghost, upper, lower_unfrozen, scan):
        bpy.data.objects.remove(obj)


def test_prepare_arches_isolate_and_clean_borders():
    """First step of the occlusion seating: only the two arches visible, ragged borders cleaned, still open."""
    scene = bpy.context.scene

    def ragged_arch(name, z):
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=60, y_segments=30, size=15.0)
        for v in bm.verts:                                   # wavy border, like a scan's
            if v.is_boundary:
                v.co.x += 0.2 * ((v.index * 7919) % 5 - 2) / 2
        a, b = bm.verts.new((40, 0, 0)), bm.verts.new((40.5, 0, 0))   # crumb: one loose triangle
        bm.faces.new((a, b, bm.verts.new((40, 0.5, 0))))
        edge = [v for v in bm.verts if v.is_boundary and abs(v.co.y + 15) < 1e-4][:2]
        bm.faces.new((edge[0], edge[1], bm.verts.new((edge[0].co.x, -16.5, 0))))   # spike on the border
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.omfs.role = 'SCAN'
        obj.location.z = z
        return obj

    upper, lower = ragged_arch("Prep upper", 600), ragged_arch("Prep lower", 590)
    other = bpy.data.objects.new("Prep other", None)
    scene.collection.objects.link(other)
    already_hidden = bpy.data.objects.new("Prep hidden", None)
    scene.collection.objects.link(already_hidden)
    already_hidden.hide_set(True)
    props = scene.omfs_occlusion
    props.upper, props.lower = upper, lower
    centre = {i: v.co.copy() for i, v in enumerate(upper.data.vertices) if abs(v.co.x) < 5 and abs(v.co.y) < 5}
    before = len(upper.data.vertices)
    original = upper.data

    assert bpy.ops.omfs.prepare_arches() == {'FINISHED'}
    layer = bpy.context.view_layer
    assert upper.visible_get() and lower.visible_get()
    assert not other.visible_get(), "everything else is hidden"
    assert all(not o.visible_get(view_layer=layer) for o in scene.objects if o not in (upper, lower))
    mesh = upper.data
    assert mesh is not original and upper.omfs_untrimmed is original, "original borders kept"
    assert len(mesh.vertices) < before
    bm = bmesh.new()
    bm.from_mesh(mesh)
    assert any(e.is_boundary for e in bm.edges), "the scan stays open"
    assert not any(occlusion_prepare._is_spike(f, radians(params.OCC_TRIM_SPIKE_ANGLE)) for f in bm.faces), "no spike"
    assert any(sum(e.is_boundary for e in f.edges) == 2 for f in bm.faces), "square corners are not spikes"
    assert len(occlusion_prepare._islands(bm)) == 1, "crumb removed"
    assert max(abs(v.co.y) for v in bm.verts) < 15.0 - params.OCC_TRIM_BAND + 0.6, "band removed at the border"
    bm.free()
    kept = {tuple(round(c, 5) for c in v.co) for v in mesh.vertices}
    assert all(tuple(round(c, 5) for c in co) in kept for co in centre.values()), "inside never moved"

    assert bpy.ops.omfs.show_hidden() == {'FINISHED'}
    assert other.visible_get() and not already_hidden.visible_get(), "exactly what was visible comes back"
    assert bpy.ops.omfs.restore_borders() == {'FINISHED'}
    assert upper.data is original and len(upper.data.vertices) == before and upper.omfs_untrimmed is None
    assert upper.data.name == "Prep upper"
    props.upper = props.lower = None
    for obj in (upper, lower, other, already_hidden):
        bpy.data.objects.remove(obj)


def test_guide_workshop_tools_are_documented():
    icons = {i.identifier for i in bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items}
    assert 'FACE_MAPS' in icons, "icon of the zone summary"
    for t in guide_workshop.TOOLS.values():
        assert t.icon in icons, f"{t.id}: unknown icon {t.icon}"
        assert t.tip and t.meshmixer, f"{t.id}: tooltip incomplete"
        tip = bpy.types.OMFS_OT_guide_tool.description(bpy.context, type("P", (), {"tool": t.id})())
        assert "Meshmixer" in tip, tip
    sections = {s[0] for s in guide_workshop.SECTIONS} | {'START'}
    assert {t.section for t in guide_workshop.TOOLS.values()} <= sections


def test_densify_keeps_the_surface():
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=10.0)
    bmesh.ops.translate(bm, vec=(-500, 0, 0), verts=bm.verts)
    mesh = bpy.data.meshes.new("Densify")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Densify", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.omfs.role = 'GUIDE'
    before = [tuple(v.co) for v in mesh.vertices]
    tree = BVHTree.FromPolygons(before, [tuple(p.vertices) for p in mesh.polygons])
    for p in mesh.polygons:
        p.select = p.center.z > 0      # the upper half
    picked = {i for p in mesh.polygons if p.select for i in p.vertices}
    guide_workshop.make_active(bpy.context, obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.omfs.densify()
    bpy.ops.object.mode_set(mode='OBJECT')
    assert [tuple(v.co) for v in mesh.vertices[:len(before)]] == before, "existing vertices never move"
    new = [v.co for v in mesh.vertices[len(before):]]
    assert len(new) > 0
    assert all(tree.find_nearest(co)[3] < 1e-4 for co in new), "new vertices lie on the old surface"
    assert all(co.z > -1e-4 for co in new), "only the selection is densified"
    assert all(len(p.vertices) == 3 for p in mesh.polygons), "still triangles (seam faces split too)"
    assert min(p.area for p in mesh.polygons) > 1e-3, "no flat sliver triangle"
    assert len(picked) < len(mesh.vertices)


def test_guide_workshop():
    scene = bpy.context.scene
    bone = box_object("Workshop bone", 'BONE', (-300, 0, 0), (40, 20, 20))
    locking.set_locked(bone, True)
    vertices_before = [tuple(v.co) for v in bone.data.vertices]
    scene.omfs_guide.source = bone

    def run(tool_id):
        error = guide_workshop.TOOLS[tool_id].run(bpy.context)
        assert error is None, f"{tool_id}: {error}"

    copy = guide_workshop.paint_copy(bpy.context)   # made by the first stroke (Sculpt needs a window)
    assert copy is not None and copy.data != bone.data and not locking.is_locked(copy)
    assert copy["omfs_copy_of"] == bone.name and guide_workshop.paint_copy(bpy.context) == copy, "made once"
    bone.hide_set(True)                              # painting: the copy stands in for the bone

    # Sculpt operators crash without a window: stand in for Mask Extract's output (a shell with a
    # Solidify modifier) and test our post-processing of it.
    guide = box_object("Extracted", 'OTHER', (-300, 0, 10), (40, 20, 0.001))
    guide.modifiers.new("Solidify", 'SOLIDIFY')
    guide_workshop.finish_extracted_guide(bpy.context, guide)
    assert guide.omfs.role == 'GUIDE' and guide.name.startswith("Guide")
    assert [m.type for m in guide.modifiers] == ['DISPLACE', 'SOLIDIFY'], [m.type for m in guide.modifiers]
    assert abs(guide.modifiers[1].thickness - params.GUIDE_THICKNESS) < 1e-6
    assert not guide.modifiers[1].use_even_offset, "even thickness spikes at sharp border folds"
    assert copy.hide_get(), "the copy is hidden after an extraction: clicks must reach the guide"
    assert not bone.hide_get(), "and the bone is back"
    assert not any(m.show_in_editmode for m in guide.modifiers), "Edit Mode shows the footprint, not the shell"
    copy.hide_set(False)
    assert scene.omfs_guide.guide == guide, "the extracted guide is the one the tools modify"
    guide_workshop.make_active(bpy.context, copy)
    run('ADD_CLEARANCE')                             # copy active: the modifier goes to the guide
    assert not copy.modifiers and bpy.context.active_object == guide
    assert [m.type for m in guide.modifiers] == ['DISPLACE', 'SOLIDIFY'], "updated, not duplicated"

    # Adjustable cylinders, Boolean, apply.
    coll = guide_workshop.workshop_collection(scene)
    tube = cylinders.add(bpy.context, 10.0, params.SLEEVE_TUBE_LENGTH, "Tube", coll, Matrix.Translation((-300, 0, 10)))
    assert abs(tube.dimensions.x - 10.0) < 1e-3
    # Adjustable: pulling the top end keeps the bottom end in place; typed sizes are exact.
    bore = cylinders.add(bpy.context, 4.1, params.SLEEVE_TUBE_LENGTH, "Bore", coll,
                         Matrix.LocRotScale(Vector((-300, 0, 10)), Euler((radians(90), 0, 0)).to_quaternion(), None))
    bottom = bore.location - cylinders.axis(bore) * bore.omfs_cylinder.length / 2
    cylinders.set_length(bore, 12.0, 'BOTTOM')
    bpy.context.view_layer.update()
    assert (bore.location - cylinders.axis(bore) * 6.0 - bottom).length < 1e-5
    bore.omfs_cylinder.diameter = 4.25
    bpy.context.view_layer.update()
    assert abs(bore.dimensions.x - 4.25) < 1e-3 and abs(bore.dimensions.z - 12.0) < 1e-3
    bpy.data.objects.remove(bore)
    guide_workshop.make_active(bpy.context, guide)
    scene.omfs_guide.cutter = tube
    run('BOOL_UNION')
    assert guide.modifiers[-1].type == 'BOOLEAN' and guide.modifiers[-1].object == tube
    run('ADD_REMESH')
    run('APPLY_MODIFIERS')
    assert not guide.modifiers and len(guide.data.polygons) > 0

    # Zones: face sets painted in Sculpt Mode are a face attribute; stand in for two strokes.
    assert guide_workshop.chosen_zone(bpy.context)[0] is None, "no zone painted yet"
    sets = copy.data.attributes.new(guide_workshop.FACE_SETS, 'INT', 'FACE')
    sets.data.foreach_set("value", [1, 1, 2, 2, 3, 1])   # base 1, then zones 2 and 3
    zone, faces = guide_workshop.chosen_zone(bpy.context)
    assert zone == 3 and list(faces) == [False, False, False, False, True, False], "last painted by default"
    scene.omfs_guide.zone = 2
    zone, faces = guide_workshop.chosen_zone(bpy.context)
    assert zone == 2 and int(faces.sum()) == 2 and "n° 2" in guide_workshop.zone_summary(bpy.context)
    guide_workshop.set_mask(copy.data, faces)
    mask = [d.value for d in copy.data.attributes[guide_workshop.MASK].data]
    in_zone = {copy.data.loops[i].vertex_index for p in copy.data.polygons if faces[p.index] for i in p.loop_indices}
    assert all((m == 1.0) == (i in in_zone) for i, m in enumerate(mask)), mask
    assert [tuple(v.co) for v in bone.data.vertices] == vertices_before, "zones live on the copy only"

    bpy.context.view_layer.objects.active = bone
    assert guide_workshop.editable_active(bpy.context)[0] == guide, "locked anatomy: the tools act on the guide"
    assert [tuple(v.co) for v in bone.data.vertices] == vertices_before, "the bone must never change"
    bpy.data.objects.remove(copy)


def test_guide_cut_edge():
    # Uses the cut from test_osteotomy: frame = world axes, plane at z = -45 (flat).
    scene = bpy.context.scene
    cut = osteotomy.find_cut(scene)
    cut.omfs_cut.profile_angle = cut.omfs_cut.frontal_angle = 0
    plane_z = (cut.matrix_world @ Vector((0, 0, osteotomy.height_on_plane(cut.omfs_cut, 0, 0)))).z
    for tool_id, check in (('CUT_KEEP_ABOVE', lambda zs: abs(min(zs) - plane_z) < 1e-3 and max(zs) > plane_z),
                           ('CUT_KEEP_BELOW', lambda zs: abs(max(zs) - plane_z) < 1e-3 and min(zs) < plane_z)):
        guide = box_object("Edge guide", 'GUIDE', (0, -40, plane_z), (20, 10, 20))
        guide_workshop.make_active(bpy.context, guide)
        assert guide_workshop.TOOLS[tool_id].run(bpy.context) is None
        assert guide_workshop.TOOLS[tool_id].run(bpy.context) is None   # again: updates, no duplicate
        assert [m.name for m in guide.modifiers] == [guide_workshop.CUT_EDGE_MODIFIER]
        assert guide_workshop.TOOLS['APPLY_MODIFIERS'].run(bpy.context) is None
        zs = [(guide.matrix_world @ v.co).z for v in guide.data.vertices]
        assert check(zs), (tool_id, plane_z, min(zs), max(zs))
        bpy.data.objects.remove(guide)


def bumpy_shell(name):
    """Open, wavy surface standing in for an extracted guide (before its thickness is applied)."""
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=20, y_segments=20, size=10)
    for v in bm.verts:
        v.co.z = 0.6 * ((v.index * 7919) % 13) / 13   # deterministic bumps
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.omfs.role = 'GUIDE'
    solidify = obj.modifiers.new("Épaisseur", 'SOLIDIFY')
    solidify.thickness, solidify.offset = 3.0, 1.0
    return obj


def test_smooth_outside_keeps_intrados():
    plain, smoothed = bumpy_shell("Plain shell"), bumpy_shell("Smoothed shell")
    inner = [tuple(v.co) for v in smoothed.data.vertices]
    guide_workshop.make_active(bpy.context, smoothed)
    assert guide_workshop.TOOLS['SMOOTH_OUTSIDE'].run(bpy.context) is None
    assert [m.type for m in smoothed.modifiers] == ['SOLIDIFY', 'SMOOTH']
    assert guide_workshop.TOOLS['HIDE_AND_SMOOTH'].run(bpy.context), "must refuse while thickness is unapplied"
    for obj in (plain, smoothed):
        guide_workshop.make_active(bpy.context, obj)
        assert guide_workshop.TOOLS['APPLY_MODIFIERS'].run(bpy.context) is None
    def nearest_distance(obj, point):
        return min((v.co - Vector(point)).length for v in obj.data.vertices)

    assert max(nearest_distance(smoothed, p) for p in inner) < 1e-5, "the intrados must not move"
    moved = max(nearest_distance(plain, v.co) for v in smoothed.data.vertices)
    assert moved > 0.05, f"the outer face should be smoothed ({moved})"
    for obj in (plain, smoothed):
        bpy.data.objects.remove(obj)


def test_select_linked_flat():
    cube = box_object("Flat cube", 'GUIDE', (500, 0, 0), (10, 10, 10))
    guide_workshop.make_active(bpy.context, cube)
    bpy.ops.object.mode_set(mode='EDIT')
    for angle, expected in ((40, 1), (100, 6)):   # cube faces meet at 90 degrees
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        cube.data.polygons[0].select = True
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.omfs.select_linked_flat(sharpness=radians(angle))
        bpy.ops.object.mode_set(mode='OBJECT')
        count = sum(f.select for f in cube.data.polygons)
        assert count == expected, (angle, count)
        bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(cube)


def assert_same_matrix(a, b, label, tol=1e-4):
    diff = max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))
    assert diff < tol, f"{label}: matrices differ by {diff}"


def test_occlusion_blocks_are_rigid():
    scene = bpy.context.scene
    upper = box_object("Upper scan", 'SCAN', (0, 0, 10), (40, 30, 8))
    lower = box_object("Lower scan", 'SCAN', (0, 0, -10), (40, 30, 8))
    bone = box_object("Mandible", 'BONE', (0, 5, -30), (60, 50, 30))
    lower.matrix_world = Matrix.Translation((1, 2, 3))   # e.g. after its registration
    bpy.context.view_layer.update()
    preop_lower, preop_bone = lower.matrix_world.copy(), bone.matrix_world.copy()
    shape = occlusion.shape_fingerprint(bone)

    props = scene.omfs_occlusion
    props.upper, props.lower = upper, lower
    block = occlusion.find_block(scene, 'LOWER')
    assert lower.parent is block and upper.parent is occlusion.find_block(scene, 'UPPER')
    props.pick_lower = bone                                # picker adds, then clears itself
    assert bone.parent is block and props.pick_lower is None
    bpy.context.view_layer.update()
    for obj, pose in ((lower, preop_lower), (bone, preop_bone)):
        assert_same_matrix(obj.matrix_world, pose, f"{obj.name} unchanged on joining")
    assert occlusion.movement_summary(scene, block) is None

    occlusion.set_ghosts(scene, block, True)
    ghost = occlusion.ghost_of(bone)
    assert ghost.data is bone.data                         # linked duplicate: no new mesh data
    motion = Matrix.Translation((0, -5, 2)) @ Matrix.Rotation(radians(4), 4, 'X')
    block.matrix_world = motion @ block.matrix_world
    bpy.context.view_layer.update()
    assert_same_matrix(bone.matrix_world, motion @ preop_bone, "bone moves with the block")
    assert_same_matrix(ghost.matrix_world, preop_bone, "ghost stays pre-op")
    assert occlusion.shape_fingerprint(bone) == shape and occlusion.shape_intact(bone)
    (_x, y, z), angle, _in_frame = occlusion.movement_summary(scene, block)
    assert abs(angle - 4) < 1e-3, angle

    occlusion.remove_member(bone)                          # back to pre-op, ghost gone
    bpy.context.view_layer.update()
    assert_same_matrix(bone.matrix_world, preop_bone, "removed bone back at pre-op")
    assert occlusion.ghost_of(bone) is None and bone.name in bpy.data.objects

    occlusion.back_to_preop(block)
    bpy.context.view_layer.update()
    assert_same_matrix(lower.matrix_world, preop_lower, "block back to pre-op")

    lower.data.vertices[0].co.x += 0.01                    # any change to the shape is caught
    lower.data.update()
    bpy.context.view_layer.update()
    assert not occlusion.shape_intact(lower)

    props.upper = props.lower = None
    for obj in (upper, lower, bone, *occlusion.members(block)):
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


def dense_box(name, role, center, size):
    obj = box_object(name, role, center, size)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=12, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def test_occlusion_springs_and_overlap():
    scene = bpy.context.scene
    props = scene.omfs_occlusion
    upper = dense_box("Upper arch", 'SCAN', (0, 0, 10), (40, 30, 8))    # occlusal surface at z = 6
    lower = dense_box("Lower arch", 'SCAN', (0, 0, -10), (40, 30, 8))   # occlusal surface at z = -6
    props.upper, props.lower = upper, lower
    block = occlusion.find_block(scene, 'UPPER')                        # the lower arch never moves
    shape = occlusion.shape_fingerprint(upper)
    # Springs "want" 1 mm of overlap and the upper 2 mm to the right (-X).
    for i, (x, y) in enumerate(((-12, -8), (12, -8), (0, 8)), start=1):
        occlusion_seating.add_point(scene, f"OCC_U{i}", f"Haut {i}", Vector((x + 2, y, 7)), upper)
        occlusion_seating.add_point(scene, f"OCC_L{i}", f"Bas {i}", Vector((x, y, -6)), lower)
    bpy.context.view_layer.update()
    assert len(occlusion_seating.point_pairs(scene)) == 3

    def bottom_of_upper():
        return min((upper.matrix_world @ v.co).z for v in upper.data.vertices)

    upside_down = Matrix.Translation((5, 3, -20)) @ Matrix.Rotation(radians(180), 4, 'X')
    for tolerance, expected, start in ((0.2, 0.2, None), (5.0, 1.0, None), (0.2, 0.2, upside_down)):
        occlusion.back_to_preop(block)   # resisted at 0.2 mm, pure point fit, then from upside down
        if start is not None:
            block.matrix_world = start @ block.matrix_world
        bpy.context.view_layer.update()
        props.tolerance_anterior = props.tolerance_posterior = tolerance
        props.spring_share = 100.0   # springs only: the magnet would also pull the box sides together
        bpy.ops.omfs.seat_occlusion()
        overlap = -6.0 - bottom_of_upper()
        assert abs(overlap - expected) < 0.1, (tolerance, overlap)
        shift = block.matrix_world.translation.x - occlusion.preop_matrix(block).translation.x
        assert abs(shift + 2.0) < 0.1, (tolerance, shift)
        assert_same_matrix(lower.matrix_world, Matrix(), "the lower arch stays")
        assert abs(props.max_overlap - expected) < 0.15, (tolerance, props.max_overlap)
        assert props.area_anterior > 0 and props.area_left > 0 and props.area_right > 0
    assert lower.data.color_attributes.get(occlusion_seating.MAP_ATTRIBUTE) is not None

    # Springs that stop 1 mm short of contact: alone they leave the gap, with the magnet it closes.
    for up, _low in occlusion_seating.point_pairs(scene):
        up.matrix_world = Matrix.Translation(up.matrix_world.translation - Vector((2, 0, 2)))   # z = 5, no shift
    gaps = {}
    for share in (100.0, 70.0):
        occlusion.back_to_preop(block)
        bpy.context.view_layer.update()
        props.tolerance_anterior = props.tolerance_posterior = 0.2
        props.spring_share = share
        bpy.ops.omfs.seat_occlusion()
        gaps[share] = bottom_of_upper() + 6.0
    assert abs(gaps[100.0] - 1.0) < 0.1, gaps
    assert gaps[70.0] < params.OCC_CONTACT, gaps                         # pulled into contact
    assert occlusion_seating.tripod(props), (props.area_right, props.area_anterior, props.area_left)
    occlusion.back_to_preop(block)
    bpy.context.view_layer.update()
    occlusion_seating.add_point(scene, "OCC_U4", "Haut 4", Vector((0, -8, 7)), upper)   # all lower points
    occlusion_seating.add_point(scene, "OCC_L4", "Bas 4", Vector((0, -8, -6)), lower)  # on y = -8 but one
    for obj in occlusion_seating.occlusion_points(scene):
        if obj.omfs.landmark in ("OCC_U3", "OCC_L3"):
            bpy.data.objects.remove(obj)
    bpy.context.view_layer.update()
    assert occlusion_seating.points_spread(occlusion_seating.seatings(scene)[0]) < occlusion_seating.MIN_SPREAD   # one line: refused
    assert occlusion.shape_fingerprint(upper) == shape          # only the block moved

    below = box_object("Edge probe", 'SCAN', (0, 0, 0), (10, 10, 10))          # point straight below a
    d, _c, _n = occlusion_seating.signed_distances(np.array([[5.0, 0.0, -9.0]]), below, 20.0)   # side face
    assert d[0] > 0, d                                                      # closest = an edge: outside
    bpy.data.objects.remove(below)
    # Canine springs (no shift wanted) against the three others (2 mm shift): the canines win more
    # as their weight grows.
    for up, _low in occlusion_seating.point_pairs(scene):
        up.matrix_world = Matrix.Translation(up.matrix_world.translation + Vector((2, 0, 2)))   # back to x + 2, z = 7
    for key, x in (('CR', -10), ('CL', 10)):
        occlusion_seating.add_point(scene, f"OCC_U{key}", key, Vector((x, -12, 7)), upper)
        occlusion_seating.add_point(scene, f"OCC_L{key}", key, Vector((x, -12, -6)), lower)
    bpy.context.view_layer.update()
    assert [occlusion_seating.pair_key(up) for up, _low in occlusion_seating.point_pairs(scene)][:2] == ['CR', 'CL']
    left = {}
    for weight in (1.0, 3.0):
        occlusion.back_to_preop(block)
        bpy.context.view_layer.update()
        props.tolerance_anterior = props.tolerance_posterior = 5.0
        props.spring_share, props.canine_weight = 100.0, weight
        bpy.ops.omfs.seat_occlusion()
        lengths = occlusion_seating.spring_lengths(scene)                 # canines first
        left[weight] = (np.mean(lengths[:2]), np.mean(lengths[2:]))
    assert left[3.0][0] < 0.7 * left[1.0][0], left                       # canines held closer
    assert left[3.0][1] > left[1.0][1], left                             # at the others' expense
    for key in ('CR', 'CL'):
        occlusion_seating.remove_pair(scene, key)

    d = np.array([-3.0, -0.4, -0.4, -0.3, 0.2])                        # one spike, then a real overlap
    area = np.array([0.01, 0.05, 0.05, 0.05, 1.0])
    assert abs(occlusion_seating.robust_overlap(d, area) - 0.4) < 1e-9

    icons = bpy.types.UILayout.bl_rna.functions['prop'].parameters['icon'].enum_items.keys()
    assert all(icon in icons for icon, _text in occlusion_seating.legend())

    colours = occlusion_seating.overlap_colours(np.array([-2.0, -0.9, -0.6, -0.2, 0.1, 0.5, 3.0, np.nan]))
    expected = [occlusion_seating.MAGENTA, occlusion_seating.RED, occlusion_seating.ORANGE, occlusion_seating.YELLOW,
                occlusion_seating.GREEN, occlusion_seating.BLUE, occlusion_seating.GREY, occlusion_seating.GREY]
    assert np.allclose(colours, expected), colours

    props.upper = props.lower = None
    for obj in occlusion_seating.occlusion_points(scene) + [upper, lower]:
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


def test_occlusion_anterior_limit():
    """Springs asking for 1 mm of overlap: the front teeth (in front of the canine points) stop at their own limit, the back teeth at theirs. Then a hand position is kept."""
    scene = bpy.context.scene
    props = scene.omfs_occlusion
    upper = dense_box("Upper arch", 'SCAN', (0, 0, 10), (46, 38, 8))             # wider: no shared edges
    lower = dense_box("Lower arch", 'SCAN', (0, -4, -10), (40, 22, 8))           # back teeth, top at z = -6
    front = dense_box("Front teeth", 'SCAN', (0, 11, -10), (40, 8, 8))           # y > 7, top at z = -6
    bm = bmesh.new()
    bm.from_mesh(lower.data)
    bm.from_mesh(front.data)
    bm.to_mesh(lower.data)
    bm.free()
    bpy.data.objects.remove(front)
    props.upper, props.lower = upper, lower
    block = occlusion.find_block(scene, 'UPPER')
    for key, x, y in (('CR', -12, 5), ('CL', 12, 5), ('1', -12, -12), ('2', 12, -12)):
        occlusion_seating.add_point(scene, f"OCC_U{key}", key, Vector((x, y, 7)), upper)
        occlusion_seating.add_point(scene, f"OCC_L{key}", key, Vector((x, y, -6)), lower)
    bpy.context.view_layer.update()
    assert occlusion_seating.zone_ready(scene)
    upper_plane, lower_plane = occlusion_seating.zone_planes(scene)
    assert upper_plane[1][1] > 0.99 and lower_plane[1][1] > 0.99                 # towards the incisors (+Y)
    assert abs(upper_plane[0][1] - (5 - params.OCC_CANINE_ZONE_MARGIN)) < 1e-6   # behind the canine tip
    assert abs(lower_plane[0][1] - 5) < 1e-6
    bpy.ops.omfs.toggle_occlusion_points()                                       # one click hides points, springs, zone lines
    assert not occlusion_seating.points_visible(scene)
    assert all(o.hide_get() for o in occlusion_seating.occlusion_points(scene))
    assert occlusion_seating.zone_ready(scene) and len(occlusion_seating.point_pairs(scene)) == 4  # still active

    def overlaps():
        """How deep the upper's underside goes below the lower's top (z = -6) over the front and back teeth."""
        points = registration.vertex_coords(upper)
        bottom = points[np.abs(points[:, 2] - points[:, 2].min()) < 3.0]
        over = np.abs(bottom[:, 0]) < 19.0
        front_z = bottom[over & (bottom[:, 1] > 8.0) & (bottom[:, 1] < 14.0)][:, 2].min()
        back_z = bottom[over & (bottom[:, 1] < 0.0) & (bottom[:, 1] > -14.0)][:, 2].min()
        return -6.0 - front_z, -6.0 - back_z

    props.spring_share = 100.0
    for anterior, posterior in ((0.0, 0.3), (0.2, 0.5)):
        occlusion.back_to_preop(block)
        bpy.context.view_layer.update()
        props.tolerance_anterior, props.tolerance_posterior = anterior, posterior
        bpy.ops.omfs.seat_occlusion()
        front_overlap, back_overlap = overlaps()
        assert front_overlap < anterior + 0.03, (anterior, front_overlap)
        assert back_overlap < posterior + 0.03, (posterior, back_overlap)
        assert back_overlap > front_overlap + 0.1, (front_overlap, back_overlap)  # the limits differ
        assert props.max_overlap_anterior < anterior + 0.05, props.max_overlap_anterior
        assert props.max_overlap_posterior < posterior + 0.05, props.max_overlap_posterior
    bpy.ops.omfs.toggle_occlusion_points()                                       # seated while hidden; shown again
    assert occlusion_seating.points_visible(scene)
    bpy.ops.omfs.toggle_overlap_map()                                            # the map hides them ...
    assert props.show_map and not occlusion_seating.points_visible(scene)
    bpy.ops.omfs.toggle_overlap_map()                                            # ... and gives them back
    assert not props.show_map and occlusion_seating.points_visible(scene)

    props.upper = None                                                           # field cleared: no zone, no error
    assert not occlusion_seating.zone_ready(scene)
    props.upper = upper

    # Without the canine springs there is no anterior zone: the posterior limit is everywhere.
    for key in ('CR', 'CL'):
        occlusion_seating.remove_pair(scene, key)
    bpy.context.view_layer.update()
    assert not occlusion_seating.zone_ready(scene)

    # Hand position, no spring at all: "from here" keeps it and the magnet only closes the gap.
    for key in ('1', '2'):
        occlusion_seating.remove_pair(scene, key)
    occlusion.back_to_preop(block)
    props.spring_share, props.tolerance_posterior = 0.0, 0.2
    block.matrix_world = Matrix.Translation((3, 0, -6)) @ block.matrix_world     # 6 mm gap: out of the magnet's range
    bpy.context.view_layer.update()
    try:
        bpy.ops.omfs.seat_occlusion(from_here=True)
    except RuntimeError as error:
        assert "Rien ne guide" in str(error)
    else:
        raise AssertionError("expected an error: nothing can move the block")

    block.matrix_world = Matrix.Translation((0, 0, -4)) @ block.matrix_world     # 2 mm gap, 3 mm to the left
    bpy.context.view_layer.update()
    bpy.ops.omfs.seat_occlusion(from_here=True)
    assert abs(block.matrix_world.translation.x - occlusion.preop_matrix(block).translation.x - 3) < 0.01
    assert abs(block.matrix_world.translation.y - occlusion.preop_matrix(block).translation.y) < 0.01
    yaw = block.matrix_world.to_3x3().col[0].y                                    # X axis swinging towards Y
    assert abs(yaw) < 1e-4, yaw                                                   # the midline did not move
    assert overlaps()[1] > -0.3, overlaps()                                       # gap closed
    assert overlaps()[1] < 0.25, overlaps()

    props.spring_share = params.OCC_SPRING_SHARE
    props.upper = props.lower = None
    for obj in [upper, lower]:
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


def test_midline_spring():
    """One named spring, listed after the canines and before the numbered ones, ordinary weight."""
    scene = bpy.context.scene
    props = scene.omfs_occlusion
    upper = dense_box("Upper arch", 'SCAN', (0, 0, 10), (46, 38, 8))
    lower = dense_box("Lower arch", 'SCAN', (0, 0, -10), (40, 30, 8))
    props.upper, props.lower = upper, lower
    for key, x in (('1', 8), ('CR', -12), ('ML', 0)):
        occlusion_seating.add_point(scene, f"OCC_U{key}", key, Vector((x, 10, 6)), upper)
        occlusion_seating.add_point(scene, f"OCC_L{key}", key, Vector((x, 10, -6)), lower)
    bpy.context.view_layer.update()
    pairs = occlusion_seating.point_pairs(scene)
    assert [occlusion_seating.pair_key(up) for up, _low in pairs] == ['CR', 'ML', '1']
    assert occlusion_seating.pair_label(pairs[1][0]) == "Midline"
    assert not occlusion_seating.is_canine(pairs[1][0])
    assert list(occlusion_seating.pair_weights(occlusion_seating.point_pairs(scene), 3.0)) == [3.0, 1.0, 1.0]
    assert not occlusion_seating.zone_ready(scene)                                # the midline makes no zone
    for key in ('1', 'CR', 'ML'):
        occlusion_seating.remove_pair(scene, key)
    props.upper = props.lower = None
    for obj in [upper, lower]:
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


def test_spring_release():
    """Springs bring the arches "macro": shorter than the release distance they pull less (10 % at 0), and
    what they lose goes to the magnet; springs only (100 %) or release 0: full strength."""
    scene = bpy.context.scene
    props = scene.omfs_occlusion
    upper = dense_box("Upper arch", 'SCAN', (0, 0, 10), (40, 30, 8))
    lower = dense_box("Lower arch", 'SCAN', (0, 0, -10), (40, 30, 8))
    props.upper, props.lower = upper, lower
    for key, gap in (('1', 0.0), ('2', 1.5), ('3', 6.0)):
        occlusion_seating.add_point(scene, f"OCC_U{key}", key, Vector((0, int(key) * 5, 6)), upper)
        occlusion_seating.add_point(scene, f"OCC_L{key}", key, Vector((0, int(key) * 5, 6 - gap)), lower)
    bpy.context.view_layer.update()
    pairs = occlusion_seating.seatings(scene)[0].pairs
    props.spring_release = 3.0
    fade = occlusion_seating.spring_fade(scene, pairs, 0.7)
    r = params.OCC_SPRING_RESIDUAL
    assert np.allclose(fade, [r, r + (1 - r) * 0.5, 1.0]), fade
    assert np.allclose(occlusion_seating.spring_fade(scene, pairs, 1.0), 1.0), "springs only: no release"
    props.spring_release = 0.0
    assert np.allclose(occlusion_seating.spring_fade(scene, pairs, 0.7), 1.0), "release off"
    props.spring_release = params.OCC_SPRING_RELEASE
    for key in ('1', '2', '3'):
        occlusion_seating.remove_pair(scene, key)
    props.upper = props.lower = None
    for obj in [upper, lower]:
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


def test_anatomy_pickers_take_segmentation_imports():
    """Teeth / bone imported from the Segmentation workspace (made_by "Segmentation") can be chosen for the scan's
    registration and the genioplasty; the workspace's own surfaces (another scene) and generated objects cannot."""
    scene = bpy.context.scene
    imported = box_object("Dents du bas", 'TEETH', (0, 0, 0), (10, 10, 10))
    imported.omfs.made_by = "Segmentation"
    other = bpy.data.scenes.new("Segmentation test")
    working = bpy.data.objects.new("Seg · Dents du bas", imported.data)
    working.omfs.role = 'TEETH'
    other.collection.objects.link(working)
    helper = box_object("Generated", 'TEETH', (0, 0, 0), (1, 1, 1))
    helper.omfs.made_by = "Splint"
    poll = registration._is_role('TEETH')
    assert poll(None, imported), "the planning scene's imported teeth"
    assert not poll(None, working), "not the Segmentation workspace's surface (another scene)"
    assert not poll(None, helper), "not a generated object"
    bone = box_object("Mandibule", 'BONE', (0, 0, 0), (10, 10, 10))
    bone.omfs.made_by = "Segmentation"
    assert osteotomy._is_bone(None, bone), "genioplasty: the imported mandible"
    scan = box_object("Scan", 'SCAN', (0, 0, 0), (10, 10, 10))
    scan.omfs_reg["target"] = working               # chosen before the fix (stored as is, bypassing the poll)
    registration.default_partners(scene, scan)
    assert scan.omfs_reg.target == imported, "the workspace surface is replaced by its import in this scene"
    for obj in (imported, helper, bone, scan, working):
        bpy.data.objects.remove(obj)
    bpy.data.scenes.remove(other)


def test_open_border_is_ignored():
    plane = box_object("Open scan", 'SCAN', (0, 0, 0), (10, 10, 1))
    bm = bmesh.new()
    bm.from_mesh(plane.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.5], context='FACES')   # open the top
    bm.to_mesh(plane.data)
    bm.free()
    border = occlusion_seating.border_polygons(plane.data)
    assert border.sum() == 4 and not border.all(), border   # the four side faces touch the hole
    bpy.data.objects.remove(plane)


def test_flip_upper_above_lower():
    scene = bpy.context.scene
    props = scene.omfs_occlusion
    upper = dense_box("Upper stacked", 'SCAN', (1, 2, 4), (40, 30, 8))   # both scanner exports teeth up,
    lower = dense_box("Lower stacked", 'SCAN', (0, 0, 4), (40, 30, 8))   # one inside the other
    for registered in (Matrix(), Matrix.Translation((10, -5, 30)) @ Matrix.Rotation(radians(20), 4, 'X')):
        lower.matrix_world = registered
        props.upper, props.lower = upper, lower
        bpy.ops.omfs.flip_upper_arch(gap=5.0)
        bpy.context.view_layer.update()
        relative = lower.matrix_world.inverted() @ upper.matrix_world   # in the lower scan's frame
        assert_close(relative.col[2].xyz, (0, 0, -1), "teeth down")
        assert_close(relative.col[1].xyz, (0, 1, 0), "incisors stay anterior")
        low_top = max(v.co.z for v in lower.data.vertices)
        up_bottom = min((relative @ v.co).z for v in upper.data.vertices)
        assert abs(up_bottom - low_top - 5.0) < 1e-3, up_bottom - low_top
        block = occlusion.find_block(scene, 'UPPER')
        assert occlusion.movement_summary(scene, block) is None            # not a surgical movement
        assert_close(block.matrix_world.translation, occlusion.world_bbox_centre(upper), "block recentred")
        props.upper = props.lower = None
    for obj in (upper, lower):
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


def test_section_plane():
    scene = bpy.context.scene
    props = scene.omfs_section
    props.use_frame, props.thickness = False, 2.0
    for orientation, normal in (('AXIAL', (0, 0, 1)), ('CORONAL', (0, -1, 0)), ('SAGITTAL', (1, 0, 0))):
        props.orientation, props.offset = orientation, 10.0
        m = section.plane_matrix(scene)
        assert_close(m.col[2].xyz, normal, f"{orientation} normal (up / anterior / left)")
        assert abs(m.to_3x3().determinant() - 1) < 1e-6, f"{orientation}: right-handed axes"
        assert_close(m.translation, Vector(normal) * 10, f"{orientation}: 10 mm along the normal")
        n = Vector(normal)
        assert section.in_slab(scene, n * 10.9) and section.in_slab(scene, n * 9.1), "inside the 2 mm slab"
        assert not section.in_slab(scene, n * 11.1) and not section.in_slab(scene, n * 8.9), "outside the slab"
        view_dir = section.view_rotation(scene) @ Vector((0, 0, 1))   # towards the viewer
        assert abs(abs(view_dir.dot(n)) - 1) < 1e-6, f"{orientation}: view perpendicular to the plane"
    props.orientation = 'AXIAL'
    assert_close(section.view_rotation(scene) @ Vector((0, 0, 1)), (0, 0, -1), "axial seen from the feet")
    props.flip_view = True
    assert_close(section.view_rotation(scene) @ Vector((0, 0, 1)), (0, 0, 1), "other side: from above")
    props.flip_view = False

    # The reference frame drives the standard planes (hook for the clinical planes).
    frame = reference_frame.find_frame(scene)
    props.use_frame, props.offset = True, 0.0
    assert_close(section.plane_matrix(scene).translation, frame.matrix_world.translation, "offset 0 = frame origin")

    # A custom object: its local XY plane.
    empty = bpy.data.objects.new("Section test", None)
    scene.collection.objects.link(empty)
    empty.matrix_world = Matrix.Translation((5, 6, 7)) @ Euler((0.4, 0, 0)).to_matrix().to_4x4()
    props.orientation, props.custom, props.offset = 'CUSTOM', empty, 3.0
    z = empty.matrix_world.col[2].xyz
    assert_close(section.plane_matrix(scene).translation, Vector((5, 6, 7)) + 3 * z, "custom plane, shifted 3 mm")
    props.custom = None
    assert section.plane_matrix(scene) is None and section.slab_planes(scene) is None
    bpy.data.objects.remove(empty)
    props.orientation, props.offset = 'AXIAL', 0.0


def section_measures(obj, scene):
    """(contour length, cap area, number of loops, open lines?) of obj on the section plane."""
    plane = section.plane_matrix(scene)
    cut = section_contours.cut(obj, plane, bpy.context.evaluated_depsgraph_get())
    length = float(np.linalg.norm(cut.segments[:, 1] - cut.segments[:, 0], axis=1).sum())
    area = section_contours.rows_area(section_contours.cap_rows(plane, cut.loops))
    return length, area, len(cut.loops), cut.open_lines


def test_section_contours():
    scene = bpy.context.scene
    props = scene.omfs_section
    props.use_frame, props.orientation, props.offset = False, 'AXIAL', 0.0
    box = dense_box("Contour box", 'BONE', (0, 0, 0), (20, 10, 10))
    length, area, loops, open_lines = section_measures(box, scene)
    assert abs(length - 60) < 1e-3 and abs(area - 200) < 1e-3 and loops == 1 and not open_lines, (length, area)
    plane = section.plane_matrix(scene)
    cut = section_contours.cut(box, plane, bpy.context.evaluated_depsgraph_get())
    points, tris = section_contours.rows_quads(plane, section_contours.cap_rows(plane, cut.loops))
    assert tris.dtype == np.int32 and points.dtype == np.float32, "GPU buffers: 4-byte ints and floats"

    props.offset = -5 + 3 * 10 / 13   # exactly through a row of vertices
    length, area, loops, _ = section_measures(box, scene)
    assert abs(length - 60) < 1e-3 and abs(area - 200) < 1e-3 and loops == 1, ("through vertices", length, area)
    props.offset = 7.0
    assert section_measures(box, scene)[0] == 0, "plane above the box: nothing"

    box.location.z = 6.0   # the box follows its object transform
    length, area, _, _ = section_measures(box, scene)
    assert abs(area - 200) < 1e-3, ("moved box", area)
    box.location.z = 0.0

    # A box with a box-shaped hole: the cap is a ring (even-odd fill).
    props.offset = 0.0
    hole = box_object("Contour hole", 'BONE', (0, 0, 0), (10, 4, 20))
    tube = box.copy()
    tube.data = box.data.copy()
    scene.collection.objects.link(tube)
    bm = bmesh.new()
    bm.from_mesh(tube.data)
    bm.from_mesh(hole.data)   # second shell inside the first
    bm.to_mesh(tube.data)
    bm.free()
    length, area, loops, _ = section_measures(tube, scene)
    assert loops == 2 and abs(area - (200 - 40)) < 1e-3, ("ring", loops, area)

    # An open surface (like a scan): outline, no cap.
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=10, y_segments=10, size=5)
    bmesh.ops.rotate(bm, verts=bm.verts, matrix=Matrix.Rotation(radians(90), 3, 'X'))
    sheet = bpy.data.objects.new("Contour sheet", bpy.data.meshes.new("Contour sheet"))
    bm.to_mesh(sheet.data)
    bm.free()
    scene.collection.objects.link(sheet)
    length, area, loops, open_lines = section_measures(sheet, scene)
    assert open_lines and loops == 0 and abs(length - 10) < 1e-3, ("open sheet", length, loops)

    for obj in (box, hole, tube, sheet):
        bpy.data.objects.remove(obj)


def test_section_overlap_and_intersection():
    scene = bpy.context.scene
    props = scene.omfs_section
    props.use_frame, props.orientation, props.offset = False, 'AXIAL', 0.0
    a = box_object("Overlap A", 'GUIDE', (0, 0, 0), (20, 10, 10))
    b = box_object("Overlap B", 'BONE', (10, 0, 0), (10, 10, 10))
    far = box_object("Overlap far", 'BONE', (0, 40, 0), (5, 5, 5))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    plane = section.plane_matrix(scene)
    hidden = [o for o in section.visible_meshes(scene) if o not in (a, b, far)]
    for obj in hidden:
        obj.hide_set(True)
    found = section_contours.overlaps(scene, plane, depsgraph)
    assert len(found) == 1 and {found[0][0], found[0][1]} == {a.name, b.name}, found
    assert abs(found[0][3] - 50) < 0.01, ("overlap 5 x 10 mm on the section", found[0][3])

    # 3D: exact common volume, without touching A or B.
    verts_before = len(a.data.vertices)
    props.inter_a, props.inter_b = a, b
    assert bpy.ops.omfs.show_intersection() == {'FINISHED'}
    overlay = analysis.find_intersection(scene)
    assert overlay.data == a.data and overlay.omfs.role == 'ANALYSIS', "overlay shares A's mesh"
    volume, closed = analysis.intersection_volume(overlay, bpy.context.evaluated_depsgraph_get())
    assert abs(volume - 500) < 0.01 and closed, ("5 x 10 x 10 mm", volume)
    b.location.x = 2.0   # live: B moves, the Boolean follows
    volume, _ = analysis.intersection_volume(overlay, bpy.context.evaluated_depsgraph_get())
    assert abs(volume - 300) < 0.01, ("after moving B 2 mm", volume)
    assert len(a.data.vertices) == verts_before and not a.modifiers, "A untouched"
    assert bpy.ops.omfs.show_intersection() == {'FINISHED'} and analysis.find_intersection(scene) is None
    assert len(a.data.vertices) == verts_before and a.data.users == 1, "removing the overlay leaves A as it was"

    # A nearly closed mesh (a few open edges, like a CT segmentation): Exact solver, snapshot.
    bm = bmesh.new()
    bm.from_mesh(b.data)
    bm.faces.ensure_lookup_table()
    bm.faces.remove(bm.faces[0])   # 4 open edges out of 12: above the threshold, so allow it here
    bm.to_mesh(b.data)
    bm.free()
    params.ANALYSIS_MAX_OPEN_EDGES, saved = 0.5, params.ANALYSIS_MAX_OPEN_EDGES
    overlay = analysis.make_intersection(scene, a, b)
    params.ANALYSIS_MAX_OPEN_EDGES = saved
    assert analysis.is_snapshot(overlay) and overlay.data != a.data and not analysis.snapshot_stale(overlay)
    b.location.x = 3.0
    bpy.context.view_layer.update()   # matrix_world follows on the next evaluation
    assert analysis.snapshot_stale(overlay), "moving B makes the snapshot stale"
    snapshot = overlay.data.name
    analysis.remove_intersection(scene)
    assert snapshot not in bpy.data.meshes and a.data.users == 1, "snapshot mesh removed with its overlay"

    # An open mesh has no inside: refused.
    sheet = bpy.data.objects.new("Overlap sheet", bpy.data.meshes.new("Overlap sheet"))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=2, y_segments=2, size=5)
    bm.to_mesh(sheet.data)
    bm.free()
    scene.collection.objects.link(sheet)
    assert isinstance(analysis.make_intersection(scene, a, sheet), str), "open mesh refused"

    props.inter_a = props.inter_b = None
    for obj in (a, b, far, sheet):
        bpy.data.objects.remove(obj)
    for obj in hidden:
        obj.hide_set(False)


def test_distance_map():
    scene = bpy.context.scene
    props = scene.omfs_section
    target = box_object("Distance target", 'TEETH', (0, 0, 0), (10, 10, 10))
    source = dense_box("Distance source", 'GUIDE', (10, 0, 0), (2, 2, 2))
    attributes_before = len(source.data.attributes)
    props.dist_source, props.dist_target = source, target
    assert bpy.ops.omfs.distance_map() == {'FINISHED'}
    d, p, q = distance_map.closest(source, bpy.context.evaluated_depsgraph_get())
    assert abs(d - 4.0) < 1e-3 and abs(p.x - 9) < 1e-3 and abs(q.x - 5) < 1e-3, ("4 mm gap", d, p, q)

    source.location.x = -4.5   # live: 0.5 mm inside the target's x = 5 face
    d, _p, _q = distance_map.closest(source, bpy.context.evaluated_depsgraph_get())
    assert abs(d - (-0.5)) < 1e-3, ("negative inside the target", d)

    source.location.x = -2.5   # nearest vertices 1.5 mm from the face: yellow with defaults 1 / 2
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = source.evaluated_get(depsgraph).data
    d = np.array([a.value for a in mesh.attributes[distance_map.DISTANCE].data])
    colours = np.array([a.color for a in mesh.attributes[distance_map.COLOUR].data])
    i = int(np.argmin(d))
    assert abs(d[i] - 1.5) < 1e-3 and np.allclose(colours[i], distance_map.YELLOW), (d[i], colours[i])
    props.dist_yellow = 1.4   # thresholds are live modifier inputs
    mesh = source.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    assert np.allclose(mesh.attributes[distance_map.COLOUR].data[i].color, distance_map.GREEN), "now green"
    props.dist_yellow = params.DIST_YELLOW

    assert len(source.data.attributes) == attributes_before, "the source mesh itself gets no attribute"
    assert bpy.ops.omfs.distance_map() == {'FINISHED'} and not source.modifiers, "one click removes the map"
    props.dist_source = props.dist_target = None
    for obj in (source, target):
        bpy.data.objects.remove(obj)


def test_snapshot_helpers():
    from types import SimpleNamespace as NS
    assert snapshot.bar_length(12.0) == 10, "10 mm bar at 12 px/mm (120 px)"
    assert snapshot.bar_length(1.0) == 100 and snapshot.bar_length(300.0) == 0.5
    assert "repository" in snapshot.folder_problem(str(REPO / "docs")), "never inside the repo"
    assert snapshot.folder_problem("") and snapshot.folder_problem("/no/such/folder")
    with tempfile.TemporaryDirectory() as folder:
        assert snapshot.folder_problem(folder) is None
        scene = bpy.context.scene
        scene.omfs_section.orientation, scene.omfs_section.offset = 'CORONAL', -12.34
        name = snapshot.snapshot_name(scene)
        assert name.startswith("section_coronal_-12.3mm_") and name.endswith(".png"), name

        # Visible part of a 100 x 60 viewport: sidebar over the right 30 px, header over the top 10 px.
        window = NS(type='WINDOW', x=0, y=0, width=100, height=60)
        area = NS(x=0, y=0, regions=[window, NS(type='UI', x=70, y=0, width=30, height=60),
                                     NS(type='TOOLS', x=0, y=0, width=1, height=60),
                                     NS(type='HEADER', x=0, y=50, width=100, height=10)])
        assert snapshot.visible_rect(area, window) == (0, 0, 70, 50), snapshot.visible_rect(area, window)

        pixels = np.zeros((50, 70, 4), np.float32)
        pixels[..., 0], pixels[..., 3] = 1, 1
        out = str(Path(folder) / "out.png")
        snapshot.save_png(pixels, out)
        result = bpy.data.images.load(out)
        assert tuple(result.size) == (70, 50), result.size
        bpy.data.images.remove(result)
    scene.omfs_section.orientation, scene.omfs_section.offset = 'AXIAL', 0.0


def test_segment_movement():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # frame axes = world axes (x left, y posterior)
    for obj in [o for o in scene.objects if o.omfs.role == 'SEGMENT']:
        obj.omfs.role = 'OTHER'                       # earlier tests' segments: not the one moved here
    seg = box_object("Chin segment", 'SEGMENT', (0, -10, -60), (40, 30, 15))
    landmarks.place(scene, 'POG', (0, -25, -58), seg)
    landmarks.place(scene, 'ME', (0, -20, -67), seg)
    bpy.context.view_layer.update()                  # matrix_world of the new landmarks
    vertices = [v.co.copy() for v in seg.data.vertices]
    movement.init_segment(scene, seg)
    props = seg.omfs_move
    assert movement.active_segment(scene) == seg and props.pivot_kind == 'POG'
    assert props.axes == 'FRAME', "no osteotomy plane: reference frame axes"
    assert seg.lock_location[0] and seg.lock_rotation[0], "G / R must not bypass the movement values"
    preop = seg.matrix_world.copy()

    props.anterior, props.up, props.left = 6.0, -2.0, 1.0
    assert_close(seg.matrix_world.translation, (1, -6, -2), "6 forward (-y), 2 down, 1 left")
    rows = {name: d for name, _p, d in movement.tracked_rows(scene, seg)}
    assert_close(rows["Pogonion"], (1, 6, -2), "pogonion moves by the translation")
    assert movement.describe(*rows["Pogonion"]) == ["6.0 mm avance", "2.0 mm bas", "1.0 mm gauche"]

    props.pitch = radians(10)                        # front up, about the pogonion
    rows = {name: d for name, _p, d in movement.tracked_rows(scene, seg)}
    assert_close(rows["Pogonion"], (1, 6, -2), "the pivot only translates")
    # Menton is 5 mm behind, 9 mm below the pogonion: front up swings it forward.
    angle = radians(10)
    y, z = 5 * np.cos(angle) - 9 * np.sin(angle), -5 * np.sin(angle) - 9 * np.cos(angle)
    assert_close(rows["Menton"], (1, 6 + (5 - y), -2 + (z + 9)), "menton about the pogonion")
    assert rows["Menton"][1] > 6 and rows["Menton"][2] < -2, "forward and down"

    for kind in ('ME', 'CENTRE', 'POG'):              # changing the pivot never moves the segment
        before = seg.matrix_world.copy()
        props.pivot_kind = kind
        assert_same_matrix(seg.matrix_world, before, f"pivot {kind}")
    assert abs(props.anterior - 6.0) < 1e-4, "back on the pogonion: same numbers"

    props.roll, props.yaw = radians(5), radians(-3)
    menton = landmarks.find(scene, 'ME').matrix_world.translation
    moved = seg.matrix_world @ preop.inverted() @ menton
    d = movement.displacement(scene, seg, menton)
    assert_close((moved - menton), (d[0], -d[1], d[2]), "labels = where the segment takes the point")

    props.yaw = radians(5)                           # + yaw: the front turns to the left
    front = seg.matrix_world @ preop.inverted() @ Vector((0, -40, -60))
    pivot_now = seg.matrix_world @ preop.inverted() @ Vector((0, -25, -58))
    assert (front - pivot_now).x > 0
    props.roll, props.yaw = 0.0, radians(0)
    props.roll = radians(5)                          # + roll: the left side goes up
    side_l = seg.matrix_world @ preop.inverted() @ Vector((20, -25, -58))
    side_r = seg.matrix_world @ preop.inverted() @ Vector((-20, -25, -58))
    assert side_l.z > side_r.z

    # The cutting guide follows the same movement, with its Boolean tools; it goes back exactly.
    guide = box_object("Cut guide", 'GUIDE', (0, -30, -55), (30, 4, 10))
    sleeve = box_object("Sleeve", 'OTHER', (0, -33, -55), (4, 4, 6))
    hole = guide.modifiers.new("Trou", 'BOOLEAN')
    hole.object = sleeve
    guide_preop, sleeve_preop = guide.matrix_world.copy(), sleeve.matrix_world.copy()
    movement.guide_follows(scene, seg, guide)
    w = seg.matrix_world @ preop.inverted()
    assert_same_matrix(guide.matrix_world, w @ guide_preop, "guide moved like the segment")
    assert_same_matrix(sleeve.matrix_world, w @ sleeve_preop, "its sleeve too")
    props.anterior = 8.0                             # moved again: re-applying starts from pre-op
    movement.guide_follows(scene, seg, guide)
    assert_same_matrix(guide.matrix_world, seg.matrix_world @ preop.inverted() @ guide_preop, "re-applied")
    movement.guide_back(guide)
    assert_same_matrix(guide.matrix_world, guide_preop, "guide back")
    assert_same_matrix(sleeve.matrix_world, sleeve_preop, "sleeve back")
    assert not movement.guide_moved(guide)
    for obj in (guide, sleeve):
        bpy.data.objects.remove(obj)

    bpy.ops.omfs.segment_preop()
    assert_same_matrix(seg.matrix_world, preop, "back to pre-op")
    assert all((v.co - c).length == 0 for v, c in zip(seg.data.vertices, vertices)), "mesh untouched"
    bpy.data.objects.remove(seg)
    for lm in ('POG', 'ME'):
        bpy.data.objects.remove(landmarks.find(scene, lm))


def test_reapplied_cut_keeps_movement():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)
    bone = box_object("Chin bone 2", 'BONE', (0, 0, -50), (60, 40, 30))
    landmarks.place(scene, 'MF_R', (-20, 0, -40), bone)
    landmarks.place(scene, 'MF_L', (20, 0, -40), bone)
    landmarks.place(scene, 'POG', (0, -20, -55), bone)
    bpy.context.view_layer.update()
    cut = osteotomy.add_cut(scene)
    cut.omfs_cut.source = bone
    segment, _proximal = osteotomy.apply_cut(bpy.context, cut)
    assert segment.omfs_move.has_preop and segment.omfs_move.pivot_kind == 'POG'
    segment.omfs_move.anterior = 4.0
    segment.omfs_move.pitch = radians(-5)
    moved = segment.matrix_world.copy()
    cut.omfs_cut.below_foramina = 6.0
    segment, _proximal = osteotomy.apply_cut(bpy.context, cut)
    assert abs(segment.omfs_move.anterior - 4.0) < 1e-6 and abs(segment.omfs_move.pitch + radians(5)) < 1e-6
    assert_same_matrix(segment.matrix_world, moved, "the new segment takes the planned position")


def test_movement_on_osteotomy_plane():
    scene = bpy.context.scene
    cut = osteotomy.find_cut(scene)                  # from test_reapplied_cut_keeps_movement
    cut.omfs_cut.profile_angle = radians(10)         # the cut rises anteriorly
    segment, _proximal = osteotomy.apply_cut(bpy.context, cut)
    props = segment.omfs_move
    assert props.axes == 'CUT'
    movement.back_to_preop(scene, segment)
    a = movement.cut_axes(scene, segment)
    assert_close(a.col[2], (0, np.sin(radians(10)), np.cos(radians(10))), "up = the cut's normal")
    assert_close(a.col[0], (1, 0, 0), "left unchanged by a profile angle")

    props.anterior = 5.0                             # slides along the cut: forward and up on the face
    pog = landmarks.find(scene, 'POG').matrix_world.translation
    on_cut = movement.displacement(scene, segment, pog)
    on_face = movement.displacement(scene, segment, pog, movement.frame_axes(scene))
    assert_close(on_cut, (0, 5, 0), "5 mm along the cut")
    assert_close(on_face, (0, 5 * np.cos(radians(10)), 5 * np.sin(radians(10))), "= forward + up on the face")

    props.pitch, props.up = radians(4), -1.0
    before = segment.matrix_world.copy()
    props.axes = 'FRAME'                             # same pose, numbers rewritten
    assert_same_matrix(segment.matrix_world, before, "changing the axes never moves the segment")
    props.axes = 'CUT'
    assert_same_matrix(segment.matrix_world, before, "and back")
    assert abs(props.anterior - 5.0) < 1e-4 and abs(props.up + 1.0) < 1e-4 and abs(props.pitch - radians(4)) < 1e-5

    frontal = movement.cut_axes(scene, segment)      # a frontal tilt keeps anterior in the midsagittal plane
    cut.omfs_cut.frontal_angle = radians(8)
    tilted = movement.cut_axes(scene, segment)
    assert abs(tilted.col[1].x) < 1e-6 and abs(tilted.col[2].dot(tilted.col[1])) < 1e-6
    assert (tilted.col[2] - frontal.col[2]).length > 0.1
    cut.omfs_cut.frontal_angle = 0.0


def test_polish_protects_intrados():
    scene = bpy.context.scene
    bone = box_object("Polish bone", 'BONE', (0, 300, 0), (40, 40, 20))   # top face at z = 10
    scene.omfs_guide.source = bone
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=8, y_segments=8, size=10.0)
    bmesh.ops.translate(bm, vec=(0, 300, 10), verts=bm.verts)
    mesh = bpy.data.meshes.new("Footprint")
    bm.to_mesh(mesh)
    bm.free()
    guide = bpy.data.objects.new("Footprint", mesh)
    scene.collection.objects.link(guide)
    guide.modifiers.new("Solidify", 'SOLIDIFY')
    guide_workshop.finish_extracted_guide(bpy.context, guide)
    assert guide_workshop.TOOLS['APPLY_MODIFIERS'].run(bpy.context) is None
    protected = guide_workshop.protect_intrados(bpy.context, guide)
    mask = np.array([d.value for d in guide.data.attributes[guide_workshop.MASK].data])
    z = np.array([v.co.z for v in guide.data.vertices])
    assert protected == int((mask == 1).sum()) and protected > 0
    assert np.allclose(z[mask == 1], 10 + params.GUIDE_CLEARANCE, atol=1e-3), "the intrados is protected"
    assert np.allclose(z[mask == 0], 10 + params.GUIDE_CLEARANCE + params.GUIDE_THICKNESS, atol=1e-3), \
        "the outer surface is free to polish"
    for obj in (guide, bone):
        bpy.data.objects.remove(obj)

    # A cap on an intra-oral scan: its intrados (facing the scan) is protected the same way.
    scan = box_object("Polish scan", 'SCAN', (0, 300, 0), (40, 40, 20))
    cap = box_object("Polish cap", 'GUIDE', (0, 300, 10 + params.GUIDE_CLEARANCE + 1.5), (20, 20, 3))
    protected = guide_workshop.protect_intrados(bpy.context, cap)
    z = np.array([v.co.z for v in cap.data.vertices]) + cap.matrix_world.translation.z
    mask = np.array([d.value for d in cap.data.attributes[guide_workshop.MASK].data])
    assert protected > 0 and np.all(z[mask == 1] < 11) and np.all(z[mask == 0] > 12), "the scan side is protected"
    for obj in (cap, scan):
        bpy.data.objects.remove(obj)


def test_sleeves_on_screws_and_mega_boolean():
    from omfs_planner import sleeves
    scene = bpy.context.scene
    guide = box_object("Sleeve guide", 'GUIDE', (0, 600, 0), (30, 10, 10))
    scene.omfs_guide.guide = guide
    screw = screws.add_screw(scene, Vector((0, 590, 0)), "a", Vector((0, 615, 0)), "b")   # along +Y, through the guide
    found, t = sleeves.nearest_screw(scene, Vector((1, 593, 50)), Vector((0, 0, -1)))
    assert found == screw and abs(t - 3) < 1e-4, "the click picks the screw, at its point under the mouse"
    assert sleeves.nearest_screw(scene, Vector((40, 593, 50)), Vector((0, 0, -1)))[0] is None, "too far: none"
    tube = sleeves.add_sleeve(bpy.context, guide, 'TUBE', screw, -2.0)
    assert (cylinders.axis(tube) - Vector((0, 1, 0))).length < 1e-6, "along the screw"
    assert (tube.matrix_world.translation - Vector((0, 588, 0))).length < 1e-4, "where it was slid to"
    sleeves.add_boolean(guide, tube, 'UNION')
    bore = sleeves.add_sleeve(bpy.context, guide, 'BORE', screw, -2.0)
    sleeves.add_boolean(guide, bore, 'DIFFERENCE')
    hole = sleeves.add_fixation_hole(bpy.context, guide, Vector((10, 600, 5)), Vector((0, 0, 1)))
    cut = box_object("Sleeve cut", 'OTHER', (0, 600, -20), (50, 50, 20))
    mod = guide.modifiers.new(guide_workshop.CUT_EDGE_MODIFIER, 'BOOLEAN')
    mod.operation, mod.object = 'DIFFERENCE', cut
    guide_workshop.order_booleans(guide)
    assert [m.object for m in guide.modifiers] == [tube, bore, cut, hole], "unions, holes, cut edge, fixation last"
    # A male of the cap crosses the tube: the tube is subtracted from the cap too.
    cap = box_object("Sleeve cap", 'GUIDE', (0, 580, 20), (20, 10, 4))
    male = box_object("Sleeve male", 'OTHER', (0, 588, 3), (2, 2, 30))
    male.omfs.made_by = pins.PART_TAG
    mod = cap.modifiers.new("male", 'BOOLEAN')
    mod.operation, mod.object = 'UNION', male
    scene.omfs_pins.cap = cap
    report, error = sleeves.mega_boolean(bpy.context)
    assert error is None and not guide.modifiers and not cap.modifiers, (report, error)
    assert tube.name in report and bore.name in report and hole.name not in report, report
    assert all(o.hide_get() for o in (tube, bore, hole, cut, male)), "the tools are put away"
    scene.omfs_pins.cap = scene.omfs_guide.guide = None
    for obj in (guide, cap, male, cut, tube, bore, hole, screw):
        bpy.data.objects.remove(obj)


def test_irrigation_hole():
    # On a finished guide (no modifiers left): two points give the hole, past both ends; it is dug last
    # and the Mega Boolean (run again) applies it. The volume removed is the channel's.
    from omfs_planner import sleeves
    scene = bpy.context.scene
    guide = box_object("Irrigation guide", 'GUIDE', (0, 750, 0), (30, 10, 10))
    scene.omfs_guide.guide = guide
    before = volume(guide)
    start, end = Vector((-15, 750, 2)), Vector((5, 750, 2))       # from the guide's side face, 20 mm in
    hole = sleeves.add_irrigation_hole(bpy.context, guide, start, end)
    assert hole.omfs_cylinder.kind == 'IRRIGATION'
    assert abs(hole.omfs_cylinder.length - (20 + 2 * params.IRRIGATION_OVERSHOOT)) < 1e-6
    assert (cylinders.axis(hole) - Vector((1, 0, 0))).length < 1e-6, "from the start to the end"
    assert (hole.matrix_world.translation - (start + end) / 2).length < 1e-6
    cut = box_object("Irrigation cut", 'OTHER', (0, 750, -20), (50, 50, 20))
    mod = guide.modifiers.new(guide_workshop.CUT_EDGE_MODIFIER, 'BOOLEAN')
    mod.operation, mod.object = 'DIFFERENCE', cut
    guide_workshop.order_booleans(guide)
    assert [m.object for m in guide.modifiers] == [cut, hole], "dug after the cut edge, with the fixation holes"
    guide.modifiers.remove(guide.modifiers[guide_workshop.CUT_EDGE_MODIFIER])
    report, error = sleeves.mega_boolean(bpy.context)
    assert error is None and not guide.modifiers, (report, error)
    removed = before - volume(guide)
    r = params.IRRIGATION_DIAMETER / 2
    inside = 20 + params.IRRIGATION_OVERSHOOT                    # the end overshoots into the guide
    expected = np.pi * r * r * inside * (np.sin(2 * np.pi / cylinders.SEGMENTS) * cylinders.SEGMENTS / (2 * np.pi))
    assert abs(removed - expected) / expected < 0.01, (removed, expected)
    scene.omfs_guide.guide = None
    for obj in (guide, hole, cut):
        bpy.data.objects.remove(obj)


def test_apply_skips_disabled_modifiers():
    """Mega Boolean / Apply: a Boolean whose cutter was deleted and one switched off in the view no longer
    stop it halfway (surgeon, 2026-11-09: "Modifier is disabled, skipping apply")."""
    scene = bpy.context.scene
    guide = box_object("Skip guide", 'GUIDE', (0, 800, 0), (30, 10, 10))
    gone = cylinders.add(bpy.context, 2.0, 20.0, "Skip gone", scene.collection, Matrix.Translation((5, 800, 0)))
    off = cylinders.add(bpy.context, 2.0, 20.0, "Skip off", scene.collection, Matrix.Translation((-5, 800, 0)))
    kept = cylinders.add(bpy.context, 2.0, 20.0, "Skip kept", scene.collection, Matrix.Translation((0, 800, 0)))
    for cutter in (gone, off, kept):
        sleeves.add_boolean(guide, cutter, 'DIFFERENCE')
    bpy.data.objects.remove(gone)                                                # deleted by hand
    guide.modifiers[f"Trou {off.name}"].show_viewport = False                    # eye closed
    before = volume(guide)
    skipped = []
    assert guide_workshop.apply_stack(bpy.context, guide, skipped) == 1
    assert len(skipped) == 2 and [m.name for m in guide.modifiers] == [f"Trou {off.name}"], skipped
    assert volume(guide) < before - 25, "the live hole was dug (2 mm through 10 mm: about 31 mm3)"
    for obj in (guide, off, kept):
        bpy.data.objects.remove(obj)


def test_applied_hole_wins_over_a_later_union():
    """Case 1 (2026-11-05): sleeve bore applied early, a pin boss united over it later filled the bore."""
    scene = bpy.context.scene
    guide = box_object("Late guide", 'GUIDE', (0, 700, 0), (30, 10, 10))
    bore = cylinders.add(bpy.context, 4.0, 20.0, "Late bore", scene.collection,
                         Matrix.Translation((0, 700, 0)) @ Matrix.Rotation(radians(90), 4, 'X'))
    sleeves.add_boolean(guide, bore, 'DIFFERENCE')
    assert guide_workshop.apply_stack(bpy.context, guide) == 1
    assert dict(guide[guide_workshop.APPLIED_DIFFERENCES]) == {bore.name: f"Trou {bore.name}"}

    def in_bore():
        """Points of the bore's axis inside the guide's material."""
        tree = guide_workshop.world_tree(guide)
        axis = [bore.matrix_world @ Vector((0, 0, z / 2)) for z in range(-19, 20)]
        return [p for p in axis if model_base.inside(tree, p)]

    assert not in_bore()
    far = box_object("Late far boss", 'OTHER', (12, 700, 8), (4, 4, 4))
    boss = box_object("Late boss", 'OTHER', (0, 708, 0), (8, 8, 8))    # over the bore's mouth
    for obj in (far, boss):
        sleeves.add_boolean(guide, obj, 'UNION')
    assert guide_workshop.redo_differences(guide) == [bore.name], "redone once, because the boss reaches it"
    assert guide_workshop.redo_differences(guide) == [], "already in the stack"
    guide_workshop.apply_stack(bpy.context, guide)
    assert not in_bore(), "the bore stays open through the boss"
    assert volume(guide) > 30 * 10 * 10, "the bosses were added"
    far2 = box_object("Late far 2", 'OTHER', (12, 700, -8), (4, 4, 4))
    sleeves.add_boolean(guide, far2, 'UNION')
    assert guide_workshop.redo_differences(guide) == [], "a union away from the bore leaves it alone"
    for obj in (guide, bore, far, boss, far2):
        bpy.data.objects.remove(obj)


def test_plane_cut():
    def evaluated_volume(obj):
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        v = bm.calc_volume(signed=True)
        bm.free()
        obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
        return v

    guide = box_object("Trim guide", 'GUIDE', (0, 800, 0), (30, 10, 10))
    plane = trim.start_cut(bpy.context, guide, Vector((10, 800, 0)), Vector((0, 1, 0)), Vector((-1, 0, 0)))
    assert abs(evaluated_volume(guide) - 2500) < 1, "the smaller side (x > 10) goes, whatever the normal said"
    assert plane.show_in_front and trim.find_cut(guide)[0] == plane
    trim.flip_cut(guide)
    assert abs(evaluated_volume(guide) - 500) < 1, "flipped: the other side goes"
    plane.matrix_world.translation.x = 0
    assert abs(evaluated_volume(guide) - 1500) < 1, "the cut follows the plane"
    trim.cancel_cut(bpy.context, guide)
    assert not guide.modifiers and trim.find_cut(guide) == (None, None)
    assert not [o for o in bpy.data.objects if o.name.startswith(trim.plane_name(guide))], "helpers removed"
    trim.start_cut(bpy.context, guide, Vector((10, 800, 0)), Vector((0, 0, 1)), Vector((1, 0, 0)))
    assert trim.apply_cut(bpy.context, guide) is None
    assert not guide.modifiers and trim.is_closed(guide.data), "applied, still closed"
    assert abs(volume(guide) - 2500) < 1
    assert not [o for o in bpy.data.objects if o.name.startswith(trim.plane_name(guide))]
    # An open shell thickened by Solidify, a tube already waiting: the cut goes between them.
    shell = box_object("Trim shell", 'GUIDE', (0, 850, 0), (30, 10, 0.001))
    shell.modifiers.new("Solidify", 'SOLIDIFY')
    tube = box_object("Trim tube", 'OTHER', (14, 850, 0), (4, 4, 4))
    sleeves.add_boolean(shell, tube, 'UNION')
    trim.start_cut(bpy.context, shell, Vector((10, 850, 0)), Vector((0, 1, 0)), Vector((1, 0, 0)))
    assert [m.name for m in shell.modifiers] == ["Solidify", guide_workshop.TRIM_MODIFIER, f"Union {tube.name}"]
    guide_workshop.order_booleans(shell)
    assert shell.modifiers[1].name == guide_workshop.TRIM_MODIFIER, "the trim stays before the other Booleans"
    assert trim.apply_cut(bpy.context, shell) is None
    assert [m.name for m in shell.modifiers] == [f"Union {tube.name}"], "the tube stays live"
    for obj in (guide, shell, tube):
        bpy.data.objects.remove(obj)


def test_erase_and_fill():
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=10.0)
    for f in bm.faces:
        f.select = f.calc_center_median().z > 8      # a cap to remove
    kept = {tuple(round(c, 6) for c in v.co) for v in bm.verts if not all(f.select for f in v.link_faces)}
    holes, left_open = trim.erase_and_fill(bm)
    assert holes == 1 and left_open == 0
    assert all(e.is_manifold for e in bm.edges), "closed again"
    after = {tuple(round(c, 6) for c in v.co) for v in bm.verts}
    assert kept <= after, "the rest of the mesh, rim included, did not move"
    patch = [v.co for v in bm.verts if tuple(round(c, 6) for c in v.co) not in kept]
    assert len(patch) > 3, "the patch is as fine as the rim"
    assert all(7.0 < co.z < 9.5 and co.length < 10 for co in patch), "a smooth membrane on the rim"
    assert bm.calc_volume(signed=True) > 0, "faces of the patch point outwards like the rest"
    bm.free()


def test_zones_by_angle_and_mask():
    box = box_object("Angle box", 'OTHER', (0, 400, 0), (10, 10, 10))
    bm = bmesh.new()
    bm.from_mesh(box.data)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)   # flat faces next to each other
    bm.to_mesh(box.data)
    bm.free()
    mesh = box.data
    assert guide_workshop.zones_by_angle_of(mesh, radians(20)) == 6, "a cube's 90 degree creases make 6 zones"
    assert guide_workshop.zones_by_angle_of(mesh, radians(100)) == 1, "no fold sharper than 100 degrees"
    guide_workshop.zones_by_angle_of(mesh, radians(20))
    sets = guide_workshop.face_sets(mesh)
    top = np.array([p.normal.z > 0.9 for p in mesh.polygons])
    assert len(set(sets[top])) == 1, "one zone per face of the cube"
    guide_workshop.mask_faces(mesh, sets == sets[top][0], 1.0)
    z = np.array([v.co.z for v in mesh.vertices])
    mask = guide_workshop.read_mask(mesh)
    assert np.all(mask[z > 4.99] == 1) and np.all(mask[z < 4.99] == 0), "clicked zone masked, nothing else"
    guide_workshop.mask_faces(mesh, sets == sets[top][0], 0.0)
    assert not guide_workshop.read_mask(mesh).any(), "Ctrl + click unmasks"
    bpy.data.objects.remove(box)


def open_scan(name, bm):
    """A scan object from a bmesh (freed), role SCAN."""
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.omfs.role = 'SCAN'
    return obj


def model_vertices_on_scan(model, scan):
    """How many of the model's vertices sit exactly on a vertex of the scan."""
    scan_points = {tuple(v.co) for v in scan.data.vertices}
    return sum(tuple(v.co) in scan_points for v in model.data.vertices)


def test_model_base_horseshoe():
    scene = bpy.context.scene
    verts, tris = make_test_case.arch_tube(arch_radius=25, tube_radius=4, z=0)
    bm = bmesh.new()
    for co in verts:
        bm.verts.new(co)
    bm.verts.ensure_lookup_table()
    for tri in tris:
        bm.faces.new([bm.verts[i] for i in tri])
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_center_median().z < -2], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    scan = open_scan("Horseshoe scan", bm)
    settings = scene.omfs_model
    settings.scan = scan
    settings.teeth_band = 3.0                     # the crown: the top 3 mm of the tube
    assert bpy.ops.omfs.make_model() == {'FINISHED'}
    model = model_base.model_of(scene, scan)
    report = model_base.last_report(model)
    assert not report["palate"] and report["bar"] is not None, report
    assert report["open_edges"] == 0, "the model is closed"
    assert report["protected_removed"] == 0 and report["protected_edge"] == 0, report
    assert model_vertices_on_scan(model, scan) == report["kept"], "kept vertices of the scan never move"
    z = np.array([v.co.z for v in model.data.vertices])
    top = max(v.co.z for v in scan.data.vertices)
    assert abs(z.min() - (top - settings.height)) < 1e-4, "flat base at the height asked"
    assert model.modifiers["Barre de renfort"].object is not None
    assert report["bar_anchored"], "both ends of the bar are inside the model"
    assert report["pieces"] == 1, "the bar joins both sides into one solid"
    settings.use_bar = False                      # settings rebuild the model
    assert model_base.last_report(model)["bar"] is None and "Barre de renfort" not in model.modifiers
    bar = bpy.data.objects.get(f"Barre — {model.name}")
    assert bar is None, "no bar left behind"
    settings.use_bar = True
    for obj in (model, bpy.data.objects[f"Barre — {model.name}"], scan):
        bpy.data.objects.remove(obj)


def horseshoe_scan(name, z, flip):
    """An open U-shaped arch scan: a half tube cut open on its base side (below, or above if flipped)."""
    verts, tris = make_test_case.arch_tube(arch_radius=25, tube_radius=4, z=z)
    bm = bmesh.new()
    for co in verts:
        bm.verts.new(co)
    bm.verts.ensure_lookup_table()
    for tri in tris:
        bm.faces.new([bm.verts[i] for i in tri])
    side = 1 if flip else -1
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if (f.calc_center_median().z - z) * side > 2], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    return open_scan(name, bm)


def test_model_pins():
    scene = bpy.context.scene
    settings = scene.omfs_model
    settings.teeth_band = 3.0
    lower_scan = horseshoe_scan("Pins LowerJaw", 0, False)
    upper_scan = horseshoe_scan("Pins UpperJaw", 9, True)
    models = []
    for scan in (lower_scan, upper_scan):
        settings.scan = scan
        assert bpy.ops.omfs.make_model() == {'FINISHED'}
        models.append(model_base.model_of(scene, scan))
    lower, upper = models
    assert model_base.pin_models(scene) == (lower, upper), "found by name"
    point, up = pins.model_base_plane(lower)
    top = max(v.co.z for v in lower_scan.data.vertices)
    assert up.z > 0.99 and abs(point.z - (top - settings.height)) < 1e-3, (point, up)
    foot = point + Vector((0, -12, 0))                 # in the horseshoe's opening, near its middle
    site = pins.add_model_site(scene, lower, upper, foot)
    s = site.omfs_pin
    p = scene.omfs_pins
    assert s.kind == 'MODEL' and s.has_stem and pins.sites(scene, 'GUIDE') == []
    stop = site.matrix_world.translation
    assert abs((stop - point).dot(up) - (p.cone_length + p.floor)) < 1e-4, "the boss stands on the base plane"
    arm_start = Vector((0, 25, 9 + 4))                 # top of the upper model's front, facing up... any point
    assert pins.set_arm(scene, site, upper, arm_start, Vector((0, -1, 0))) is None
    dg = bpy.context.evaluated_depsgraph_get()
    for model in (lower, upper):
        bm = bmesh.new()
        bm.from_mesh(model.evaluated_get(dg).to_mesh())
        assert len(model_base.bmesh_pieces(bm)) == 1, f"{model.name}: the pin parts join it into one solid"
        assert all(e.is_manifold for e in bm.edges), model.name
        bm.free()
    for kind in ("bossage", "logement", "tige", "mâle", "bras"):
        assert bpy.data.objects.get(pins.part_name(site, kind)) is not None, kind
    # Printed too tight, stacks already applied: a wider clearance still widens the bore.
    for model in (lower, upper):
        guide_workshop.apply_stack(bpy.context, model)
    axis = site.matrix_world.to_3x3() @ Vector((0, 0, 1))
    probe = stop - axis + axis.orthogonal().normalized() * (p.base_diameter / 2 + p.clearance + 0.05)
    assert model_base.inside(guide_workshop.world_tree(lower), probe), "in the boss, just past the bore"
    default = p.clearance
    p.clearance = default + 0.25
    guide_workshop.apply_stack(bpy.context, lower)
    assert not model_base.inside(guide_workshop.world_tree(lower), probe), "the bore is wider"
    assert trim.is_closed(lower.data), "and the model still closed"
    p.clearance = default
    pins.remove_site(site)
    assert not any(m.type == 'BOOLEAN' and m.name.startswith("Pin modèle") for m in lower.modifiers)
    settings.teeth_band = params.MODEL_TEETH_BAND
    for obj in [o for o in bpy.data.objects if o.name.startswith("Barre —")] + [lower, upper, lower_scan, upper_scan]:
        bpy.data.objects.remove(obj)


def test_model_base_keeps_the_palate():
    scene = bpy.context.scene
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=20)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_center_median().z < 0], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])   # a palate is seen from inside the dome
    scan = open_scan("Palate scan", bm)
    settings = scene.omfs_model
    settings.scan = scan
    settings.height = 10.0                        # less than the dome: must be overridden
    settings.flip = False
    assert bpy.ops.omfs.make_model() == {'FINISHED'}
    model = model_base.model_of(scene, scan)
    report = model_base.last_report(model)
    assert report["palate"] and report["bar"] is None, report
    assert report["height"] >= 20 + params.MODEL_UNDER_PALATE - 1e-3, "the base goes under the whole palate"
    assert report["open_edges"] == 0
    assert model_vertices_on_scan(model, scan) == report["kept"]
    settings.height = params.MODEL_HEIGHT
    for obj in (model, scan):
        bpy.data.objects.remove(obj)


def u_arch_scan(name):
    """Open scan: a U-shaped ridge of teeth (top, h = 0), a palate 14 mm deep inside the U, the vestibule
    dropping outside. Returns (scan, depth of each vertex below the ridge, inside the U)."""
    bm = bmesh.new()
    curve = np.array([(22 * np.cos(t), 20 * np.sin(t) - 2) for t in np.linspace(0, np.pi, 200)]
                     + [(s * 22, y) for s in (-1, 1) for y in np.linspace(-2, -25, 50)])
    for x in np.arange(-35, 35.01, 0.5):
        for y in np.arange(-30, 30.01, 0.5):
            d = np.min(np.hypot(curve[:, 0] - x, curve[:, 1] - y))
            h = -0.2 * d if d <= 2 else -0.4 - 13.6 * min(1.0, (d - 2) / 6)
            bm.verts.new((x, y, h))
    bm.verts.ensure_lookup_table()
    nx, ny = len(np.arange(-35, 35.01, 0.5)), len(np.arange(-30, 30.01, 0.5))
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            bm.faces.new((bm.verts[a], bm.verts[a + ny], bm.verts[a + ny + 1], bm.verts[a + 1]))
    return open_scan(name, bm)


def test_model_base_upper_without_palate():
    """"Inclure le palais" off: the palate goes, the upper model is a horseshoe closed by the bar (8 mm)."""
    scene = bpy.context.scene
    scan = u_arch_scan("U arch scan")
    settings = scene.omfs_model
    settings.scan, settings.flip = scan, False
    assert abs(settings.bar_diameter - 8.0) < 1e-6 and params.MODEL_BAR_DIAMETER == 8.0, "bar doubled"
    co = registration.vertex_coords(scan)
    deep = (co[:, 2] < -12) & (np.abs(co[:, 0]) < 14) & (co[:, 1] > -20) & (co[:, 1] < 10)   # the palate's vault
    ridge = co[:, 2] > -0.2                                                    # the teeth
    reports = {}
    for include in (True, False):
        settings.include_palate = include
        assert bpy.ops.omfs.make_model() == {'FINISHED'}
        model = model_base.model_of(scene, scan)
        reports[include] = report = model_base.last_report(model)
        assert report["open_edges"] == 0, report
        present = {tuple(np.round(p, 4)) for p in registration.vertex_coords(model)}
        assert all(tuple(np.round(p, 4)) in present for p in co[ridge]), "every tooth vertex kept"
        vault = sum(tuple(np.round(p, 4)) in present for p in co[deep])
        if include:
            assert report["palate"] and report["bar"] is None and vault == deep.sum(), (report, vault)
        else:
            assert not report["palate"] and report["palate_removed"], report
            assert vault == 0, f"{vault} palate vertices left"
            assert report["bar"] is not None, "closed by the bar, like the lower"
            assert abs(report["height"] - settings.height) < 1e-6, "the base is the set height, not under a palate"
    settings.include_palate = True
    for obj in [o for o in bpy.data.objects if o.name.startswith("Barre —")] + [model_base.model_of(scene, scan), scan]:
        bpy.data.objects.remove(obj)


def test_bite_check():
    scene = bpy.context.scene

    def surface(name, z):
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=60, y_segments=30, size=1.0)
        for v in bm.verts:
            x, y = v.co.x * 20, v.co.y * 10                # 40 x 20 mm
            v.co = (x, y, z + np.sin(x * 0.7) * np.cos(y * 0.9) + 0.3 * np.sin(x * 0.23 + y))   # cusps
        return open_scan(name, bm)

    upper = surface("Test UpperJaw", 1.0)
    lower = surface("Test LowerJaw", -1.0)
    bm = bmesh.new()                        # the bite: a strip of each arch, captured together
    for arch in (upper, lower):
        part = bmesh.new()
        part.from_mesh(arch.data)
        bmesh.ops.delete(part, geom=[f for f in part.faces if not -15 < f.calc_center_median().x < 0],
                         context='FACES')
        mesh = bpy.data.meshes.new("part")
        part.to_mesh(mesh)
        part.free()
        bm.from_mesh(mesh)
        bpy.data.meshes.remove(mesh)
    bite = open_scan("Test BiteScan", bm)
    bite.matrix_world = Matrix.Translation((40, -30, 25)) @ Matrix.Rotation(radians(70), 4, Vector((1, 2, 3)))
    bite_data = [tuple(v.co) for v in bite.data.vertices]
    settings = scene.omfs_model
    settings.bite, settings.upper, settings.lower = bite, upper, lower
    assert bpy.ops.omfs.check_bite() == {'FINISHED'}
    result = model_base.last_bite(scene)
    assert result["ok"], f"the bite's shape fits wherever the bite object sits: {result}"
    assert result["gap"] < 0.05, result
    assert not model_base.bite_stale(scene, result)
    assert [tuple(v.co) for v in bite.data.vertices] == bite_data, "the bite's mesh is never changed"
    lower.location.z -= 1.0                 # the arches no longer in the bite's occlusion
    bpy.context.view_layer.update()
    assert model_base.bite_stale(scene, result), "moving an arch makes the check stale"
    bpy.ops.omfs.check_bite()
    result = model_base.last_bite(scene)
    assert not result["ok"], result
    settings.bite = settings.upper = settings.lower = None
    del scene["omfs_bite_check"]
    for obj in (bpy.data.objects[f"{bite.name} (recalé, contrôle)"], bite, upper, lower):
        bpy.data.objects.remove(obj)


def test_up_axis_from_the_arches():
    """No reference frame: up comes from the arches, whatever file axis the scanner used (case1: Y)."""
    scene = bpy.data.scenes.new("No frame")              # earlier tests left a frame in the main scene
    props = scene.omfs_splint
    upper, lower = tube_scan("Up upper", 9), tube_scan("Up lower", -9)
    turn = Matrix.Rotation(radians(-90), 4, 'X')        # the occlusal axis becomes the file's Y
    for obj in (upper, lower):
        obj.matrix_world = turn @ obj.matrix_world
        scene.collection.objects.link(obj)
    props.kind, props.upper, props.lower = 'FINAL', upper, lower
    assert reference_frame.find_frame(scene) is None
    up = splint.up_axis(scene)
    assert up.dot(turn.to_3x3() @ Vector((0, 0, 1))) > 0.99, up
    # A cap on one open scan: up points at its teeth, away from its open border.
    bm = bmesh.new()
    bm.from_mesh(lower.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_center_median().z < -9], context='FACES')
    bm.to_mesh(lower.data)
    bm.free()
    props.kind, props.cap_arch = 'CHAPE', lower
    up = splint.up_axis(scene)
    assert up.dot(turn.to_3x3() @ Vector((0, 0, 1))) > 0.99, f"teeth side of an open scan: {up}"
    for obj in (upper, lower):
        bpy.data.objects.remove(obj)
    bpy.data.scenes.remove(scene)


def test_arch_curve():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower = box_object("Lower arch scan", 'SCAN', (0, -400, -5), (60, 50, 10))   # occlusal surface z = 0
    upper = box_object("Upper arch scan", 'SCAN', (0, -400, 9), (60, 50, 10))    # occlusal surface z = 4
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower, props.centre = 'FINAL', upper, lower, True
    assert splint.arches(scene) == [upper, lower]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    clicks = [(-25, -390, 0), (-15, -410, 0), (0, -420, 4), (15, -410, 0), (25, -390, 0)]   # on either arch
    for c in clicks:
        placed = splint.add_point(scene, Vector(c), depsgraph)
        assert abs(placed.z - 2.0) < 1e-4, f"halfway between the arches: {placed}"
    curve = splint.find_curve(scene)
    points = splint.curve_points(curve)
    assert len(points) == 5 and curve.omfs.role == 'ARCH'
    assert all((p - Vector((c[0], c[1], 2.0))).length < 1e-4 for p, c in zip(points, clicks)), "through each click"
    chord = sum(((points[i + 1] - points[i]).length for i in range(4)))
    assert splint.curve_length(curve) >= chord - 1e-3
    splint.remove_last_point(scene)
    assert len(splint.curve_points(curve)) == 4

    props.kind, props.cap_arch = 'CHAPE', lower      # a cap: points stay on the surface clicked
    assert splint.arches(scene) == [lower]
    assert abs(splint.add_point(scene, Vector((0, -400, 0)), depsgraph).z) < 1e-6
    bpy.ops.omfs.arch_curve_clear()
    assert splint.find_curve(scene) is None
    for obj in (lower, upper):
        bpy.data.objects.remove(obj)


def tube_scan(name, z):
    verts, tris = make_test_case.arch_tube(arch_radius=45, tube_radius=5, z=z)   # crown 10 mm wide, 16 high
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], tris)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.omfs.role = 'SCAN'
    obj.location.y = 500
    return obj


def cover(scene, upper, lower, top=12.0, bottom=6.0, angles=(20, 90, 160)):
    """Splint coverage points on the tubes' buccal side: the top surface at `top`, the bottom at `bottom`."""
    for obj in splint.coverage_points(scene):
        bpy.data.objects.remove(obj)
    for a in angles:
        u = radians(a)
        for arch, z, side in ((upper, top, 'U'), (lower, bottom, 'L')):
            splint.add_coverage_point(scene, arch, Vector((50 * np.cos(u), 500 + 50 * np.sin(u), z)), side)
    bpy.context.view_layer.update()


def test_splint_blank():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)   # 2 mm gap: bite at z = 9
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower = 'FINAL', upper, lower
    props.edge_round = props.splint_edge_round = 0.0   # sharp: flat faces measured exactly (rim tested below)
    bpy.ops.omfs.arch_curve_clear()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    assert "point" in splint.build_blank(scene, depsgraph), "no coverage point yet: refused"
    cover(scene, upper, lower)                       # covers the upper teeth up to z 12, the lower down to 6
    points = splint.coverage_points(scene)
    assert len(points) == 6 and points[0].parent is upper and all(p.omfs.role == 'MARKER' for p in points)

    blank = splint.build_blank(scene, depsgraph)
    assert not isinstance(blank, str), blank
    mesh = blank.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    assert all(e.is_manifold for e in bm.edges) and bm.calc_volume() > 0, "closed, outward normals"
    bm.free()
    # Crown band z 6 to 12. The 16-sided tubes have a vertex ring at 67.5 degrees in that band (z 7.4),
    # 5 * cos(67.5) = 1.91 mm off their axis: plus the splint's clearances, 0.5 buccal and 2.0 lingual.
    info = splint.blank_info(blank)
    expected = 2 * 5 * np.cos(radians(67.5)) + params.SPLINT_BUCCAL_CLEARANCE + params.SPLINT_LINGUAL_CLEARANCE
    assert abs(info["width_min"] - expected) < 0.3 and info["width_max"] < expected + 0.8, (info, expected)   # max: the tube's open ends
    n = len(mesh.vertices) // 2
    z = np.array([v.co.z for v in mesh.vertices])
    assert np.allclose(z[:n], 12, atol=1e-3) and np.allclose(z[n:], 6, atol=1e-3), "the surfaces through the points"
    assert 125 < info["length"] < 150, f"an automatic arch, end to end of the half torus (141 mm): {info['length']}"
    line = splint.centreline(scene, depsgraph, 1.0)
    r = np.hypot(line[0][:, 0], line[0][:, 1] - 500)
    assert np.abs(r[5:-5] - 45).max() < 1.5, "the centre line runs in the middle of the crowns"

    props.splint_buccal_margin = 4.0                 # only the buccal side grows, live
    assert abs(splint.blank_info(blank)["width_min"] - (expected + 4.0 - params.SPLINT_BUCCAL_CLEARANCE)) < 0.3
    assert not splint.blank_stale(scene, blank)
    up = splint.coverage_points(scene, 'U')[1]       # the front upper point raised 2 mm: a surface, not a slab
    up.matrix_world = Matrix.Translation(up.matrix_world.translation + Vector((0, 0, 2)))
    bpy.context.view_layer.update()
    assert splint.blank_stale(scene, blank), "a moved point asks for an update"
    splint.build_blank(scene, depsgraph)
    co = np.array([tuple(v.co) for v in blank.data.vertices])[:n if len(blank.data.vertices) // 2 == n else None]
    top = np.array([tuple(v.co) for v in blank.data.vertices])[:len(blank.data.vertices) // 2]
    front = top[top[:, 1] > 540][:, 2].max()
    side = top[top[:, 1] < 510][:, 2].max()
    assert front > 13.5 and abs(side - 12) < 0.3, (front, side)
    props.splint_buccal_margin = params.SPLINT_BUCCAL_CLEARANCE

    props.kind, props.cap_arch = 'CHAPE', lower      # a cap: unchanged, on its clicked curve
    bpy.ops.omfs.arch_curve_clear()
    for a in (10, 50, 90, 130, 170):
        u = radians(a)
        splint.add_point(scene, Vector((45 * np.cos(u), 500 + 45 * np.sin(u), 8)), depsgraph)
    assert splint.find_blank(scene) == blank and splint.build_blank(scene, depsgraph) == blank
    z = [v.co.z for v in blank.data.vertices]
    assert abs(min(z) - (8 - params.SPLINT_CAP_DEPTH)) < 1e-3 and abs(max(z) - 11) < 1e-3, \
        "0.8 mm down onto the teeth, 3 mm above"
    curve_length = splint.curve_length(splint.find_curve(scene))
    assert abs(splint.blank_info(blank)["length"] - curve_length) < 0.5, "the cap stops at the ends of its curve"
    bpy.ops.omfs.remove_splint_point(name="")
    assert not splint.coverage_points(scene)
    for obj in (lower, upper, blank):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_splint_imprint():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)   # lower top z 8, upper bottom z 10
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower = 'FINAL', upper, lower
    props.edge_round = props.splint_edge_round = 0.0
    bpy.ops.omfs.arch_curve_clear()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    def click(z):
        for a in (10, 50, 90, 130, 170):
            u = radians(a)
            splint.add_point(scene, Vector((45 * np.cos(u), 500 + 45 * np.sin(u), z)), depsgraph)

    def faces(blank):
        n = len(blank.data.vertices) // 2
        z = np.array([v.co.z for v in blank.data.vertices])
        return z[:n], z[n:]

    cover(scene, upper, lower)
    props.imprint = True
    blank = splint.build_blank(scene, depsgraph)
    top, bottom = faces(blank)
    gap = params.SPLINT_CLEARANCE
    assert abs(bottom.max() - (8 + gap)) < 0.02 and abs(top.min() - (10 - gap)) < 0.02, (bottom.max(), top.min())
    assert abs(bottom.min() - 6) < 1e-3 and abs(top.max() - 12) < 1e-3, "flat where no tooth is met"
    bm = bmesh.new()
    bm.from_mesh(blank.data)
    assert all(e.is_manifold for e in bm.edges) and bm.calc_volume() > 0
    bm.free()
    info = splint.blank_info(blank)
    assert info["contact_area"] == 0 and abs(info["thinnest"] - (2 - 2 * gap)) < 0.05, info

    upper.location.z = -2.5                          # arches overlapping by 0.5 mm: holes at the contacts
    bpy.context.view_layer.update()
    blank = splint.build_blank(scene, depsgraph)
    top, bottom = faces(blank)
    assert (top >= bottom - 1e-6).all() and splint.blank_info(blank)["contact_area"] > 0

    solid = splint.solidify(scene)                   # frozen into its own object, holes at the contacts
    assert not isinstance(solid, str), solid
    assert solid != blank and blank.hide_get() and splint.find_solids(scene) == [solid]
    info = splint.blank_info(solid)
    assert info["welded"] > 0 and info["open_edges"] == 0, info
    assert len(solid.data.vertices) < len(blank.data.vertices), "the contact patches are gone"
    bm = bmesh.new()
    bm.from_mesh(solid.data)
    assert bm.calc_volume() > 0
    bm.free()
    count = len(solid.data.vertices)
    props.splint_buccal_margin = 3.0                 # the settings no longer touch the solid
    assert len(solid.data.vertices) == count and len(blank.data.vertices) != count
    props.splint_buccal_margin = params.SPLINT_BUCCAL_CLEARANCE
    bpy.data.objects.remove(solid)
    blank.hide_set(False)

    upper.location.z = 0
    props.kind, props.cap_arch = 'CHAPE', upper      # a cap hanging under the upper arch
    bpy.ops.omfs.arch_curve_clear()
    click(10)
    bpy.context.view_layer.update()
    assert splint.cap_on_upper(scene)
    blank = splint.build_blank(scene, depsgraph)
    top, bottom = faces(blank)
    assert abs(top.max() - (10 + params.SPLINT_CAP_DEPTH)) < 1e-3 and abs(bottom.min() - 7) < 1e-3
    assert abs(top.min() - (10 - gap)) < 0.02 and abs(bottom.max() - 7) < 1e-3, "only the top is pressed"
    props.imprint = False

    # The clearance is a rolled ball: a vertical tooth wall (cols 0-9 at 1 mm, nothing beyond) is
    # cleared sideways too, and the step down happens outside the tooth.
    step = np.where(np.arange(14) < 10, 1.0, -np.inf)[None, :].repeat(5, axis=0)
    out = splint.dilate(step, 0.1, 0.1, 0.05, np.full(5, 0.05))[2]
    assert np.allclose(out[:10], 1.1) and abs(out[10] - (1 + 0.1 * np.sqrt(0.75))) < 1e-9
    assert abs(out[11] - 1.0) < 1e-9 and np.isneginf(out[12])
    bpy.ops.omfs.remove_splint_point(name="")
    for obj in (lower, upper, blank):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_splint_rounded_rim_and_wire_holes():
    """Rounded rim on the blank; wire holes only in the buccal flange, never on the teeth, one Boolean."""
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)   # bite at z = 9
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower = 'FINAL', upper, lower
    props.splint_edge_round, props.splint_buccal_margin, props.imprint = 1.0, 6.0, True
    bpy.ops.omfs.arch_curve_clear()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cover(scene, upper, lower)
    blank = splint.build_blank(scene, depsgraph)
    n = len(blank.data.vertices) // 2
    z = np.array([v.co.z for v in blank.data.vertices])
    assert abs(z[:n].max() - 12) < 1e-3 and abs(z[:n].min() - (10 - params.SPLINT_CLEARANCE)) < 0.02
    rim = z[0]                                        # first row, lingual edge: a corner of the outline
    assert abs(rim - 11) < 1e-3 and abs(z[n] - 7) < 1e-3, (rim, z[n])   # 1 mm down / up at the very edge
    bm = bmesh.new()
    bm.from_mesh(blank.data)
    assert all(e.is_manifold for e in bm.edges) and bm.calc_volume() > 0
    volume = bm.calc_volume()
    bm.free()

    flange = splint_holes.Flange(scene, blank, depsgraph)
    need = params.SPLINT_WIRE_HOLE_DIAMETER / 2 + params.SPLINT_WIRE_HOLE_TEETH_GAP
    _c, error = flange.place(Vector((0, 543, 9)), params.SPLINT_WIRE_HOLE_DIAMETER)       # lingual of the curve
    assert error and "lingual" in error
    hole, error = splint_holes.place_hole(scene, blank, Vector((0, 545.5, 9)), params.SPLINT_WIRE_HOLE_DIAMETER,
                                          depsgraph)                                       # on the teeth: pushed
    assert error is None, error
    centre = hole.matrix_world.translation
    assert centre.y > 545.5 and flange.teeth_distance(np.array(centre)) >= need - 1e-6
    assert abs(centre.z - 9) < 1e-3 and (hole.matrix_world.col[2].xyz - Vector((0, 0, 1))).length < 1e-6
    for tube in (lower, upper):                       # across the axis, over the splint's height: clear
        co = np.array([tube.matrix_world @ v.co for v in tube.data.vertices])
        band = (co[:, 2] > 5.5) & (co[:, 2] < 12.5)
        assert np.hypot(co[band, 0] - centre.x, co[band, 1] - centre.y).min() >= need - 1e-6
    _h, error = splint_holes.place_hole(scene, blank, Vector((47 * np.cos(radians(60)), 500 + 47 * np.sin(radians(60)), 9)),
                                        params.SPLINT_WIRE_HOLE_DIAMETER, depsgraph)
    assert error is None, error
    mods = [m for m in blank.modifiers if m.type == 'BOOLEAN' and m.operand_type == 'COLLECTION']
    assert len(mods) == 1 and len(splint_holes.holes(scene)) == 2, "one Boolean for every hole"
    evaluated = blank.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bm = bmesh.new()
    bm.from_mesh(evaluated.to_mesh())
    evaluated.to_mesh_clear()
    expected = 2 * np.pi * (params.SPLINT_WIRE_HOLE_DIAMETER / 2) ** 2 * 6      # two holes through 6 mm
    assert all(e.is_manifold for e in bm.edges) and abs((volume - bm.calc_volume()) - expected) < 0.25 * expected
    bm.free()
    props.wire_hole_diameter = 2.0                    # every hole follows
    assert all(abs(max(v.co.x for v in h.data.vertices) - 1.0) < 1e-4 for h in splint_holes.holes(scene))

    props.splint_buccal_margin = 1.0                  # flange too narrow: refused, never on the teeth
    blank = splint.build_blank(scene, depsgraph)
    _c, error = splint_holes.Flange(scene, blank, depsgraph).place(Vector((0, 546, 9)), 1.5)
    assert error and "bord" in error, error
    bpy.ops.omfs.remove_wire_hole()
    assert not splint_holes.holes(scene)
    props.wire_hole_diameter = params.SPLINT_WIRE_HOLE_DIAMETER
    props.splint_buccal_margin, props.imprint = params.SPLINT_BUCCAL_CLEARANCE, False
    props.edge_round, props.splint_edge_round = params.SPLINT_EDGE_ROUND, params.SPLINT_SPLINT_EDGE_ROUND
    bpy.ops.omfs.remove_splint_point(name="")
    for obj in (lower, upper, blank):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_splint_text_and_checks():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower, props.imprint = 'FINAL', upper, lower, False
    bpy.ops.omfs.arch_curve_clear()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cover(scene, upper, lower)
    blank = splint.build_blank(scene, depsgraph)     # flat top at z 12, bottom at 6

    def volume(obj):
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        v = bm.calc_volume()
        bm.free()
        return v

    before = volume(blank)
    settings = scene.omfs_text_next
    settings.text, settings.height, settings.depth, settings.mode = "AB 12", 3.0, 0.5, 'ENGRAVE'
    # Two clicks on the top, at the front of the arch (y 544, the middle of the band), reading along +x.
    label, warning = text_labels.add_label(blank, Vector((-5, 544, 12)), Vector((5, 544, 12)), Vector((0, 0, 1)), settings)
    assert warning is None, warning
    bm = bmesh.new()
    bm.from_mesh(label.data)
    assert all(e.is_manifold for e in bm.edges) and bm.calc_volume() > 0, "the cutter is closed"
    bm.free()
    z = [v.co.z for v in label.data.vertices]
    assert abs(min(z) - 11.5) < 1e-3 and abs(max(z) - (12 + params.TEXT_OVERLAP)) < 1e-3
    xs = [v.co.x for v in label.data.vertices]
    assert max(xs) - min(xs) <= 10 + 1e-3 and label.omfs_text.letter_height <= 3 + 1e-6
    engraved = volume(blank)
    assert engraved < before, "letters cut into the top"
    assert abs(text_labels.wall_under(label) - 5.5) < 1e-3, "12 - 0.5 - 6 mm left under the letters"
    result = splint.check(blank, bpy.context.evaluated_depsgraph_get())
    assert result["open_edges"] == 0 and result["ok"], result

    label.omfs_text.mode = 'EMBOSS'                  # live: the same letters now stand out
    assert volume(blank) > before and blank.modifiers[0].operation == 'UNION'
    props.splint_buccal_margin = 2.0                 # rebuilt blank: the text follows
    assert abs(max(v.co.z for v in label.data.vertices) - 12.5) < 1e-3 and volume(blank) > before

    solid = splint.solidify(scene)                   # text placed before solidifying comes along
    assert text_labels.labels_on(solid) == [label]
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "splint.stl")
        bpy.ops.omfs.export_part(filepath=path, part_name=solid.name)
        assert Path(path).stat().st_size > 1000

    text_labels.remove_label(label)
    assert not blank.modifiers and not solid.modifiers
    props.splint_buccal_margin = params.SPLINT_BUCCAL_CLEARANCE
    bpy.ops.omfs.remove_splint_point(name="")
    for obj in (lower, upper, blank, solid):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_universal_text():
    """The same text tool on any printable part (a guide, a model...), never on raw anatomy."""
    scene = bpy.context.scene
    guide = box_object("Text guide", 'GUIDE', (0, 700, 0), (30, 10, 10))      # top face at z = 5
    bone = box_object("Text bone", 'BONE', (0, 750, 0), (30, 10, 10))
    assert text_labels.can_carry(guide) and not text_labels.can_carry(bone), "CBCT data is never written on"
    locking.set_locked(guide, True)
    assert not text_labels.can_carry(guide), "locked: not editable"
    locking.set_locked(guide, False)
    settings = scene.omfs_text_next
    settings.text, settings.height, settings.depth, settings.mode = "G1", 4.0, 0.6, 'ENGRAVE'
    label, warning = text_labels.add_label(guide, Vector((-6, 700, 5)), Vector((6, 700, 5)), Vector((0, 0, 1)), settings)
    assert warning is None, warning
    result = text_labels.print_check(guide, bpy.context.evaluated_depsgraph_get())
    assert result["ok"] and abs(result["texts"][0]["wall"] - 9.4) < 1e-3, result
    bpy.context.view_layer.objects.active = label
    assert text_labels.target(bpy.context) == guide, "the active text's part"
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "guide.stl")
        assert bpy.ops.omfs.export_part(filepath=path, part_name=guide.name) == {'FINISHED'}
        assert Path(path).stat().st_size > 1000

    # On a dental model: the text follows when the model is made again.
    model_settings = scene.omfs_model
    model_settings.teeth_band = 3.0
    scan = horseshoe_scan("Text LowerJaw", 0, False)
    model_settings.scan = scan
    bpy.ops.omfs.make_model()
    model = model_base.model_of(scene, scan)
    base, up = pins.model_base_plane(model)
    front = Vector((0, -29, base.z + 5))              # on the front wall, outside the horseshoe
    hit = model.ray_cast(front, Vector((0, 1, 0)))
    assert hit[0], "the model's front wall"
    text_labels.add_label(model, hit[1] + Vector((-5, 0, 0)), hit[1] + Vector((5, 0, 0)), hit[2], settings)
    model_label = text_labels.labels_on(model)[0]
    model_settings.height = 18.0                      # rebuilds the model
    assert text_labels.labels_on(model) == [model_label]
    assert text_labels.print_check(model, bpy.context.evaluated_depsgraph_get())["open_edges"] == 0
    model_settings.height, model_settings.teeth_band = params.MODEL_HEIGHT, params.MODEL_TEETH_BAND
    for obj in [o for o in bpy.data.objects if o.name.startswith("Barre —")] + [label, model_label, guide, bone, model, scan]:
        bpy.data.objects.remove(obj)


def test_used_tools_are_put_away():
    scene = bpy.context.scene
    part = box_object("Tidy part", 'GUIDE', (0, 800, 0), (20, 10, 10))
    cutter = box_object("Tidy cutter", 'OTHER', (0, 800, 5), (4, 4, 4))
    spare = box_object("Tidy spare", 'OTHER', (30, 800, 0), (4, 4, 4))       # never used: stays
    bone = box_object("Tidy bone", 'BONE', (0, 830, 0), (10, 10, 10))
    home = cutter.users_collection[0]
    mod = part.modifiers.new("Trou", 'BOOLEAN')
    mod.object, mod.operation = cutter, 'DIFFERENCE'
    part.modifiers.new("Analyse", 'BOOLEAN').object = bone                     # anatomy used by a Boolean
    assert tidy.tidy(scene) == 0, "in use: nothing moves"
    bpy.context.view_layer.objects.active = part
    bpy.ops.object.modifier_apply(modifier="Trou")
    part.modifiers.remove(part.modifiers["Analyse"])
    assert tidy.tidy(scene) == 1
    used = bpy.data.collections[tidy.COLLECTION]
    assert used in cutter.users_collection and home not in cutter.users_collection, "applied: put away"
    assert used not in spare.users_collection and used not in bone.users_collection, "never anatomy, never unused"
    part.modifiers.new("Encore", 'BOOLEAN').object = cutter                    # needed again: back home
    assert tidy.tidy(scene) == 1 and home in cutter.users_collection and used not in cutter.users_collection
    for obj in (part, cutter, spare, bone):
        bpy.data.objects.remove(obj)


def test_planning_aids_are_put_away():
    scene = bpy.context.scene

    def away(obj):
        return tidy.COLLECTION in [c.name for c in obj.users_collection]

    frame = reference_frame.find_frame(scene) or reference_frame.build(scene, FRAME_POINTS)
    plane = frame.children[0]
    plane.hide_set(False)
    tidy.tidy(scene)
    assert not away(plane), "a shown plane stays"
    plane.hide_set(True)
    tidy.tidy(scene)
    assert away(plane) and not away(frame), "a hidden plane goes away, never the frame"
    plane.hide_set(False)
    tidy.tidy(scene)
    assert not away(plane) and plane.parent == frame, "shown again: back"

    curve = bpy.data.objects.new("Aids curve", bpy.data.curves.new("Aids curve", 'CURVE'))
    curve.omfs.role = 'ARCH'
    blank = bpy.data.objects.new("Aids blank", bpy.data.meshes.new("Aids blank"))
    blank.omfs.made_by = splint.BLANK_TAG
    solid = bpy.data.objects.new("Aids solid", bpy.data.meshes.new("Aids solid"))
    solid.omfs.made_by = splint.SOLID_TAG
    for obj in (curve, blank):
        scene.collection.objects.link(obj)
    blank.hide_set(True)
    tidy.tidy(scene)
    assert not away(curve), "no solid splint yet: the curve is needed"
    scene.collection.objects.link(solid)
    tidy.tidy(scene)
    assert away(curve), "solidified, blank hidden: the curve goes away"
    blank.hide_set(False)
    tidy.tidy(scene)
    assert not away(curve), "the blank shown again (to change it): the curve comes back"
    for obj in (curve, blank, solid):
        bpy.data.objects.remove(obj)


def test_bicortical_screws():
    scene = bpy.context.scene
    segment = box_object("Screw segment", 'SEGMENT', (0, 900, 0), (20, 10, 10))     # front face y = 895
    mandible = box_object("Screw mandible", 'BONE', (0, 915, 0), (20, 10, 10))      # back face y = 920
    guide = box_object("Screw guide", 'GUIDE', (0, 892, 0), (20, 2, 10))            # in front of the segment
    bpy.context.view_layer.update()
    entry = screws.first_bone(bpy.context, Vector((0, 850, 1)), Vector((0, 1, 0)))
    assert entry[1] == segment and abs(entry[0].y - 895) < 1e-4, "the guide in front is passed through"
    exit_ = screws.first_bone(bpy.context, Vector((0, 960, -1)), Vector((0, -1, 0)))
    assert exit_[1] == mandible and abs(exit_[0].y - 920) < 1e-4
    screw = screws.add_screw(scene, entry[0], entry[1].name, exit_[0], exit_[1].name)
    expected = (exit_[0] - entry[0]).length
    assert abs(screws.length(screw) - expected) < 1e-6
    ends = [screw.matrix_world @ v.co for v in screw.data.vertices]
    along = Vector(exit_[0] - entry[0]).normalized()
    heights = [(e - entry[0]).dot(along) for e in ends]
    assert abs(min(heights)) < 1e-4 and abs(max(heights) - expected) < 1e-4, "from the entry to the exit"
    radius = max((e - entry[0] - along * (e - entry[0]).dot(along)).length for e in ends)
    assert abs(radius - params.SCREW_DIAMETER / 2) < 1e-4
    scene.omfs_screws.diameter = 2.4                 # every screw follows
    radius = max((screw.matrix_world @ v.co - entry[0] - along * (screw.matrix_world @ v.co - entry[0]).dot(along)).length
                 for v in screw.data.vertices)
    assert abs(radius - 1.2) < 1e-4
    scene.omfs_screws.diameter = params.SCREW_DIAMETER
    second = screws.add_screw(scene, entry[0] + Vector((5, 0, 0)), segment.name, exit_[0], mandible.name)
    assert [o.name for o in screws.screws(scene)] == ["Vis 1", "Vis 2"]
    for obj in (screw, second):
        screws.remove_screw(obj)
    for obj in (segment, mandible, guide):
        bpy.data.objects.remove(obj)


def test_deleted_work_copy_is_gone():
    scene = bpy.context.scene
    verts, tris = make_test_case.box((0, 0, 0), (10, 10, 10))
    mesh = bpy.data.meshes.new("Bone box")
    mesh.from_pydata(verts, [], tris)
    bone = bpy.data.objects.new("Bone box", mesh)
    scene.collection.objects.link(bone)
    scene.omfs_guide.source = bone
    copy = guide_workshop.paint_copy(bpy.context)
    for coll in list(copy.users_collection):        # deleted in the viewport: the panel's pointer keeps it
        coll.objects.unlink(copy)
    assert guide_workshop.current_copy(bpy.context) is None
    again = guide_workshop.paint_copy(bpy.context)
    assert again is not None and again != copy, "a new copy, not the deleted one"
    scene.omfs_guide.work_copy = None
    for obj in (copy, again, bone):
        bpy.data.objects.remove(obj)


def test_boolean_order():
    guide = box_object("Order guide", 'GUIDE', (0, 0, 0), (10, 10, 2))
    tool_obj = box_object("Order tool", 'OTHER', (0, 0, 0), (2, 2, 6))
    guide.modifiers.new("Dégagement", 'DISPLACE')
    for name, operation in (("Trou a", 'DIFFERENCE'), (guide_workshop.CUT_EDGE_MODIFIER, 'DIFFERENCE'),
                            ("Union b", 'UNION'), ("Trou c", 'DIFFERENCE'), ("Union d", 'UNION')):
        mod = guide.modifiers.new(name, 'BOOLEAN')
        mod.operation, mod.object = operation, tool_obj
    guide_workshop.order_booleans(guide)
    assert [m.name for m in guide.modifiers] == ["Dégagement", "Union b", "Union d", "Trou a", "Trou c",
                                                  guide_workshop.CUT_EDGE_MODIFIER], "tubes, holes, cut edge"
    for obj in (guide, tool_obj):
        bpy.data.objects.remove(obj)


def test_pins():
    # Pure geometry: a closed conical pin of the right volume; an arm leaving at 90 degrees and
    # arriving along the pin's axis.
    scene = bpy.context.scene
    d = pins.dims(scene)
    verts, faces = pins.revolve(d["male"], (0, 0, 0), (0, 0, 1))
    mesh = bpy.data.meshes.new("pin test")
    mesh.from_pydata(verts.tolist(), [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    assert all(e.is_manifold for e in bm.edges)
    (z0, r0), (_z, r1), (_s, rs), (z3, _rs) = d["male"]
    expected = np.pi * (-z0) / 3 * (r0 * r0 + r0 * r1 + r1 * r1) + np.pi * rs * rs * z3
    assert abs(bm.calc_volume() - expected) / expected < 0.01, (bm.calc_volume(), expected)
    bm.free()
    bpy.data.meshes.remove(mesh)
    path = pins.arm_path((0, 0, 0), (0, -1, 0), (10, -8, 12), (0, 0, 1))
    start, end = path[1] - path[0], path[-1] - path[-2]
    assert np.dot(start / np.linalg.norm(start), (0, -1, 0)) > 0.99, "leaves the cap at 90 degrees"
    assert np.dot(end / np.linalg.norm(end), (0, 0, -1)) > 0.99, "arrives down the pin's axis"

    # On a guide under an osteotomy plane: the female's top lies in the plane, the pin plugs in.
    reference_frame.build(scene, FRAME_POINTS)
    bone = box_object("Pin bone", 'BONE', (0, 0, -50), (60, 40, 30))
    landmarks.place(scene, 'MF_R', (-20, 0, -40), bone)
    landmarks.place(scene, 'MF_L', (20, 0, -40), bone)
    landmarks.place(scene, 'POG', (0, -20, -55), bone)
    bpy.context.view_layer.update()
    for old in [o for o in scene.objects if o.omfs.role == 'CUT']:
        bpy.data.objects.remove(old)
    cut = osteotomy.add_cut(scene)
    cut.omfs_cut.source = bone
    bpy.context.view_layer.update()
    guide = box_object("Pin guide", 'GUIDE', (0, -22, -50), (30, 4, 20))   # crosses the plane (z = -45), like a real guide
    cap = box_object("Pin cap", 'GUIDE', (0, -15, -30), (30, 10, 4))
    point, normal = pins.osteotomy_plane(scene, guide)
    click = Vector((0, -24, -50))                    # on the guide's front face
    site = pins.add_site(scene, guide, pins.guide_start(scene, guide, click, Vector((0, -1, 0))))
    assert not isinstance(site, str), site
    assert not site.omfs_pin.has_stem, "against the guide: no bridge"
    stop = site.matrix_world.translation
    assert abs((stop - point).dot(normal)) < 1e-4, "the stop is on the osteotomy plane"
    axis = site.matrix_world.to_3x3() @ Vector((0, 0, 1))
    assert axis.dot(normal) > 0.9999, "the pin is perpendicular to the plane"
    boss = bpy.data.objects[pins.part_name(site, "bossage")]
    tops = [boss.matrix_world @ v.co for v in boss.data.vertices]
    assert max((t - point).dot(normal) for t in tops) < 1e-4, "the boss stops at the plane"
    assert [m.operation for m in guide.modifiers] == ['UNION', 'DIFFERENCE'], "boss, then bore"

    scene.omfs_pins.cap = cap
    assert guide_workshop.TOOLS['PIN_MALES'].run(bpy.context) is None
    assert pins.set_arm(scene, site, cap, Vector((0, -20, -30)), Vector((0, 0, -1))) is None
    assert sorted(m.object.name for m in cap.modifiers) == sorted(
        [pins.part_name(site, "mâle"), pins.part_name(site, "bras")])
    male = bpy.data.objects[pins.part_name(site, "mâle")]
    heights = [(male.matrix_world @ v.co - point).dot(normal) for v in male.data.vertices]
    p = scene.omfs_pins
    assert abs(min(heights) - (-p.cone_length + p.tip_gap)) < 1e-4, "the tip stays short of the bottom"

    # Bone rest: a click on the bone above the cut; the block joins the closest arm to the bone, its
    # foot on the bone surface, and is united to the cap. Below the cut: refused.
    bone_point = Vector((6, -20, -40))               # bone front face, 5 mm above the plane (z = -45)
    assert pins.nearest_arm(scene, bone_point) == site
    assert "ostéotomie" in pins.set_rest(scene, site, bone, Vector((6, -20, -44.5))), "below the margin: refused"
    assert not site.omfs_pin.has_rest
    assert pins.set_rest(scene, site, bone, bone_point) is None
    rest = bpy.data.objects[pins.part_name(site, "appui osseux")]
    assert any(m.object == rest and m.operation == 'UNION' for m in cap.modifiers), "united to the cap"
    _v, _f, feet = pins.rest_shape(scene, site)
    assert np.allclose(feet[:, 1], -20, atol=1e-4), "the foot lies on the bone's face"
    assert feet[:, 2].min() > -45 + params.BONE_REST_CUT_MARGIN - 1e-6
    p.rest_thickness = 6.0                            # settings rebuild the rest too
    assert len(rest.data.vertices) > len(_v)
    p.rest_thickness = params.BONE_REST_THICKNESS

    p.clearance = 0.2                                # settings rebuild every part
    bore = bpy.data.objects[pins.part_name(site, "logement")]
    radius = max((bore.matrix_world @ v.co - stop).cross(axis).length for v in bore.data.vertices)
    assert abs(radius - (p.base_diameter / 2 + 0.2)) < 1e-3
    p.clearance = params.PIN_CLEARANCE
    pins.remove_site(site)
    assert not guide.modifiers and not cap.modifiers and not pins.sites(scene)

    # Slid 12 mm away from the guide's front: the stop stays in the plane, a bridge joins it to the
    # guide, as wide as the boss, its top flush with the plane.
    away = pins.add_site(scene, guide, pins.guide_start(scene, guide, click, Vector((0, -1, 0))) + Vector((0, -12, 0)))
    stop = away.matrix_world.translation
    assert abs((stop - point).dot(normal)) < 1e-4, "slid: the stop is still on the osteotomy plane"
    assert away.omfs_pin.has_stem
    bridge = bpy.data.objects[pins.part_name(away, "pont")]
    corners = [bridge.matrix_world @ v.co for v in bridge.data.vertices]
    tops = sorted((c - point).dot(normal) for c in corners)
    assert abs(tops[-1]) < 1e-4 and abs(tops[-4]) < 1e-4, "the bridge's top is in the plane"
    p = scene.omfs_pins
    assert abs((corners[0] - corners[3]).length - (p.base_diameter + 2 * p.wall)) < 1e-3, "as wide as the boss"
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    bm.from_mesh(guide.evaluated_get(dg).to_mesh())
    assert len(model_base.bmesh_pieces(bm)) == 1, "guide, bridge and boss: one solid"
    bm.free()

    # The guide follows the segment after the pins were placed: the Empty moves with it, the parts
    # are rebuilt there (not moved twice) and the arm reaches the pin where it is now.
    scene.omfs_pins.cap = cap
    pins.set_arm(scene, away, cap, Vector((0, -20, -30)), Vector((0, 0, -1)))
    assert away in movement.companions(guide) and bridge not in movement.companions(guide)
    shift = Matrix.Translation((2, -3, 4))
    before = [bridge.matrix_world @ v.co for v in bridge.data.vertices]
    for obj in movement.companions(guide):
        obj[movement.GUIDE_PREOP] = [v for row in obj.matrix_world for v in row]
        obj.matrix_world = shift @ obj.matrix_world
    movement.rebuild_pins(scene)
    after = [bridge.matrix_world @ v.co for v in bridge.data.vertices]
    assert all((a - shift @ b).length < 1e-4 for a, b in zip(after, before)), "the bridge went with the guide"
    arm = bpy.data.objects[pins.part_name(away, "bras")]
    back = away.matrix_world.translation + axis * p.back_length
    assert min((arm.matrix_world @ v.co - back).length for v in arm.data.vertices) < p.arm_diameter, \
        "the arm reaches the pin where it is now"
    movement.guide_back(guide)
    again = [bridge.matrix_world @ v.co for v in bridge.data.vertices]
    assert all((a - b).length < 1e-4 for a, b in zip(again, before)), "back with the guide"
    pins.remove_site(away)
    assert not guide.modifiers

    # A pin right over the guide's body: its male overlaps the guide itself, so the Mega Boolean
    # hollows the guide for it (shoulder + clearance, open along the axis); a pin beside it does not.
    from omfs_planner import sleeves
    over = pins.add_site(scene, guide, Vector((0, -22, -45)))
    beside = pins.add_site(scene, guide, Vector((0, -40, -45)))
    pins.add_males(scene, cap)
    assert sleeves.hollow_for_males(bpy.context, guide) == [over.name], "only the male inside the guide"
    channel = bpy.data.objects[pins.part_name(over, "passage du mâle")]
    mod = next(m for m in guide.modifiers if m.object == channel)
    assert mod.operation == 'DIFFERENCE'
    heights = [(channel.matrix_world @ v.co - point).dot(normal) for v in channel.data.vertices]
    assert abs(min(heights)) < 1e-4 and abs(max(heights) - params.PIN_MALE_CHANNEL) < 1e-4, "from the plane, up the axis"
    for site in (over, beside):
        pins.remove_site(site)
    assert not guide.modifiers and not cap.modifiers
    for obj in (guide, cap, bone, cut):
        bpy.data.objects.remove(obj)
    for lm in ('MF_R', 'MF_L', 'POG'):
        bpy.data.objects.remove(landmarks.find(scene, lm))


def test_bone_rest_block():
    # Pure geometry: an arm along X 10 mm above a sphere of bone; the block is as wide as the tube,
    # closed, facing out, and every point of its foot lies on the bone surface.
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=5, radius=10.0)
    bmesh.ops.translate(bm, vec=(0, 0, -10), verts=bm.verts)
    mesh = bpy.data.meshes.new("rest bone")
    bm.to_mesh(mesh)
    bm.free()
    bone = bpy.data.objects.new("rest bone", mesh)
    bpy.context.scene.collection.objects.link(bone)
    bone.location = (3, 0, 0)                         # world vs local: the cast must handle both
    bpy.context.view_layer.update()
    path = np.linspace((-10, 0, 10), (10, 0, 10), 41) + np.array((3, 0, 0))
    shape = pins.rest_block(path, 2.0, 4.0, (3, 0, 0), pins.bone_cast(bone, 0.0))
    assert not isinstance(shape, str), shape
    verts, faces, feet = shape
    tree = BVHTree.FromObject(bone, bpy.context.evaluated_depsgraph_get())
    to_local = bone.matrix_world.inverted()
    assert max(tree.find_nearest(to_local @ Vector(f))[3] for f in feet) < 1e-4, "the foot fits the bone"
    assert np.ptp(feet[:, 1]) > 3.99 and np.ptp(feet[:, 0]) > 3.99, "as wide as the tube, 4 mm along it"
    assert feet[:, 2].max() < 1e-4 and feet[:, 2].min() < -0.3, "the foot follows the curve, not a plane"
    block = bpy.data.meshes.new("rest block")
    block.from_pydata(verts.tolist(), [], faces)
    bm = bmesh.new()
    bm.from_mesh(block)
    assert all(e.is_manifold for e in bm.edges), "closed"
    vol = bm.calc_volume(signed=True)
    assert 16 * 10.6 < vol < 16 * 11.6, f"faces out, from the arm to the bone ({vol})"
    bm.free()
    bpy.data.meshes.remove(block)
    with_gap = pins.rest_block(path, 2.0, 4.0, (3, 0, 0), pins.bone_cast(bone, 0.2))[2]
    assert abs(with_gap[:, 2].max() - 0.2) < 1e-4, "the gap lifts the foot off the bone"
    assert isinstance(pins.rest_block(path, 2.0, 4.0, (3, 0, 9), pins.bone_cast(bone, 0.0)), str), "arm on the bone"
    bpy.data.objects.remove(bone)
    bpy.data.meshes.remove(mesh)


def test_empty_folder_is_an_error():
    with tempfile.TemporaryDirectory() as folder:
        try:
            bpy.ops.omfs.import_case(directory=folder)
        except RuntimeError as error:   # operators report errors as RuntimeError in scripts
            assert "No STL, OBJ or PLY" in str(error)
        else:
            raise AssertionError("expected an error for an empty folder")


def remove_print_scene():
    scene = print_scene.print_scene()
    if scene is not None:
        for part in print_scene.parts(scene):
            print_scene.remove_part(scene, part)
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj)
        bpy.data.scenes.remove(scene)


def test_print_profiles_and_settings():
    """Printer and resin profiles are data; the PrusaSlicer settings and the expected .goo values come from them."""
    printers, resins = print_profiles.profiles("printers"), print_profiles.profiles("resins")
    printer = printers["elegoo_mars_4_ultra"]
    assert (printer["pixels_x"], printer["pixels_y"], printer["goo_version"]) == (8520, 4320, 30) and printer["mirror_x"]
    assert len(resins) >= 2 and resins[print_profiles.favourite_resin()]["favourite"]
    resin = resins["elegoo_abs_like_v3_translucent"]
    supports = {"enable": True, "pad": False, "lift": 5.0, "density": 120, "tip": 0.5}
    config = print_chain.slicer_config(printer, resin, supports)
    assert config["display_pixels_x"] == 8520 and config["layer_height"] == 0.03 and config["pad_enable"] == 0
    assert config["material_ow_faded_layers"] == 6 and config["material_ow_support_points_density_relative"] == 120, \
        "the per-material overrides must be set too, or PrusaSlicer keeps its defaults"
    assert "FILEFORMAT_GOO" in config["printer_notes"] and "LiftHeight2_4" in config["printer_notes"]
    preview = print_chain.slicer_config(printer, resin, supports, preview=True)
    assert preview["display_pixels_x"] == 1065 and preview["display_width"] == config["display_width"], "same plate, coarse pixels"
    expected = print_chain.expected_goo(printer, resin)
    assert expected["Version"] == 30 and expected["BottomExposureTime"] == 40 and expected["RetractSpeed2"] == 60
    assert print_chain.settings_differences({k: str(v) for k, v in expected.items()}, expected) == []
    assert print_chain.settings_differences(dict({k: str(v) for k, v in expected.items()}, ExposureTime="9"), expected)[0][0] == "ExposureTime"
    assert 60 < print_chain.print_time(800, resin) / 60 < 300


def test_print_3mf_shares_vertices():
    """The 3MF given to PrusaSlicer: welded vertices (a triangle soup crashed it), blockers and per-object options."""
    import zipfile
    soup = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], float)     # 2 triangles, 6 vertices
    tris = np.arange(6).reshape(2, 3)
    parts = [print_chain.Part("a", soup, tris, [(soup + (0, 0, 1), tris)], (30.0, 20.0)),
             print_chain.Part("raft", soup, tris, [], (50.0, 20.0), {"supports_enable": 0})]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "plate.3mf"
        print_chain.write_3mf(path, parts)
        with zipfile.ZipFile(path) as z:
            model, config = z.read("3D/3dmodel.model").decode(), z.read("Metadata/Slic3r_PE_model.config").decode()
    assert model.count("<vertex ") == 4 + 4 + 4, "4 shared vertices per volume, not 6"
    assert model.count("<triangle ") == 6 and model.count("<item ") == 2 and "30.00000 20.00000" in model
    assert config.count("SupportBlocker") == 1 and config.count("ModelPart") == 2 and 'key="supports_enable" value="0"' in config


def test_print_scene_copies_and_checks():
    """Parts are copied to the print scene (the plan is untouched); the face that fits the bone is a bearing
    surface, turned up and tilted, with its no-support slab; raft, build volume and file name."""
    scene = bpy.context.scene
    bone = dense_box("Print bone", 'BONE', (0, 0, -5), (40, 30, 10))            # top at z = 0
    guide = dense_box("Print guide", 'GUIDE', (0, 0, 1.25), (20, 12, 2.5))      # sits on the bone
    locking.set_locked(bone, True)
    before = (tuple(guide.matrix_world.translation), len(guide.data.vertices), volume(guide))
    assert guide in print_scene.printable(scene) and bone not in print_scene.printable(scene)
    guide.omfs_print.pick = True
    bpy.ops.omfs.print_prepare()
    assert (tuple(guide.matrix_world.translation), len(guide.data.vertices), volume(guide)) == before, "the plan is untouched"
    plate = print_scene.print_scene()
    assert plate is not None and plate is not scene and guide.name in scene.objects and guide.name not in plate.objects
    part = print_scene.parts(plate)[0]
    assert part.omfs_print.source == guide.name and part.data is not guide.data
    flagged = print_scene.get_bearing(part.data)
    normals = np.array([p.normal for p in part.data.polygons])
    assert flagged.any() and (normals[flagged][:, 2] < -0.99).all(), "only the face on the bone"
    assert abs(sum(p.area for p in part.data.polygons if flagged[p.index]) - 20 * 12) < 1.0
    blockers = [c for c in part.children if c.omfs_print.kind == 'BLOCKER']
    assert len(blockers) == 1 and print_scene.open_edges(blockers[0].data) == 0, "one closed slab"
    up = print_scene.rotation_of(part) @ np.array((0.0, 0.0, -1.0))             # where the bearing normal points now
    assert abs(up[2] - np.cos(np.radians(params.PRINT_TILT_DEFAULT))) < 1e-3, f"bearing up, tilted ({up})"
    assert part.lock_location[2] and all(part.lock_scale)

    plate.omfs_print.supports = False                                            # no external tool in this test
    print_scene.update_all_now(plate)
    low, _high = print_scene.world_box(part)
    assert abs(low[2]) < 1e-4 and not part.omfs_print.outside, "without supports the part sits on the plate"
    assert not print_scene.of_kind(plate, 'RAFT'), "the raft needs supports"
    plate.omfs_print.supports = True
    vertices, triangles = print_scene.raft_arrays(plate)
    n = len(vertices) // 2
    foot, top = vertices[:n], vertices[n:]
    assert np.allclose(foot[:, 2], 0) and np.allclose(top[:, 2], plate.omfs_print.raft_thickness)
    assert (np.linalg.norm(top[:, :2], axis=1) > np.linalg.norm(foot[:, :2], axis=1)).all(), "the top overhangs: bevelled edge"
    edges = np.sort(np.vstack((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])), axis=1)
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all(), "raft closed"
    plate.omfs_print.supports = False

    part.location.x = 200.0                                                      # off the plate
    print_scene.check_volume(plate)
    assert part.omfs_print.outside and np.allclose(part.color, print_scene.OUTSIDE_COLOUR, atol=1e-3)
    assert any(not ok and "Hors du volume" in text for ok, text in print_scene.checks(plate))
    part.location.x = 0.0
    print_scene.check_volume(plate)
    assert not part.omfs_print.outside
    name = print_scene.file_name(plate)
    assert name.startswith("CAS01_Print_guide_") and name.endswith(".goo")

    remove_print_scene()
    guide.omfs_print.pick = False
    for obj in (guide, bone):
        bpy.data.objects.remove(obj)


def test_print_test_piece_to_goo():
    """Integration, only when PrusaSlicer and UVtools are installed: the synthetic test piece gets supports that
    stay off its bearing face, and exports to a .goo whose settings match the profiles, with nothing floating."""
    mesh = print_scene.test_piece_mesh()
    assert print_scene.open_edges(mesh) == 0 and 1300 < print_scene.mesh_volume(mesh) < 1700
    flagged = print_scene.get_bearing(mesh)
    assert flagged.any() and all(p.normal.z < -0.9 for p in mesh.polygons if flagged[p.index])
    bpy.data.meshes.remove(mesh)
    try:
        print_scene.tools()
    except print_chain.ToolMissing:
        print("  (PrusaSlicer / UVtools not installed: export not tested)")
        return
    bpy.ops.omfs.print_test_piece()
    plate = print_scene.print_scene()
    part = print_scene.parts(plate)[0]
    print_scene.update_all_now(plate)
    supports = part.omfs_print.supports
    assert part.omfs_print.state == 'OK' and supports is not None and len(supports.data.vertices) > 1000
    assert 4.0 < part.omfs_print.lift < 6.0 and abs(print_scene.world_box(part)[0][2] - part.omfs_print.lift) < 1e-3
    # no support above the bearing face (it is on top): nothing of the preview reaches the part's upper half
    low, high = print_scene.world_box(part)
    assert print_scene.world_box(supports)[1][2] < high[2] - 1.0, "supports stay under the part"
    assert all(ok for ok, _text in print_scene.checks(plate)), print_scene.checks(plate)
    assert len(print_scene.of_kind(plate, 'RAFT')) == 1
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / print_scene.file_name(plate)
        result = print_scene.export_now(plate, path)
        assert result.problems == [], result.problems
        assert path.exists() and path.with_suffix(".json").exists()
        found = print_chain.goo_properties(path, print_scene.tools()[1])
        assert found["Version"] == "30" and found["MachineName"] == "Elegoo Mars 4 Ultra" and found["LayerHeight"] == "0.03"
        assert abs(result.layers - print_scene.estimates(plate)["layers"]) < 30, "the estimate before export is close"
    remove_print_scene()


def synthetic_segmentation(folder):
    """A small DentalSeg-like probabilities file (no patient data): a "mandible" bar with a piece of "maxilla"
    stuck to its end (the error seen on a real scan), a separate "vertebra" piece of mandible, and an axis order
    and sign like a real CBCT (voxel axes permuted and flipped, 0.4 x 0.4 x 0.6 mm)."""
    shape = (40, 30, 24)
    data = np.zeros((5,) + shape, np.uint8)
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')

    def blob(k, low, high):
        inside = (x >= low[0]) & (x < high[0]) & (y >= low[1]) & (y < high[1]) & (z >= low[2]) & (z < high[2])
        data[k][inside] = 255
    blob(1, (4, 10, 8), (26, 18, 14))          # mandible bar
    blob(0, (26, 10, 8), (32, 18, 14))         # "maxilla" glued to its end: really mandible
    blob(1, (4, 22, 16), (10, 27, 21))         # a separate piece labelled mandible
    blob(0, (14, 2, 2), (36, 8, 20))           # the real maxilla, apart
    blob(4, (8, 13, 10), (22, 15, 12))         # a canal inside the mandible
    data[4][(x >= 8) & (x < 22) & (y >= 13) & (y < 15) & (z >= 10) & (z < 12)] = 255
    data[1][data[4] > 127] = 0
    for k in range(5):                         # soft edges, like probabilities
        soft = data[k].astype(float)
        for axis in range(3):
            soft = (np.roll(soft, 1, axis) + 2 * soft + np.roll(soft, -1, axis)) / 4
        data[k] = np.clip(soft, 0, 255).astype(np.uint8)
    directions = np.array(((0.0, 0.4, 0.0), (0.0, 0.0, -0.4), (-0.6, 0.0, 0.0)))   # i -> +Y, j -> -Z, k -> -X
    origin = np.array((50.0, -100.0, 20.0))
    path = Path(folder) / "synthetic_probabilites.nrrd"
    volume_scene.write_nrrd(path, volume_scene.Volume(data, directions, origin))
    return path, data, directions, origin


def test_volume_segmentation_workspace():
    """A probabilities .nrrd opens as one surface per structure in the patient's coordinates; pieces are counted;
    a mislabelled piece is given to the right structure (and undone); the result is imported with its roles."""
    plan = bpy.context.scene
    with tempfile.TemporaryDirectory() as folder:
        path, data, directions, origin = synthetic_segmentation(folder)
        volume = volume_scene.read_nrrd(path)
        assert volume.probabilities and volume.data.shape == data.shape and (volume.data == data).all()
        assert np.allclose(volume.directions, directions) and np.allclose(volume.origin, origin)
        bpy.ops.omfs.volume_open(filepath=str(path))
    scene = volume_scene.volume_scene()
    assert scene is not None and scene is not plan and scene.omfs_volume.source_scene == plan
    mandible, maxilla = volume_scene.structure(scene, 1), volume_scene.structure(scene, 0)
    assert mandible.name.startswith(volume_scene.PREFIX) and volume_scene.structure_name(mandible) == "Mandibule"
    assert np.allclose(mandible.color, volume_scene.STRUCTURES[1][2], atol=1e-3), "its own colour, not the role's"
    co = np.empty(len(mandible.data.vertices) * 3)
    mandible.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    voxels = volume.to_world(np.argwhere(data[1] > 127))
    assert (co.min(axis=0) > voxels.min(axis=0) - 0.7).all() and (co.max(axis=0) < voxels.max(axis=0) + 0.7).all(), \
        "the surface wraps the voxels, in patient coordinates"
    mandible.omfs_vol.min_piece = maxilla.omfs_vol.min_piece = 0.0     # count every piece (these are small)
    bpy.context.view_layer.update()
    sizes = volume_scene.count_pieces(scene, mandible)
    assert len(sizes) == 2 and sizes[0] > sizes[1] > 0, sizes
    assert len(volume_scene.count_pieces(scene, maxilla)) == 2

    glued = volume.to_world((28.0, 14.0, 11.0))            # inside the "maxilla" piece glued to the mandible
    message = volume_scene.correct(scene, maxilla, Vector(glued), 1)
    assert "donné à « Mandibule »" in message, message
    bpy.context.view_layer.update()
    assert len(volume_scene.count_pieces(scene, maxilla)) == 1, "the glued piece left the maxilla"
    after = volume_scene.count_pieces(scene, mandible)
    assert len(after) == 2 and after[0] > sizes[0] + 15, "the mandible grew by that piece, still two pieces"
    separate = volume.to_world((7.0, 24.0, 18.0))
    assert "supprimé" in volume_scene.correct(scene, mandible, Vector(separate), None)
    bpy.context.view_layer.update()
    assert len(volume_scene.count_pieces(scene, mandible)) == 1, "one mandible"
    bpy.ops.omfs.volume_undo()
    bpy.context.view_layer.update()
    assert len(volume_scene.count_pieces(scene, mandible)) == 2, "undo puts the piece back"

    mandible.omfs_vol.min_piece = 50.0                      # hides the small piece without touching the voxels
    bpy.context.view_layer.update()
    assert len(volume_scene.count_pieces(scene, mandible)) == 1
    bpy.ops.omfs.volume_import()
    made = {o.name: o for o in plan.objects if o.omfs.made_by == "Segmentation"}
    assert {"Maxillaire", "Mandibule", "Canal mandibulaire"} <= set(made), sorted(made)
    assert made["Mandibule"].omfs.role == 'BONE' and made["Canal mandibulaire"].omfs.role == 'NERVE'
    assert locking.is_locked(made["Mandibule"]) and made["Mandibule"].data.attributes.get(volume_scene.ISLAND) is None
    bpy.ops.omfs.volume_import()                            # again: replaces, never duplicates
    assert sum(1 for o in plan.objects if o.omfs.made_by == "Segmentation" and o.name.startswith("Mandibule")) == 1
    for obj in [o for o in plan.objects if o.omfs.made_by == "Segmentation"]:
        bpy.data.objects.remove(obj)
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj)
    bpy.data.scenes.remove(scene)


def test_ortho_osteotomies():
    """Le Fort I: a line of points across the front of a "maxilla", pushed back, separates the lower part (distal)
    from the rest; a bone already apart (a "vertebra") stays fixed. BSSO: a cut each side of a "mandible" bar gives
    the distal body and two proximal parts. The bones' own meshes are never changed."""
    scene = bpy.context.scene
    maxilla = dense_box("Maxillaire", 'BONE', (0, 0, 0), (60, 50, 40))               # front face at y = -25
    vertebra = dense_box("Vertebre", 'BONE', (0, 60, -15), (10, 10, 10))
    joined = bpy.data.meshes.new("Maxillaire")                                         # one object, two bones apart
    bm = bmesh.new()
    for obj in (maxilla, vertebra):
        part = obj.data.copy()
        part.transform(obj.matrix_world)
        bm.from_mesh(part)
        bpy.data.meshes.remove(part)
    bm.to_mesh(joined)
    bm.free()
    bpy.data.objects.remove(vertebra)
    old = maxilla.data
    maxilla.data = joined
    bpy.data.meshes.remove(old)
    maxilla.location = (0, 0, 0)
    bpy.context.view_layer.update()
    faces_before = len(maxilla.data.polygons)
    assert ortho_osteotomy.bone(scene, 'MAXILLA') == maxilla, "found by its name"
    points = [Vector((x, -25.0, -5.0)) for x in (-28, -10, 10, 28)]                   # across the front, 5 mm below centre
    cut = ortho_osteotomy.make_cut(scene, 'LEFORT', maxilla, points, Vector((0, 1, 0)))
    assert len(cut.data.polygons) == 0 and len(cut.data.edges) > 3, "before the stops: the line alone"
    pieces, warning = ortho_osteotomy.separate(bpy.context, maxilla)
    assert not pieces and "stops" in warning, "no separation without the two posterior stops"
    for side, x in (('R', -30.0), ('L', 30.0)):                                       # pterygomaxillary junctions
        ortho_osteotomy.set_stop(scene, cut, side, Vector((x, 15.0, -5.0)))
    stop = ortho_osteotomy.stop_point(cut, 'R')
    assert np.allclose(stop.color, ortho_osteotomy.STOP_COLOUR), "stop blue (not the role's colour)"
    assert np.allclose(ortho_osteotomy.front_points(cut)[0].color, ortho_osteotomy.FRONT_COLOUR)
    verts = np.array([v.co for v in cut.data.vertices])
    assert abs(verts[:, 1].max() - 15.0) < 1e-4, "the first plane stops at the stop plane"
    assert abs(verts[:, 2].min() - (-5.0 - params.ORTHO_PTM_DROP)) < 1e-4, "then down at 90 degrees"
    assert verts[:, 0].max() > 30 + params.ORTHO_END_EXTENSION_LEFORT - 3, "wide laterally"
    cut = ortho_osteotomy.make_cut(scene, 'LEFORT', maxilla, points, Vector((0, 1, 0)))   # redrawn: stops kept
    assert ortho_osteotomy.stops_ready(cut)
    pieces, warning = ortho_osteotomy.separate(bpy.context, maxilla)
    assert warning is None, warning
    named = {p.name: p for p in pieces}
    distal, proximal = named["Maxillaire distal (Le Fort I)"], named["Maxillaire proximal"]
    assert distal.omfs.role == 'SEGMENT' and proximal.omfs.role == 'BONE' and locking.is_locked(proximal)
    vd, cd = ortho_osteotomy.mesh_volume(distal.data)
    vp, _cp = ortho_osteotomy.mesh_volume(proximal.data)
    k = params.ORTHO_KERF / 2
    expected = 60 * (40 - k) * (15 - k)                       # below the plane, in front of the stop, minus half kerfs
    assert abs(vd - expected) < 300, (vd, expected, cd)
    assert vp > 60 * 50 * 24 and cd[2] < -10, "behind the stop (pterygoids) and the vertebra stay with the fixed part"
    assert maxilla.hide_get() and len(maxilla.data.polygons) == faces_before, "source hidden, never changed"
    cut = ortho_osteotomy.find_cut(scene, 'LEFORT')
    assert cut.hide_get() and all(p.hide_get() for p in cut.children), "separated: the cut and its points hidden"
    bpy.ops.omfs.ortho_rejoin(which='MAXILLA')
    assert not ortho_osteotomy.pieces_of(scene, maxilla) and not maxilla.hide_get()
    assert not cut.hide_get() and not any(p.hide_get() for p in cut.children), "rejoined: shown again to adjust"

    # a line clicked round the front and back along the lateral walls, past the stops: still one clean cut
    # (the surface goes through the line but covers the whole width, up to the stops; it never folds back)
    around = [Vector(p) for p in ((-30, 22, -5), (-30, 0, -5), (-28, -25, -5), (0, -25, -5), (28, -25, -5),
                                  (30, 0, -5), (30, 22, -5))]
    cut = ortho_osteotomy.make_cut(scene, 'LEFORT', maxilla, around, Vector((0, 1, 0)))
    assert max(v.co.y for v in cut.data.vertices) < 15.0 + 1e-4, "never past the stops"
    pieces, warning = ortho_osteotomy.separate(bpy.context, maxilla)
    assert warning is None and abs(ortho_osteotomy.mesh_volume(pieces[0].data)[0] - expected) < 300, warning
    ortho_osteotomy.rejoin(scene, maxilla)

    # stops in front of the bone's back but the line too short laterally: the cut stops inside, and says so
    short = ortho_osteotomy.make_cut(scene, 'LEFORT', maxilla, [Vector((x, -25.0, -5.0)) for x in (-5, 5)], Vector((0, 1, 0)))
    params.ORTHO_END_EXTENSION_LEFORT, wide = 4.0, params.ORTHO_END_EXTENSION_LEFORT
    ortho_osteotomy.rebuild_sheet(short)
    pieces, warning = ortho_osteotomy.separate(bpy.context, maxilla)
    params.ORTHO_END_EXTENSION_LEFORT = wide
    assert warning is not None and "incomplète" in warning and len(pieces) == 1
    ortho_osteotomy.rejoin(scene, maxilla)

    mandible = dense_box("Mandibule", 'BONE', (0, 0, -60), (100, 14, 24))              # a bar across, y -7..7, z -72..-48
    canal = dense_box("Canal mandibulaire", 'NERVE', (0, 0, -62), (100, 2, 2))      # the nerve, mid-thickness (y = 0)
    bpy.context.view_layer.update()
    for kind, sign in (('BSSO_R', -1.0), ('BSSO_L', 1.0)):
        cut = ortho_osteotomy.make_split(scene, kind, mandible)                       # clicked one by one, live
        for z in (-50.0, -70.0):                                                      # blue: buccal face (y = -7)
            ortho_osteotomy.add_split_point(scene, cut, Vector((sign * 25, -7.0, z)), Vector((0, 1, 0)))
        props = cut.omfs_ortho
        assert props.nerve_found and abs(props.nerve_depth - 7.0) < 0.3, (props.nerve_found, props.nerve_depth)
        verts = np.array([v.co for v in cut.data.vertices])
        assert abs(verts[:, 1].min() - (-7 - params.ORTHO_BSSO_OUTSIDE)) < 1e-3, "starts 10 mm outside the bone"
        assert abs(verts[:, 1].max() - 0.0) < 0.3, "the buccal cut stops at the nerve"
        assert verts[:, 2].min() < -72 - 3 and verts[:, 2].max() > -48 + 3, "the whole height of the bone"
        ortho_osteotomy.add_split_point(scene, cut, Vector((sign * 30, 7.0, -40)))     # a wrong click...
        ortho_osteotomy.remove_last_point(cut)                                         # ...Backspace
        for x, y in ((35, 2.0), (42, 7.0)):                                            # orange: back, to the lingual face
            ortho_osteotomy.add_split_point(scene, cut, Vector((sign * x, y, -60)))
        assert len(ortho_osteotomy.vertical_points(cut)) == 2 and len(ortho_osteotomy.front_points(cut)) == 2
        verts = np.array([v.co for v in cut.data.vertices])
        assert verts[:, 1].max() > 7 + 2, "past the lingual cortex"
    pieces, warning = ortho_osteotomy.separate(bpy.context, mandible)
    assert warning is None, warning
    names = sorted(p.name for p in pieces)
    assert names == ["Mandibule distale", "Mandibule proximal droit", "Mandibule proximal gauche"], names
    body = next(p for p in pieces if p.omfs.role == 'SEGMENT')
    assert abs(ortho_osteotomy.mesh_volume(body.data)[1][0]) < 1.0, "the body is the middle piece"
    right = next(p for p in pieces if p.name.endswith("droit"))
    assert ortho_osteotomy.mesh_volume(right.data)[1][0] < -30, "right = patient's right (-X)"
    for which in ('MAXILLA', 'MANDIBLE'):
        bpy.ops.omfs.ortho_rejoin(which=which)
    for kind in ('LEFORT', 'BSSO_R', 'BSSO_L'):
        ortho_osteotomy.remove_cut(ortho_osteotomy.find_cut(scene, kind))
    bpy.data.objects.remove(canal)
    assert not [o for o in scene.objects if o.omfs.made_by == ortho_osteotomy.MADE_BY], "points and sheets removed"
    for obj in (maxilla, mandible):
        bpy.data.objects.remove(obj)


SKIPPED = set()

failures = 0
for name, test in list(globals().items()):
    if name in SKIPPED:
        print(f"SKIP {name}")
    elif name.startswith("test_") and callable(test) and os.environ.get("OMFS_ONLY", name) in name:
        try:
            test()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()

print(f"{failures} failure(s)")
sys.exit(1 if failures else 0)
