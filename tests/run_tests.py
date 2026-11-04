"""Headless tests, run inside Blender:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/run_tests.py

Uses synthetic meshes only (tools/make_test_case.py), written to a temporary folder.
"""

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
import bmesh  # noqa: E402
from omfs_planner import locking  # noqa: E402
from omfs_planner import analysis, distance_map, guide_workshop, landmarks, movement, occlusion, occlusion_seating, osteotomy, params, reference_frame, registration, roles, section, section_contours, snapshot, splint, text_labels, cylinders, pins, model_base  # noqa: E402

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


def test_guide_workshop():
    scene = bpy.context.scene
    bone = box_object("Workshop bone", 'BONE', (-300, 0, 0), (40, 20, 20))
    locking.set_locked(bone, True)
    vertices_before = [tuple(v.co) for v in bone.data.vertices]
    scene.omfs_guide.source = bone

    def run(tool_id):
        error = guide_workshop.TOOLS[tool_id].run(bpy.context)
        assert error is None, f"{tool_id}: {error}"

    run('WORK_COPY')
    copy = scene.omfs_guide.work_copy
    assert copy is not None and copy.data != bone.data and bone.hide_get() and not locking.is_locked(copy)

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
    assert not any(m.show_in_editmode for m in guide.modifiers), "Edit Mode shows the footprint, not the shell"
    run('TOGGLE_WORK_COPY')
    assert not copy.hide_get()
    assert scene.omfs_guide.guide == guide, "the extracted guide is the one the tools modify"
    guide_workshop.make_active(bpy.context, copy)
    run('ADD_CLEARANCE')                             # copy active: the modifier goes to the guide
    assert not copy.modifiers and bpy.context.active_object == guide
    assert [m.type for m in guide.modifiers] == ['DISPLACE', 'SOLIDIFY'], "updated, not duplicated"

    # Cylinders at the 3D cursor, Boolean, apply.
    scene.cursor.location = (-300, 0, 10)
    run('ADD_TUBE')
    tube = bpy.context.active_object
    assert abs(tube.dimensions.x - (guide_workshop.sleeve_bore_diameter() + 2 * params.SLEEVE_TUBE_WALL)) < 1e-3
    assert scene.omfs_guide.cutter == tube, "the new cylinder is the Boolean tool"
    # Adjustable: pulling the top end keeps the bottom end in place; typed sizes are exact.
    scene.cursor.rotation_euler = (radians(90), 0, 0)                  # axis along -Y
    run('ADD_BORE')
    bore = bpy.context.active_object
    bottom = bore.location - cylinders.axis(bore) * bore.omfs_cylinder.length / 2
    cylinders.set_length(bore, 12.0, 'BOTTOM')
    bpy.context.view_layer.update()
    assert (bore.location - cylinders.axis(bore) * 6.0 - bottom).length < 1e-5
    bore.omfs_cylinder.diameter = 4.25
    bpy.context.view_layer.update()
    assert abs(bore.dimensions.x - 4.25) < 1e-3 and abs(bore.dimensions.z - 12.0) < 1e-3
    bpy.data.objects.remove(bore)
    scene.cursor.rotation_euler = (0, 0, 0)
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
    run('REMOVE_WORK_COPY')
    assert scene.omfs_guide.work_copy is None and not bone.hide_get()
    assert [tuple(v.co) for v in bone.data.vertices] == vertices_before, "the bone must never change"


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
    block = occlusion.find_block(scene, 'LOWER')
    shape = occlusion.shape_fingerprint(lower)
    # Springs "want" 1 mm of overlap and a 2 mm shift to the left (+X).
    for i, (x, y) in enumerate(((-12, -8), (12, -8), (0, 8)), start=1):
        occlusion_seating.add_point(scene, f"OCC_U{i}", f"Haut {i}", Vector((x + 2, y, 7)), upper)
        occlusion_seating.add_point(scene, f"OCC_L{i}", f"Bas {i}", Vector((x, y, -6)), lower)
    bpy.context.view_layer.update()
    assert len(occlusion_seating.point_pairs(scene)) == 3

    def top_of_lower():
        return max((lower.matrix_world @ v.co).z for v in lower.data.vertices)

    upside_down = Matrix.Translation((5, 3, -20)) @ Matrix.Rotation(radians(180), 4, 'X')
    for tolerance, expected, start in ((0.2, 0.2, None), (5.0, 1.0, None), (0.2, 0.2, upside_down)):
        occlusion.back_to_preop(block)   # resisted at 0.2 mm, pure point fit, then from upside down
        if start is not None:
            block.matrix_world = start @ block.matrix_world
        bpy.context.view_layer.update()
        props.tolerance = tolerance
        props.spring_share = 100.0   # springs only: the magnet would also pull the box sides together
        bpy.ops.omfs.seat_occlusion()
        overlap = top_of_lower() - 6.0
        assert abs(overlap - expected) < 0.1, (tolerance, overlap)
        shift = block.matrix_world.translation.x - occlusion.preop_matrix(block).translation.x
        assert abs(shift - 2.0) < 0.1, (tolerance, shift)
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
        props.tolerance, props.spring_share = 0.2, share
        bpy.ops.omfs.seat_occlusion()
        gaps[share] = 6.0 - top_of_lower()
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
    assert occlusion_seating.points_spread(scene) < occlusion_seating.MIN_SPREAD   # one line: refused
    assert occlusion.shape_fingerprint(lower) == shape          # only the block moved

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
        props.tolerance, props.spring_share, props.canine_weight = 5.0, 100.0, weight
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

    colours = occlusion_seating.overlap_colours(np.array([-2.0, -1.2, -0.7, -0.2, 0.1, 0.5, 3.0, np.nan]))
    expected = [occlusion_seating.MAGENTA, occlusion_seating.RED, occlusion_seating.ORANGE, occlusion_seating.YELLOW,
                occlusion_seating.GREEN, occlusion_seating.BLUE, occlusion_seating.GREY, occlusion_seating.GREY]
    assert np.allclose(colours, expected), colours

    props.upper = props.lower = None
    for obj in occlusion_seating.occlusion_points(scene) + [upper, lower]:
        bpy.data.objects.remove(obj)
    for arch_id in ('UPPER', 'LOWER'):
        bpy.data.objects.remove(occlusion.find_block(scene, arch_id))


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
    settings.use_bar = False                      # settings rebuild the model
    assert model_base.last_report(model)["bar"] is None and "Barre de renfort" not in model.modifiers
    bar = bpy.data.objects.get(f"Barre — {model.name}")
    assert bar is None, "no bar left behind"
    settings.use_bar = True
    for obj in (model, bpy.data.objects[f"Barre — {model.name}"], scan):
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


def test_bite_check():
    scene = bpy.context.scene

    def surface(name, z, size=(30, 20)):
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=30, y_segments=20, size=1.0)
        for v in bm.verts:
            v.co = (v.co.x * size[0] / 2, v.co.y * size[1] / 2, z + 0.8 * np.sin(v.co.x * 9))   # cusps
        return open_scan(name, bm)

    upper = surface("Test UpperJaw", 1.0)
    lower = surface("Test LowerJaw", -1.0)
    bm = bmesh.new()                        # the bite: a strip of each arch's surface, one object
    for arch in (upper, lower):
        part = bmesh.new()
        part.from_mesh(arch.data)
        bmesh.ops.delete(part, geom=[f for f in part.faces if f.calc_center_median().x > 0], context='FACES')
        mesh = bpy.data.meshes.new("part")
        part.to_mesh(mesh)
        part.free()
        bm.from_mesh(mesh)
        bpy.data.meshes.remove(mesh)
    bite = open_scan("Test BiteScan", bm)
    settings = scene.omfs_model
    settings.bite, settings.upper, settings.lower = bite, upper, lower
    assert bpy.ops.omfs.check_bite() == {'FINISHED'}
    result = model_base.last_bite(scene)
    assert result["ok"], result
    assert not model_base.bite_stale(scene, result)
    lower.location.z -= 0.5                 # the arches no longer in the bite's occlusion
    bpy.context.view_layer.update()
    assert model_base.bite_stale(scene, result), "moving an arch makes the check stale"
    bpy.ops.omfs.check_bite()
    result = model_base.last_bite(scene)
    assert not result["ok"] and result["parts"]["upper"]["ok"] and not result["parts"]["lower"]["ok"], result
    settings.bite = settings.upper = settings.lower = None
    del scene["omfs_bite_check"]
    for obj in (bite, upper, lower):
        bpy.data.objects.remove(obj)


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


def test_splint_blank():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)   # 2 mm gap: bite at z = 9
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower, props.centre = 'FINAL', upper, lower, True
    bpy.ops.omfs.arch_curve_clear()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for a in (10, 50, 90, 130, 170):
        u = radians(a)
        splint.add_point(scene, Vector((45 * np.cos(u), 500 + 45 * np.sin(u), 8)), depsgraph)
    assert all(abs(p.z - 9) < 1e-3 for p in splint.curve_points(splint.find_curve(scene)))

    blank = splint.build_blank(scene, depsgraph)
    assert not isinstance(blank, str), blank
    mesh = blank.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    assert all(e.is_manifold for e in bm.edges) and bm.calc_volume() > 0, "closed, outward normals"
    bm.free()
    # Crown band: 3 mm on each side of the bite (z 6 to 12). The 16-sided tubes have a vertex ring at
    # 67.5 degrees in that band (z 7.4), 5 * cos(67.5) = 1.91 mm off their axis: plus the 1.5 mm margins.
    info = splint.blank_info(blank)
    expected = 2 * (5 * np.cos(radians(67.5)) + 1.5)
    assert abs(info["width_max"] - expected) < 0.4 and abs(info["width_min"] - expected) < 0.6, info
    top = [v.co.z for v in mesh.vertices if v.index in set(range(len(mesh.vertices) // 2))]
    assert abs(min(top) - 12) < 1e-3 and abs(max(top) - 12) < 1e-3, "top = bite + 3 mm"
    curve_length = splint.curve_length(splint.find_curve(scene))
    assert abs(info["length"] - curve_length) < 0.5, "the blank stops at the ends of the curve"

    props.buccal_margin = 4.0                        # segmental: only the buccal side grows, live
    assert abs(splint.blank_info(blank)["width_max"] - (expected + 2.5)) < 0.4
    assert not splint.blank_stale(scene, blank)
    splint.add_point(scene, Vector((45, 500, 8)), depsgraph)
    assert splint.blank_stale(scene, blank), "a changed curve asks for an update"
    props.kind, props.cap_arch = 'CHAPE', lower      # a cap: points on the lower occlusal surface (z = 8)
    bpy.ops.omfs.arch_curve_clear()
    for a in (10, 50, 90, 130, 170):
        u = radians(a)
        splint.add_point(scene, Vector((45 * np.cos(u), 500 + 45 * np.sin(u), 8)), depsgraph)
    assert splint.find_blank(scene) == blank and splint.build_blank(scene, depsgraph) == blank
    z = [v.co.z for v in blank.data.vertices]
    assert abs(min(z) - (8 - params.SPLINT_CAP_DEPTH)) < 1e-3 and abs(max(z) - 11) < 1e-3, \
        "0.8 mm down onto the teeth, 3 mm above"
    for obj in (lower, upper, blank):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_splint_imprint():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)   # lower top z 8, upper bottom z 10
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower, props.centre = 'FINAL', upper, lower, True
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

    click(9)
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
    props.buccal_margin = 3.0                        # the settings no longer touch the solid
    assert len(solid.data.vertices) == count and len(blank.data.vertices) != count
    props.buccal_margin = params.SPLINT_BUCCAL_MARGIN
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
    for obj in (lower, upper, blank):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_splint_text_and_checks():
    scene = bpy.context.scene
    reference_frame.build(scene, FRAME_POINTS)       # up = world Z
    lower, upper = tube_scan("Lower tube", 0), tube_scan("Upper tube", 18)
    bpy.context.view_layer.update()
    props = scene.omfs_splint
    props.kind, props.upper, props.lower, props.centre, props.imprint = 'FINAL', upper, lower, True, False
    bpy.ops.omfs.arch_curve_clear()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for a in (10, 50, 90, 130, 170):
        u = radians(a)
        splint.add_point(scene, Vector((45 * np.cos(u), 500 + 45 * np.sin(u), 8)), depsgraph)
    blank = splint.build_blank(scene, depsgraph)     # flat top at z 12, bottom at 6

    def volume(obj):
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        v = bm.calc_volume()
        bm.free()
        return v

    before = volume(blank)
    settings = props.label
    settings.text, settings.height, settings.depth, settings.mode = "AB 12", 3.0, 0.5, 'ENGRAVE'
    # Two clicks on the top, at the front of the arch (y 545), reading along +x.
    label, warning = text_labels.add_label(blank, Vector((-5, 545, 12)), Vector((5, 545, 12)), Vector((0, 0, 1)), settings)
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
    props.buccal_margin = 2.0                        # rebuilt blank: the text follows
    assert abs(max(v.co.z for v in label.data.vertices) - 12.5) < 1e-3 and volume(blank) > before

    solid = splint.solidify(scene)                   # text placed before solidifying comes along
    assert text_labels.labels_on(solid) == [label]
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "splint.stl")
        bpy.ops.omfs.export_splint(filepath=path, part_name=solid.name)
        assert Path(path).stat().st_size > 1000

    text_labels.remove_label(label)
    assert not blank.modifiers and not solid.modifiers
    props.buccal_margin = params.SPLINT_BUCCAL_MARGIN
    for obj in (lower, upper, blank, solid):
        bpy.data.objects.remove(obj)
    bpy.ops.omfs.arch_curve_clear()


def test_deleted_work_copy_is_gone():
    scene = bpy.context.scene
    verts, tris = make_test_case.box((0, 0, 0), (10, 10, 10))
    mesh = bpy.data.meshes.new("Bone box")
    mesh.from_pydata(verts, [], tris)
    bone = bpy.data.objects.new("Bone box", mesh)
    scene.collection.objects.link(bone)
    scene.omfs_guide.source = bone
    assert guide_workshop.TOOLS['WORK_COPY'].run(bpy.context) is None
    copy = guide_workshop.current_copy(bpy.context)
    for coll in list(copy.users_collection):        # deleted in the viewport: the panel's pointer keeps it
        coll.objects.unlink(copy)
    assert guide_workshop.current_copy(bpy.context) is None
    assert guide_workshop.TOOLS['ZONE_PAINT'].run(bpy.context) == "Créez d'abord une copie de travail"
    scene.omfs_guide.work_copy = None
    bpy.data.objects.remove(copy)
    bpy.data.objects.remove(bone)


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
    guide = box_object("Pin guide", 'GUIDE', (0, -22, -60), (30, 4, 10))
    cap = box_object("Pin cap", 'GUIDE', (0, -15, -30), (30, 10, 4))
    point, normal = pins.osteotomy_plane(scene, guide)
    click = Vector((0, -24, -58))                    # on the guide's front face
    site = pins.add_site(scene, guide, click, Vector((0, -1, 0)))
    assert not isinstance(site, str), site
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

    p.clearance = 0.2                                # settings rebuild every part
    bore = bpy.data.objects[pins.part_name(site, "logement")]
    radius = max((bore.matrix_world @ v.co - stop).cross(axis).length for v in bore.data.vertices)
    assert abs(radius - (p.base_diameter / 2 + 0.2)) < 1e-3
    p.clearance = params.PIN_CLEARANCE
    pins.remove_site(site)
    assert not guide.modifiers and not cap.modifiers and not pins.sites(scene)
    for obj in (guide, cap, bone, cut):
        bpy.data.objects.remove(obj)
    for lm in ('MF_R', 'MF_L', 'POG'):
        bpy.data.objects.remove(landmarks.find(scene, lm))


def test_empty_folder_is_an_error():
    with tempfile.TemporaryDirectory() as folder:
        try:
            bpy.ops.omfs.import_case(directory=folder)
        except RuntimeError as error:   # operators report errors as RuntimeError in scripts
            assert "No STL, OBJ or PLY" in str(error)
        else:
            raise AssertionError("expected an error for an empty folder")


failures = 0
for name, test in list(globals().items()):
    if name.startswith("test_") and callable(test):
        try:
            test()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()

print(f"{failures} failure(s)")
sys.exit(1 if failures else 0)
