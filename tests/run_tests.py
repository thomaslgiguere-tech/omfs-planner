"""Headless tests, run inside Blender:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/run_tests.py

Uses synthetic meshes only (tools/make_test_case.py), written to a temporary folder.
"""

import sys
import tempfile
import traceback
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "extension"), str(REPO / "tools")]

import make_test_case  # noqa: E402
import omfs_planner  # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402
from omfs_planner import landmarks, reference_frame, roles  # noqa: E402

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
