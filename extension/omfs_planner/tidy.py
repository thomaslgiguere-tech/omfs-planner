"""Used tools are put away: once a helper object's Boolean modifiers are applied (or removed), it moves
to a hidden "Outils utilisés" collection, so the outliner only shows what is still useful.

A tool is any object that has been the operand of a Boolean modifier: pin parts, text cutters,
cylinders, the model's bar, the cut-edge box, a cutter of the surgeon's own... It is marked the first
time it is seen in a modifier; when no modifier uses it any more, it moves. Nothing is deleted: if a
modifier uses it again (Ctrl+Z, a new Boolean), it goes back to the collection it came from. Anatomy,
models, guides, splints and planning objects are never moved, even when used by a Boolean (the
intersection volume of Section & Analysis uses the bone, for instance).

Planning aids go there too when they are not needed: the reference planes while they are hidden
(they come back when shown), the arch curve once a splint is solidified and its blank hidden (it
comes back with the blank). The reference frame itself never moves: movements are expressed in it.

A timer checks once a second, in Object Mode only: applying with Ctrl+A in Blender's own menu tidies
up too, not only the OMFS buttons.
"""

import bpy

COLLECTION = "Outils utilisés"
USED = "omfs_tool"              # custom property: this object has been a Boolean operand
HOME = "omfs_tool_home"         # the collection it came from
HIDDEN = "omfs_tool_hidden"     # whether it was hidden when put away (a relinked object shows again)
NEVER = {'BONE', 'TEETH', 'NERVE', 'SKIN', 'SCAN', 'SEGMENT', 'GUIDE', 'LANDMARK', 'REFERENCE', 'CUT',
         'BLOCK', 'ARCH', 'ANALYSIS'}
NEVER_MADE_BY = {"dental model", "splint blank", "splint solid", "workshop"}
INTERVAL = 1.0                  # seconds between checks


def can_be_tool(obj):
    return obj.omfs.role not in NEVER and obj.omfs.made_by not in NEVER_MADE_BY


def used_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
        layer = bpy.context.view_layer.layer_collection.children.get(coll.name)
        if layer is not None:
            layer.hide_viewport = True                 # out of sight; the eye in the outliner shows it
    return coll


def operands(scene):
    """Objects used right now by a Boolean modifier of an object in the scene."""
    return {m.object for o in scene.objects for m in o.modifiers if m.type == 'BOOLEAN' and m.object is not None}


def put_away(obj, coll):
    obj[HOME] = obj.users_collection[0].name if obj.users_collection else ""
    obj[HIDDEN] = obj.hide_get()
    for other in list(obj.users_collection):
        other.objects.unlink(obj)
    coll.objects.link(obj)


def bring_back(obj, coll, scene):
    home = bpy.data.collections.get(obj.get(HOME, ""))
    (home if home is not None and home in scene.collection.children_recursive else scene.collection).objects.link(obj)
    coll.objects.unlink(obj)
    obj.hide_set(obj.get(HIDDEN, False))   # case1: the cut box (400 mm) came back visible over the guide


def unneeded_aids(scene):
    """{planning aid: True if it can go away now} for the reference planes and the arch curve."""
    from . import reference_frame, splint
    aids = {}
    frame = reference_frame.find_frame(scene)
    if frame is not None:
        for plane in frame.children:
            aids[plane] = plane.hide_get()
    curve = splint.find_curve(scene)
    if curve is not None:
        blank = splint.find_blank(scene)
        aids[curve] = bool(splint.find_solids(scene)) and (blank is None or blank.hide_get())
    return aids


def tidy(scene):
    """Mark the tools in use, put away the used ones no longer needed and the planning aids not
    needed now, bring back those needed again. Returns the number of objects moved."""
    in_use = operands(scene)
    for obj in in_use:
        if can_be_tool(obj):
            obj[USED] = True
    away = {obj: obj not in in_use for obj in scene.objects if obj.get(USED) and can_be_tool(obj)}
    away.update(unneeded_aids(scene))
    coll = bpy.data.collections.get(COLLECTION)
    moved = 0
    for obj, should_go in away.items():
        is_away = coll is not None and coll in obj.users_collection
        if should_go and not is_away:
            coll = used_collection(scene)
            put_away(obj, coll)
            moved += 1
        elif is_away and not should_go:
            bring_back(obj, coll, scene)
            moved += 1
    return moved


def _tick():
    context = bpy.context
    try:
        if context.mode == 'OBJECT' and context.scene is not None:
            tidy(context.scene)
    except (AttributeError, ReferenceError, RuntimeError):
        pass                                           # no window yet, file loading...
    return INTERVAL


def register():
    bpy.app.timers.register(_tick, first_interval=INTERVAL, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
