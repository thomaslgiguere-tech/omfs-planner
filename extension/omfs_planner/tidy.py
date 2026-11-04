"""Used tools are put away: once a helper object's Boolean modifiers are applied (or removed), it moves
to a hidden "Outils utilisés" collection, so the outliner only shows what is still useful.

A tool is any object that has been the operand of a Boolean modifier: pin parts, text cutters,
cylinders, the model's bar, the cut-edge box, a cutter of the surgeon's own... It is marked the first
time it is seen in a modifier; when no modifier uses it any more, it moves. Nothing is deleted: if a
modifier uses it again (Ctrl+Z, a new Boolean), it goes back to the collection it came from. Anatomy,
models, guides, splints and planning objects are never moved, even when used by a Boolean (the
intersection volume of Section & Analysis uses the bone, for instance).

A timer checks once a second, in Object Mode only: applying with Ctrl+A in Blender's own menu tidies
up too, not only the OMFS buttons.
"""

import bpy

COLLECTION = "Outils utilisés"
USED = "omfs_tool"              # custom property: this object has been a Boolean operand
HOME = "omfs_tool_home"         # the collection it came from
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


def tidy(scene):
    """Mark the tools in use, put away the used ones no longer needed, bring back the ones needed
    again. Returns the number of objects moved."""
    in_use = operands(scene)
    moved = 0
    for obj in in_use:
        if can_be_tool(obj):
            obj[USED] = True
    coll = bpy.data.collections.get(COLLECTION)
    for obj in list(scene.objects):
        if not obj.get(USED) or not can_be_tool(obj):
            continue
        put_away = coll is not None and coll in obj.users_collection
        if obj not in in_use and not put_away:
            coll = used_collection(scene)
            obj[HOME] = obj.users_collection[0].name if obj.users_collection else ""
            for other in list(obj.users_collection):
                other.objects.unlink(obj)
            coll.objects.link(obj)
            moved += 1
        elif obj in in_use and put_away:
            home = bpy.data.collections.get(obj.get(HOME, ""))
            (home if home is not None and home.name in scene.collection.children_recursive
             else scene.collection).objects.link(obj)
            coll.objects.unlink(obj)
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
