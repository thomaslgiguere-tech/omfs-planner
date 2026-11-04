"""Guide design workshop ("Confection de guide"): Blender's native tools, one click away.

A learning bench, not an automatic guide builder: each button runs a native Blender tool
(sculpt face sets and mask, Edit Mode selection, modifiers, Boolean...) and its tooltip explains what it does,
its keyboard shortcut and the closest Meshmixer tool. The texts are in French, for the surgeon.

Anatomy is never edited. Zones are painted "on the bone": the surgeon picks the bone (or teeth) and
paints, and the painting really happens on a copy put in its place automatically (the bone hidden
while painting, shown again after the extraction). Face sets and the mask are values stored on the
mesh, and a Sculpt stroke can still move vertices (Shift = smooth): on the CBCT mesh itself that
would break "CBCT mesh data is never modified".
"""

from collections import namedtuple
from math import radians

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import cylinders, landmarks, locking, osteotomy, params, pins

COLLECTION = "Guide design"
FACE_SETS = ".sculpt_face_set"   # face attribute: the face set number of each face
MASK = ".sculpt_mask"            # vertex attribute: the sculpt mask, 0 to 1
BRUSHES = {   # section 7: essentials brush -> (label, icon, what it does, Meshmixer)
    "Smooth": ("Pinceau Smooth (adoucir)", 'MOD_SMOOTH',
               "Lisse la surface sous le pinceau: bosses, marches, bords irréguliers.", "Sculpt > Robust Smooth"),
    "Flatten/Contrast": ("Pinceau Flatten (aplatir)", 'MESH_PLANE',
                         "Ramène la surface vers un plan moyen (Ctrl: au contraire, accentue).", "Sculpt > Flatten"),
    "Clay Strips": ("Pinceau Clay Strips (ajouter)", 'ADD',
                    "Ajoute de la matière par bandes (Ctrl: en retire), pour renforcer ou combler.",
                    "Sculpt > Draw / Inflate"),
    "Grab": ("Pinceau Grab (tirer)", 'VIEW_PAN',
             "Attrape la surface et la déplace avec la souris, sans ajouter de matière.", "Sculpt > Drag"),
}
# Shown under every tool of section 7: the keys worth knowing with a brush in hand (Sculpt Mode).
SCULPT_KEYS = ("Raccourcis en Sculpture (souris dans la vue):\n"
               "• Ctrl + glisser: effet inverse (+ ↔ −): Clay Strips retire, Flatten accentue, Mask efface\n"
               "• F puis bouger la souris: taille du pinceau (clic pour valider)\n"
               "• Maj+F puis bouger la souris: force du pinceau\n"
               "• Maj + glisser: lisser, quel que soit le pinceau\n"
               "• Ctrl+Z: annuler le dernier trait")
SHELL_GROUP = "Coque (Solidify)"   # filled by the Solidify modifier, see smooth_outside
# Boolean solver on the guide. Exact, after the pins' unions (boss tops flush with the plane, the
# bridges' too), left a few open edges; the cut box's Difference on that then kept part of the box
# (case1: a 400 mm cube over the guide). Manifold gave a clean closed guide on the same stack.
SOLVER = 'MANIFOLD'

# operator: a dedicated operator to draw instead of omfs.guide_tool (when the tool needs its own
# adjustable settings in the Adjust Last Operation panel).
Tool = namedtuple("Tool", "id label icon section tip meshmixer shortcut run operator")
TOOLS = {}

# Sections, in panel order: (id, title, short explanation shown above the buttons)
SECTIONS = (
    ('ZONES', "1. Zones (Face Sets)",
     "Peindre en mode Sculpture les zones utiles (appui du guide, fenêtre de coupe, fourreaux...): "
     "chaque trait crée une zone d'une nouvelle couleur, au bord net. Choisir une zone, puis "
     "l'extraire en coque. On peint « sur l'os »: une copie identique prend sa place le temps de "
     "peindre (l'os CBCT n'est jamais touché), puis l'os revient après l'extraction."),
    ('MASK', "1b. Masque (bord lissé, optionnel)",
     "Pour un bord de guide plus régulier: convertir la zone en masque, lisser son bord, puis "
     "extraire. Le masque ne déplace aucun sommet: c'est une valeur posée sur la surface peinte."),
    ('EDIT', "2. Mode Édition (sélection, extrusion)",
     "Travailler directement les sommets, arêtes et faces de l'objet actif (le guide, pas l'os). "
     "Tab entre et sort du mode Édition."),
    ('SHAPE', "3. Épaisseur et dégagement (modificateurs)",
     "Un modificateur change la forme affichée sans toucher au maillage, tant qu'il n'est pas "
     "appliqué. On peut donc régler les valeurs à volonté, puis appliquer à la fin."),
    ('BOOLEAN', "4. Vis, manchons et Méga Boolean",
     "Les vis finales (segment en position planifiée) donnent l'axe des manchons: cliquer près d'une vis, "
     "glisser le cylindre le long de son axe, cliquer. Flèches et diamètre restent réglables jusqu'au "
     "Méga Boolean, qui applique tout dans l'ordre: unions, différences, trait de coupe, trous de fixation."),
    ('OSTEO', "5. Trait de coupe (bord du guide)",
     "Coupe le guide exactement dans le plan d'ostéotomie et garde un seul côté: la face plane "
     "obtenue guide la lame. Si la coupe change dans Osteotomy, recliquez pour recaler le bord."),
    ('PINS', "6. Tubes and pins (guide ↔ chape)",
     "Le guide de coupe et la chape s'emboîtent: femelles coniques dans le guide (le dessus du bossage "
     "est exactement dans le plan d'ostéotomie: c'est la butée), pins mâles, puis un bras courbe qui "
     "part de la chape à 90° et arrive dans l'axe de chaque pin. Tout reste modifiable jusqu'à Appliquer."),
    ('SCULPT', "7. Sculpture (adoucir, intrados protégé)",
     "Pinceaux de Sculpture sur n'importe quel objet libre (le guide, la chape, un bras...). Les sommets "
     "masqués (gris foncé) ne bougent jamais: protéger l'intrados par la distance à l'os, ou découper "
     "l'objet en zones aux arêtes vives (angle réglable) et cliquer les zones à masquer."),
    ('CHECK', "8. Vérifier et finir",
     "Avant d'imprimer: le guide doit être fermé (étanche), sans arête libre."),
)


def tool(tool_id, label, icon, section, tip, meshmixer, shortcut="", operator=None):
    """Register a workshop tool. The function returns None, or an error message for the user."""
    def register_tool(run):
        TOOLS[tool_id] = Tool(tool_id, label, icon, section, tip, meshmixer, shortcut, run, operator)
        return run
    return register_tool


def tools_in(section):
    return [t for t in TOOLS.values() if t.section == section]


# --- Helpers ---------------------------------------------------------------------------

def props(context):
    return context.scene.omfs_guide


def viewport(context):
    """Context override on the 3D view's main region, for tools that follow the mouse."""
    area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
        a for a in context.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return context.temp_override(area=area, region=region)


def workshop_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def make_active(context, obj):
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for other in context.selected_objects:
        other.select_set(False)
    context.view_layer.objects.active = obj
    obj.select_set(True)


def current_copy(context):
    """The painting copy, or None if there is none or it was deleted from the scene (the panel's
    pointer keeps a deleted object alive in the file, outside every collection)."""
    copy = props(context).work_copy
    return copy if copy is not None and copy.name in context.scene.objects else None


def paint_source(context):
    """The bone or teeth zones are painted on: the one chosen, else the first bone (or segment)."""
    source = props(context).source
    if source is not None and source.name in context.scene.objects:
        return source
    return next((o for role in ('BONE', 'SEGMENT') for o in context.scene.objects
                 if o.type == 'MESH' and o.omfs.role == role), None)


def paint_copy(context):
    """The copy zones are painted on, made the first time for the chosen bone (or None): same mesh
    and colour, in the bone's place. The bone itself is never painted (see the module docstring)."""
    source = paint_source(context)
    if source is None:
        return None
    copy = current_copy(context)
    if copy is not None and copy.get("omfs_copy_of") == source.name:
        return copy
    copy = next((o for o in context.scene.objects if o.get("omfs_copy_of") == source.name), None)
    if copy is None:
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        copy = bpy.data.objects.new(f"{source.name} (zones)", source.data.copy())
        copy.data.name = copy.name
        workshop_collection(context.scene).objects.link(copy)
        copy.matrix_world = source.matrix_world
        copy.color = source.color
        copy.omfs.role = 'OTHER'
        copy.omfs.made_by = "workshop"
        copy["omfs_copy_of"] = source.name
    props(context).work_copy = copy
    return copy


def show_source(context):
    """Back to the bone: the painting copy hidden, the bone shown (after an extraction, Object Mode)."""
    copy = current_copy(context)
    if copy is None:
        return
    copy.hide_set(True)
    source = context.scene.objects.get(copy.get("omfs_copy_of", ""))
    if source is not None:
        source.hide_set(False)


def is_guide(obj):
    return obj is not None and obj.type == 'MESH' and obj.omfs.role == 'GUIDE'


def target_guide(context):
    """The guide chosen at the top of the panel, else the last one extracted (None if none)."""
    chosen = props(context).guide
    if is_guide(chosen) and chosen.name in context.scene.objects:
        return chosen
    guides = [o for o in context.scene.objects if is_guide(o)]
    return guides[-1] if guides else None


def editable_active(context):
    """The object the modelling tools act on: the active object if it is a free mesh (the guide,
    a cylinder...), else the guide. The painting copy is only for painting zones, and anatomy is
    locked: with either active, the tools go to the guide instead."""
    obj = context.active_object
    if (obj is not None and obj.type == 'MESH' and not locking.is_locked(obj)
            and obj != current_copy(context)):
        return obj, None
    guide = target_guide(context)
    if guide is None:
        return None, "Pas encore de guide: extrayez une zone (section 1)"
    guide.hide_set(False)
    make_active(context, guide)
    return guide, None


def need_mode(context, mode):
    obj, error = editable_active(context)
    if error:
        return error
    if context.mode != mode:
        bpy.ops.object.mode_set(mode={'EDIT_MESH': 'EDIT'}.get(mode, mode))
    return None


def add_modifier(context, kind, name, **settings):
    obj, error = editable_active(context)
    if error:
        return error
    mod = obj.modifiers.get(name) or obj.modifiers.new(name, kind)
    for key, value in settings.items():
        setattr(mod, key, value)
    return None


def sleeve_bore_diameter():
    return params.SLEEVE_OUTER_DIAMETER + params.SLEEVE_FIT_CLEARANCE


# --- 0. Start ----------------------------------------------------------------------------

@tool('OBJECT_MODE', "Revenir en mode Objet", 'OBJECT_DATAMODE', 'START',
      "Quitte le mode Sculpture, Édition ou Peinture. En mode Objet on manipule des objets entiers "
      "(et l'outil Navigate fait tourner la vue au glisser).",
      "Quitter un outil (Accept / Cancel)", "Tab (depuis Édition), Ctrl+Tab (menu des modes)")
def object_mode(context):
    painting = context.active_object is not None and context.active_object == current_copy(context)
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if painting:
        show_source(context)
    return None


# --- 1. Zones (face sets) ------------------------------------------------------------------

def need_sculpt(context):
    copy = paint_copy(context)
    if copy is None:
        return "Choisissez l'os (ou les dents) sur lequel peindre"
    copy.hide_set(False)   # hidden after an extraction, so the guide can be clicked
    source = context.scene.objects.get(copy["omfs_copy_of"])
    if source is not None:
        source.hide_set(True)   # the copy is in its place, painted instead
    if context.active_object != copy:
        make_active(context, copy)
    if context.mode != 'SCULPT':
        bpy.ops.object.mode_set(mode='SCULPT')
    return None


def face_sets(mesh):
    """Face set number of each face, or None if no zone was ever painted."""
    attribute = mesh.attributes.get(FACE_SETS)
    if attribute is None:
        return None
    values = np.empty(len(mesh.polygons), np.int32)
    attribute.data.foreach_get("value", values)
    return values


def chosen_zone(context):
    """(face set number, faces in it) of the chosen zone (the last painted one by default),
    or (None, error message)."""
    copy = current_copy(context)
    if copy is None:
        return None, "Peignez d'abord une zone"
    sets = face_sets(copy.data)
    if sets is None or len(sets) == 0:
        return None, "Peignez d'abord une zone"
    zone = props(context).zone
    if zone not in sets:
        zone = int(sets.max())   # each new stroke gets the next number: the last painted
    return zone, sets == zone


def zone_summary(context):
    """One line for the panel: which zone the buttons below act on."""
    zone, faces = chosen_zone(context)
    if zone is None:
        return faces
    which = "dernière peinte" if props(context).zone != zone else f"n° {zone}"
    return f"Zone choisie: {which} ({int(faces.sum())} faces)"


def zone_to_mask(context):
    """Mask exactly the vertices of the chosen zone (everything else unmasked)."""
    error = need_sculpt(context)
    if error:
        return error
    zone, faces = chosen_zone(context)
    if zone is None:
        return faces
    bpy.ops.object.mode_set(mode='OBJECT')   # write the attribute on the mesh itself
    set_mask(current_copy(context).data, faces)
    bpy.ops.object.mode_set(mode='SCULPT')
    return None


def set_mask(mesh, faces):
    """Mask = 1 on every vertex of the faces given (boolean per face), 0 elsewhere."""
    write_mask(mesh, np.zeros(len(mesh.vertices), np.float32))
    mask_faces(mesh, faces, 1.0)


def read_mask(mesh):
    attribute = mesh.attributes.get(MASK)
    mask = np.zeros(len(mesh.vertices), np.float32)
    if attribute is not None:
        attribute.data.foreach_get("value", mask)
    return mask


def write_mask(mesh, mask):
    """Write the sculpt mask (Object Mode: in Sculpt Mode the attribute is not read back)."""
    attribute = mesh.attributes.get(MASK) or mesh.attributes.new(MASK, 'FLOAT', 'POINT')
    attribute.data.foreach_set("value", mask)
    mesh.update()


def mask_faces(mesh, faces, value):
    """Set the mask to `value` on every vertex of the faces given (boolean per face), keep the rest."""
    sizes = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    loops = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    mask = read_mask(mesh)
    mask[loops[np.repeat(faces, sizes)]] = value
    write_mask(mesh, mask)


@tool('ZONE_PAINT', "Peindre une zone", 'BRUSH_DATA', 'ZONES',
      "Pinceau Face Set Paint: chaque trait crée une nouvelle zone (nouvelle couleur). Ctrl + glisser "
      "agrandit la zone sous la souris au début du trait. F puis bouger la souris change la taille. "
      "Évitez Maj + glisser: il lisse le bord en déplaçant des sommets.",
      "Select (pinceau) + Create Facegroup", "Ctrl+glisser = continuer la zone, F = taille")
def zone_paint(context):
    error = need_sculpt(context)
    if error:
        return error
    props(context).zone = 0   # the zone about to be painted becomes the chosen one
    return activate_brush("Face Set Paint")


@tool('ZONE_LASSO', "Zone au lasso", 'MOD_MASK', 'ZONES',
      "Dessinez un contour: les faces à l'intérieur (y compris à l'arrière) forment une nouvelle zone.",
      "Select > Lasso + Create Facegroup", "")
def zone_lasso(context):
    error = need_sculpt(context)
    if error:
        return error
    props(context).zone = 0
    bpy.ops.wm.tool_set_by_id(name="builtin.lasso_face_set")
    return None


@tool('ZONE_PICK', "Choisir la zone (clic)", 'EYEDROPPER', 'ZONES',
      "Cliquez une zone dans la vue: les boutons suivants (agrandir, isoler, extraire...) agiront sur "
      "elle. Par défaut, c'est la dernière zone peinte.",
      "Select > Select Facegroup (clic)", "", operator="omfs.pick_zone")
def zone_pick(context):
    with viewport(context):
        bpy.ops.omfs.pick_zone('INVOKE_DEFAULT')
    return None


def zone_edit(mode):
    def run(context):
        error = need_sculpt(context)
        if error:
            return error
        zone, faces = chosen_zone(context)
        if zone is None:
            return faces
        with viewport(context):
            bpy.ops.sculpt.face_set_edit(active_face_set=zone, mode=mode, modify_hidden=False)
        return None
    return run


tool('ZONE_GROW', "Agrandir la zone", 'ADD', 'ZONES', "Ajoute une rangée de faces autour de la zone choisie.",
     "Modify > Expand Ring", "Ctrl+W (zone sous la souris)")(zone_edit('GROW'))
tool('ZONE_SHRINK', "Réduire la zone", 'REMOVE', 'ZONES', "Retire une rangée de faces au bord de la zone choisie.",
     "Modify > Contract Ring", "Ctrl+Alt+W (zone sous la souris)")(zone_edit('SHRINK'))


@tool('ZONE_ISOLATE', "Isoler la zone", 'HIDE_OFF', 'ZONES',
      "Cache tout sauf la zone choisie: plus lisible et plus fluide sur une mandibule CT. Rien n'est "
      "supprimé (Tout montrer).",
      "Hide / Show (Facegroups)", "Maj+H (zone sous la souris), H = cacher la zone")
def zone_isolate(context):
    error = need_sculpt(context)
    if error:
        return error
    zone, faces = chosen_zone(context)
    if zone is None:
        return faces
    with viewport(context):
        bpy.ops.paint.hide_show_all(action='SHOW')
        bpy.ops.sculpt.face_set_change_visibility(mode='TOGGLE', active_face_set=zone)
    return None


@tool('ZONE_SHOW_ALL', "Tout montrer", 'HIDE_ON', 'ZONES', "Réaffiche toutes les faces cachées.",
      "Show All", "Alt+H")
def zone_show_all(context):
    error = need_sculpt(context)
    if error:
        return error
    with viewport(context):
        bpy.ops.paint.hide_show_all(action='SHOW')
    return None


@tool('ZONE_EXTRACT', "Extraire la zone en coque (guide)", 'MOD_SOLIDIFY', 'ZONES',
      "Crée un NOUVEL objet à partir de la zone choisie, collé à l'os, avec un dégagement et une "
      "épaisseur réglables (modificateurs Displace + Solidify, section 3). L'os ne change pas.",
      "Edit > Extract (Offset + Thickness)", "")
def zone_extract(context):
    return zone_to_mask(context) or mask_extract(context)


# --- 1b. Mask ------------------------------------------------------------------------------

@tool('ZONE_TO_MASK', "Zone → masque", 'MOD_MASK', 'MASK',
      "Masque exactement les sommets de la zone choisie (section 1), pour en lisser le bord avant "
      "d'extraire. Remplace le masque existant.",
      "Select > Select Facegroup", "")
def zone_mask(context):
    return zone_to_mask(context)


def activate_brush(brush):
    try:
        bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS',
                                     relative_asset_identifier=f"brushes/essentials_brushes-mesh_sculpt.blend/Brush/{brush}")
    except RuntimeError:
        return f"Pinceau {brush} introuvable: choisissez-le dans la barre d'outils de gauche"
    return None


def mask_tools(prefix, section, need):
    """The mask tools of a section; `need(context)` brings the right object into Sculpt Mode
    (the painted bone in 1b, the active free mesh in 7) or returns an error."""
    def run(action):
        def tool_run(context):
            return need(context) or action(context)
        return tool_run

    def mask_filter(kind):
        def action(context):
            bpy.ops.sculpt.mask_filter(filter_type=kind, auto_iteration_count=True)
        return run(action)

    tool(prefix + 'PAINT', "Peindre le masque", 'BRUSH_DATA', section,
         "Pinceau Mask: le glisser peint en gris foncé. Ctrl + glisser efface. "
         "F puis bouger la souris change la taille du pinceau. Ne déforme rien.",
         "Select (pinceau de sélection)", "Ctrl+glisser = effacer, F = taille")(
        run(lambda context: activate_brush("Mask")))
    tool(prefix + 'LASSO', "Masque au lasso", 'MOD_MASK', section,
         "Dessinez un contour: tout ce qui est à l'intérieur est masqué (y compris l'arrière). "
         "Ctrl + lasso retire du masque.",
         "Select > Lasso", "")(run(lambda context: bpy.ops.wm.tool_set_by_id(name="builtin.lasso_mask") and None))
    tool(prefix + 'GROW', "Agrandir", 'ADD', section, "Élargit le bord du masque d'une rangée de sommets.",
         "Modify > Expand Ring", "")(mask_filter('GROW'))
    tool(prefix + 'SHRINK', "Réduire", 'REMOVE', section, "Rétrécit le bord du masque d'une rangée de sommets.",
         "Modify > Contract Ring", "")(mask_filter('SHRINK'))
    tool(prefix + 'SMOOTH', "Lisser le bord", 'MOD_SMOOTH', section,
         "Adoucit le contour du masque.", "Modify > Smooth Boundary", "Maj + glisser avec le pinceau")(
        mask_filter('SMOOTH'))
    tool(prefix + 'INVERT', "Inverser", 'ARROW_LEFTRIGHT', section, "Le masqué devient libre et inversement.",
         "Modify > Invert", "Ctrl+I (en Sculpture)")(
        run(lambda context: bpy.ops.paint.mask_flood_fill(mode='INVERT') and None))
    tool(prefix + 'CLEAR', "Effacer le masque", 'X', section, "Retire tout le masque pour recommencer.",
         "Select > Clear selection", "Alt+M (en Sculpture)")(
        run(lambda context: bpy.ops.paint.mask_flood_fill(mode='VALUE', value=0.0) and None))


mask_tools('MASK_', 'MASK', need_sculpt)


@tool('MASK_EXTRACT', "Extraire en coque (guide)", 'MOD_SOLIDIFY', 'MASK',
      "Crée un NOUVEL objet à partir de la zone masquée, collé à l'os, avec un dégagement et une "
      "épaisseur réglables (modificateurs Displace + Solidify, section 3). L'os ne change pas.",
      "Edit > Extract (Offset + Thickness)", "")
def mask_extract(context):
    error = need_sculpt(context)
    if error:
        return error
    before = set(bpy.data.objects)
    bpy.ops.sculpt.paint_mask_extract(mask_threshold=0.5, add_boundary_loop=True, smooth_iterations=4,
                                      apply_shrinkwrap=True, add_solidify=True)
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        return "Rien à extraire: peignez d'abord un masque"
    bpy.ops.object.mode_set(mode='OBJECT')
    finish_extracted_guide(context, new[0])
    return None


def finish_extracted_guide(context, guide):
    """Name, file and colour the extracted shell; set its clearance and thickness modifiers."""
    guide.name = guide.data.name = "Guide"
    for coll in guide.users_collection:
        coll.objects.unlink(guide)
    workshop_collection(context.scene).objects.link(guide)
    guide.omfs.role = 'GUIDE'
    make_active(context, guide)
    props(context).guide = guide   # the modelling tools act on it from now on
    solidify = next((m for m in guide.modifiers if m.type == 'SOLIDIFY'), None)
    if solidify is not None:
        solidify.name = "Épaisseur"
        solidify.thickness = params.GUIDE_THICKNESS
        solidify.offset = 1.0              # grow outwards, away from the bone
        # Even Thickness divides by the angle at sharp folds of the border: spikes of tens of mm.
        # Without it the thickness only thins at those folds; same thickness elsewhere.
        solidify.use_even_offset = False
        solidify.use_quality_normals = True
    clearance = guide.modifiers.new("Dégagement", 'DISPLACE')
    clearance.strength = params.GUIDE_CLEARANCE
    clearance.mid_level = 0.0
    clearance.direction = 'NORMAL'
    guide.modifiers.move(len(guide.modifiers) - 1, 0)   # gap first, then thickness
    # In Edit Mode, show the surface being edited (the footprint on the bone), not the shell
    # built on it: the shell would hide the vertices.
    for mod in guide.modifiers:
        mod.show_in_editmode = False
    # The guide's footprint lies exactly on the painted surface: back to the bone, the copy hidden
    # (or every click lands on it). Painting a zone shows it again.
    show_source(context)


# --- 2. Edit Mode ------------------------------------------------------------------------

@tool('EDIT_MODE', "Entrer en mode Édition", 'EDITMODE_HLT', 'EDIT',
      "Montre les sommets, arêtes et faces de l'objet actif pour les sélectionner et les modifier.",
      "Select tool (sélection de triangles)", "Tab")
def edit_mode(context):
    copy = current_copy(context)
    if copy is not None and context.active_object not in (None, copy):
        show_source(context)   # editing the guide: its vertices lie on the painted surface
    return need_mode(context, 'EDIT_MESH')


def select_mode(kind):
    def run(context):
        error = need_mode(context, 'EDIT_MESH')
        if error:
            return error
        bpy.ops.mesh.select_mode(type=kind)
        return None
    return run


tool('SEL_VERT', "Sommets", 'VERTEXSEL', 'EDIT', "Sélectionner des sommets (points).",
     "Aucun (Meshmixer sélectionne des triangles)", "1")(select_mode('VERT'))
tool('SEL_EDGE', "Arêtes", 'EDGESEL', 'EDIT', "Sélectionner des arêtes (traits entre sommets).",
     "Aucun", "2")(select_mode('EDGE'))
tool('SEL_FACE', "Faces", 'FACESEL', 'EDIT', "Sélectionner des faces (triangles).",
     "Select (triangles)", "3")(select_mode('FACE'))


def select_tool(tool_name):
    def run(context):
        error = need_mode(context, 'EDIT_MESH')
        if error:
            return error
        bpy.ops.wm.tool_set_by_id(name=tool_name)
        return None
    return run


tool('SEL_CIRCLE', "Pinceau de sélection", 'MESH_CIRCLE', 'EDIT',
     "Glisser = sélectionne sous le cercle, Ctrl + glisser = désélectionne. Molette = taille.",
     "Select (pinceau)", "")(select_tool("builtin.select_circle"))
tool('SEL_LASSO', "Lasso", 'MOD_CURVE', 'EDIT', "Dessiner un contour pour sélectionner l'intérieur.",
     "Select > Lasso", "")(select_tool("builtin.select_lasso"))


def mesh_op(op_name, **kwargs):
    def run(context):
        error = need_mode(context, 'EDIT_MESH')
        if error:
            return error
        module, name = op_name.split(".")
        getattr(getattr(bpy.ops, module), name)(**kwargs)
        return None
    return run


tool('SEL_MORE', "Agrandir la sélection", 'ADD', 'EDIT', "Ajoute une rangée autour de la sélection.",
     "Modify > Expand Ring", "Ctrl + pavé +")(mesh_op("mesh.select_more"))
tool('SEL_LESS', "Réduire la sélection", 'REMOVE', 'EDIT', "Retire une rangée au bord de la sélection.",
     "Modify > Contract Ring", "Ctrl + pavé -")(mesh_op("mesh.select_less"))
tool('SEL_LINKED', "Sélection liée", 'LINKED', 'EDIT', "Étend à toute la pièce connectée.",
     "Modify > Select Connected", "Ctrl+L")(mesh_op("mesh.select_linked"))
tool('SEL_LINKED_FLAT', "Sélection liée par angle", 'MOD_EDGESPLIT', 'EDIT',
     "Part de la face sélectionnée et s'étend tant que l'angle entre faces voisines reste sous le seuil "
     f"({params.SELECT_FLAT_ANGLE:.0f}° par défaut). Ex.: cliquer une face de l'intrados, puis ce bouton: "
     "la sélection couvre l'intrados et s'arrête au rebord. Réglez l'angle en direct dans le panneau "
     "« Adjust Last Operation » en bas à gauche (F9).",
     "Aucun équivalent direct (Select > Expand, sans seuil d'angle)", "Select > Select Linked > Linked Flat Faces",
     operator="omfs.select_linked_flat")(lambda c: None)
tool('SEL_INVERT', "Inverser la sélection", 'ARROW_LEFTRIGHT', 'EDIT', "Sélectionne tout le reste.",
     "Modify > Invert", "Ctrl+I")(mesh_op("mesh.select_all", action='INVERT'))
tool('SEL_BOUNDARY', "Bord de la sélection", 'SELECT_SUBTRACT', 'EDIT',
     "Ne garde que la boucle du bord de la zone sélectionnée (utile pour lisser un bord).",
     "Modify > Select Boundary", "")(mesh_op("mesh.region_to_loop"))


@tool('EXTRUDE', "Extruder le long des normales", 'MOD_SOLIDIFY', 'EDIT',
      "Pousse la sélection perpendiculairement à la surface: bougez la souris, tapez une valeur en mm "
      "(ex. 2) puis Entrée. Clic droit annule.",
      "Edit > Extrude", "Alt+E > Extrude Faces Along Normals")
def extrude(context):
    error = need_mode(context, 'EDIT_MESH')
    if error:
        return error
    with viewport(context):
        bpy.ops.mesh.extrude_region_shrink_fatten('INVOKE_DEFAULT')
    return None


def has_unapplied_thickness(obj):
    return any(m.type == 'SOLIDIFY' for m in obj.modifiers)


@tool('HIDE_AND_SMOOTH', "Cacher la sélection + pinceau Smooth", 'HIDE_ON', 'EDIT',
      "Cache les faces sélectionnées (ex. l'intrados) puis passe en mode Sculpture avec le pinceau "
      "Smooth: polissez le reste au glisser, les faces cachées ne bougent pas. F = taille du pinceau. "
      "« Tout réafficher » à la fin. Uniquement après avoir appliqué l'Épaisseur.",
      "Freeze (sélection) + Sculpt > Robust Smooth", "H, puis Ctrl+Tab > Sculpt, Maj+S")
def hide_and_smooth(context):
    obj, error = editable_active(context)
    if error:
        return error
    if has_unapplied_thickness(obj):
        return ("Appliquez d'abord les modificateurs: tant que l'Épaisseur n'est pas appliquée, "
                "le maillage visible EST l'intrados (utilisez plutôt « Lisser l'extérieur seulement »)")
    if context.mode != 'EDIT_MESH':
        return "Sélectionnez d'abord en mode Édition les faces à protéger"
    bpy.ops.mesh.hide(unselected=False)
    bpy.ops.object.mode_set(mode='SCULPT')
    return activate_brush("Smooth")


@tool('REVEAL', "Tout réafficher", 'HIDE_OFF', 'EDIT',
      "Réaffiche les faces cachées (en mode Édition ou Sculpture).", "Unfreeze", "Alt+H")
def reveal(context):
    obj, error = editable_active(context)
    if error:
        return error
    if context.mode == 'SCULPT':
        bpy.ops.paint.hide_show_all(action='SHOW')
    else:
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.reveal(select=False)
    return None


tool('SMOOTH_VERTS', "Lisser la sélection", 'MOD_SMOOTH', 'EDIT',
     "Adoucit les sommets sélectionnés (bosses, bords irréguliers).",
     "Deform > Smooth", "")(mesh_op("mesh.vertices_smooth", factor=0.5, repeat=5))
tool('FILL_HOLES', "Combler les trous", 'MOD_TRIANGULATE', 'EDIT',
     "Ferme les trous entourés par la sélection (sélectionnez tout avec A pour tout fermer).",
     "Inspector > Auto Repair", "")(mesh_op("mesh.fill_holes", sides=0))
tool('SEPARATE', "Séparer la sélection", 'UNLINKED', 'EDIT',
     "Détache la sélection dans un nouvel objet.",
     "Edit > Separate", "P > Selection")(mesh_op("mesh.separate", type='SELECTED'))


# --- 3. Modifiers ------------------------------------------------------------------------

tool('ADD_CLEARANCE', "Dégagement (0,10 mm)", 'MOD_DISPLACE', 'SHAPE',
     "Modificateur Displace: écarte la surface de l'os le long des normales (jeu d'ajustement). "
     "Valeur réglable dans la liste ci-dessous.",
     "Edit > Offset", "")(lambda c: add_modifier(c, 'DISPLACE', "Dégagement", strength=params.GUIDE_CLEARANCE,
                                                mid_level=0.0, direction='NORMAL'))
tool('ADD_THICKNESS', "Épaisseur (3 mm)", 'MOD_SOLIDIFY', 'SHAPE',
     "Modificateur Solidify: donne une épaisseur à la coque, vers l'extérieur.",
     "Edit > Extrude / Offset avec Connected", "")(lambda c: add_modifier(
         c, 'SOLIDIFY', "Épaisseur", thickness=params.GUIDE_THICKNESS, offset=1.0, use_even_offset=True))
tool('ADD_SMOOTH', "Lissage", 'MOD_SMOOTH', 'SHAPE',
     "Modificateur Smooth: adoucit toute la forme (à garder léger pour ne pas perdre l'ajustement).",
     "Deform > Smooth", "")(lambda c: add_modifier(c, 'SMOOTH', "Lissage", factor=0.5, iterations=5))
@tool('SMOOTH_OUTSIDE', "Lisser l'extérieur seulement", 'MOD_SMOOTH', 'SHAPE',
      "Lisse la face extérieure et le rebord créés par l'Épaisseur, sans toucher l'intrados (qui reste "
      "ajusté à l'os). Non destructif: force réglable dans la liste (Factor, Iterations).",
      "Deform > Smooth avec l'intrados en Freeze", "")
def smooth_outside(context):
    obj, error = editable_active(context)
    if error:
        return error
    solidify = next((m for m in obj.modifiers if m.type == 'SOLIDIFY'), None)
    if solidify is None:
        return "Ajoutez d'abord une Épaisseur (ou extrayez le guide)"
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if SHELL_GROUP not in obj.vertex_groups:
        obj.vertex_groups.new(name=SHELL_GROUP)   # empty: Solidify fills it
    solidify.shell_vertex_group = SHELL_GROUP
    name = "Lissage extérieur"
    smooth = obj.modifiers.get(name) or obj.modifiers.new(name, 'SMOOTH')
    smooth.vertex_group = SHELL_GROUP
    # With the thickness grown outwards (offset > 0), Solidify's "shell" group holds the vertices
    # left on the original surface, i.e. the intrados: smooth everything else. (Its "rim" group
    # would also catch the intrados border, so it isn't used.)
    smooth.invert_vertex_group = solidify.offset > 0
    smooth.factor, smooth.iterations = 0.5, 10
    after = list(obj.modifiers).index(solidify) + 1
    obj.modifiers.move(list(obj.modifiers).index(smooth), after)   # right after the thickness
    return None


tool('ADD_REMESH', "Remesh étanche (0,2 mm)", 'MOD_REMESH', 'SHAPE',
     "Modificateur Remesh (voxels): reconstruit un maillage fermé et régulier, prêt à imprimer. "
     "Plus la taille est petite, plus c'est fidèle (et lourd).",
     "Edit > Make Solid", "")(lambda c: add_modifier(c, 'REMESH', "Remesh", mode='VOXEL',
                                                    voxel_size=params.GUIDE_REMESH_VOXEL))


@tool('APPLY_MODIFIERS', "Appliquer les modificateurs", 'CHECKMARK', 'SHAPE',
      "Grave définitivement les modificateurs dans le maillage (dans l'ordre de la liste; les Boolean "
      "sont d'abord remis dans l'ordre: tubes, puis trous, puis trait de coupe). Les cylindres et "
      "boîtes utilisés sont ensuite cachés, pas supprimés. À faire à la fin, avant l'export. "
      "Annulable avec Ctrl+Z.",
      "Accept (dans un outil)", "Ctrl+A dans le menu des modificateurs")
def apply_modifiers(context):
    obj, error = editable_active(context)
    if error:
        return error
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    order_booleans(obj)
    tools = [m.object for m in obj.modifiers if m.type == 'BOOLEAN' and m.object is not None]
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for tool_obj in tools:                # their job is done: out of the way, kept (Ctrl+Z, reuse)
        tool_obj.hide_set(True)
    return None


FITS_ON = {'BONE', 'SEGMENT', 'TEETH', 'SCAN'}   # surfaces a printed part fits on: its intrados faces them


def intrados(obj, surfaces, depsgraph):
    """Boolean per vertex of obj: closer than GUIDE_PROTECT_DISTANCE to one of the surfaces (the side
    of a guide that fits the bone, of a cap that fits the teeth of the intra-oral scan)."""
    near = np.zeros(len(obj.data.vertices), bool)
    for surface in surfaces:
        tree = BVHTree.FromObject(surface, depsgraph, deform=False)   # mesh data only, in its local coordinates
        to_surface = surface.matrix_world.inverted() @ obj.matrix_world
        scale = surface.matrix_world.to_scale().x  # 1 for our meshes (raw file coordinates)
        for i, v in enumerate(obj.data.vertices):
            if not near[i]:
                _location, _normal, _index, distance = tree.find_nearest(to_surface @ v.co)
                near[i] = distance is not None and distance * scale < params.GUIDE_PROTECT_DISTANCE
    return near


def fitted_surfaces(context, obj):
    """The bone, segment, teeth and intra-oral scans near the object (its bounding box grown by the
    protect distance): the guide's intrados faces the bone, the cap's faces the scan."""
    def box(o):
        corners = np.array([tuple(o.matrix_world @ Vector(c)) for c in o.bound_box])
        return corners.min(0), corners.max(0)
    low, high = box(obj)
    low, high = low - params.GUIDE_PROTECT_DISTANCE, high + params.GUIDE_PROTECT_DISTANCE
    found = []
    for o in context.scene.objects:
        if o.type == 'MESH' and o.omfs.role in FITS_ON and o != obj:
            a, b = box(o)
            if (a <= high).all() and (b >= low).all():
                found.append(o)
    return found


def protect_intrados(context, obj):
    """Mask the intrados of an applied guide or cap (Object Mode). Returns the number of protected
    vertices, or an error message."""
    surfaces = fitted_surfaces(context, obj)
    if not surfaces:
        return f"Ni os ni scan près de {obj.name}"
    near = intrados(obj, surfaces, context.evaluated_depsgraph_get())
    if not near.any():
        return (f"Aucune face de {obj.name} à moins de {params.GUIDE_PROTECT_DISTANCE:g} mm de "
                + ", ".join(o.name for o in surfaces))
    attribute = obj.data.attributes.get(MASK) or obj.data.attributes.new(MASK, 'FLOAT', 'POINT')
    attribute.data.foreach_set("value", near.astype(np.float32))
    obj.data.update()
    return int(near.sum())


# --- 7. Sculpt ---------------------------------------------------------------------------

def sculpt_on(context):
    """The active free mesh in Sculpt Mode, its modifiers applied first (brushes work on the real
    mesh; the Boolean tools used are hidden, as in Appliquer)."""
    obj, error = editable_active(context)
    if error:
        return error
    if obj.modifiers:
        error = apply_modifiers(context)
        if error:
            return error
    if context.mode != 'SCULPT':
        bpy.ops.object.mode_set(mode='SCULPT')
    return None


@tool('POLISH_OUTSIDE', "Sculpter guide ou chape, intrados protégé", 'SCULPTMODE_HLT', 'SCULPT',
      "Applique les modificateurs (la coque devient un vrai maillage), protège l'intrados par un masque "
      "(tout ce qui est à moins de 1 mm de l'os, des dents ou du scan intra-oral: l'intrados du guide comme "
      "celui de la chape, en gris foncé), puis passe en Sculpture avec le pinceau "
      "Smooth. Polissez au glisser; les autres pinceaux de la barre de gauche (Clay Strips pour ajouter, "
      "Flatten pour aplatir...) sont aussi sans danger pour l'ajustement. F = taille, Maj+F = force.",
      "Sculpt > Robust Smooth avec l'intrados en Freeze", "Maj = lisser avec n'importe quel pinceau")
def polish_outside(context):
    obj, error = editable_active(context)
    if error:
        return error
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    error = sculpt_on(context)
    if error:
        return error
    bpy.ops.object.mode_set(mode='OBJECT')
    protected = protect_intrados(context, obj)
    if isinstance(protected, str):
        return protected
    bpy.ops.object.mode_set(mode='SCULPT')
    return activate_brush("Smooth")


@tool('SCULPT_OBJECT', "Sculpter l'objet actif (sans masque)", 'SCULPTMODE_HLT', 'SCULPT',
      "Passe l'objet actif (guide, chape, bras... jamais l'anatomie) en mode Sculpture avec le pinceau "
      "Smooth, après avoir appliqué ses modificateurs. Le masque existant est gardé: effacez-le ou "
      "protégez l'intrados avec les boutons ci-dessous.",
      "Sculpt (menu de gauche)", "Ctrl+Tab > Sculpt Mode")
def sculpt_object(context):
    return sculpt_on(context) or activate_brush("Smooth")


def zones_by_angle_of(mesh, angle):
    """Face set per face: faces joined across every edge where they fold less than `angle`
    (radians), so each sharp crease is a zone border. Written on the mesh (Object Mode)."""
    count = len(mesh.polygons)
    normals = np.empty(count * 3, np.float32)
    mesh.polygons.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)
    sizes = np.empty(count, np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    edges = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("edge_index", edges)
    faces = np.repeat(np.arange(count), sizes)
    order = np.argsort(edges, kind='stable')
    edges, faces = edges[order], faces[order]
    pair = np.flatnonzero(edges[:-1] == edges[1:])      # the two faces of each manifold edge
    a, b = faces[pair], faces[pair + 1]
    cos = np.einsum('ij,ij->i', normals[a], normals[b])
    joined = cos > np.cos(angle)
    a, b = a[joined], b[joined]
    labels = np.arange(count)
    while True:   # each face takes the smallest label among its joined neighbours, until stable
        low = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]                                  # jump along the chain of labels
        if np.array_equal(new, labels):
            break
        labels = new
    sets = (np.unique(labels, return_inverse=True)[1] + 1).astype(np.int32)
    attribute = mesh.attributes.get(FACE_SETS) or mesh.attributes.new(FACE_SETS, 'INT', 'FACE')
    attribute.data.foreach_set("value", sets)
    mesh.update()
    return int(sets.max())


@tool('ZONES_BY_ANGLE', "Découper en zones par angle aigu", 'MOD_EDGESPLIT', 'SCULPT',
      "Face Sets par arêtes vives: l'objet est découpé en zones (couleurs) partout où deux faces "
      "voisines font un pli plus grand que l'angle réglé ci-dessus. Avec un guide, l'intrados (bordé "
      "par le rebord) devient une zone à lui seul: cliquez-le ensuite pour le masquer. Ne déplace aucun "
      "sommet; remplace les zones existantes de l'objet.",
      "Select > Crease Angle (sélection arrêtée aux plis)", "Sculpt > Face Sets > Initialize")
def zones_by_angle(context):
    error = sculpt_on(context)
    if error:
        return error
    obj = context.active_object
    bpy.ops.object.mode_set(mode='OBJECT')
    zones_by_angle_of(obj.data, props(context).sharp_angle)
    bpy.ops.object.mode_set(mode='SCULPT')
    return None


@tool('MASK_ZONE_CLICK', "Masquer les zones cliquées", 'EYEDROPPER', 'SCULPT',
      "Cliquez les zones à masquer (gris foncé): chaque clic ajoute la zone au masque, Ctrl + clic la "
      "retire. Clic droit ou Échap pour finir. Les zones viennent du découpage par angle ou du pinceau "
      "Face Set Paint (section 1).",
      "Select > Select Facegroup, puis Freeze", "", operator="omfs.mask_zone_click")
def mask_zone_click(context):
    with viewport(context):
        bpy.ops.omfs.mask_zone_click('INVOKE_DEFAULT')
    return None


@tool('MASK_FROM_SELECTION', "Masque = sélection (mode Édition)", 'RESTRICT_SELECT_OFF', 'SCULPT',
      "Masque exactement les sommets sélectionnés en mode Édition (section 2: par angle, liés, lasso...) "
      "et rien d'autre, puis passe en Sculpture. Pour l'inverse, Inverser ensuite.",
      "Select, puis Freeze", "")
def mask_from_selection(context):
    obj, error = editable_active(context)
    if error:
        return error
    if context.mode != 'EDIT_MESH' or context.edit_object != obj:
        return "Sélectionnez d'abord en mode Édition (section 2) les sommets à masquer"
    bpy.ops.object.mode_set(mode='OBJECT')   # the selection is flushed to the mesh, the mask written on it
    selected = np.empty(len(obj.data.vertices), bool)
    obj.data.vertices.foreach_get("select", selected)
    if not selected.any():
        bpy.ops.object.mode_set(mode='EDIT')
        return "Rien n'est sélectionné"
    write_mask(obj.data, selected.astype(np.float32))
    return sculpt_on(context)


def brush_tool(brush):
    label, icon, tip, meshmixer = BRUSHES[brush]
    return tool('BRUSH_' + brush.split('/')[0].replace(' ', '_').upper(), label, icon, 'SCULPT',
                tip + " Les sommets masqués ne bougent pas. F = taille, Maj+F = force, Maj = lisser.",
                meshmixer, "Maj = lisser avec n'importe quel pinceau")(
        lambda context, brush=brush: sculpt_on(context) or activate_brush(brush))


for _brush in BRUSHES:
    brush_tool(_brush)

mask_tools('SCULPT_MASK_', 'SCULPT', sculpt_on)


@tool('POLISH_FILTER', "Lisser tout le non-masqué (1 passe)", 'MOD_SMOOTH', 'SCULPT',
      "Lisse d'un coup toute la surface non masquée (les sommets masqués ne bougent pas: sans masque, "
      "l'intrados bouge aussi). Une passe par clic; Ctrl+Z annule la dernière.",
      "Deform > Smooth sur tout, intrados en Freeze", "Sculpt > Filtre (Mesh Filter: Smooth)")
def polish_filter(context):
    error = sculpt_on(context)
    if error:
        return error
    with viewport(context):
        bpy.ops.sculpt.mesh_filter(type='SMOOTH', strength=0.5)
    return None


# --- 4. Cylinders and Boolean ------------------------------------------------------------

tool('ADD_TUBE', "Tube de manchon (sur une vis)", 'MESH_CYLINDER', 'BOOLEAN',
     f"Cliquez près d'une vis finale: un cylindre plein (trou + 2 × paroi de {params.SLEEVE_TUBE_WALL} mm, "
     f"long de {params.SLEEVE_TUBE_LENGTH} mm) apparaît sur l'axe de la vis; bougez la souris pour le glisser "
     "le long de l'axe (il ne le quitte jamais), cliquez pour le fixer. Uni au guide aussitôt. Ensuite: flèches "
     "bleues aux bouts = longueur, orange = diamètre, ou tapez les valeurs. Échap retire celui qui glisse.",
     "Meshmix > primitives (cylindre), aligné à la main", "", operator="omfs.add_sleeve_tube")(lambda c: None)
tool('ADD_BORE', "Trou de manchon (sur une vis)", 'MESH_CYLINDER', 'BOOLEAN',
     f"Même geste que le tube: un cylindre du diamètre du manchon + jeu ({params.SLEEVE_OUTER_DIAMETER} + "
     f"{params.SLEEVE_FIT_CLEARANCE} mm), {params.SLEEVE_BORE_OVERSHOOT:g} mm plus long que le tube à chaque "
     "bout, sur l'axe de la vis, soustrait du guide aussitôt. Flèches et diamètre réglables ensuite.",
     "Meshmix > primitives (cylindre), aligné à la main", "", operator="omfs.add_sleeve_bore")(lambda c: None)
tool('FIXATION_HOLE', "Trou de fixation du guide (clic)", 'DOT', 'BOOLEAN',
     f"Cliquez sur le guide où une vis le tiendra à l'os: un trou de {params.FIXATION_HOLE_DIAMETER} mm (vis "
     "Synthes 1,85 mm + jeu) le traverse selon la normale à la surface. Soustrait en tout dernier.",
     "Meshmix > primitives (cylindre) + Boolean Difference", "", operator="omfs.add_fixation_hole")(lambda c: None)


def transform(op_name):
    def run(context):
        if context.active_object is None:
            return "Sélectionnez l'objet à déplacer"
        if locking.is_locked(context.active_object):
            return "Cet objet est verrouillé (anatomie)"
        with viewport(context):
            getattr(bpy.ops.transform, op_name)('INVOKE_DEFAULT')
        return None
    return run


tool('MOVE', "Déplacer", 'ORIENTATION_VIEW', 'BOOLEAN',
     "Bougez la souris; tapez X, Y ou Z pour bloquer un axe de l'objet (deux fois = axe global), puis une valeur "
     "en mm et Entrée. Clic droit annule.",
     "Transform (flèches)", "G, puis X / Y / Z, puis un nombre")(transform("translate"))
tool('ROTATE', "Tourner", 'DRIVER_ROTATIONAL_DIFFERENCE', 'BOOLEAN',
     "Comme Déplacer: X / Y / Z pour l'axe, une valeur en degrés, Entrée.",
     "Transform (anneaux)", "R, puis X / Y / Z, puis un nombre")(transform("rotate"))


@tool('SNAP', "Aimanter à la surface", 'SNAP_ON', 'BOOLEAN',
      "Active/désactive l'aimantation: en déplaçant, l'objet colle à la surface sous la souris et "
      "s'aligne sur sa normale.",
      "Transform > Snap / Align", "Maj+Tab")
def snap(context):
    ts = context.scene.tool_settings
    ts.use_snap = not ts.use_snap
    if ts.use_snap:
        ts.snap_elements = {'FACE_PROJECT'}
        ts.use_snap_align_rotation = True
    return None


def boolean(operation, name):
    def run(context):
        obj, error = editable_active(context)
        if error:
            return error
        cutter = props(context).cutter
        if cutter is None or cutter == obj:
            return "Choisissez l'objet outil (le cylindre) dans « Avec »"
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        mod = obj.modifiers.new(f"{name} {cutter.name}", 'BOOLEAN')
        mod.operation, mod.object, mod.solver = operation, cutter, SOLVER
        cutter.display_type = 'WIRE'      # see through the tool, it no longer counts as a solid
        order_booleans(obj)
        return None
    return run


def boolean_rank(mod):
    """Order of the Booleans in the stack: tubes added first, then holes dug (through the tubes too),
    then the cut edge (nothing sticks out past the osteotomy line), the guide's fixation holes last."""
    if mod.name == CUT_EDGE_MODIFIER:
        return 3
    if cylinders.is_cylinder(mod.object) and mod.object.omfs_cylinder.kind == 'FIXATION':
        return 4
    return 1 if mod.operation == 'UNION' else 2


def order_booleans(obj):
    """Sort the Boolean modifiers by boolean_rank in the slots they occupy; the others stay put."""
    slots = [i for i, m in enumerate(obj.modifiers) if m.type == 'BOOLEAN']
    wanted = sorted((obj.modifiers[i] for i in slots), key=boolean_rank)
    for slot, mod in zip(slots, [m.name for m in wanted]):
        obj.modifiers.move(obj.modifiers.find(mod), slot)


tool('BOOL_UNION', "Unir au guide", 'SELECT_EXTEND', 'BOOLEAN',
     "Modificateur Boolean Union: le cylindre « Avec » est fusionné au guide actif.",
     "Edit > Boolean Union", "")(boolean('UNION', "Union"))
tool('BOOL_DIFFERENCE', "Soustraire du guide", 'SELECT_SUBTRACT', 'BOOLEAN',
     "Modificateur Boolean Difference: le cylindre « Avec » est creusé dans le guide actif.",
     "Edit > Boolean Difference", "")(boolean('DIFFERENCE', "Trou"))
tool('MEGA_BOOLEAN', "Méga Boolean (guide puis chape)", 'MOD_BOOLEAN', 'BOOLEAN',
     "Applique dans l'ordre tous les Boolean du guide: unions (bossages, ponts, tubes), différences "
     "(logements, trous de manchon), trait de coupe, puis les trous de fixation. Un tube ou un trou qui "
     "croise une pin de la chape (mâle ou bras) est aussi soustrait de la chape, puis la chape est appliquée "
     "(unions, puis différences). Les outils sont cachés, pas supprimés. Annulable avec Ctrl+Z.",
     "Edit > Boolean, un par un dans le bon ordre", "", operator="omfs.mega_boolean")(lambda c: None)


# --- 5. Osteotomy line ---------------------------------------------------------------------

CUT_EDGE_MODIFIER = "Trait de coupe"


def cut_edge(keep_above):
    def run(context):
        obj, error = editable_active(context)
        if error:
            return error
        cut = osteotomy.find_cut(context.scene)
        if cut is None:
            return "Ajoutez d'abord le plan d'ostéotomie (panneau Osteotomy)"
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        name = f"{CUT_EDGE_MODIFIER} ({obj.name})"
        verts, faces = osteotomy.half_space_mesh_data(cut.omfs_cut, below=keep_above)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        box = context.scene.objects.get(name)
        if box is None:
            box = bpy.data.objects.new(name, mesh)
            workshop_collection(context.scene).objects.link(box)
            box.omfs.role = 'MARKER'      # helper: landmark clicks go through it
            box.display_type = 'WIRE'
        else:
            old_mesh, box.data = box.data, mesh
            bpy.data.meshes.remove(old_mesh)
        box.matrix_world = cut.matrix_world
        box.hide_set(True)                # still used by the Boolean while hidden
        mod = obj.modifiers.get(CUT_EDGE_MODIFIER) or obj.modifiers.new(CUT_EDGE_MODIFIER, 'BOOLEAN')
        mod.operation, mod.object, mod.solver = 'DIFFERENCE', box, SOLVER
        order_booleans(obj)
        return None
    return run


tool('CUT_KEEP_ABOVE', "Garder au-dessus du trait", 'TRIA_UP', 'OSTEO',
     "Retire tout ce qui est sous le plan d'ostéotomie: le bord inférieur du guide suit le trait de coupe.",
     "Edit > Plane Cut (garder le dessus)", "")(cut_edge(keep_above=True))
tool('CUT_KEEP_BELOW', "Garder en dessous du trait", 'TRIA_DOWN', 'OSTEO',
     "Retire tout ce qui est au-dessus du plan d'ostéotomie: le bord supérieur du guide suit le trait "
     "de coupe (guide sur le segment).",
     "Edit > Plane Cut (garder le dessous)", "")(cut_edge(keep_above=False))


# --- 6. Tubes and pins ----------------------------------------------------------------------

tool('PIN_FEMALES', "Ajouter les femelles (clic, glisser, clic)", 'MESH_CYLINDER', 'PINS',
     "Cliquez sur le guide près de l'endroit voulu, bougez la souris pour placer la femelle, cliquez pour "
     "confirmer; recommencez (souvent 3), clic droit pour finir. Elle glisse dans le plan d'ostéotomie: "
     "le dessus du bossage reste exactement dans le plan. Éloignée du guide, un pont l'y relie (toute la moitié du bossage côté guide, dessus dans le plan: plus de surface de coupe).",
     "Aucun (le plus proche: Meshmix > pièce à insérer)", "", operator="omfs.add_pin_female")(lambda c: None)


@tool('PIN_MALES', "Ajouter les pins mâles", 'SNAP_PEEL_OBJECT', 'PINS',
      "Une pin conique dans chaque femelle, avec le jeu; son épaulement repose sur le plan "
      "d'ostéotomie. Elle est unie à la chape choisie ci-dessus.",
      "Aucun", "")
def pin_males(context):
    cap = context.scene.omfs_pins.cap
    if cap is None:
        return "Choisissez la chape ci-dessus"
    if not pins.sites(context.scene):
        return "Ajoutez d'abord les femelles"
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    pins.add_males(context.scene, cap)
    return None


tool('PIN_ARMS', "Bras depuis la chape (un clic par pin)", 'CURVE_PATH', 'PINS',
     "Pour chaque pin à tour de rôle, cliquez sur la chape où part son bras: il sort à 90° de la "
     "surface et arrive sur le dos de la pin dans son axe, par une courbe douce (Bézier).",
     "Aucun (le plus proche: Meshmix > tube)", "", operator="omfs.add_pin_arm")(lambda c: None)


# --- 7. Check ----------------------------------------------------------------------------

@tool('NON_MANIFOLD', "Montrer les défauts", 'ERROR', 'CHECK',
      "Passe en mode Édition et sélectionne les arêtes non étanches (bords libres, faces en trop). "
      "Rien de sélectionné = maillage fermé.",
      "Analysis > Inspector", "Maj+Ctrl+Alt+M")
def non_manifold(context):
    error = need_mode(context, 'EDIT_MESH')
    if error:
        return error
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    return None


@tool('STATS', "Statistiques", 'INFO', 'CHECK',
      "Affiche en haut à gauche le nombre de sommets et de triangles.",
      "Analysis > Stats", "")
def stats(context):
    overlay = context.space_data.overlay if context.space_data and context.space_data.type == 'VIEW_3D' else None
    if overlay is None:
        return "Pas de vue 3D"
    overlay.show_stats = not overlay.show_stats
    return None


@tool('MARK_GUIDE', "Marquer comme guide", 'MOD_BUILD', 'CHECK',
      "Donne le rôle Guide (couleur orange) à l'objet actif, pour l'export.",
      "Aucun", "")
def mark_guide(context):
    obj, error = editable_active(context)
    if error:
        return error
    obj.omfs.role = 'GUIDE'
    return None


# --- Recipe ------------------------------------------------------------------------------

RECIPE = (
    ("Guide de coupe et chape (génioplastie)", (
        "Osteotomy: simuler le trait de coupe (plan d'ostéotomie).",
        "[1] Choisir l'os dans « Peindre sur », peindre la zone du guide sur la corticale vestibulaire, de "
        "part et d'autre du trait (l'os CBCT n'est jamais modifié), puis Extraire la zone en coque.",
        "[3] Régler Dégagement et Épaisseur; [5] Garder au-dessus ou en dessous du trait (forme et étendue).",
        "Cacher le guide; Mouvement du segment: simuler la génioplastie.",
        "[4] Positionnement de vis bicorticale: poser les vis finales (segment en position planifiée).",
        "[6] Retour au guide: ajouter les femelles des pins (un peu à l'écart des vis de préférence).",
        "Mouvement du segment: Appliquer le même mouvement au guide de coupe (pins comprises).",
        "[6] Bras depuis la chape: ils rejoignent les pins là où elles sont maintenant.",
        "[4] Tubes et trous de manchon sur les vis: clic près de la vis, glisser le long de l'axe, clic; "
        "ajuster longueur et diamètre avec les flèches.",
        "[4] Trous de fixation du guide (clic sur le guide).",
        "[4] Méga Boolean: unions, différences (aussi sur les pins de la chape s'ils se croisent), trait de "
        "coupe, trous de fixation.",
        "[7] Sculpter guide ou chape, intrados protégé; [8] Montrer les défauts, Marquer comme guide.",
    )),
)


# --- Properties and operator -------------------------------------------------------------

def _on_guide(context):
    guide = props(context).guide
    if guide is not None and context.mode == 'OBJECT':
        guide.hide_set(False)
        make_active(context, guide)


def _is_source(_props, obj):
    return obj.type == 'MESH' and obj.omfs.role in {'BONE', 'SEGMENT', 'TEETH', 'SCAN'}


class OMFS_GuideProperties(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(name="Peindre sur", type=bpy.types.Object, poll=_is_source,
                                      description="Os, segment ou dents sur lequel le guide s'appuie (par "
                                                  "défaut l'os): on y peint les zones, l'os CBCT n'est jamais modifié")
    guide: bpy.props.PointerProperty(name="Guide", type=bpy.types.Object, poll=lambda _p, o: is_guide(o),
                                     update=lambda self, context: _on_guide(context),
                                     description="Guide que les outils modifient (le dernier extrait, par défaut)")
    work_copy: bpy.props.PointerProperty(name="Surface peinte", type=bpy.types.Object)
    zone: bpy.props.IntProperty(name="Zone", min=0, description="Face set chosen by clic (0 = the last painted)")
    sharp_angle: bpy.props.FloatProperty(
        name="Angle aigu", subtype='ANGLE', min=radians(1), max=radians(180),
        default=radians(params.SCULPT_SHARP_ANGLE),
        description="Découper en zones: deux faces voisines qui font plus que cet angle sont dans des "
                    "zones différentes (petit angle = plus de zones, bords plus fins).\n" + SCULPT_KEYS)
    cutter: bpy.props.PointerProperty(name="Avec", type=bpy.types.Object,
                                      poll=lambda _p, o: o.type == 'MESH' and not locking.is_locked(o),
                                      description="Objet outil du Boolean (ex. un cylindre)")


def tool_description(t):
    lines = [t.tip]
    if t.shortcut:
        lines.append(f"Raccourci Blender: {t.shortcut}")
    lines.append(f"Meshmixer: {t.meshmixer}")
    if t.section == 'SCULPT':
        lines.append(SCULPT_KEYS)
    return "\n".join(lines)


class OMFS_OT_select_linked_flat(bpy.types.Operator):
    """Extend the selection across faces meeting at less than an angle"""
    bl_idname = "omfs.select_linked_flat"
    bl_label = "Sélection liée par angle"
    bl_options = {'REGISTER', 'UNDO'}

    sharpness: bpy.props.FloatProperty(name="Angle maximal", subtype='ANGLE', min=radians(1), max=radians(180),
                                       default=radians(params.SELECT_FLAT_ANGLE))

    @classmethod
    def description(cls, context, properties):
        return tool_description(TOOLS['SEL_LINKED_FLAT'])

    def execute(self, context):
        error = need_mode(context, 'EDIT_MESH')
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.faces_select_linked_flat(sharpness=self.sharpness)
        return {'FINISHED'}


class OMFS_OT_pick_zone(bpy.types.Operator):
    """Click a painted zone (face set) to choose it (Esc or right-click to cancel)"""
    bl_idname = "omfs.pick_zone"
    bl_label = "Choisir la zone"

    @classmethod
    def description(cls, context, properties):
        return tool_description(TOOLS['ZONE_PICK'])

    def invoke(self, context, event):
        copy = current_copy(context)
        if copy is None:
            self.report({'WARNING'}, "Peignez d'abord une zone")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        self.area.header_text_set("Cliquez une zone peinte   |   Échap / clic droit: annuler")
        context.window.cursor_modal_set('EYEDROPPER')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.finish(context)
            return {'CANCELLED'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        copy = current_copy(context)
        face = landmarks.object_face_hit(context, *under_mouse, copy)
        sets = face_sets(copy.data)
        if face is None or sets is None:
            self.report({'WARNING'}, "Cliquez sur une zone peinte")
            return {'RUNNING_MODAL'}
        props(context).zone = int(sets[face])
        self.finish(context)
        for area in context.screen.areas:   # the sidebar shows the chosen zone
            area.tag_redraw()
        return {'FINISHED'}


class OMFS_OT_mask_zone_click(bpy.types.Operator):
    """Click zones (face sets) of the active object to mask them (Ctrl: unmask; Esc or right-click to finish)"""
    bl_idname = "omfs.mask_zone_click"
    bl_label = "Masquer les zones cliquées"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def description(cls, context, properties):
        return tool_description(TOOLS['MASK_ZONE_CLICK'])

    def invoke(self, context, event):
        error = sculpt_on(context)
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        if face_sets(context.active_object.data) is None:
            self.report({'WARNING'}, "Aucune zone sur cet objet: « Découper en zones par angle aigu » d'abord")
            return {'CANCELLED'}
        self.area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if self.area is None:
            return {'CANCELLED'}
        self.area.header_text_set("Clic: masquer la zone   |   Ctrl + clic: démasquer   |   Échap / clic droit: finir")
        context.window.cursor_modal_set('EYEDROPPER')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        self.area.header_text_set(None)
        context.window.cursor_modal_restore()

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.finish(context)
            return {'FINISHED'}
        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}   # orbit, pan and zoom keep working
        under_mouse = landmarks.viewport_region_under_mouse(self.area, event)
        if under_mouse is None:
            return {'PASS_THROUGH'}
        obj = context.active_object
        face = landmarks.object_face_hit(context, *under_mouse, obj)
        sets = face_sets(obj.data)
        if face is None or sets is None:
            return {'RUNNING_MODAL'}
        bpy.ops.object.mode_set(mode='OBJECT')
        mask_faces(obj.data, sets == sets[face], 0.0 if event.ctrl else 1.0)
        bpy.ops.object.mode_set(mode='SCULPT')
        return {'RUNNING_MODAL'}


class OMFS_OT_guide_tool(bpy.types.Operator):
    """Native Blender tool for guide design"""
    bl_idname = "omfs.guide_tool"
    bl_label = "Guide Tool"
    bl_options = {'REGISTER', 'UNDO'}

    tool: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        t = TOOLS.get(properties.tool)
        return tool_description(t) if t else cls.__doc__

    def execute(self, context):
        t = TOOLS.get(self.tool)
        if t is None:
            return {'CANCELLED'}
        error = t.run(context)
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        return {'FINISHED'}


classes = (OMFS_GuideProperties, OMFS_OT_select_linked_flat, OMFS_OT_pick_zone, OMFS_OT_mask_zone_click,
           OMFS_OT_guide_tool)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_guide = bpy.props.PointerProperty(type=OMFS_GuideProperties)


def unregister():
    del bpy.types.Scene.omfs_guide
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
