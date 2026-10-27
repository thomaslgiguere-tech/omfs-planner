"""Guide design workshop ("Confection de guide"): Blender's native tools, one click away.

A learning bench, not an automatic guide builder: each button runs a native Blender tool
(sculpt mask, Edit Mode selection, modifiers, Boolean...) and its tooltip explains what it does,
its keyboard shortcut and the closest Meshmixer tool. The texts are in French, for the surgeon.

Anatomy is never edited: the work happens on a copy of the chosen bone or teeth.
"""

from collections import namedtuple

import bpy

from . import locking, params

COLLECTION = "Guide design"
MASK_BRUSH = "brushes/essentials_brushes-mesh_sculpt.blend/Brush/Mask"

Tool = namedtuple("Tool", "id label icon section tip meshmixer shortcut run")
TOOLS = {}

# Sections, in panel order: (id, title, short explanation shown above the buttons)
SECTIONS = (
    ('MASK', "1. Masque (surface de contact)",
     "Peindre en mode Sculpture la zone d'os que le guide doit toucher, puis l'extraire en coque. "
     "Le masque ne déplace aucun sommet: c'est une couleur posée sur la copie de travail."),
    ('EDIT', "2. Mode Édition (sélection, extrusion)",
     "Travailler directement les sommets, arêtes et faces de l'objet actif (le guide, pas l'os). "
     "Tab entre et sort du mode Édition."),
    ('SHAPE', "3. Épaisseur et dégagement (modificateurs)",
     "Un modificateur change la forme affichée sans toucher au maillage, tant qu'il n'est pas "
     "appliqué. On peut donc régler les valeurs à volonté, puis appliquer à la fin."),
    ('BOOLEAN', "4. Cylindres et Boolean (manchons)",
     "Placer le curseur 3D sur l'os, y ajouter un cylindre aux dimensions du manchon, l'orienter, "
     "puis l'unir au guide (tube) ou le soustraire (trou)."),
    ('CHECK', "5. Vérifier et finir",
     "Avant d'imprimer: le guide doit être fermé (étanche), sans arête libre."),
)


def tool(tool_id, label, icon, section, tip, meshmixer, shortcut=""):
    """Register a workshop tool. The function returns None, or an error message for the user."""
    def register_tool(run):
        TOOLS[tool_id] = Tool(tool_id, label, icon, section, tip, meshmixer, shortcut, run)
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


def editable_active(context):
    """The active object if it may be edited (a mesh that isn't locked anatomy), else an error."""
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return None, "Sélectionnez d'abord un maillage (le guide ou la copie de travail)"
    if locking.is_locked(obj):
        return None, f"{obj.name} est verrouillé (anatomie): travaillez sur une copie de travail"
    return obj, None


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


def add_cylinder(context, diameter, name):
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    cursor = context.scene.cursor
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=diameter / 2, depth=params.SLEEVE_TUBE_LENGTH,
                                        location=cursor.location, rotation=cursor.rotation_euler)
    obj = context.active_object
    obj.name = obj.data.name = name
    for coll in obj.users_collection:
        coll.objects.unlink(obj)
    workshop_collection(context.scene).objects.link(obj)
    return None


# --- 0. Start ----------------------------------------------------------------------------

@tool('WORK_COPY', "Créer une copie de travail", 'DUPLICATE', 'START',
      "Copie l'os (ou les dents) choisi dans un nouvel objet modifiable et masque l'original. "
      "Tout le travail (masque, extraction) se fait sur cette copie: l'anatomie reste intacte.",
      "Edit > Duplicate", "Maj+D (en mode Objet)")
def work_copy(context):
    source = props(context).source
    if source is None:
        return "Choisissez d'abord l'objet de départ"
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    copy = bpy.data.objects.new(f"{source.name} (copie de travail)", source.data.copy())
    copy.data.name = copy.name
    workshop_collection(context.scene).objects.link(copy)
    copy.matrix_world = source.matrix_world
    copy.omfs.role = 'OTHER'
    copy.omfs.made_by = "workshop"
    source.hide_set(True)
    make_active(context, copy)
    props(context).work_copy = copy
    return None


@tool('REMOVE_WORK_COPY', "Retirer la copie de travail", 'TRASH', 'START',
      "Supprime la copie de travail (et son masque) et réaffiche l'original. Le guide extrait, lui, "
      "est conservé. Annulable avec Ctrl+Z.",
      "Supprimer un objet dans l'Object Browser", "X (en mode Objet)")
def remove_work_copy(context):
    copy = props(context).work_copy
    if copy is None:
        return "Aucune copie de travail"
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    source = props(context).source
    mesh = copy.data
    bpy.data.objects.remove(copy)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    if source is not None:
        source.hide_set(False)
    return None


@tool('OBJECT_MODE', "Revenir en mode Objet", 'OBJECT_DATAMODE', 'START',
      "Quitte le mode Sculpture, Édition ou Peinture. En mode Objet on manipule des objets entiers "
      "(et l'outil Navigate fait tourner la vue au glisser).",
      "Quitter un outil (Accept / Cancel)", "Tab (depuis Édition), Ctrl+Tab (menu des modes)")
def object_mode(context):
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    return None


# --- 1. Mask -----------------------------------------------------------------------------

def need_sculpt(context):
    copy = props(context).work_copy
    if copy is None:
        return "Créez d'abord une copie de travail"
    if context.active_object != copy:
        make_active(context, copy)
    if context.mode != 'SCULPT':
        bpy.ops.object.mode_set(mode='SCULPT')
    return None


@tool('MASK_PAINT', "Peindre le masque", 'BRUSH_DATA', 'MASK',
      "Pinceau Mask: le glisser peint la zone de contact en gris foncé. Ctrl + glisser efface. "
      "F puis bouger la souris change la taille du pinceau. Ne déforme rien.",
      "Select (pinceau de sélection)", "Ctrl+glisser = effacer, F = taille")
def mask_paint(context):
    error = need_sculpt(context)
    if error:
        return error
    try:
        bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS', relative_asset_identifier=MASK_BRUSH)
    except RuntimeError:
        return "Pinceau Mask introuvable: choisissez-le dans la barre d'outils de gauche"
    return None


@tool('MASK_LASSO', "Masque au lasso", 'MOD_MASK', 'MASK',
      "Dessinez un contour: tout ce qui est à l'intérieur est masqué (y compris l'arrière). "
      "Ctrl + lasso retire du masque.",
      "Select > Lasso", "")
def mask_lasso(context):
    error = need_sculpt(context)
    if error:
        return error
    bpy.ops.wm.tool_set_by_id(name="builtin.lasso_mask")
    return None


def mask_filter(kind):
    def run(context):
        error = need_sculpt(context)
        if error:
            return error
        bpy.ops.sculpt.mask_filter(filter_type=kind, auto_iteration_count=True)
        return None
    return run


tool('MASK_GROW', "Agrandir", 'ADD', 'MASK', "Élargit le bord du masque d'une rangée de sommets.",
     "Modify > Expand Ring", "")(mask_filter('GROW'))
tool('MASK_SHRINK', "Réduire", 'REMOVE', 'MASK', "Rétrécit le bord du masque d'une rangée de sommets.",
     "Modify > Contract Ring", "")(mask_filter('SHRINK'))
tool('MASK_SMOOTH', "Lisser le bord", 'MOD_SMOOTH', 'MASK',
     "Adoucit le contour du masque: le bord du guide extrait sera plus régulier.",
     "Modify > Smooth Boundary", "Maj + glisser avec le pinceau")(mask_filter('SMOOTH'))


@tool('MASK_INVERT', "Inverser", 'ARROW_LEFTRIGHT', 'MASK', "Le masqué devient libre et inversement.",
      "Modify > Invert", "Ctrl+I (en Sculpture)")
def mask_invert(context):
    error = need_sculpt(context)
    if error:
        return error
    bpy.ops.paint.mask_flood_fill(mode='INVERT')
    return None


@tool('MASK_CLEAR', "Effacer le masque", 'X', 'MASK', "Retire tout le masque pour recommencer.",
      "Select > Clear selection", "Alt+M (en Sculpture)")
def mask_clear(context):
    error = need_sculpt(context)
    if error:
        return error
    bpy.ops.paint.mask_flood_fill(mode='VALUE', value=0.0)
    return None


@tool('MASK_EXTRACT', "Extraire en coque (guide)", 'MOD_SOLIDIFY', 'MASK',
      "Crée un NOUVEL objet à partir de la zone masquée, collé à l'os, avec un dégagement et une "
      "épaisseur réglables (modificateurs Displace + Solidify, section 3). La copie de travail ne change pas.",
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
    solidify = next((m for m in guide.modifiers if m.type == 'SOLIDIFY'), None)
    if solidify is not None:
        solidify.name = "Épaisseur"
        solidify.thickness = params.GUIDE_THICKNESS
        solidify.offset = 1.0              # grow outwards, away from the bone
        solidify.use_even_offset = True
    clearance = guide.modifiers.new("Dégagement", 'DISPLACE')
    clearance.strength = params.GUIDE_CLEARANCE
    clearance.mid_level = 0.0
    clearance.direction = 'NORMAL'
    guide.modifiers.move(len(guide.modifiers) - 1, 0)   # gap first, then thickness


# --- 2. Edit Mode ------------------------------------------------------------------------

@tool('EDIT_MODE', "Entrer en mode Édition", 'EDITMODE_HLT', 'EDIT',
      "Montre les sommets, arêtes et faces de l'objet actif pour les sélectionner et les modifier.",
      "Select tool (sélection de triangles)", "Tab")
def edit_mode(context):
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
tool('ADD_REMESH', "Remesh étanche (0,2 mm)", 'MOD_REMESH', 'SHAPE',
     "Modificateur Remesh (voxels): reconstruit un maillage fermé et régulier, prêt à imprimer. "
     "Plus la taille est petite, plus c'est fidèle (et lourd).",
     "Edit > Make Solid", "")(lambda c: add_modifier(c, 'REMESH', "Remesh", mode='VOXEL',
                                                    voxel_size=params.GUIDE_REMESH_VOXEL))


@tool('APPLY_MODIFIERS', "Appliquer les modificateurs", 'CHECKMARK', 'SHAPE',
      "Grave définitivement les modificateurs dans le maillage (dans l'ordre de la liste). "
      "À faire à la fin, avant l'export. Annulable avec Ctrl+Z.",
      "Accept (dans un outil)", "Ctrl+A dans le menu des modificateurs")
def apply_modifiers(context):
    obj, error = editable_active(context)
    if error:
        return error
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return None


# --- 4. Cylinders and Boolean ------------------------------------------------------------

@tool('CURSOR_TOOL', "Placer le curseur 3D", 'PIVOT_CURSOR', 'BOOLEAN',
      "Active l'outil Curseur: cliquez sur l'os pour y poser le curseur 3D (cercle rouge et blanc), "
      "orienté selon la surface. Les cylindres seront créés là, perpendiculaires à l'os.",
      "Aucun (le plus proche: Drop Pivot)", "Maj + clic droit (sans changer d'outil)")
def cursor_tool(context):
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    tool_props = context.workspace.tools.from_space_view3d_mode('OBJECT').operator_properties("view3d.cursor3d")
    tool_props.use_depth = True          # on the surface under the mouse
    tool_props.orientation = 'GEOM'      # Z axis along the surface normal
    return None


tool('ADD_BORE', "Cylindre: trou du manchon", 'MESH_CYLINDER', 'BOOLEAN',
     f"Ajoute au curseur un cylindre du diamètre du manchon + jeu "
     f"({params.SLEEVE_OUTER_DIAMETER} + {params.SLEEVE_FIT_CLEARANCE} mm), long de "
     f"{params.SLEEVE_TUBE_LENGTH} mm. À soustraire du guide.",
     "Meshmix > primitives (cylindre)", "Maj+A > Mesh > Cylinder")(
         lambda c: add_cylinder(c, sleeve_bore_diameter(), "Trou manchon"))
tool('ADD_TUBE', "Cylindre: tube autour du manchon", 'MESH_CYLINDER', 'BOOLEAN',
     f"Ajoute au curseur un cylindre plein (trou + 2 × paroi de {params.SLEEVE_TUBE_WALL} mm). "
     "À unir au guide, puis soustraire le trou.",
     "Meshmix > primitives (cylindre)", "Maj+A > Mesh > Cylinder")(
         lambda c: add_cylinder(c, sleeve_bore_diameter() + 2 * params.SLEEVE_TUBE_WALL, "Tube manchon"))


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
     "Bougez la souris; tapez X, Y ou Z pour bloquer un axe (deux fois = axe local), puis une valeur "
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
        mod.operation, mod.object, mod.solver = operation, cutter, 'EXACT'
        cutter.display_type = 'WIRE'      # see through the tool, it no longer counts as a solid
        return None
    return run


tool('BOOL_UNION', "Unir au guide", 'SELECT_EXTEND', 'BOOLEAN',
     "Modificateur Boolean Union: le cylindre « Avec » est fusionné au guide actif.",
     "Edit > Boolean Union", "")(boolean('UNION', "Union"))
tool('BOOL_DIFFERENCE', "Soustraire du guide", 'SELECT_SUBTRACT', 'BOOLEAN',
     "Modificateur Boolean Difference: le cylindre « Avec » est creusé dans le guide actif.",
     "Edit > Boolean Difference", "")(boolean('DIFFERENCE', "Trou"))


# --- 5. Check ----------------------------------------------------------------------------

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
    ("Guide de coupe (sur l'os)", (
        "Départ: la mandibule. Créer une copie de travail.",
        "[1] Peindre le masque sur la corticale vestibulaire du menton, de part et d'autre du "
        "trait de coupe. Lisser le bord, Agrandir ou Réduire au besoin.",
        "[1] Extraire en coque: un objet « Guide » apparaît avec Dégagement (0,10 mm) et Épaisseur (3 mm).",
        "[3] Régler les valeurs dans la liste des modificateurs; un Lissage léger si besoin.",
        "[4] Pour chaque vis: Placer le curseur 3D sur l'os, Cylindre tube, Déplacer et Tourner "
        "selon l'axe de la vis, choisir le tube dans « Avec », Unir au guide.",
        "[4] Cylindre trou au même endroit, le choisir dans « Avec », Soustraire du guide.",
        "[3] Remesh étanche, puis Appliquer les modificateurs.",
        "[5] Montrer les défauts (rien ne doit être sélectionné), Marquer comme guide.",
    )),
    ("Guide de repositionnement (sur les dents)", (
        "Départ: le scan intra-oral recalé. Créer une copie de travail.",
        "[1] Masquer les faces occlusales et vestibulaires des dents, pas la gencive.",
        "[1] Extraire en coque, puis [3] régler Dégagement et Épaisseur.",
        "[2] En mode Édition, sélectionner une zone et l'extruder vers le guide de coupe (bras de connexion).",
        "[4] Boolean avec le guide de coupe placé en position finale, pour l'emboîtement.",
        "[3] Remesh étanche, Appliquer; [5] Vérifier, Marquer comme guide.",
    )),
)


# --- Properties and operator -------------------------------------------------------------

def _is_source(_props, obj):
    return obj.type == 'MESH' and obj.omfs.role in {'BONE', 'SEGMENT', 'TEETH', 'SCAN'}


class OMFS_GuideProperties(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(name="Départ", type=bpy.types.Object, poll=_is_source,
                                      description="Os, segment ou dents sur lequel le guide s'appuie")
    work_copy: bpy.props.PointerProperty(name="Copie de travail", type=bpy.types.Object)
    cutter: bpy.props.PointerProperty(name="Avec", type=bpy.types.Object,
                                      poll=lambda _p, o: o.type == 'MESH' and not locking.is_locked(o),
                                      description="Objet outil du Boolean (ex. un cylindre)")


class OMFS_OT_guide_tool(bpy.types.Operator):
    """Native Blender tool for guide design"""
    bl_idname = "omfs.guide_tool"
    bl_label = "Guide Tool"
    bl_options = {'REGISTER', 'UNDO'}

    tool: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        t = TOOLS.get(properties.tool)
        if t is None:
            return cls.__doc__
        lines = [t.tip]
        if t.shortcut:
            lines.append(f"Raccourci Blender: {t.shortcut}")
        lines.append(f"Meshmixer: {t.meshmixer}")
        return "\n".join(lines)

    def execute(self, context):
        t = TOOLS.get(self.tool)
        if t is None:
            return {'CANCELLED'}
        error = t.run(context)
        if error:
            self.report({'WARNING'}, error)
            return {'CANCELLED'}
        return {'FINISHED'}


classes = (OMFS_GuideProperties, OMFS_OT_guide_tool)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omfs_guide = bpy.props.PointerProperty(type=OMFS_GuideProperties)


def unregister():
    del bpy.types.Scene.omfs_guide
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
