"""Sidebar tab "OMFS" in the 3D viewport (press N to show the sidebar)."""

import textwrap

import bpy
from mathutils import Vector

from . import analysis, cylinders, distance_map, pins, guide_workshop, landmarks, locking, movement, occlusion, occlusion_seating, osteotomy, params, reference_frame, registration, roles, section, section_contours, snapshot, splint, text_labels, view


def draw_text(layout, context, text):
    """Labels don't wrap in Blender: cut the text to the sidebar width."""
    chars = max(20, int(context.region.width / (6.5 * context.preferences.system.ui_scale)))
    col = layout.column(align=True)
    col.scale_y = 0.8
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, chars) or [""]:
            col.label(text=line)


def tool_button(layout, tool_id, text=None):
    t = guide_workshop.TOOLS[tool_id]
    if t.operator:   # own operator, so its settings show in Adjust Last Operation (F9)
        layout.operator(t.operator, text=t.label if text is None else text, icon=t.icon)
    else:
        layout.operator("omfs.guide_tool", text=t.label if text is None else text, icon=t.icon).tool = tool_id


class OMFS_PT_case(bpy.types.Panel):
    bl_label = "Case"
    bl_idname = "OMFS_PT_case"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        layout.operator("omfs.import_case", icon='IMPORT')
        layout.operator("omfs.setup_scene", icon='SETTINGS')

        meshes = [obj for obj in context.scene.objects
                  if obj.type == 'MESH' and obj.omfs.role not in roles.NOT_SURFACES]
        if not meshes:
            return
        box = layout.box()
        for obj in meshes:
            row = box.row(align=True)
            row.label(text=obj.name)
            row.prop(obj.omfs, "role", text="")
            locked = locking.is_locked(obj)
            op = row.operator("omfs.toggle_lock", text="", icon='LOCKED' if locked else 'UNLOCKED',
                              depress=locked)
            op.object_name = obj.name


class OMFS_PT_view(bpy.types.Panel):
    bl_label = "View"
    bl_idname = "OMFS_PT_view"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        grid = layout.grid_flow(columns=2, align=True)
        for view_id, label, _axis in view.VIEWS:
            grid.operator("omfs.view", text=label).view = view_id
        layout.operator("view3d.view_all", text="Frame All", icon='ZOOM_ALL')
        col = layout.column(align=True)
        col.label(text=f"{view.CENTRE_KEY}: centre on point under mouse", icon='INFO')
        col.label(text="Navigate tool: drag orbit, Shift + drag pan", icon='BLANK1')
        layout.operator("wm.tool_set_by_id", text="Navigate Tool", icon='ORIENTATION_VIEW').name = "omfs.navigate"
        if not view.navigation_preferences_ok(context.preferences):
            layout.operator("omfs.navigation_preferences", icon='PREFERENCES')


class OMFS_PT_section(bpy.types.Panel):
    bl_label = "Section & Analysis"
    bl_idname = "OMFS_PT_section"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.omfs_section, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.omfs_section
        box = layout.box()
        box.active = props.enabled
        box.label(text="Section plane", icon='MOD_MIRROR')
        box.row().prop(props, "orientation", expand=True)
        if props.orientation == 'CUSTOM':
            box.prop(props, "custom", text="")
        elif reference_frame.find_frame(scene):
            box.prop(props, "use_frame")
        else:
            box.label(text="Scanner axes (no reference frame yet)", icon='INFO')
        if section.plane_matrix(scene) is None:
            box.label(text="Pick the plane object", icon='ERROR')
            return
        row = box.row(align=True)
        row.prop(props, "offset")
        row.operator("omfs.section_centre", text="", icon='PIVOT_MEDIAN')
        box.prop(props, "thickness")
        row = box.row(align=True)
        row.prop(props, "slab_only", toggle=True, icon='MOD_BEVEL')
        row.prop(props, "show_plane", toggle=True, icon='HIDE_OFF' if props.show_plane else 'HIDE_ON')
        row = box.row(align=True)
        row.operator("omfs.section_view", icon='VIEW_ORTHO')
        row.prop(props, "flip_view", toggle=True, icon='ARROW_LEFTRIGHT')

        box = layout.box()
        box.active = props.enabled
        box.label(text="Section contours", icon='MOD_OUTLINE')
        row = box.row(align=True)
        row.prop(props, "show_contours", toggle=True)
        sub = row.row(align=True)
        sub.active = props.slab_only   # a cap is only visible once the slab hides what covers it
        sub.prop(props, "show_caps", toggle=True)
        row.prop(props, "show_overlap", toggle=True)
        if props.show_overlap and props.enabled:
            plane = section.active_plane(scene)
            pairs = section_contours.overlaps(scene, plane, context.view_layer.depsgraph) if plane else []
            col = box.column(align=True)
            for name_a, name_b, _rows, area in pairs:
                row = col.row()
                row.alert = True
                row.label(text=f"{name_a} × {name_b}: {area:.1f} mm²", icon='ERROR')
            if not pairs:
                col.label(text="No overlap on the section", icon='CHECKMARK')

        box = layout.box()
        box.active = props.enabled
        box.label(text="Snapshot", icon='IMAGE_DATA')
        box.prop(props, "show_caption", toggle=True, icon='FONT_DATA')
        box.prop(props, "snapshot_dir", text="")
        problem = snapshot.folder_problem(props.snapshot_dir)
        row = box.row()
        row.enabled = problem is None and props.enabled
        row.operator("omfs.section_snapshot", icon='RENDER_STILL')
        if problem and props.snapshot_dir:
            warn = box.column()
            warn.alert = True
            draw_text(warn, context, problem)

        box = layout.box()
        box.label(text="Intersection volume (3D)", icon='MOD_BOOLEAN')
        overlay = analysis.find_intersection(scene)
        col = box.column(align=True)
        col.enabled = overlay is None
        col.prop(props, "inter_a")
        col.prop(props, "inter_b")
        box.operator("omfs.show_intersection", text="Remove Intersection" if overlay else "Show Intersection",
                     icon='X' if overlay else 'MOD_BOOLEAN', depress=overlay is not None)
        if overlay and analysis.is_snapshot(overlay):
            if analysis.snapshot_stale(overlay):
                box.operator("omfs.refresh_intersection", text="Moved: Refresh (a few seconds)", icon='ERROR')
            else:
                box.label(text="Snapshot (mesh with open edges): Exact solver", icon='INFO')
        measured = analysis.cached_volume(overlay) if overlay else None
        if measured:
            volume, closed = measured
            row = box.row()
            row.alert = volume > 0
            row.label(text=f"Common volume: {volume:.1f} mm³", icon='ERROR' if volume > 0 else 'CHECKMARK')
            if not closed:
                box.label(text="Result not closed: volume approximate", icon='INFO')

        box = layout.box()
        box.label(text="Distance map", icon='DRIVER_DISTANCE')
        source, _mod = distance_map.find_map(scene)
        col = box.column(align=True)
        col.enabled = source is None
        col.prop(props, "dist_source")
        col.prop(props, "dist_target")
        col = box.column(align=True)
        col.prop(props, "dist_red")
        col.prop(props, "dist_yellow")
        box.operator("omfs.distance_map", text="Remove Distance Map" if source else "Show Distance Map",
                     icon='X' if source else 'DRIVER_DISTANCE', depress=source is not None)
        found = distance_map.cached_closest(source) if source else None
        if found:
            d = found[0]
            row = box.row(align=True)
            row.alert = d < props.dist_red
            row.label(text=f"Minimum: {d:.2f} mm" if d >= 0 else f"Inside the target: {-d:.2f} mm deep",
                      icon='ERROR' if d < props.dist_red else 'CHECKMARK')
            row.prop(props, "dist_marker", text="", icon='EMPTY_SINGLE_ARROW')
            leg = box.column(align=True)
            leg.scale_y = 0.8
            for icon, text in distance_map.legend(props):
                leg.label(text=text, icon=icon)


class OMFS_PT_registration(bpy.types.Panel):
    bl_label = "Teeth Registration"
    bl_idname = "OMFS_PT_registration"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scans = [o for o in context.scene.objects if o.type == 'MESH' and o.omfs.role == 'SCAN']
        if not scans:
            layout.label(text="No intraoral scan in this case")
            return
        for scan in scans:
            self.draw_scan(context, layout.box(), scan)

    def draw_scan(self, context, box, scan):
        props = scan.omfs_reg
        name = scan.name
        box.label(text=name, icon='MESH_DATA')
        if occlusion.is_block(scan.parent) and occlusion.movement_summary(context.scene, scan.parent):
            col = box.column()
            col.alert = True
            draw_text(col, context, f"{scan.parent.name} déplacé : Retour préop (Assise de l'occlusion) "
                                    "avant de recaler, sinon le recalage mélange pré et postop.")
        col = box.column(align=True)
        col.prop(props, "target")
        col.prop(props, "bone")

        box.label(text="1. Point pairs (3 or more)")
        pairs = registration.point_pairs(context.scene, scan)
        col = box.column(align=True)
        for scan_point, _cbct_point in pairs:
            number = int(scan_point.omfs.landmark[5:])
            row = col.row(align=True)
            row.label(text=f"Pair {number}", icon='CHECKMARK')
            op = row.operator("omfs.delete_point_pair", text="", icon='X')
            op.scan_name, op.number = name, number
        col.operator("omfs.add_point_pair", icon='ADD').scan_name = name
        row = box.row()
        row.enabled = len(pairs) >= 3
        row.operator("omfs.align_points", icon='SNAP_ON').scan_name = name
        if props.points_rms >= 0:
            box.label(text=f"Points RMS: {props.points_rms:.2f} mm")
        if pairs:
            visible = registration.points_visible(context.scene, scan)
            op = box.operator("omfs.toggle_point_pairs", text="Hide Point Pairs" if visible else "Show Point Pairs",
                              icon='HIDE_OFF' if visible else 'HIDE_ON')
            op.scan_name = name

        box.label(text="2. Crowns used (gingiva excluded)")
        col = box.column(align=True)
        col.operator("omfs.suggest_crowns", icon='AUTO').scan_name = name
        row = col.row(align=True)
        op = row.operator("omfs.paint_crowns", text="Paint: use", icon='BRUSH_DATA')
        op.scan_name, op.include = name, True
        op = row.operator("omfs.paint_crowns", text="Paint: ignore", icon='BRUSH_DATA')
        op.scan_name, op.include = name, False
        if context.mode == 'PAINT_WEIGHT':
            col.operator("omfs.finish_painting", icon='CHECKMARK').scan_name = name
        if props.crown_count:
            box.label(text=f"{props.crown_count} crown vertices")

        box.label(text="3. Refine on the crowns")
        box.operator("omfs.refine_icp", icon='PIVOT_BOUNDBOX').scan_name = name
        if props.rms >= 0:
            res = box.column(align=True)
            warn = max(props.mean, props.rms) > params.REG_WARN_ERROR
            res.alert = warn
            res.label(text=f"Mean {props.mean:.2f} mm   RMS {props.rms:.2f} mm",
                      icon='ERROR' if warn else 'CHECKMARK')
            res.label(text=f"95% of crown points within {props.p95:.2f} mm")
            box.operator("omfs.toggle_distance_map", icon='COLOR')
        box.operator("omfs.reset_scan", icon='LOOP_BACK').scan_name = name


class OMFS_PT_landmarks(bpy.types.Panel):
    bl_label = "Landmarks"
    bl_idname = "OMFS_PT_landmarks"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        group = None
        for landmark_id, name, _description, lm_group in landmarks.LANDMARKS:
            if lm_group != group:
                group = lm_group
                col = layout.box().column(align=True)
                col.label(text=group)
            obj = landmarks.find(context.scene, landmark_id)
            row = col.row(align=True)
            row.label(text=name, icon='CHECKMARK' if obj else 'BLANK1')
            op = row.operator("omfs.place_landmark", text="Redo" if obj else "Place",
                              icon='RESTRICT_SELECT_OFF')
            op.landmark_id = landmark_id
            if obj:
                row.operator("omfs.delete_landmark", text="", icon='X').landmark_id = landmark_id
        if landmarks.all_landmarks(context.scene):
            visible = landmarks.landmarks_visible(context.scene)
            layout.operator("omfs.toggle_landmarks", text="Hide Landmarks" if visible else "Show Landmarks",
                            icon='HIDE_OFF' if visible else 'HIDE_ON')


class OMFS_PT_reference_frame(bpy.types.Panel):
    bl_label = "Reference Frame"
    bl_idname = "OMFS_PT_reference_frame"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        missing = reference_frame.missing_landmarks(context.scene)
        frame = reference_frame.find_frame(context.scene)
        if missing:
            col = layout.column(align=True)
            col.label(text="Place first:", icon='ERROR')
            for lm in missing:
                col.label(text=landmarks.BY_ID[lm][1], icon='BLANK1')
            return
        layout.operator("omfs.build_reference_frame", text="Update Frame" if frame else "Build Frame",
                        icon='ORIENTATION_GLOBAL')
        if frame is None:
            return
        visible = reference_frame.planes_visible(frame)
        layout.operator("omfs.toggle_reference_planes", text="Hide Planes" if visible else "Show Planes",
                        icon='HIDE_OFF' if visible else 'HIDE_ON')
        col = layout.column(align=True)
        col.label(text="Positive: left, anterior, up")
        col.label(text="Origin: nasion on Frankfort")


class OMFS_PT_osteotomy(bpy.types.Panel):
    bl_label = "Osteotomy"
    bl_idname = "OMFS_PT_osteotomy"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if osteotomy.anchor_matrix(scene) is None:
            layout.label(text="Build the frame and place both mental foramina", icon='ERROR')
            return
        cut = osteotomy.find_cut(scene)
        if cut is None:
            layout.operator("omfs.add_cut", icon='MOD_BOOLEAN')
            return
        visible = osteotomy.cut_helpers_visible(scene)
        layout.operator("omfs.toggle_cut_helpers",
                        text="Hide Cut Plane & Stars" if visible else "Show Cut Plane & Stars",
                        icon='HIDE_OFF' if visible else 'HIDE_ON')
        if not osteotomy.anchor_is_current(scene, cut):
            layout.operator("omfs.update_cut", text="Frame or foramina changed: Update Cut", icon='ERROR')

        props = cut.omfs_cut
        col = layout.column(align=True)
        col.prop(props, "source")
        col.separator()
        col.prop(props, "below_foramina")
        col.prop(props, "profile_angle")
        col.prop(props, "frontal_angle")
        col.prop(props, "extent_r")
        col.prop(props, "extent_l")

        box = layout.box()
        box.label(text="Distance above the cut", icon='DRIVER_DISTANCE')
        for label, distance, margin in osteotomy.cached_safety_rows(scene, cut):
            row = box.row()
            if distance is None:
                row.label(text=label)
                row.label(text="not in cut extent")
                continue
            row.alert = distance < margin
            row.label(text=label, icon='ERROR' if distance < margin else 'CHECKMARK')
            row.label(text="CROSSES THE CUT" if distance < 0 else f"{distance:.1f} mm")

        if props.source is None:
            return
        applied = props.applied_state
        if not applied:
            layout.operator("omfs.apply_cut", icon='MOD_BOOLEAN')
        elif applied != repr(osteotomy.state(props)):
            layout.operator("omfs.apply_cut", text="Cut changed: Re-apply", icon='ERROR')
        else:
            layout.operator("omfs.apply_cut", text="Re-apply Cut", icon='FILE_REFRESH')


class OMFS_PT_movement(bpy.types.Panel):
    bl_label = "Mouvement du segment"
    bl_idname = "OMFS_PT_movement"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if reference_frame.find_frame(scene) is None:
            layout.label(text="Build the reference frame first", icon='ERROR')
            return
        obj = movement.active_segment(scene)
        if obj is None:
            layout.label(text="Apply the cut first (Osteotomy)", icon='INFO')
            return
        if sum(movement.is_segment(o) for o in scene.objects) > 1:
            layout.prop(scene.omfs_movement, "segment", text="")
        layout.label(text=obj.name, icon='MESH_DATA')
        props = obj.omfs_move
        if not props.has_preop:
            layout.label(text="No pre-op pose: re-apply the cut", icon='ERROR')
            return

        row = layout.row(align=True)
        row.prop(props, "locked", toggle=True, icon='LOCKED' if props.locked else 'UNLOCKED')
        row.operator("omfs.segment_preop", icon='LOOP_BACK')

        layout.prop(props, "axes")
        if props.axes == 'CUT' and movement.cut_axes(scene, obj) is None:
            layout.label(text="No osteotomy plane: reference frame axes", icon='ERROR')
        col = layout.column(align=True)
        col.label(text="Translation of the pivot (mm)")
        col.prop(props, "anterior")
        col.prop(props, "up")
        col.prop(props, "left")
        col = layout.column(align=True)
        col.label(text="Rotation about the pivot")
        col.prop(props, "pitch", text="Pitch (+ front up)")
        col.prop(props, "roll", text="Roll (+ left up)")
        col.prop(props, "yaw", text="Yaw (+ front to left)")
        layout.prop(props, "pivot_kind")

        rows = movement.tracked_rows(scene, obj)
        if not rows:
            layout.label(text="Place the pogonion and menton to follow them", icon='INFO')
        on_cut = movement.axes(scene, obj) != movement.frame_axes(scene)
        facial = movement.tracked_rows(scene, obj, movement.frame_axes(scene)) if on_cut else rows
        for (name, _point, d), (_n, _p, d_facial) in zip(rows, facial):
            box = layout.box()
            box.label(text=f"{name}: {Vector(d).length:.1f} mm net", icon='EMPTY_SINGLE_ARROW')
            col = box.column(align=True)
            if on_cut:
                col.label(text="On the osteotomy plane:")
            for text in movement.describe(*d):
                col.label(text=text)
            if on_cut:   # what it means on the face (Frankfort, midsagittal, coronal)
                col.separator()
                col.label(text="In the reference frame:")
                for text in movement.describe(*d_facial):
                    col.label(text=text)
        layout.prop(props, "show_labels")

        guide = guide_workshop.target_guide(context)
        box = layout.box()
        box.label(text=f"Guide de coupe: {guide.name}" if guide else "Pas encore de guide", icon='MOD_SHRINKWRAP')
        box.enabled = guide is not None
        box.operator("omfs.guide_follow_segment", icon='CON_FOLLOWPATH')
        if guide is not None and movement.guide_moved(guide):
            row = box.row()
            row.label(text="Déplacé (re-cliquez après un nouveau mouvement)", icon='INFO')
            row.operator("omfs.guide_preop", text="", icon='LOOP_BACK')


class OMFS_PT_occlusion(bpy.types.Panel):
    bl_label = "Assise de l'occlusion"
    bl_idname = "OMFS_PT_occlusion"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.omfs_occlusion
        box = layout.box()
        box.label(text="Empreintes optiques", icon='MESH_DATA')
        if not any(o.type == 'MESH' and o.omfs.role == 'SCAN' for o in context.scene.objects):
            warn = box.column()
            warn.alert = True
            draw_text(warn, context, "Aucun objet de rôle Scan : réglez le rôle des empreintes sur Scan dans "
                                     "le panneau Case (ou importez avec Import Case).")
        col = box.column(align=True)
        col.prop(props, "upper")
        col.prop(props, "lower")
        if props.upper is not None and props.lower is not None:
            self.draw_preposition(context, layout.box(), props)
        if occlusion_seating.arches_ready(context.scene):
            self.draw_seating(context, layout.box(), props)

    def draw_seating(self, context, box, props):
        scene = context.scene
        box.label(text="Assise : le bas sur le haut fixe", icon='SNAP_FACE')
        if props.upper.omfs_reg.rms < 0:
            warn = box.column()
            warn.alert = True
            draw_text(warn, context, "Arcade du haut non recalée : elle est encore dans les coordonnées du "
                                     "scanner, et le bas (avec son os) y sera assis.")
        box.label(text="1. Ressorts (3 paires ou plus)")
        pairs = occlusion_seating.point_pairs(scene)
        lengths = dict(zip((occlusion_seating.pair_key(up) for up, _low in pairs), occlusion_seating.spring_lengths(scene)))
        col = box.column(align=True)
        for key, label in occlusion_seating.CANINES:   # the two canine springs, always listed
            row = col.row(align=True)
            if key in lengths:
                row.label(text=f"{label} : {lengths[key]:.2f} mm", icon='FORCE_FORCE')
                row.operator("omfs.add_occlusion_pair", text="", icon='FILE_REFRESH').key = key
                row.operator("omfs.delete_occlusion_pair", text="", icon='X').key = key
            else:
                row.operator("omfs.add_occlusion_pair", text=f"Ressort {label.lower()}", icon='ADD').key = key
        col.prop(props, "canine_weight")
        col = box.column(align=True)
        for up, _low in pairs:
            key = occlusion_seating.pair_key(up)
            if key.isdigit():
                row = col.row(align=True)
                row.label(text=f"Ressort {key} : {lengths[key]:.2f} mm", icon='FORCE_HARMONIC')
                row.operator("omfs.delete_occlusion_pair", text="", icon='X').key = key
        col.operator("omfs.add_occlusion_pair", icon='ADD')
        if occlusion_seating.occlusion_points(scene):
            visible = occlusion_seating.points_visible(scene)
            col.operator("omfs.toggle_occlusion_points", text="Cacher les ressorts" if visible else "Afficher les ressorts",
                         icon='HIDE_OFF' if visible else 'HIDE_ON')

        box.label(text="2. Asseoir")
        col = box.column(align=True)
        col.prop(props, "spring_share")
        col.prop(props, "tolerance")
        row = box.row()
        row.enabled = len(occlusion_seating.point_pairs(scene)) >= 3
        row.operator("omfs.seat_occlusion", icon='PLAY')
        block = occlusion.find_block(scene, 'LOWER')
        summary = occlusion.movement_summary(scene, block)
        if summary is None:
            box.label(text="Arcade du bas : position de départ", icon='CHECKMARK')
        else:
            (x, y, z), angle, in_frame = summary
            names = ("Gauche", "Avant", "Haut") if in_frame else ("X", "Y", "Z")
            box.label(text="  ".join(f"{n} {v:+.1f}" for n, v in zip(names, (x, y, z))) + f" mm   {angle:.1f}°",
                      icon='ORIENTATION_GLOBAL')
        row = box.row(align=True)
        row.operator("omfs.select_block", text="Ajuster le bas (G / R)", icon='RESTRICT_SELECT_OFF').arch = 'LOWER'
        row.operator("omfs.block_preop", text="Départ", icon='LOOP_BACK').arch = 'LOWER'
        shown = bool(occlusion.ghosts(block))
        box.operator("omfs.toggle_ghost", text="Cacher le fantôme" if shown else "Fantôme de départ",
                     icon='GHOST_ENABLED' if shown else 'GHOST_DISABLED').arch = 'LOWER'

        box.label(text="3. Chevauchement")
        box.operator("omfs.toggle_overlap_map", text="Cacher la carte" if props.show_map else "Carte de chevauchement",
                     icon='COLOR', depress=props.show_map)
        if props.max_overlap >= 0:
            current = occlusion_seating.map_is_current(scene)
            res = box.column(align=True)
            overlap = props.max_overlap
            res.alert = overlap > params.OCC_OVERLAP_RARE
            icon = 'ERROR' if overlap > params.OCC_OVERLAP_RARE else 'CHECKMARK'
            res.label(text=f"Chevauchement max {overlap:.2f} mm", icon=icon)
            if overlap > params.OCC_OVERLAP_MAX:
                res.label(text=f"Au-delà de {params.OCC_OVERLAP_MAX:g} mm : meulage important", icon='BLANK1')
            elif overlap > params.OCC_OVERLAP_RARE:
                res.label(text=f"Plus de {params.OCC_OVERLAP_RARE:g} mm : rare, à justifier", icon='BLANK1')
            if overlap > 3.0:
                res.label(text="Vérifier : normales inversées ou bords du scan ?", icon='BLANK1')
            col = box.column(align=True)
            col.label(text="Surface de contact (mm²)")
            row = col.row(align=True)
            for name, area in (("D", props.area_right), ("Ant", props.area_anterior), ("G", props.area_left)):
                row.label(text=f"{name} {area:.0f}", icon='CHECKMARK' if area >= params.OCC_TRIPOD_MIN_AREA else 'X')
            stable = occlusion_seating.tripod(props)
            col.label(text="Tripode stable" if stable else "Pas de tripode : contacts manquants",
                      icon='CHECKMARK' if stable else 'ERROR')
            if not current:
                box.operator("omfs.refresh_overlap_map", text="Position changée : actualiser", icon='FILE_REFRESH')
        if props.show_map:
            leg = box.column(align=True)
            leg.scale_y = 0.8
            for icon, text in occlusion_seating.legend():
                leg.label(text=text, icon=icon)

    def draw_preposition(self, context, box, props):
        box.label(text="Préposition de l'arcade du haut", icon='ORIENTATION_GIMBAL')
        draw_text(box, context, "Avant l'assise. Change la position de départ du scan, pas son bloc "
                                "(Retour préop la garde).")
        col = box.column(align=True)
        col.operator("omfs.flip_upper_arch", icon='FILE_REFRESH')
        adjusting = not locking.is_locked(props.upper)
        col.operator("omfs.adjust_upper_arch", text="Terminer l'ajustement" if adjusting else "Ajuster à la main",
                     icon='CHECKMARK' if adjusting else 'UNLOCKED', depress=adjusting)
        if adjusting:
            draw_text(box, context, "G : déplacer, R : tourner (X / Y / Z pour un axe). Puis Terminer.")


class OMFS_PT_splint(bpy.types.Panel):
    bl_label = "Object creator: Splint / Chape"
    bl_idname = "OMFS_PT_splint"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.omfs_splint
        layout.row().prop(props, "kind", expand=True)
        col = layout.column(align=True)
        if props.kind == 'CHAPE':
            col.prop(props, "cap_arch")
        else:
            col.prop(props, "upper", text="Haut")
            col.prop(props, "lower", text="Bas")
            upper, lower = splint.upper_arch(scene), splint.lower_arch(scene)
            if props.upper is None and upper is not None:
                col.label(text=f"Haut: {upper.name} (assise)", icon='LINKED')
            if props.lower is None and lower is not None:
                col.label(text=f"Bas: {lower.name} (assise)", icon='LINKED')
            hint = ("Arcade du haut déplacée, arcade du bas en préop." if props.kind == 'INTERMEDIATE'
                    else "Les deux arcades en occlusion finale (Assise de l'occlusion).")
            draw_text(layout, context, f"Placez d'abord les arcades: {hint} Trous permis aux points de contact.")

        box = layout.box()
        box.label(text="1. Courbe d'arcade", icon='CURVE_BEZCURVE')
        draw_text(box, context, "Cliquez 5 à 7 points sur les faces occlusales, d'une dernière molaire à "
                                "l'autre en passant par les incisives. La courbe passe par chaque point.")
        if props.kind != 'CHAPE':
            box.prop(props, "centre")
        curve = splint.find_curve(scene)
        n = len(splint.curve_points(curve))
        box.operator("omfs.arch_curve_click", text="Ajouter des points" if n else "Cliquer les points",
                     icon='RESTRICT_SELECT_OFF')
        if n:
            box.label(text=f"{n} points, {splint.curve_length(curve):.0f} mm", icon='CHECKMARK' if n >= 5 else 'INFO')
            row = box.row(align=True)
            row.operator("omfs.arch_curve_undo", text="Dernier", icon='BACK')
            row.operator("omfs.arch_curve_edit", text="Modifier", icon='EDITMODE_HLT')
            row.operator("omfs.arch_curve_clear", text="", icon='TRASH')
            if curve.mode == 'EDIT':
                draw_text(box, context, "G: déplacer (aimanté aux surfaces), X: supprimer. Tab pour terminer.")

        box = layout.box()
        box.label(text="2. Bloc paramétrique", icon='MESH_CUBE')
        box.enabled = n >= 2
        col = box.column(align=True)
        col.label(text="Débord au-delà des couronnes")
        col.prop(props, "buccal_margin")
        col.prop(props, "lingual_margin")
        col = box.column(align=True)
        col.label(text="Profondeur sur les dents (depuis le milieu de l'occlusion)")
        if props.kind == 'CHAPE':
            col.prop(props, "depth_up", text="Au-dessous" if splint.cap_on_upper(scene) else "Au-dessus")
            col.prop(props, "cap_depth")
        else:
            col.prop(props, "depth_up")
            col.prop(props, "depth_low")
        box.prop(props, "resolution")
        blank = splint.find_blank(scene)
        if blank is None:
            box.operator("omfs.build_splint_blank", icon='MESH_CUBE')
        else:
            stale = splint.blank_stale(scene, blank)
            box.operator("omfs.build_splint_blank", text="Courbe ou arcades modifiées: mettre à jour" if stale
                         else "Reconstruire le bloc", icon='ERROR' if stale else 'FILE_REFRESH')
        info = splint.blank_info(blank) if blank else {}
        if info:
            box.label(text=f"{info['length']:.0f} mm de long, {info['width_min']:.1f} à "
                           f"{info['width_max']:.1f} mm de large")

        box = layout.box()
        box.label(text="3. Empreinte des dents (projection)", icon='MOD_SHRINKWRAP')
        box.enabled = blank is not None
        draw_text(box, context, "Le dessus et le dessous du bloc sont poussés le long de l'axe d'insertion "
                                "(le haut du référentiel) jusqu'aux dents: les contre-dépouilles sont comblées.")
        box.prop(props, "imprint", toggle=True, icon='MOD_SHRINKWRAP')
        box.prop(props, "clearance")
        if info.get("imprint"):
            box.label(text=f"Épaisseur min. {info['thinnest']:.2f} mm", icon='CHECKMARK')
            if info["contact_area"] > 0:
                box.label(text=f"Contacts (trous): {info['contact_area']:.1f} mm²", icon='INFO')

        box = layout.box()
        box.label(text="4. Objet solide", icon='MESH_DATA')
        box.enabled = blank is not None
        draw_text(box, context, "Fige la gouttière en un objet à part: les réglages ne la changent plus, "
                                "trous aux contacts. Le texte déjà placé suit. Le bloc reste, caché.")
        box.operator("omfs.solidify_splint", icon='MESH_DATA')
        for solid in splint.find_solids(scene):
            info = splint.blank_info(solid)
            row = box.row()
            row.label(text=solid.name, icon='CHECKMARK' if info.get("open_edges") == 0 else 'ERROR')
            row.label(text="fermé" if info.get("open_edges") == 0 else f"{info.get('open_edges')} arêtes ouvertes")
        if blank is not None and blank.hide_get():
            box.operator("omfs.show_splint_blank", icon='HIDE_OFF')

        box = layout.box()
        box.label(text="5. Texte et vérification", icon='FONT_DATA')
        part = splint.current_part(context)
        box.enabled = part is not None
        if part is None:
            return
        box.label(text=f"Sur: {part.name}", icon='OBJECT_DATA')
        active = context.view_layer.objects.active
        if text_labels.is_label(active) and active.omfs_text.part == part:
            t = active.omfs_text                     # the selected text: its own settings, live
            col = box.column(align=True)
            col.prop(t, "text")
            col.row(align=True).prop(t, "mode", expand=True)
            col.prop(t, "height")
            col.prop(t, "depth")
            row = box.row()
            row.label(text=f"Lettres: {t.letter_height:.1f} mm",
                      icon='CHECKMARK' if t.letter_height >= params.TEXT_MIN_HEIGHT else 'ERROR')
            row.operator("omfs.remove_text", text="", icon='TRASH').label_name = active.name
        else:
            draw_text(box, context, "Cliquez où le texte commence, puis où il finit: les lettres remplissent "
                                    "l'espace entre les deux clics et se lisent de l'extérieur.")
            col = box.column(align=True)
            col.prop(props.label, "text")
            col.row(align=True).prop(props.label, "mode", expand=True)
            col.prop(props.label, "height")
            col.prop(props.label, "depth")
            op = box.operator("omfs.place_text", icon='RESTRICT_SELECT_OFF')
            op.part_name, op.settings_path = part.name, "omfs_splint.label"
        labels = text_labels.labels_on(part)
        if labels:
            col = box.column(align=True)
            for label in labels:
                col.label(text=label.name, icon='SMALL_CAPS')
            draw_text(box, context, "Sélectionnez un texte (cadre filaire) pour le modifier.")

        col = box.column(align=True)
        row = col.row(align=True)
        row.operator("omfs.check_splint", icon='CHECKMARK').part_name = part.name
        row.operator("omfs.export_splint", icon='EXPORT').part_name = part.name
        result = splint.last_check(part)
        if result:
            ok = result["open_edges"] == 0 and result["volume"] > 0
            col.label(text="Maillage fermé" if ok else f"{result['open_edges']} arêtes ouvertes",
                      icon='CHECKMARK' if ok else 'ERROR')
            if result.get("thinnest") is not None:
                col.label(text=f"Épaisseur min. {result['thinnest']:.2f} mm", icon='INFO')
            for x in result["texts"]:
                if x["wall"] is not None:
                    good = x["wall"] >= params.PRINT_MIN_WALL
                    col.label(text=f"{x['name']}: {x['wall']:.2f} mm sous la gravure",
                              icon='CHECKMARK' if good else 'ERROR')
            col.label(text="Prêt à imprimer" if result["ok"] else "À corriger",
                      icon='CHECKMARK' if result["ok"] else 'ERROR')


class OMFS_PT_guide(bpy.types.Panel):
    bl_label = "Confection de guide"
    bl_idname = "OMFS_PT_guide"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.omfs_guide
        draw_text(layout, context, "Atelier d'outils natifs de Blender. Survolez un bouton pour son "
                                   "explication, son raccourci et l'équivalent Meshmixer.")
        box = layout.box()
        box.label(text="Copie de travail (zones)", icon='DUPLICATE')
        row = box.row(align=True)
        copy = guide_workshop.current_copy(context)
        if copy is None:
            row.prop(props, "source", text="")
            tool_button(row, 'WORK_COPY', text="Copier")
        else:
            row.label(text=copy.name)
            hidden = copy.hide_get()
            row.operator("omfs.guide_tool", text="", icon='HIDE_ON' if hidden else 'HIDE_OFF',
                         depress=not hidden).tool = 'TOGGLE_WORK_COPY'
            tool_button(row, 'REMOVE_WORK_COPY', text="")
        box = layout.box()
        guide = guide_workshop.target_guide(context)
        box.label(text="Guide à modifier", icon='MOD_BUILD')
        if guide is None:
            box.label(text="Pas encore de guide: extrayez une zone (section 1)")
        else:
            box.prop(props, "guide", text="", icon='MESH_DATA')
            if props.guide is None:
                box.label(text=f"Par défaut: {guide.name}")
        obj = context.active_object
        layout.label(text=f"Actif: {obj.name if obj else '-'}   Mode: {context.mode.replace('_', ' ').title()}")
        if context.mode != 'OBJECT':
            tool_button(layout, 'OBJECT_MODE')


def draw_pins_settings(layout, context):
    p = context.scene.omfs_pins
    layout.prop(p, "cap")
    col = layout.column(align=True)
    col.label(text="Pin conique (butée = épaulement sur le plan)")
    for name in ("base_diameter", "tip_diameter", "cone_length", "clearance", "tip_gap", "shoulder_diameter",
                 "back_length"):
        col.prop(p, name)
    col = layout.column(align=True)
    col.label(text="Bossage femelle et bras")
    for name in ("wall", "floor", "embed", "arm_diameter"):
        col.prop(p, name)
    for site in pins.sites(context.scene):
        s = site.omfs_pin
        row = layout.row()
        state = "bras" if s.has_arm else ("mâle" if s.has_male else "femelle")
        row.label(text=f"{site.name}: {state}", icon='CHECKMARK' if s.has_arm else 'DOT')
        row.operator("omfs.remove_pin", text="", icon='TRASH').site_name = site.name


def make_guide_section(section_id, title, text):
    """One collapsible sub-panel per workshop section."""
    def draw(self, context):
        layout = self.layout
        draw_text(layout, context, text)
        if section_id == 'BOOLEAN':
            layout.prop(context.scene.omfs_guide, "cutter")
            active = context.active_object
            if cylinders.is_cylinder(active):
                col = layout.column(align=True)
                col.label(text=f"{active.name}: flèches bleues = longueur, orange = diamètre", icon='MESH_CYLINDER')
                col.prop(active.omfs_cylinder, "diameter")
                col.prop(active.omfs_cylinder, "length")
        if section_id in {'ZONES', 'MASK'} and guide_workshop.current_copy(context) is not None:
            layout.label(text=guide_workshop.zone_summary(context), icon='FACE_MAPS')
        if section_id == 'PINS':
            draw_pins_settings(layout, context)
        col = layout.column(align=True)
        for t in guide_workshop.tools_in(section_id):
            tool_button(col, t.id)
        if section_id == 'SHAPE':
            draw_modifiers(layout, context.active_object)

    return type(f"OMFS_PT_guide_{section_id.lower()}", (bpy.types.Panel,), {
        "bl_label": title, "bl_idname": f"OMFS_PT_guide_{section_id.lower()}", "bl_parent_id": "OMFS_PT_guide",
        "bl_space_type": 'VIEW_3D', "bl_region_type": 'UI', "bl_category": "OMFS",
        "bl_options": {'DEFAULT_CLOSED'}, "draw": draw,
    })


MODIFIER_FIELDS = {   # the one or two settings worth showing for each modifier type
    'DISPLACE': ("strength",), 'SOLIDIFY': ("thickness", "offset"), 'SMOOTH': ("factor", "iterations"),
    'REMESH': ("voxel_size",), 'BOOLEAN': ("object", "operation"),
}


def draw_modifiers(layout, obj):
    """The active object's modifier stack (the Properties editor is hidden in the OMFS layout)."""
    if obj is None or obj.type != 'MESH' or not obj.modifiers:
        return
    box = layout.box()
    box.label(text=f"Modificateurs de {obj.name} (dans l'ordre)", icon='MODIFIER')
    for mod in obj.modifiers:
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(mod, "show_viewport", text="")
        row.label(text=mod.name)
        op = row.operator("object.modifier_remove", text="", icon='X')
        op.modifier = mod.name
        for field in MODIFIER_FIELDS.get(mod.type, ()):
            col.prop(mod, field)


class OMFS_PT_guide_recipe(bpy.types.Panel):
    bl_label = "Recette pas à pas"
    bl_idname = "OMFS_PT_guide_recipe"
    bl_parent_id = "OMFS_PT_guide"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMFS"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        draw_text(layout, context, "[n] = numéro de la section où se trouve l'outil.")
        for title, steps in guide_workshop.RECIPE:
            box = layout.box()
            box.label(text=title, icon='TEXT')
            for i, step in enumerate(steps, start=1):
                draw_text(box, context, f"{i}. {step}")


GUIDE_SECTION_PANELS = tuple(make_guide_section(*s) for s in guide_workshop.SECTIONS)

classes = (OMFS_PT_case, OMFS_PT_view, OMFS_PT_section, OMFS_PT_registration, OMFS_PT_landmarks, OMFS_PT_reference_frame, OMFS_PT_osteotomy,
           OMFS_PT_movement, OMFS_PT_occlusion, OMFS_PT_splint,
           OMFS_PT_guide, *GUIDE_SECTION_PANELS, OMFS_PT_guide_recipe)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
