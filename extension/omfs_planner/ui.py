"""Sidebar tab "OMFS" in the 3D viewport (press N to show the sidebar)."""

import bpy

from . import landmarks, locking, osteotomy, params, reference_frame, registration, roles, view


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


classes = (OMFS_PT_case, OMFS_PT_view, OMFS_PT_registration, OMFS_PT_landmarks, OMFS_PT_reference_frame, OMFS_PT_osteotomy)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
