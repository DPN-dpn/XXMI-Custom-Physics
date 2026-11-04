import json
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    ensure_object_visible,
    world_vertex_positions,
    mesh_edge_arrays,
    label_components,
    get_target_mesh_objects,
    ROLE_RIGID_VG,
)

VG_RIGID = ROLE_RIGID_VG


def get_rigid_data(item):
    """RigidItem의 JSON 문자열 데이터를 딕셔너리로 파싱합니다."""
    if not item or not item.data_json:
        return {}
    try:
        return json.loads(item.data_json)
    except Exception:
        return {}


def set_rigid_data(item, data):
    """RigidItem의 JSON 문자열 데이터를 갱신하고 정점 수를 동기화합니다."""
    if not item:
        return
    item.data_json = json.dumps(data)
    item.vert_count = sum(len(v) for v in data.values())


class XXMI_UL_rigid_items(bpy.types.UIList):
    """강체 부품 목록을 그리는 UIList입니다."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text="", icon='PINNED')
            row.prop(item, "name", text="", emboss=False)
            bone_label = "자동"
            if item.target_bone == 'BREAST_L':
                bone_label = "Breast.L"
            elif item.target_bone == 'BREAST_R':
                bone_label = "Breast.R"
            elif item.target_bone == 'CENTER':
                bone_label = "중앙(L+R)"
            sub = row.row()
            sub.alignment = 'RIGHT'
            sub.label(text=f"[{bone_label} | {item.vert_count}개]")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='PINNED')


class XXMI_OT_rigid_item_add(bpy.types.Operator):
    bl_idname = "object.xxmi_rigid_item_add"
    bl_label = "강체 슬롯 추가"
    bl_description = "새로운 강체 부품(단추, 무전기, 리본 등) 관리 슬롯을 추가합니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.rigid_items.add()
        item.name = f"강체 {len(props.rigid_items)}"
        props.rigid_item_index = len(props.rigid_items) - 1

        was_edit = (context.mode == 'EDIT_MESH')
        target_objs = get_target_mesh_objects(context)
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        data = {}
        if target_objs:
            for obj in target_objs:
                mesh = obj.data
                n_verts = len(mesh.vertices)
                if n_verts == 0:
                    continue
                sel_mask = np.zeros(n_verts, dtype=bool)
                mesh.vertices.foreach_get("select", sel_mask)
                sel_indices = np.nonzero(sel_mask)[0].tolist()
                if sel_indices:
                    data[obj.name] = sel_indices
                    vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)
                    vg_rigid.add(sel_indices, 1.0, 'REPLACE')

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if data:
            set_rigid_data(item, data)
            self.report({'INFO'}, f"새 강체 슬롯 '{item.name}' 생성 및 선택 정점 {item.vert_count}개 할당 완료!")
        else:
            self.report({'INFO'}, f"새 강체 슬롯 '{item.name}'이 추가되었습니다.")

        return {'FINISHED'}


class XXMI_OT_rigid_item_remove(bpy.types.Operator):
    bl_idname = "object.xxmi_rigid_item_remove"
    bl_label = "강체 슬롯 삭제"
    bl_description = "선택된 강체 슬롯을 목록에서 삭제하고 정점 강체 등록을 해제합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.rigid_item_index < len(props.rigid_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        idx = props.rigid_item_index
        item = props.rigid_items[idx]
        data = get_rigid_data(item)

        was_edit = (context.mode == 'EDIT_MESH')
        target_objs = get_target_mesh_objects(context)
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH':
                vg_rigid = obj.vertex_groups.get(VG_RIGID)
                if vg_rigid:
                    vg_rigid.remove(v_indices)

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        name = item.name
        props.rigid_items.remove(idx)
        props.rigid_item_index = max(0, min(idx, len(props.rigid_items) - 1))
        self.report({'INFO'}, f"강체 슬롯 '{name}'을 삭제했습니다.")
        return {'FINISHED'}


class XXMI_OT_rigid_item_assign(bpy.types.Operator):
    bl_idname = "object.xxmi_rigid_item_assign"
    bl_label = "선택 정점 이 강체에 할당"
    bl_description = "에디트 모드에서 현재 선택된 정점들을 이 강체 슬롯에 등록합니다 (여러 메쉬 오브젝트 지원)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.rigid_item_index < len(props.rigid_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.rigid_items[props.rigid_item_index]
        data = get_rigid_data(item)

        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        assigned_any = False
        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0].tolist()
            if not sel_indices:
                continue

            existing = set(data.get(obj.name, []))
            existing.update(sel_indices)
            data[obj.name] = sorted(list(existing))
            assigned_any = True

            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)
            vg_rigid.add(sel_indices, 1.0, 'REPLACE')

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if not assigned_any:
            self.report({'ERROR'}, "에디트 모드에서 강체로 등록할 정점을 먼저 선택해주세요.")
            return {'CANCELLED'}

        set_rigid_data(item, data)
        self.report({'INFO'}, f"'{item.name}' 슬롯에 정점 할당 완료! (현재 총 {item.vert_count}개 정점)")
        return {'FINISHED'}


class XXMI_OT_rigid_item_remove_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_rigid_item_remove_verts"
    bl_label = "선택 정점 강체 제외"
    bl_description = "현재 선택된 정점을 이 강체 슬롯에서 제외합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.rigid_item_index < len(props.rigid_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.rigid_items[props.rigid_item_index]
        data = get_rigid_data(item)

        target_objs = get_target_mesh_objects(context)
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        removed_total = 0
        for obj in target_objs:
            if obj.name not in data:
                continue
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = set(np.nonzero(sel_mask)[0].tolist())
            if not sel_indices:
                continue

            current_set = set(data[obj.name])
            to_remove = current_set.intersection(sel_indices)
            if to_remove:
                current_set.difference_update(to_remove)
                data[obj.name] = sorted(list(current_set))
                removed_total += len(to_remove)
                vg_rigid = obj.vertex_groups.get(VG_RIGID)
                if vg_rigid:
                    vg_rigid.remove(list(to_remove))

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        set_rigid_data(item, data)
        self.report({'INFO'}, f"'{item.name}' 슬롯에서 {removed_total}개 정점을 제외했습니다.")
        return {'FINISHED'}


class XXMI_OT_rigid_item_select_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_rigid_item_select_verts"
    bl_label = "강체 정점 선택"
    bl_description = "이 강체 슬롯에 등록된 정점들을 에디트 모드에서 뷰포트에 선택 표시합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.rigid_item_index < len(props.rigid_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.rigid_items[props.rigid_item_index]
        data = get_rigid_data(item)

        if not data:
            self.report({'WARNING'}, f"'{item.name}' 슬롯에 등록된 정점이 없습니다.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        valid_objs = []
        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH' and v_indices:
                ensure_object_visible(context, obj)
                obj.select_set(True)
                valid_objs.append(obj)

        if not valid_objs:
            self.report({'ERROR'}, "등록된 메쉬 오브젝트를 씬에서 찾을 수 없습니다.")
            return {'CANCELLED'}

        context.view_layer.objects.active = valid_objs[0]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')

        for obj in valid_objs:
            v_indices = data.get(obj.name, [])
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            for vi in v_indices:
                if vi < len(bm.verts):
                    bm.verts[vi].select_set(True)
            bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"'{item.name}' 슬롯의 정점 {item.vert_count}개를 선택했습니다.")
        return {'FINISHED'}


class XXMI_OT_mark_selection_rigid(bpy.types.Operator):
    bl_idname = "object.xxmi_mark_selection_rigid"
    bl_label = "선택 부위를 강체로 등록"
    bl_description = "선택된 정점들을 강체(XXMI_Role_Rigid)로 등록하여 물리 동작 시 형태를 유지합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        objs = get_target_mesh_objects(context)
        return len(objs) > 0

    def execute(self, context):
        target_objs = get_target_mesh_objects(context)
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        total_marked = 0
        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0]
            if len(sel_indices) == 0:
                continue

            world = world_vertex_positions(obj)
            e0, e1 = mesh_edge_arrays(mesh)
            lab_all = label_components(n_verts, e0, e1)
            uniq_labs, lab_counts = np.unique(lab_all, return_counts=True)

            for lab_id, cnt in zip(uniq_labs, lab_counts):
                if cnt <= 2500:
                    isl_mask = (lab_all == lab_id)
                    n_sel = np.count_nonzero(sel_mask & isl_mask)
                    if n_sel > 0 and (n_sel >= 4 or (n_sel / cnt) >= 0.10):
                        sel_mask[isl_mask] = True

            kd_local = KDTree(n_verts)
            for vi, wpos in enumerate(world):
                kd_local.insert(Vector(wpos), vi)
            kd_local.balance()

            cur_sel = np.nonzero(sel_mask)[0].tolist()
            for si in cur_sel:
                hits = kd_local.find_range(Vector(world[si]), 0.0005)
                for _, hi, _ in hits:
                    sel_mask[hi] = True

            final_sel = np.nonzero(sel_mask)[0].tolist()
            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)
            vg_rigid.add(final_sel, 1.0, 'REPLACE')
            total_marked += len(final_sel)

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if total_marked == 0:
            self.report({'ERROR'}, "강체로 등록할 정점들을 에디트 모드에서 선택해주세요.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"총 {total_marked}개 정점을 강체로 등록 완료했습니다.")
        return {'FINISHED'}


class XXMI_OT_unmark_selection_rigid(bpy.types.Operator):
    bl_idname = "object.xxmi_unmark_selection_rigid"
    bl_label = "선택 부위 강체 해제"
    bl_description = "선택된 정점을 강체 목록에서 제외합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        objs = get_target_mesh_objects(context)
        return any(o.vertex_groups.get(VG_RIGID) is not None for o in objs)

    def execute(self, context):
        target_objs = get_target_mesh_objects(context)
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        total_unmarked = 0
        for obj in target_objs:
            vg_rigid = obj.vertex_groups.get(VG_RIGID)
            if not vg_rigid:
                continue
            sel = [v.index for v in obj.data.vertices if v.select]
            if not sel:
                continue
            vg_rigid.remove(sel)
            total_unmarked += len(sel)

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        self.report({'INFO'}, f"총 {total_unmarked}개 정점의 강체 등록을 해제했습니다.")
        return {'FINISHED'}


class XXMI_OT_select_rigid_vertices(bpy.types.Operator):
    bl_idname = "object.xxmi_select_rigid_vertices"
    bl_label = "강체 정점 다시 선택"
    bl_description = "등록된 강체 정점들을 에디트 모드에서 다시 선택합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and o.vertex_groups.get(VG_RIGID) is not None
            for o in bpy.data.objects
        )

    def execute(self, context):
        target_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and o.vertex_groups.get(VG_RIGID) is not None
        ]

        if not target_objs:
            self.report({'WARNING'}, "등록된 강체 정점이 없습니다.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        for obj in target_objs:
            ensure_object_visible(context, obj)
            obj.select_set(True)

        context.view_layer.objects.active = target_objs[0]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')

        total_selected = 0
        for obj in target_objs:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            vg_rigid = obj.vertex_groups.get(VG_RIGID)
            if not vg_rigid:
                continue
            g_idx = vg_rigid.index
            dvert_lay = bm.verts.layers.deform.verify()

            for v in bm.verts:
                dvert = v[dvert_lay]
                is_r = (g_idx in dvert and dvert[g_idx] > 0.5)
                v.select_set(is_r)
                if is_r:
                    total_selected += 1

            bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"총 {total_selected}개 강체 정점을 다시 선택했습니다.")
        return {'FINISHED'}


classes = (
    XXMI_UL_rigid_items,
    XXMI_OT_rigid_item_add,
    XXMI_OT_rigid_item_remove,
    XXMI_OT_rigid_item_assign,
    XXMI_OT_rigid_item_remove_verts,
    XXMI_OT_rigid_item_select_verts,
    XXMI_OT_mark_selection_rigid,
    XXMI_OT_unmark_selection_rigid,
    XXMI_OT_select_rigid_vertices,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

