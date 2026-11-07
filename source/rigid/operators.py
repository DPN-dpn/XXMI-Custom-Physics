import bpy
import bmesh
import numpy as np
from ..utils import (
    ensure_object_visible,
    ensure_all_objects_in_object_mode,
    get_target_mesh_objects,
    ROLE_RIGID_VG,
)
from .data import get_rigid_data, set_rigid_data

VG_RIGID = ROLE_RIGID_VG


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

        target_objs = get_target_mesh_objects(context)
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

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

        if was_edit and target_objs:
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

        target_objs = get_target_mesh_objects(context)
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH':
                vg_rigid = obj.vertex_groups.get(VG_RIGID)
                if vg_rigid:
                    vg_rigid.remove(v_indices)

        if was_edit and target_objs:
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
    bl_label = "선택 정점 강체 재할당"
    bl_description = "에디트 모드에서 현재 선택된 정점들로 이 강체 슬롯의 정점 목록을 새로 교체(재할당)합니다 (여러 메쉬 오브젝트 지원)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.rigid_item_index < len(props.rigid_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.rigid_items[props.rigid_item_index]
        old_data = get_rigid_data(item)

        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        new_data = {}
        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0].tolist()
            if sel_indices:
                new_data[obj.name] = sel_indices

        if not new_data:
            if was_edit:
                for obj in target_objs:
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "에디트 모드에서 강체 슬롯에 재할당할 정점을 먼저 선택해주세요.")
            return {'CANCELLED'}

        # 다른 강체 슬롯에 등록된 정점 파악 (보호용)
        other_rigid_by_obj = {}
        for idx_r, r_item in enumerate(props.rigid_items):
            if idx_r == props.rigid_item_index:
                continue
            r_d = get_rigid_data(r_item)
            for oname, vis in r_d.items():
                if vis:
                    other_rigid_by_obj.setdefault(oname, set()).update(vis)

        affected_obj_names = set(old_data.keys()) | set(new_data.keys())
        for oname in affected_obj_names:
            obj = bpy.data.objects.get(oname)
            if not obj or obj.type != 'MESH':
                continue
            vg_rigid = obj.vertex_groups.get(VG_RIGID)
            old_vis = set(old_data.get(oname, [])) - other_rigid_by_obj.get(oname, set())
            if vg_rigid and old_vis:
                vg_rigid.remove(list(old_vis))
            new_vis = new_data.get(oname, [])
            if new_vis:
                if not vg_rigid:
                    vg_rigid = obj.vertex_groups.new(name=VG_RIGID)
                vg_rigid.add(new_vis, 1.0, 'REPLACE')

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        set_rigid_data(item, new_data)
        self.report({'INFO'}, f"'{item.name}' 슬롯의 정점을 현재 선택된 {item.vert_count}개 정점으로 재할당했습니다!")
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
            ensure_all_objects_in_object_mode(context)

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

        # 1. data_json이 비어있는 경우 XXMI_Role_Rigid 그룹으로부터 자동 복구 시도
        if not data:
            recovered_rigid = {}
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    vg = obj.vertex_groups.get(VG_RIGID)
                    if vg:
                        vis = []
                        for vi in range(len(obj.data.vertices)):
                            try:
                                if vg.weight(vi) > 0.5:
                                    vis.append(vi)
                            except Exception:
                                pass
                        if vis:
                            recovered_rigid[obj.name] = vis
            if recovered_rigid:
                data = recovered_rigid
                set_rigid_data(item, data)

        if not data:
            self.report({'WARNING'}, f"'{item.name}' 슬롯에 등록된 정점이 없습니다.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        resolved_objs = []
        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if not obj or obj.type != 'MESH':
                for o in bpy.data.objects:
                    if o.type == 'MESH' and o.vertex_groups.get(VG_RIGID):
                        obj = o
                        break
                if not obj and context.active_object and context.active_object.type == 'MESH':
                    obj = context.active_object

            if obj and obj.type == 'MESH' and v_indices:
                ensure_object_visible(context, obj)
                obj.select_set(True)
                if obj not in resolved_objs:
                    resolved_objs.append((obj, v_indices))

        if not resolved_objs:
            self.report({'ERROR'}, "등록된 메쉬 오브젝트를 씬에서 찾을 수 없습니다.")
            return {'CANCELLED'}

        first_obj = resolved_objs[0][0]
        context.view_layer.objects.active = first_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')

        total_sel = 0
        for obj, v_indices in resolved_objs:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            for vi in v_indices:
                if vi < len(bm.verts):
                    bm.verts[vi].select_set(True)
                    total_sel += 1
            bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"'{item.name}' 슬롯의 정점 {total_sel}개를 선택했습니다.")
        return {'FINISHED'}


classes = (
    XXMI_OT_rigid_item_add,
    XXMI_OT_rigid_item_remove,
    XXMI_OT_rigid_item_assign,
    XXMI_OT_rigid_item_remove_verts,
    XXMI_OT_rigid_item_select_verts,
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
