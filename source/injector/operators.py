import bpy
import numpy as np
import shutil
import struct
import os
import math
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from ..utils import ensure_object_visible, read_vertex_group_arrays
from ..simulation.operators import stop_physics_preview

from .common import (
    _clean_injected_lines,
    _find_position_buffers,
    _read_buf_positions_np,
    _read_buf_positions,
    _find_best_transform_for_object,
    _export_physics_state_buf,
    _export_physics_config_json,
    _load_physics_from_mod,
)

from .zzz import (
    export_physics_motion_hlsl_zzz,
    export_vertex_physics_hlsl_zzz,
    inject_physics_zzz,
)

from .hsr import (
    export_physics_motion_hlsl_hsr,
    export_vertex_physics_hlsl_hsr,
    inject_physics_hsr,
)


class XXMI_OT_apply_custom_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_apply_custom_physics"
    bl_label = "모드에 바스트모핑 주입"
    bl_description = "가슴 물리 웨이트가 등록된 모든 메쉬 오브젝트를 모드의 각 Position.buf와 매칭하여 실시간 HLSL 및 버퍼를 주입합니다"

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and (o.vertex_groups.get("XXMI_Breast_L") or o.vertex_groups.get("XXMI_Breast_R"))
            for o in bpy.data.objects
        )

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_mod_input = props.target_mod_dir.strip()
        target_game = getattr(props, "target_game", "ZZZ")

        if props.is_preview_running:
            stop_physics_preview(context)

        if not target_mod_input or not os.path.exists(target_mod_input):
            self.report({'ERROR'}, "타겟 모드 폴더(또는 .ini 파일) 경로를 올바르게 지정해주세요.")
            return {'CANCELLED'}

        # 앵커 본(XXMI_Breast_Rig)이 존재하면 주입 전 웨이트를 본 위치 기준으로 강제 재베이크 동기화
        rig = getattr(props, "anchor_obj", None) or bpy.data.objects.get("XXMI_Breast_Rig")
        if rig and rig.type == 'ARMATURE':
            from ..breast.operators import _rebake_mesh_weights_from_rig
            _rebake_mesh_weights_from_rig(context, rig, props)

        # 1. 가슴 웨이트를 가진 모든 메쉬 오브젝트 수집 및 가중치 캐시 구축
        breast_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and (o.vertex_groups.get("XXMI_Breast_L") or o.vertex_groups.get("XXMI_Breast_R"))
        ]

        if not breast_objs:
            self.report({'ERROR'}, "1단계에서 가슴 선택 영역을 먼저 등록해주세요 (XXMI_Breast_L/R 그룹 없음).")
            return {'CANCELLED'}

        mesh_caches = []
        for obj in breast_objs:
            ensure_object_visible(context, obj)
            n_verts = len(obj.data.vertices)
            if n_verts == 0:
                continue

            group_map = {}
            for gname, key in (("XXMI_Breast_L", "l"), ("XXMI_Breast_R", "r"), ("XXMI_Role_Rigid", "rigid")):
                vg = obj.vertex_groups.get(gname)
                if vg:
                    group_map[vg.index] = key
            arrs = read_vertex_group_arrays(obj, group_map)
            zeros = np.zeros(n_verts, dtype=np.float32)
            w_l = arrs.get("l", zeros)
            w_r = arrs.get("r", zeros)
            is_rigid = (arrs.get("rigid", zeros) > 0.5).astype(np.float32)

            weighted = np.nonzero((w_l > 1e-5) | (w_r > 1e-5))[0]
            if len(weighted) == 0:
                continue

            co = np.empty(n_verts * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)
            wco = co[weighted]

            n_sample = min(300, len(wco))
            sample_idx = np.linspace(0, len(wco) - 1, n_sample).astype(np.int64)
            sample_obj = [Vector(p) for p in wco[sample_idx].tolist()]

            mesh_caches.append({
                "obj": obj,
                "wco": wco,
                "w_l": w_l[weighted],
                "w_r": w_r[weighted],
                "rigid": is_rigid[weighted],
                "sample_obj": sample_obj,
            })

        if not mesh_caches:
            self.report({'ERROR'}, "가슴 가중치가 설정된 정점이 없습니다. 1단계에서 가슴 선택 영역을 먼저 등록해주세요.")
            return {'CANCELLED'}

        global_peak_weight = 0.0
        for mcache in mesh_caches:
            global_peak_weight = max(global_peak_weight, float(mcache["w_l"].max()), float(mcache["w_r"].max()))
        if global_peak_weight < 0.05:
            global_peak_weight = 1.0

        ini_files = []
        if os.path.isfile(target_mod_input) and target_mod_input.lower().endswith('.ini'):
            ini_files.append(target_mod_input)
        elif os.path.isdir(target_mod_input):
            for root, dirs, files in os.walk(target_mod_input):
                dirs[:] = [d for d in dirs if not d.lower().startswith('disabled')]
                for f in files:
                    if f.lower().endswith('.ini'):
                        ini_files.append(os.path.join(root, f))

        if not ini_files:
            self.report({'ERROR'}, "지정한 경로에서 .ini 파일을 찾을 수 없습니다.")
            return {'CANCELLED'}

        total_injected_buffers = 0
        matched_obj_names = set()

        for ini_path in ini_files:
            mod_folder = os.path.dirname(ini_path)
            custom_physics_dir = os.path.join(mod_folder, "CustomPhysics")
            os.makedirs(custom_physics_dir, exist_ok=True)
            pos_buffers = _find_position_buffers(mod_folder, ini_path)
            if not pos_buffers:
                continue

            matched_in_ini = []

            for idx, res_name, raw_fname, buf_v_count, norm_path, blend_sec, draw_sec in pos_buffers:
                buf_np = _read_buf_positions_np(norm_path)
                n_buf = len(buf_np)
                if n_buf == 0:
                    continue
                buf_positions = [Vector(p) for p in buf_np.tolist()]

                kd_buf = KDTree(n_buf)
                for bp_i, bp_co in enumerate(buf_positions):
                    kd_buf.insert(bp_co, bp_i)
                kd_buf.balance()

                valid_matches = []
                for mcache in mesh_caches:
                    b_mat, b_score = _find_best_transform_for_object(kd_buf, mcache["sample_obj"], mcache["obj"])
                    if b_mat is not None and b_score <= 0.05:
                        valid_matches.append((mcache, b_mat, b_score))

                if not valid_matches:
                    continue

                for mcache, _, _ in valid_matches:
                    matched_obj_names.add(mcache["obj"].name)

                # 모든 매칭 오브젝트의 가중치 정점을 버퍼 좌표계로 변환하여 하나의 KD-Tree로 통합
                pts_chunks = []
                wl_chunks = []
                wr_chunks = []
                rig_chunks = []
                for mcache, b_mat, _ in valid_matches:
                    M = np.array(b_mat, dtype=np.float64)
                    pts_chunks.append(mcache["wco"] @ M[:3, :3].T + M[:3, 3])
                    wl_chunks.append(mcache["w_l"])
                    wr_chunks.append(mcache["w_r"])
                    rig_chunks.append(mcache["rigid"])

                all_pts = np.concatenate(pts_chunks, axis=0)
                all_wl = np.concatenate(wl_chunks)
                all_wr = np.concatenate(wr_chunks)
                all_rig = np.concatenate(rig_chunks)

                tol = min(0.02, max(2e-4, 2.0 * max(s for _, _, s in valid_matches)))

                kd_w = KDTree(len(all_pts))
                for k, p in enumerate(all_pts.tolist()):
                    kd_w.insert(p, k)
                kd_w.balance()

                # 가중치 정점 영역(AABB) 밖의 버퍼 정점은 검색 없이 0으로 처리 (대형 모드 속도 최적화)
                lo = all_pts.min(axis=0) - tol
                hi = all_pts.max(axis=0) + tol
                in_box = np.all((buf_np >= lo) & (buf_np <= hi), axis=1)
                cand_idx = np.nonzero(in_box)[0]

                wl_out = np.zeros(n_buf, dtype=np.float32)
                wr_out = np.zeros(n_buf, dtype=np.float32)
                rig_out = np.zeros(n_buf, dtype=np.float32)

                for i in cand_idx.tolist():
                    hits = kd_w.find_range(buf_positions[i], tol)
                    if not hits:
                        continue
                    k = max((h[1] for h in hits), key=lambda kk: all_wl[kk] + all_wr[kk])
                    wl_out[i] = all_wl[k]
                    wr_out[i] = all_wr[k]
                    rig_out[i] = all_rig[k]

                # ── UV 심(Seam) 및 분리된 경계 정점 찢어짐 방지 (Coincident Vertex Synchronization) ──
                # 버퍼 내에서 3D 공간 상 동일 위치(0.5mm 이내)를 공유하는 분리 정점들은
                # 셰이더 상에서 100% 동일하게 움직이도록 가중치를 동기화합니다 (버텍스 개수 변경 없음).
                active_mask = (wl_out > 0.0001) | (wr_out > 0.0001) | (rig_out > 0.5)
                active_indices = np.nonzero(active_mask)[0]
                if len(active_indices) > 0:
                    for ai in active_indices.tolist():
                        p = buf_positions[ai]
                        hits = kd_buf.find_range(p, 0.0005)
                        if len(hits) > 1:
                            max_l = max(wl_out[h[1]] for h in hits)
                            max_r = max(wr_out[h[1]] for h in hits)
                            max_rig = max(rig_out[h[1]] for h in hits)
                            for _, hi, _ in hits:
                                wl_out[hi] = max_l
                                wr_out[hi] = max_r
                                rig_out[hi] = max_rig

                max_wl = float(wl_out.max())
                max_wr = float(wr_out.max())
                peak_l_idx = int(wl_out.argmax())
                peak_r_idx = int(wr_out.argmax())

                # 버퍼 정점별 가중치 매핑: stride 16 (float4: wl, wr, is_rigid, 0.0)
                map_bytes = np.stack(
                    [wl_out, wr_out, rig_out, np.zeros(n_buf, dtype=np.float32)], axis=1
                ).astype('<f4').tobytes()

                has_breast = (max_wl > 0.001 or max_wr > 0.001)

                if has_breast:
                    map_buf_path = os.path.join(custom_physics_dir, f"CustomPhysicsMap_{idx}.buf")
                    with open(map_buf_path, 'wb') as f:
                        f.write(map_bytes)

                    # 흉골/몸통 중심 정점 탐색
                    best_root_idx = 0
                    breast_mask = (wl_out > 0.1) | (wr_out > 0.1)
                    if breast_mask.any():
                        center_breast = buf_np[breast_mask].mean(axis=0)
                        root_dist = np.linalg.norm(buf_np - center_breast, axis=1)
                        root_dist[(wl_out >= 0.01) | (wr_out >= 0.01)] = np.inf
                        if np.isfinite(root_dist).any():
                            best_root_idx = int(root_dist.argmin())

                    # 런타임 물리 상태 버퍼 초기화 (.buf)
                    _export_physics_state_buf(custom_physics_dir, idx)

                    # 게임별 분기 쉐이더 내보내기 (ZZZ vs HSR)
                    if target_game == 'HSR':
                        export_physics_motion_hlsl_hsr(custom_physics_dir, idx, best_root_idx, props)
                        export_vertex_physics_hlsl_hsr(custom_physics_dir, idx, n_buf, props, global_peak_weight, best_root_idx)
                    else:
                        export_physics_motion_hlsl_zzz(custom_physics_dir, idx, best_root_idx, props)
                        export_vertex_physics_hlsl_zzz(custom_physics_dir, idx, n_buf, props, global_peak_weight)

                    total_injected_buffers += 1

                matched_in_ini.append((idx, res_name, raw_fname, n_buf, has_breast, peak_l_idx, peak_r_idx, blend_sec, draw_sec))

            if matched_in_ini:
                breast_entries = [e for e in matched_in_ini if e[4]]
                if breast_entries:
                    master_entry = max(breast_entries, key=lambda e: e[3])
                    master_idx = master_entry[0]
                else:
                    master_idx = 0

                if target_game == 'HSR':
                    inject_physics_hsr(ini_path, matched_in_ini, props, master_idx)
                else:
                    inject_physics_zzz(ini_path, matched_in_ini, props, master_idx)

                _export_physics_config_json(custom_physics_dir, props, context)

        if total_injected_buffers > 0:
            names_str = ", ".join(matched_obj_names)
            game_label = "붕괴: 스타레일" if target_game == 'HSR' else "젠레스 존 제로"
            self.report({'INFO'}, f"성공: [{game_label}] {total_injected_buffers}개 Position 버퍼에 가슴 물리 주입 완료! (매칭 오브젝트: {names_str})")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "모드 폴더에서 매칭되는 Position.buf를 찾지 못했습니다. 올바른 캐릭터 모드 폴더인지 확인해주세요.")
            return {'CANCELLED'}


class XXMI_OT_rollback_custom_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_rollback_custom_physics"
    bl_label = "바스트모핑 효과 롤백"
    bl_description = "모드 폴더에 주입된 가슴 물리 쉐이더 및 버퍼를 모두 제거하고 .ini 파일을 100% 원본으로 복원합니다"

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_mod_input = props.target_mod_dir.strip()

        if props.is_preview_running:
            stop_physics_preview(context)

        if not target_mod_input or not os.path.exists(target_mod_input):
            self.report({'ERROR'}, "타겟 모드 폴더(또는 .ini 파일) 경로를 올바르게 지정해주세요.")
            return {'CANCELLED'}

        ini_files = []
        if os.path.isfile(target_mod_input) and target_mod_input.lower().endswith('.ini'):
            ini_files.append(target_mod_input)
        elif os.path.isdir(target_mod_input):
            for root, dirs, files in os.walk(target_mod_input):
                for f in files:
                    if f.lower().endswith('.ini'):
                        ini_files.append(os.path.join(root, f))

        cleaned_count = 0
        removed_files = 0

        for ini_path in ini_files:
            mod_folder = os.path.dirname(ini_path)
            with open(ini_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()

            clean_lines, modified = _clean_injected_lines(raw_lines)
            if modified:
                with open(ini_path, 'w', encoding='utf-8') as f:
                    f.writelines(clean_lines)
                cleaned_count += 1

            # 1. CustomPhysics 서브폴더 삭제
            custom_physics_dir = os.path.join(mod_folder, "CustomPhysics")
            if os.path.isdir(custom_physics_dir):
                shutil.rmtree(custom_physics_dir, ignore_errors=True)
                removed_files += 1

            # 2. 레거시(모드 루트에 남아있던 파일들)도 함께 정리
            for fname in os.listdir(mod_folder):
                fl = fname.lower()
                if (fl.startswith("customphysics") and (fl.endswith(".hlsl") or fl.endswith(".buf"))) or \
                   (fl.startswith("customphysicsmotion") and (fl.endswith(".hlsl") or fl.endswith(".buf"))) or \
                   (fl.startswith("customphysicsstate") and (fl.endswith(".hlsl") or fl.endswith(".buf"))) or \
                   (fl in ("xxmi_physics_config.json", "customphysicsconfig.json")):
                    try:
                        os.remove(os.path.join(mod_folder, fname))
                        removed_files += 1
                    except Exception:
                        pass

        self.report({'INFO'}, f"롤백 완료: {cleaned_count}개 .ini 복원, {removed_files}개 생성 파일 삭제됨.")
        return {'FINISHED'}


class XXMI_OT_load_physics_from_mod(bpy.types.Operator):
    bl_idname = "object.xxmi_load_physics_from_mod"
    bl_label = "모드 폴더에서 모핑 정보 불러오기"
    bl_description = "모드 폴더 내의 설정 파일(JSON) 또는 주입된 HLSL을 읽어 물리 파라미터와 앵커 본 정보를 복원합니다"
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_dir = self.directory.strip() if self.directory else props.target_mod_dir.strip()

        if not target_dir or not os.path.exists(target_dir):
            self.report({'ERROR'}, "모드 폴더 경로를 지정해주세요.")
            return {'CANCELLED'}

        if os.path.isfile(target_dir):
            target_dir = os.path.dirname(target_dir)

        props.target_mod_dir = target_dir
        success, msg = _load_physics_from_mod(target_dir, context, props)
        if not success:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        self.report({'INFO'}, f"성공: {msg}")
        return {'FINISHED'}

    def invoke(self, context, event):
        props = context.scene.xxmi_custom_physics_props
        if props.target_mod_dir and os.path.isdir(props.target_mod_dir):
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class XXMI_OT_physics_dir_picker(bpy.types.Operator):
    bl_idname = "object.xxmi_physics_dir_picker"
    bl_label = "모드 폴더 선택"
    bl_description = "바스트모핑을 주입할 타겟 모드 폴더를 선택합니다"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        props.target_mod_dir = self.directory
        if os.path.exists(self.directory):
            cfg_p = os.path.join(self.directory, "XXMI_Physics_Config.json")
            has_hlsl = any(f.lower().startswith("customphysics") for f in os.listdir(self.directory))
            if os.path.exists(cfg_p) or has_hlsl:
                success, msg = _load_physics_from_mod(self.directory, context, props)
                if success:
                    self.report({'INFO'}, f"폴더 선택 및 모핑 정보 자동 동기화: {msg}")
                    return {'FINISHED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


classes = (
    XXMI_OT_apply_custom_physics,
    XXMI_OT_rollback_custom_physics,
    XXMI_OT_load_physics_from_mod,
    XXMI_OT_physics_dir_picker,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
