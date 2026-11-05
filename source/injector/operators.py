import bpy
import numpy as np
import shutil
import struct
import os
import math
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from ..utils import ensure_object_visible, ensure_all_objects_in_object_mode, read_vertex_group_arrays
from ..simulation.engine import stop_physics_preview

from .ini_parser import (
    _clean_injected_lines,
    _find_position_buffers,
    backup_ini_file,
)
from .buffer_io import (
    _read_buf_positions_np,
    _read_buf_positions_and_normals_np,
    _read_buf_positions,
    _find_best_transform_for_object,
    _export_physics_state_buf,
    _export_physics_config_json,
)
from .loader import (
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
    bl_label = "모드에 물리 주입"
    bl_description = "현재 설정된 물리 가중치와 파라미터를 캐릭터 모드에 주입하여 인게임에서 동작하게 합니다"

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and any(vg.name.startswith("XXMI_") for vg in o.vertex_groups)
            for o in bpy.data.objects
        )

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_mod_input = props.target_mod_dir.strip()
        target_game = getattr(props, "target_game", "ZZZ")

        if props.is_preview_running:
            stop_physics_preview(context)

        ensure_all_objects_in_object_mode(context)

        if not target_mod_input or not os.path.exists(target_mod_input):
            self.report({'ERROR'}, "모드 폴더 경로를 먼저 지정해주세요.")
            return {'CANCELLED'}

        # 물리 리그(XXMI_Physics_Rig)가 존재하면 주입 전 웨이트를 본 위치 기준으로 강제 재베이크 동기화
        rig = getattr(props, "anchor_obj", None) or bpy.data.objects.get("XXMI_Physics_Rig") or bpy.data.objects.get("XXMI_Breast_Rig")
        if rig and rig.type == 'ARMATURE':
            from ..bone.weighting import rebake_mesh_weights_from_rig
            rebake_mesh_weights_from_rig(context, rig, props)

        # 1. 물리 가중치를 가진 모든 메쉬 오브젝트 수집 및 가중치 캐시 구축
        active_bones = [it.name for it in getattr(props, "bone_items", []) if it.name.strip()]
        b0_name = active_bones[0] if len(active_bones) > 0 else "Bone_0"
        b1_name = active_bones[1] if len(active_bones) > 1 else "Bone_1"
        vg_b0 = f"XXMI_{b0_name}"
        vg_b1 = f"XXMI_{b1_name}"
        all_bone_vgs = {f"XXMI_{bn}" for bn in active_bones}
        target_vgs = all_bone_vgs | {vg_b0, vg_b1, "XXMI_Breast_L", "XXMI_Breast_R", "XXMI_Role_Rigid"}

        physics_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and any(o.vertex_groups.get(vgn) for vgn in target_vgs)
        ]

        if not physics_objs:
            self.report({'ERROR'}, "등록된 물리 본 정점이 없습니다. 먼저 1패널에서 정점을 할당해주세요.")
            return {'CANCELLED'}

        mesh_caches = []
        for obj in physics_objs:
            ensure_object_visible(context, obj)
            n_verts = len(obj.data.vertices)
            if n_verts == 0:
                continue

            group_map = {}
            for gname, key in (
                (vg_b0, "l"), ("XXMI_Breast_L", "l"),
                (vg_b1, "r"), ("XXMI_Breast_R", "r"),
                ("XXMI_Role_Rigid", "rigid")
            ):
                vg = obj.vertex_groups.get(gname)
                if vg and vg.index not in group_map:
                    group_map[vg.index] = key
            arrs = read_vertex_group_arrays(obj, group_map)
            zeros = np.zeros(n_verts, dtype=np.float32)
            w_l = arrs.get("l", zeros)
            w_r = arrs.get("r", zeros)
            is_rigid = (arrs.get("rigid", zeros) > 0.5).astype(np.float32)

            weighted = np.nonzero((w_l > 1e-5) | (w_r > 1e-5) | (is_rigid > 0.5))[0]
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
            self.report({'ERROR'}, "물리 가중치가 계산되지 않았습니다. 먼저 3패널에서 [본 & 강체 가중치 자동 계산 & 블렌딩]을 진행해주세요.")
            return {'CANCELLED'}

        global_peak_weight = 0.0
        for mcache in mesh_caches:
            global_peak_weight = max(global_peak_weight, float(mcache["w_l"].max()), float(mcache["w_r"].max()))
        if global_peak_weight < 0.05:
            global_peak_weight = 1.0

        # 2. 대상 INI 파일 수집
        ini_files = []
        if os.path.isfile(target_mod_input) and target_mod_input.lower().endswith('.ini'):
            ini_files.append(target_mod_input)
        elif os.path.isdir(target_mod_input):
            for root, dirs, files in os.walk(target_mod_input):
                for f in files:
                    if f.lower().endswith('.ini'):
                        ini_files.append(os.path.join(root, f))

        if not ini_files:
            self.report({'ERROR'}, "지정된 경로에서 .ini 파일을 찾을 수 없습니다.")
            return {'CANCELLED'}

        total_injected_buffers = 0
        total_modified_inis = 0

        for ini_path in ini_files:
            mod_folder = os.path.dirname(ini_path)

            pos_buffers = _find_position_buffers(ini_path)
            if not pos_buffers:
                continue

            # 3. .ini 파일 백업 (영구 원본 + 타임스탬프 백업)
            backup_ini_file(ini_path)

            # 4. CustomPhysics 서브폴더 생성
            custom_physics_dir = os.path.join(mod_folder, "CustomPhysics")
            os.makedirs(custom_physics_dir, exist_ok=True)

            # 5. 파라미터 및 본 메타데이터 JSON 내보내기
            _export_physics_config_json(custom_physics_dir, props, context)

            matched_in_ini = []

            for item in pos_buffers:
                if len(item) == 7:
                    idx, res_name, raw_fname, buf_v_count, norm_path, blend_sec, draw_sec = item
                    buf_path = norm_path
                else:
                    idx, res_name, raw_fname, blend_sec, draw_sec = item
                    buf_path = os.path.join(mod_folder, raw_fname)

                if not os.path.exists(buf_path):
                    continue

                buf_np, buf_norm = _read_buf_positions_and_normals_np(buf_path)
                n_buf = len(buf_np)
                if n_buf == 0:
                    continue

                kd_buf = KDTree(n_buf)
                for vi, p in enumerate(buf_np):
                    kd_buf.insert(Vector(p), vi)
                kd_buf.balance()

                # 버퍼와 가장 잘 매칭되는 블렌더 메쉬 탐색
                best_cache = None
                best_mat = None
                best_score = float('inf')

                for mcache in mesh_caches:
                    mat, score = _find_best_transform_for_object(kd_buf, mcache["sample_obj"], mcache["obj"])
                    if score < best_score:
                        best_score = score
                        best_mat = mat
                        best_cache = mcache
                    if score < 1e-4:
                        break

                if best_score > 0.10:
                    matched_in_ini.append((idx, res_name, raw_fname, n_buf, False, 0, 0, blend_sec, draw_sec))
                    continue

                # KD-Tree 기반 정점 가중치 보간
                inv_mat = best_mat.inverted()
                buf_in_obj = np.empty_like(buf_np)
                for vi in range(n_buf):
                    buf_in_obj[vi] = (inv_mat @ Vector(buf_np[vi]))[:]

                wco = best_cache["wco"]
                kd_mesh = KDTree(len(wco))
                for vi, p in enumerate(wco):
                    kd_mesh.insert(Vector(p), vi)
                kd_mesh.balance()

                w_l_src = best_cache["w_l"]
                w_r_src = best_cache["w_r"]
                rigid_src = best_cache["rigid"]

                wl_out = np.zeros(n_buf, dtype=np.float32)
                wr_out = np.zeros(n_buf, dtype=np.float32)
                rig_out = np.zeros(n_buf, dtype=np.float32)

                tol = 0.008
                for vi in range(n_buf):
                    pt = Vector(buf_in_obj[vi])
                    hits = kd_mesh.find_range(pt, tol)
                    if hits:
                        min_d = hits[0][2]
                        if min_d < 1e-5:
                            hi = hits[0][1]
                            wl_out[vi] = w_l_src[hi]
                            wr_out[vi] = w_r_src[hi]
                            rig_out[vi] = rigid_src[hi]
                        else:
                            inv_d_sum = sum(1.0 / max(h[2], 1e-5) for h in hits)
                            wl_out[vi] = sum(w_l_src[h[1]] * (1.0 / max(h[2], 1e-5)) for h in hits) / inv_d_sum
                            wr_out[vi] = sum(w_r_src[h[1]] * (1.0 / max(h[2], 1e-5)) for h in hits) / inv_d_sum
                            rig_out[vi] = 1.0 if any(rigid_src[h[1]] > 0.5 for h in hits) else 0.0

                # 1. 좌우 본 영향이 겹치는 중앙 정점 가중치 정규화 (변위 2배 폭발 방지 및 자연스러운 블렌딩)
                sum_w = wl_out + wr_out
                overlap_mask = (wl_out > 0.001) & (wr_out > 0.001) & (sum_w > global_peak_weight)
                if np.any(overlap_mask):
                    scale = global_peak_weight / np.maximum(1e-5, sum_w[overlap_mask])
                    wl_out[overlap_mask] *= scale
                    wr_out[overlap_mask] *= scale

                # 2. Position.buf 3D 버퍼 공간 수직 레이어 Anti-Clipping 후처리
                # 보간 오차로 인해 속살 가중치가 겉옷보다 커지는 역전 현상을 최종 0개로 소멸
                # (수평 단면 거리 3.5mm 이내에서 앞뒤로 겹쳐 있는 수직 레이어만 매칭하여 그라데이션 100% 무손실 보존)
                phys_indices = np.nonzero((wl_out > 0.005) | (wr_out > 0.005))[0]
                if len(phys_indices) > 1:
                    kd_phys = KDTree(len(phys_indices))
                    for k, bi in enumerate(phys_indices):
                        kd_phys.insert(Vector(buf_np[bi]), k)
                    kd_phys.balance()

                    for w_arr in (wl_out, wr_out):
                        if w_arr.max() <= 0.005:
                            continue
                        sub_indices = [k for k in range(len(phys_indices)) if w_arr[phys_indices[k]] > 0.005]
                        if len(sub_indices) < 2:
                            continue

                        # 표면 법선(Normal) 기반 다층 레이어 Anti-Clipping 후처리
                        # 아랫가슴(Underboob), 윗가슴, 가슴골 등 곡면 경사 방향에 무관하게 항상 인체 외측을 정확히 판별
                        orig_w_arr = np.copy(w_arr)

                        for k in sub_indices:
                            bi = phys_indices[k]
                            cur_w = float(orig_w_arr[bi])
                            if cur_w <= 0.005:
                                continue
                            cur_p = buf_np[bi]
                            cur_n = buf_norm[bi]

                            hits = kd_phys.find_range(Vector(cur_p), 0.020)
                            if not hits:
                                continue

                            for _, hit_k, _ in hits:
                                if hit_k == k:
                                    continue
                                hit_bi = phys_indices[hit_k]
                                hit_p = buf_np[hit_bi]
                                hit_n = buf_norm[hit_bi]

                                # 1. 동일한 방향을 바라보는 다층 레이어인지 확인 (노멀 내적 > 0.5)
                                dot_n = float(cur_n[0]*hit_n[0] + cur_n[1]*hit_n[1] + cur_n[2]*hit_n[2])
                                if dot_n < 0.5:
                                    continue

                                # 2. 노멀 방향(법선 벡터)으로의 거리: cur에서 hit로 향하는 거리
                                delta = hit_p - cur_p
                                d_norm = float(delta[0]*cur_n[0] + delta[1]*cur_n[1] + delta[2]*cur_n[2])

                                # 3. 탄젠트 평면(접평면) 상의 수평 거리: 7.5mm 이내 (삼각망 폴리곤 간격 수용)
                                d2 = float(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2])
                                d_perp_sq = max(0.0, d2 - d_norm * d_norm)
                                if d_perp_sq > (0.0075 * 0.0075):
                                    continue

                                # 4. 진정한 바깥쪽 레이어(겉옷) 판정: 법선(Normal) 방향으로 0.3mm 이상 외부에 위치
                                target_w = min(1.0, cur_w * 1.08)  # 8% 안전 마진: 바닐라 스키닝 비틀림 오차를 흡수하는 에어갭 보장
                                if d_norm >= 0.0003 and w_arr[hit_bi] < target_w:
                                    w_arr[hit_bi] = target_w

                max_wl = float(wl_out.max())
                max_wr = float(wr_out.max())
                max_rig = float(rig_out.max())
                peak_l_idx = int(wl_out.argmax()) if max_wl > 1e-4 else 0
                peak_r_idx = int(wr_out.argmax()) if max_wr > 1e-4 else 0

                # CustomPhysicsMap.buf (stride 16: float4 -> wl, wr, is_rigid, root_dist) 바이너리 빌드
                map_bytes = np.column_stack(
                    [wl_out, wr_out, rig_out, np.zeros(n_buf, dtype=np.float32)]
                ).astype('<f4').tobytes()

                has_physics = (max_wl > 0.001 or max_wr > 0.001 or max_rig > 0.001)

                if has_physics:
                    map_buf_path = os.path.join(custom_physics_dir, f"CustomPhysicsMap_{idx}.buf")
                    with open(map_buf_path, 'wb') as f:
                        f.write(map_bytes)

                    # 흉골/몸통 중심 정점 탐색
                    best_root_idx = 0
                    physics_mask = (wl_out > 0.1) | (wr_out > 0.1)
                    if physics_mask.any():
                        center_phys = buf_np[physics_mask].mean(axis=0)
                        root_dist = np.linalg.norm(buf_np - center_phys, axis=1)
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

                matched_in_ini.append((idx, res_name, raw_fname, n_buf, has_physics, peak_l_idx, peak_r_idx, blend_sec, draw_sec))

            if matched_in_ini:
                physics_entries = [e for e in matched_in_ini if e[4]]
                if physics_entries:
                    master_entry = max(physics_entries, key=lambda e: e[3])
                    master_idx = master_entry[0]
                else:
                    master_entry = matched_in_ini[0]
                    master_idx = master_entry[0]

                if target_game == 'HSR':
                    inject_physics_hsr(ini_path, matched_in_ini, props, master_idx)
                else:
                    inject_physics_zzz(ini_path, matched_in_ini, props, master_idx)
                total_modified_inis += 1

        if total_injected_buffers == 0:
            self.report({'WARNING'}, "모드 버퍼와 매칭되는 물리 정점을 찾지 못했습니다. 모드 폴더의 원본 Position.buf 파일들을 확인해주세요.")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"물리 주입 성공! {total_modified_inis}개 INI 파일 수정, {total_injected_buffers}개 메쉬 버퍼에 물리 쉐이더가 정상 주입되었습니다."
        )
        return {'FINISHED'}


class XXMI_OT_rollback_custom_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_rollback_custom_physics"
    bl_label = "물리 효과 롤백 (원본 복원)"
    bl_description = "주입된 물리 효과를 완전히 제거하고 모드를 원본 상태로 복원합니다"

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_mod_input = props.target_mod_dir.strip()

        if props.is_preview_running:
            stop_physics_preview(context)

        if not target_mod_input or not os.path.exists(target_mod_input):
            self.report({'ERROR'}, "모드 폴더 경로를 먼저 지정해주세요.")
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
            bak_path = ini_path + ".bak"

            with open(ini_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()

            clean_lines, modified = _clean_injected_lines(raw_lines)
            if modified:
                with open(ini_path, 'w', encoding='utf-8') as f:
                    f.writelines(clean_lines)
                cleaned_count += 1
            elif os.path.exists(bak_path):
                try:
                    shutil.copy2(bak_path, ini_path)
                    cleaned_count += 1
                except Exception:
                    pass

            # 1. CustomPhysics 서브폴더 삭제
            custom_physics_dir = os.path.join(mod_folder, "CustomPhysics")
            if os.path.isdir(custom_physics_dir):
                shutil.rmtree(custom_physics_dir, ignore_errors=True)
                removed_files += 1

            # 2. 루트에 남아있을 수 있는 커스텀 물리 파일 정리
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

        self.report({'INFO'}, f"롤백 완료: 모드가 주입 전 원본 상태로 복원되었습니다. ({cleaned_count}개 .ini 복원, {removed_files}개 임시 파일 정리)")
        return {'FINISHED'}


class XXMI_OT_load_physics_from_mod(bpy.types.Operator):
    bl_idname = "object.xxmi_load_physics_from_mod"
    bl_label = "모드 폴더에서 물리 설정 불러오기"
    bl_description = "지정된 모드 폴더의 설정 파일에서 물리 가중치와 파라미터를 복원합니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_dir = props.target_mod_dir.strip() if props.target_mod_dir else ""

        if not target_dir or not os.path.exists(target_dir):
            self.report({'ERROR'}, "모드 폴더 경로를 먼저 지정해주세요.")
            return {'CANCELLED'}

        if os.path.isfile(target_dir):
            target_dir = os.path.dirname(target_dir)

        props.target_mod_dir = target_dir
        success, msg = _load_physics_from_mod(target_dir, context, props)
        if not success:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        self.report({'INFO'}, f"{msg}")
        return {'FINISHED'}


class XXMI_OT_physics_dir_picker(bpy.types.Operator):
    bl_idname = "object.xxmi_physics_dir_picker"
    bl_label = "모드 폴더 선택"
    bl_description = "모드 파일이 위치한 폴더를 선택합니다"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        props.target_mod_dir = self.directory
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
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
