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
        num_bones = max(1, len(active_bones))

        from ..bone.rig import get_rig_bone_info
        bone_info = get_rig_bone_info(rig) if rig else {}

        bone_heads_world = []
        for bi in range(num_bones):
            bn = active_bones[bi] if bi < len(active_bones) else None
            h = bone_info.get(bn, {}).get("head") if bn else None
            if h is None and bi == 0 and "L" in bone_info:
                h = bone_info["L"].get("head")
            elif h is None and bi == 1 and "R" in bone_info:
                h = bone_info["R"].get("head")
            bone_heads_world.append(Vector(h) if h is not None else None)

        all_bone_vgs = {f"XXMI_{bn}" for bn in active_bones}
        target_vgs = all_bone_vgs | {"XXMI_Breast_L", "XXMI_Breast_R", "XXMI_Role_Rigid"}

        physics_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and any(o.vertex_groups.get(vgn) for vgn in target_vgs)
        ]

        if not physics_objs:
            self.report({'ERROR'}, "등록된 물리 본 정점이 없습니다. 먼저 1패널에서 정점을 할당해주세요.")
            return {'CANCELLED'}

        mesh_caches = []
        all_world_pts = []
        all_bone_weights = [[] for _ in range(num_bones)]
        all_rigid_weights = []

        for obj in physics_objs:
            ensure_object_visible(context, obj)
            n_verts = len(obj.data.vertices)
            if n_verts == 0:
                continue

            group_map = {}
            for bi, bn in enumerate(active_bones):
                vg = obj.vertex_groups.get(f"XXMI_{bn}")
                if vg:
                    group_map[vg.index] = f"bone_{bi}"

            # 레거시 버텍스 그룹 호환
            if 0 < num_bones and "bone_0" not in group_map.values():
                vg_l = obj.vertex_groups.get("XXMI_Breast_L")
                if vg_l:
                    group_map[vg_l.index] = "bone_0"
            if 1 < num_bones and "bone_1" not in group_map.values():
                vg_r = obj.vertex_groups.get("XXMI_Breast_R")
                if vg_r:
                    group_map[vg_r.index] = "bone_1"

            vg_rig = obj.vertex_groups.get("XXMI_Role_Rigid")
            if vg_rig:
                group_map[vg_rig.index] = "rigid"

            arrs = read_vertex_group_arrays(obj, group_map)
            zeros = np.zeros(n_verts, dtype=np.float32)

            w_bones_list = [arrs.get(f"bone_{bi}", zeros) for bi in range(num_bones)]
            is_rigid = (arrs.get("rigid", zeros) > 0.5).astype(np.float32)

            co = np.empty(n_verts * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)

            # 오브젝트 로컬 -> 월드 좌표 변환
            mw = np.array(obj.matrix_world, dtype=np.float64)
            co_h = np.column_stack([co, np.ones(n_verts, dtype=np.float64)])
            co_world = (co_h @ mw.T)[:, :3]

            # 메쉬 전체 정점을 통합 월드 좌표계에 등록 (가중치 0.0 정점 포함)
            # 0.0 정점을 포함해야 버퍼 매칭 시 외곽 경계선 주변 체간부로 가중치가 무단 번지는 현상을 100% 원천 차단
            all_world_pts.append(co_world)
            for bi in range(num_bones):
                all_bone_weights[bi].append(w_bones_list[bi])
            all_rigid_weights.append(is_rigid)

            active_any = (is_rigid > 0.5)
            for wb in w_bones_list:
                active_any = active_any | (wb > 1e-5)
            weighted = np.nonzero(active_any)[0]
            if len(weighted) < 10:
                sample_idx = np.linspace(0, n_verts - 1, min(300, n_verts)).astype(np.int64)
            else:
                n_sample = min(300, len(weighted))
                sample_idx = weighted[np.linspace(0, len(weighted) - 1, n_sample).astype(np.int64)]
            sample_obj = [Vector(p) for p in co[sample_idx].tolist()]

            mesh_caches.append({
                "obj": obj,
                "sample_obj": sample_obj,
            })

        if not all_world_pts:
            self.report({'ERROR'}, "물리 가중치가 계산되지 않았습니다. 먼저 3패널에서 [본 & 강체 가중치 자동 계산 & 블렌딩]을 진행해주세요.")
            return {'CANCELLED'}

        unified_world_pts = np.vstack(all_world_pts)
        unified_bone_weights = [np.concatenate(all_bone_weights[bi]) for bi in range(num_bones)]
        unified_rigid_weights = np.concatenate(all_rigid_weights)

        # 블렌더 월드 정점 기반 통합 KD-Tree 구축
        kd_unified = KDTree(len(unified_world_pts))
        for vi, p in enumerate(unified_world_pts):
            kd_unified.insert(Vector(p), vi)
        kd_unified.balance()

        global_peak_weight = 0.0
        for wb in unified_bone_weights:
            if len(wb) > 0:
                global_peak_weight = max(global_peak_weight, float(wb.max()))
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

                # KD-Tree 기반 정점 가중치 보간 (통합 KD-Tree 활용)
                inv_mat = best_mat.inverted()
                buf_in_world = np.empty_like(buf_np)
                for vi in range(n_buf):
                    buf_in_world[vi] = (inv_mat @ Vector(buf_np[vi]))[:]

                w_out_list = [np.zeros(n_buf, dtype=np.float32) for _ in range(num_bones)]
                rig_out = np.zeros(n_buf, dtype=np.float32)

                for vi in range(n_buf):
                    pt = Vector(buf_in_world[vi])
                    res = kd_unified.find(pt)
                    if res is not None and res[0] is not None:
                        near_p, near_idx, near_d = res
                        if near_d < 0.015:  # 1.5cm 이내 정점만 가중치 매칭
                            if near_d < 0.0008:  # 0.8mm 이내 정확 일치 (동일 정점)
                                for bi in range(num_bones):
                                    w_out_list[bi][vi] = unified_bone_weights[bi][near_idx]
                                rig_out[vi] = unified_rigid_weights[near_idx]
                            else:
                                hits = kd_unified.find_range(pt, 0.005)
                                if hits:
                                    inv_d_sum = sum(1.0 / max(h[2], 1e-5) for h in hits)
                                    for bi in range(num_bones):
                                        w_out_list[bi][vi] = sum(unified_bone_weights[bi][h[1]] * (1.0 / max(h[2], 1e-5)) for h in hits) / inv_d_sum
                                    rig_out[vi] = 1.0 if any(unified_rigid_weights[h[1]] > 0.5 for h in hits) else 0.0
                                else:
                                    for bi in range(num_bones):
                                        w_out_list[bi][vi] = unified_bone_weights[bi][near_idx]
                                    rig_out[vi] = unified_rigid_weights[near_idx]

                # 1. 인접 본 영향이 겹치는 부위 가중치 정규화 (변위 폭발 방지 및 자연스러운 블렌딩)
                sum_w = sum(w_out_list)
                overlap_mask = (sum_w > global_peak_weight)
                if np.any(overlap_mask):
                    scale = global_peak_weight / np.maximum(1e-5, sum_w[overlap_mask])
                    for bi in range(num_bones):
                        w_out_list[bi][overlap_mask] *= scale

                # 2. .ib 인덱스 버퍼 기반 삼각망 찢어짐(Tearing) 완화 (Boundary Gradient Taper):
                ib_candidates = [
                    os.path.splitext(buf_path)[0] + ".ib",
                    os.path.join(mod_folder, raw_fname.replace("Position.buf", ".ib")),
                    os.path.join(mod_folder, raw_fname.replace("Position", ".ib")),
                ]
                base_stem = os.path.splitext(raw_fname)[0].replace("Position", "").replace("Blend", "").replace("TexCoord", "").lower()
                for root, _, files in os.walk(mod_folder):
                    for f in files:
                        if f.lower().endswith('.ib'):
                            f_stem = os.path.splitext(f)[0].lower()
                            if base_stem in f_stem or f_stem in base_stem:
                                p_cand = os.path.join(root, f)
                                if p_cand not in ib_candidates:
                                    ib_candidates.append(p_cand)

                all_tris_list = []
                for p_ib in ib_candidates:
                    if os.path.exists(p_ib):
                        try:
                            with open(p_ib, 'rb') as f_ib:
                                ib_raw = f_ib.read()
                            ib_tris = np.frombuffer(ib_raw, dtype=np.uint32)
                            if len(ib_tris) == 0 or ib_tris.max() >= n_buf:
                                ib_tris = np.frombuffer(ib_raw, dtype=np.uint16)
                            if len(ib_tris) >= 3 and ib_tris.max() < n_buf:
                                all_tris_list.append(ib_tris.reshape(-1, 3))
                        except Exception:
                            pass

                if all_tris_list:
                    tris = np.vstack(all_tris_list)
                    adj_buf = [[] for _ in range(n_buf)]
                    for t0, t1, t2 in tris:
                        adj_buf[t0].append(t1); adj_buf[t0].append(t2)
                        adj_buf[t1].append(t0); adj_buf[t1].append(t2)
                        adj_buf[t2].append(t0); adj_buf[t2].append(t1)

                    for w_arr in w_out_list:
                        for _ in range(3):
                            active_b = np.nonzero(w_arr > 0.01)[0]
                            for vi in active_b:
                                nbrs = adj_buf[vi]
                                if not nbrs:
                                    continue
                                min_nb = min(w_arr[nb] for nb in nbrs)
                                diff = w_arr[vi] - min_nb
                                if diff > 0.12:
                                    w_arr[vi] -= (diff - 0.12) * 0.40

                # 3. [절대 최종 단계]: Position.buf 내 모든 동일 위치(UV 심/하드 엣지 분리점) 100% 완전 동기화
                # 3DMigoto 버퍼 상의 분리 정점(Coincident Vertices) 가중치 불일치로 인한 구멍 및 찢어짐 원천 차단
                kd_co = KDTree(n_buf)
                for vi, p in enumerate(buf_np):
                    kd_co.insert(Vector(p), vi)
                kd_co.balance()

                parent = list(range(n_buf))
                def find_p(i):
                    path = []
                    while parent[i] != i:
                        path.append(i)
                        i = parent[i]
                    for node in path:
                        parent[node] = i
                    return i
                def union_p(i, j):
                    pi, pj = find_p(i), find_p(j)
                    if pi != pj:
                        parent[pi] = pj

                active_any_buf = (rig_out > 0.0001)
                for wb in w_out_list:
                    active_any_buf = active_any_buf | (wb > 0.0001)
                active_buf_idx = np.nonzero(active_any_buf)[0]

                for vi in active_buf_idx:
                    hits = kd_co.find_range(Vector(buf_np[vi]), 0.0002)
                    for _, hit_k, _ in hits:
                        union_p(vi, hit_k)

                co_groups_map = {}
                for vi in range(n_buf):
                    root = find_p(vi)
                    co_groups_map.setdefault(root, []).append(vi)

                for root, g_list in co_groups_map.items():
                    if len(g_list) > 1 and any(active_any_buf[v] for v in g_list):
                        max_rig = max(float(rig_out[v]) for v in g_list)
                        max_w = [max(float(w_out_list[bi][v]) for v in g_list) for bi in range(num_bones)]
                        for v in g_list:
                            rig_out[v] = max_rig
                            for bi in range(num_bones):
                                w_out_list[bi][v] = max_w[bi]

                max_bone_weights = [float(wb.max()) for wb in w_out_list]
                has_physics = any(mw > 0.001 for mw in max_bone_weights) or (float(rig_out.max()) > 0.001)

                if has_physics:
                    # CustomPhysicsMap_{idx}.buf 바이너리 빌드 (stride: (num_bones + 1) * 4)
                    map_cols = [*w_out_list, rig_out]
                    map_bytes = np.column_stack(map_cols).astype('<f4').tobytes()
                    map_buf_path = os.path.join(custom_physics_dir, f"CustomPhysicsMap_{idx}.buf")
                    with open(map_buf_path, 'wb') as f:
                        f.write(map_bytes)

                    # 슬롯별(본 0 ~ 본 N-1) 독립 신체 앵커 정점 탐색
                    non_phys_mask = (sum(w_out_list) < 0.01)
                    root_indices = []

                    for bi in range(num_bones):
                        root_idx = 0
                        wb = w_out_list[bi]
                        b_head_w = bone_heads_world[bi] if bi < len(bone_heads_world) else None

                        if b_head_w is not None:
                            c = np.array((best_mat @ b_head_w)[:], dtype=np.float64)
                        elif (wb > 0.05).any():
                            c = buf_np[wb > 0.05].mean(axis=0)
                        elif wb.max() > 1e-4:
                            c = buf_np[wb.argmax()]
                        else:
                            c = buf_np.mean(axis=0)

                        dists = np.linalg.norm(buf_np - c, axis=1)
                        dists[~non_phys_mask] = np.inf
                        if np.isfinite(dists).any():
                            root_idx = int(dists.argmin())
                        root_indices.append(root_idx)

                    # 런타임 물리 상태 버퍼 초기화 (.buf, num_bones * 3 * 16 바이트)
                    _export_physics_state_buf(custom_physics_dir, idx, num_bones)

                    # 게임별 분기 쉐이더 내보내기 (ZZZ vs HSR)
                    if target_game == 'HSR':
                        export_vertex_physics_hlsl_hsr(custom_physics_dir, idx, n_buf, props, global_peak_weight, root_indices)
                    else:
                        export_physics_motion_hlsl_zzz(custom_physics_dir, idx, root_indices, props)
                        export_vertex_physics_hlsl_zzz(custom_physics_dir, idx, n_buf, props, global_peak_weight)

                    total_injected_buffers += 1

                peak_0 = int(w_out_list[0].argmax()) if max_bone_weights[0] > 1e-4 else 0
                peak_1 = int(w_out_list[1].argmax()) if (num_bones > 1 and max_bone_weights[1] > 1e-4) else peak_0
                matched_in_ini.append((idx, res_name, raw_fname, n_buf, has_physics, peak_0, peak_1, blend_sec, draw_sec))

            if matched_in_ini:
                physics_entries = [e for e in matched_in_ini if e[4]]
                if physics_entries:
                    master_entry = max(physics_entries, key=lambda e: e[3])
                    master_idx = master_entry[0]
                else:
                    master_entry = matched_in_ini[0]
                    master_idx = master_entry[0]

                if target_game == 'HSR':
                    inject_physics_hsr(ini_path, matched_in_ini, props, master_idx, num_bones)
                else:
                    inject_physics_zzz(ini_path, matched_in_ini, props, master_idx, num_bones)
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
