import bpy
import numpy as np
import shutil
import struct
import os
import re
import math
import json
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from ..utils import ensure_object_visible, get_local_axes, read_vertex_group_arrays


def _extract_res_name(rhs):
    val = rhs.strip()
    vl = val.lower()
    for prefix in ("ref ", "copy "):
        if vl.startswith(prefix):
            return val[len(prefix):].strip()
    return val


def _clean_injected_lines(lines):
    out_lines = []
    skip_block = False
    skip_vb_block = False
    skip_motion_block = False
    modified = False

    for line in lines:
        l = line.strip()

        if "; [XXMI-PHYSICS-START]" in l:
            skip_block = True
            modified = True
            if out_lines and out_lines[-1].strip() == "":
                out_lines.pop()
            continue
        if "; [XXMI-PHYSICS-END]" in l:
            skip_block = False
            continue
        if skip_block:
            continue

        if "; [XXMI-PHYSICS-VB-START]" in l:
            skip_vb_block = True
            modified = True
            continue
        if "; [XXMI-PHYSICS-VB-END]" in l:
            skip_vb_block = False
            continue
        if skip_vb_block:
            if "; [XXMI-PHYSICS-ORIG]" in l:
                orig_cmd = l.split("; [XXMI-PHYSICS-ORIG]", 1)[1].strip()
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}{orig_cmd}\n")
            continue

        if "; [XXMI-PHYSICS-MOTION-START]" in l:
            skip_motion_block = True
            modified = True
            continue
        if "; [XXMI-PHYSICS-MOTION-END]" in l:
            skip_motion_block = False
            continue
        if skip_motion_block:
            continue

        ll = l.lower()
        if "resourcephysicsanimated_" in ll or "customshaderphysics_" in ll or "customshaderphysicsmotion_" in ll or "resourceskinnedmesh_" in ll or "resourcelivemesh_" in ll or "resourcephysicsstate_" in ll:
            modified = True
            continue

        out_lines.append(line)

    return out_lines, modified


def backup_ini_file(ini_path):
    """
    물리 주입 전 원본 .ini 파일의 백업본(.bak)을 생성합니다.
    이미 백업본이 존재하면 최초 원본 보존을 위해 덮어쓰지 않습니다.
    단, 현재 .ini 파일이 이미 주입된 상태이고 .bak이 없는 경우라면
    클린업한 원본 구문으로 백업본을 안전하게 생성합니다.
    """
    if not ini_path or not os.path.exists(ini_path):
        return None
    bak_path = ini_path + ".bak"
    if os.path.exists(bak_path):
        return bak_path

    try:
        with open(ini_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        clean_lines, modified = _clean_injected_lines(raw_lines)
        with open(bak_path, 'w', encoding='utf-8') as f:
            f.writelines(clean_lines if modified else raw_lines)
        return bak_path
    except Exception as e:
        print(f"[XXMI] INI 백업 생성 실패 ({ini_path}): {e}")
        return None


def _parse_ini_sections(lines):
    sections = {}
    current_sec = None
    for line in lines:
        l = line.strip()
        if l.startswith('[') and l.endswith(']'):
            current_sec = l[1:-1].strip()
            sections[current_sec] = []
        elif current_sec:
            sections[current_sec].append(line)
    return sections


def _find_draw_section(sections, blend_sec):
    prefix = blend_sec.lower()
    if prefix.endswith('blend'):
        prefix = prefix[:-5]
    elif prefix.endswith('position'):
        prefix = prefix[:-8]

    cand_draw_sec = None
    for s_name, s_lines in sections.items():
        s_lower = s_name.lower()
        if s_lower == blend_sec.lower():
            continue
        if any(s_lower.endswith(suf) for suf in ['blend', 'texcoord', 'vertexlimitraise', 'diffuse', 'lightmap', 'materialmap', 'normalmap', 'ib']):
            continue
        if s_lower.startswith(prefix):
            cand_draw_sec = s_name
            if any('drawindexed' in l.lower() for l in s_lines):
                return s_name

    return cand_draw_sec or blend_sec


def _find_position_buffers(target_dir, mod_ini_path):
    with open(mod_ini_path, 'r', encoding='utf-8') as f:
        ini_lines = f.readlines()

    clean_lines, _ = _clean_injected_lines(ini_lines)
    sections = _parse_ini_sections(clean_lines)

    res_to_filename = {}
    for sec_name, s_lines in sections.items():
        if sec_name.lower().startswith('resource') and not sec_name.lower().startswith('resourcecustomphysicsextracted'):
            for line in s_lines:
                parts = line.split('=', 1)
                if len(parts) == 2 and parts[0].strip().lower() == 'filename':
                    fname = parts[1].strip()
                    res_to_filename[sec_name] = fname

    draw_sec_vb0 = {}
    for sec_name, s_lines in sections.items():
        if not sec_name.lower().startswith('resource'):
            for line in s_lines:
                parts = line.split('=', 1)
                if len(parts) == 2 and parts[0].strip().lower() == 'vb0':
                    res_raw = _extract_res_name(parts[1])
                    if res_raw:
                        draw_sec_vb0[sec_name] = res_raw

    matched_buffers = []
    seen_fnames = set()
    idx = 0

    base_dir = os.path.dirname(mod_ini_path)

    for blend_sec, res_name in draw_sec_vb0.items():
        actual_res = None
        for r_name in res_to_filename.keys():
            if r_name.lower() == res_name.lower():
                actual_res = r_name
                break

        if not actual_res:
            continue

        raw_fname = res_to_filename[actual_res]
        full_path = os.path.join(base_dir, raw_fname)
        norm_path = os.path.normpath(full_path)

        if not os.path.isfile(norm_path):
            continue

        if norm_path.lower() in seen_fnames:
            continue

        # Position.buf 판별 (stride 40)
        file_size = os.path.getsize(norm_path)
        if file_size % 40 == 0 and file_size > 0:
            seen_fnames.add(norm_path.lower())
            draw_sec = _find_draw_section(sections, blend_sec)
            matched_buffers.append((idx, actual_res, raw_fname, file_size // 40, norm_path, blend_sec, draw_sec))
            idx += 1

    return matched_buffers


def _read_buf_positions_np(buf_path):
    """Position.buf (stride 40)의 위치(xyz)만 (n, 3) float64 numpy 배열로 빠르게 읽습니다."""
    data = np.fromfile(buf_path, dtype='<f4')
    n = len(data) // 10
    return data[:n * 10].reshape(n, 10)[:, :3].astype(np.float64)


def _read_buf_positions(buf_path):
    return [Vector(p) for p in _read_buf_positions_np(buf_path).tolist()]


def _find_best_transform_for_object(kd_buf, sample_obj, obj):
    """
    블렌더 오브젝트(로컬 좌표)의 가중치 정점 샘플이 Position.buf 정점들과 가장 잘 겹치는 변환 행렬을 추정합니다.
    (buffer = mat @ local). 오브젝트가 하나의 .buf에 병합된 여러 파츠 중 일부여도 정확히 매칭됩니다.
    """
    mw = obj.matrix_world
    rot_x_neg90 = Matrix.Rotation(math.radians(-90), 4, 'X')
    rot_x_pos90 = Matrix.Rotation(math.radians(90), 4, 'X')
    scale_100 = Matrix.Scale(100.0, 4)
    scale_001 = Matrix.Scale(0.01, 4)

    candidate_mats = [
        Matrix.Identity(4),
        mw,
        rot_x_neg90,
        rot_x_pos90,
        rot_x_neg90 @ mw,
        rot_x_pos90 @ mw,
    ]

    for s_mat in [scale_100, scale_001]:
        candidate_mats.append(s_mat)
        candidate_mats.append(s_mat @ mw)
        candidate_mats.append(rot_x_neg90 @ s_mat)
        candidate_mats.append(rot_x_pos90 @ s_mat)

    best_mat = None
    best_score = 999.0

    for mat in candidate_mats:
        dists = []
        for op in sample_obj:
            _, _, dist = kd_buf.find(mat @ op)
            dists.append(dist)

        dists.sort()
        p80_idx = max(0, int(len(dists) * 0.8) - 1)
        score = dists[p80_idx]
        if score < best_score:
            best_score = score
            best_mat = mat
            if score < 1e-4:
                break

    return best_mat, best_score


def _classify_ini_line(stripped):
    low = stripped.lower()
    if not low or low.startswith(';'):
        return 'skip'
    if re.match(r'^if\b', low):
        return 'if'
    if re.match(r'^(else\s+if|elif)\b', low) or re.match(r'^else\b', low):
        return 'else'
    if re.match(r'^endif\b', low):
        return 'endif'
    if low.startswith('drawindexed') or low.startswith('draw'):
        return 'draw'
    return 'stmt'


def _find_motion_insert_index(lines, sec_name_lower):
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            name = s[1:-1].strip().lower()
            if start is None:
                if name == sec_name_lower:
                    start = i + 1
            else:
                end = i
                break
    if start is None:
        return None

    stack = []
    first_draw = None
    for i in range(start, end):
        kind = _classify_ini_line(lines[i].strip())
        if kind == 'if':
            stack.append(i)
        elif kind == 'endif':
            if stack:
                stack.pop()
        elif kind == 'draw':
            first_draw = i
            break

    if first_draw is None:
        return None
    if not stack:
        return first_draw

    outer = stack[0]
    has_preamble = False
    for i in range(outer + 1, end):
        kind = _classify_ini_line(lines[i].strip())
        if kind == 'stmt':
            has_preamble = True
        elif kind in ('if', 'draw'):
            return i if has_preamble else outer
        elif kind in ('else', 'endif'):
            break
    return outer


def _export_physics_state_buf(export_dir, idx):
    out_path = os.path.join(export_dir, f"CustomPhysicsState_{idx}.buf")
    with open(out_path, 'wb') as f:
        f.write(bytearray(5 * 16))


def _export_physics_config_json(export_dir, props, context):
    """모드 폴더에 현재 물리 파라미터 및 본 앵커 좌표를 담은 XXMI_Physics_Config.json을 저장합니다."""
    rig = props.anchor_obj or bpy.data.objects.get("XXMI_Breast_Rig")
    bones_data = {}
    if rig and rig.type == 'ARMATURE':
        from ..breast.operators import _get_rig_bone_info
        binfo = _get_rig_bone_info(rig)
        for side, b in binfo.items():
            bones_data[b["bname"]] = {
                "head": [round(float(v), 5) for v in b["head"]],
                "tail": [round(float(v), 5) for v in b["tail"]],
                "radius": round(float(b["radius"]), 5),
            }

    config = {
        "version": "1.0",
        "strength": float(getattr(props, "strength", 1.0)),
        "speed": float(getattr(props, "speed", 1.1)),
        "damping": float(getattr(props, "damping", 0.85)),
        "inertia": float(getattr(props, "inertia", 1.2)),
        "idle_motion": float(getattr(props, "idle_motion", 0.4)),
        "falloff_power": float(getattr(props, "falloff_power", 1.6)),
        "target_game": str(getattr(props, "target_game", "ZZZ")),
        "debug_mode": bool(getattr(props, "debug_mode", False)),
        "debug_toggle_key": str(getattr(props, "debug_toggle_key", "h")),
        "bones": bones_data,
    }

    cfg_path = os.path.join(export_dir, "XXMI_Physics_Config.json")
    try:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"XXMI_Physics_Config.json 저장 실패: {e}")


def _load_physics_from_mod(mod_folder, context, props):
    """
    모드 폴더(또는 CustomPhysics 서브폴더)의 XXMI_Physics_Config.json 또는 주입된 HLSL/INI/Map.buf를 읽어
    물리 시뮬레이션 파라미터, 가상 뼈대(XXMI_Breast_Rig), 가슴 및 강체(XXMI_Role_Rigid) 정점 그룹을 100% 복원합니다.
    """
    physics_dir = os.path.join(mod_folder, "CustomPhysics")
    if not os.path.isdir(physics_dir):
        physics_dir = mod_folder

    cfg_path = os.path.join(physics_dir, "XXMI_Physics_Config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(mod_folder, "XXMI_Physics_Config.json")

    loaded_source = ""
    bones_data = {}

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            props.strength = float(cfg.get("strength", props.strength))
            props.speed = float(cfg.get("speed", props.speed))
            props.damping = float(cfg.get("damping", props.damping))
            props.inertia = float(cfg.get("inertia", props.inertia))
            props.idle_motion = float(cfg.get("idle_motion", props.idle_motion))
            props.falloff_power = float(cfg.get("falloff_power", props.falloff_power))
            if cfg.get("target_game") in ('ZZZ', 'HSR'):
                props.target_game = cfg["target_game"]
            props.debug_mode = bool(cfg.get("debug_mode", props.debug_mode))
            props.debug_toggle_key = str(cfg.get("debug_toggle_key", props.debug_toggle_key))
            bones_data = cfg.get("bones", {})
            loaded_source = "XXMI_Physics_Config.json"
        except Exception as e:
            loaded_source = f"JSON 읽기 오류 ({e})"

    if not loaded_source or "오류" in loaded_source:
        parsed_any = False
        search_dirs = [physics_dir] if physics_dir == mod_folder else [physics_dir, mod_folder]
        for sdir in search_dirs:
            for fname in os.listdir(sdir):
                fl = fname.lower()
                if fl.startswith("customphysicsmotion") and fl.endswith(".hlsl"):
                    p = os.path.join(sdir, fname)
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        m_k = re.search(r'STIFF_K\s*=\s*([\d.]+)', txt)
                        m_c = re.search(r'DAMP_C\s*=\s*([\d.]+)', txt)
                        m_i = re.search(r'INERTIA_SCALE\s*=\s*([\d.]+)', txt)
                        m_idle = re.search(r'IDLE_SCALE\s*=\s*([\d.]+)', txt)

                        if m_k:
                            k = float(m_k.group(1))
                            omega_0 = math.sqrt(k)
                            props.speed = round(omega_0 / 12.0, 2)
                            if m_c:
                                c = float(m_c.group(1))
                                props.damping = round(c / (2.0 * omega_0 * 0.55), 2)
                        if m_i:
                            props.inertia = round(float(m_i.group(1)), 2)
                        if m_idle:
                            props.idle_motion = round(float(m_idle.group(1)), 2)
                        parsed_any = True
                    except Exception:
                        pass

                elif fl.startswith("customphysics_") and fl.endswith(".hlsl"):
                    p = os.path.join(sdir, fname)
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        m_s = re.search(r'STRENGTH_SCALE\s*=\s*([\d.]+)', txt)
                        if m_s:
                            props.strength = round(float(m_s.group(1)), 2)
                            parsed_any = True
                    except Exception:
                        pass

                elif fl.endswith(".ini"):
                    p = os.path.join(sdir, fname)
                    try:
                        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                        for line in lines:
                            if "KeyXXMIPhysToggle" in line:
                                props.debug_mode = True
                                parsed_any = True
                            if props.debug_mode and line.strip().lower().startswith("key ="):
                                k_val = line.split("=", 1)[1].strip()
                                if k_val:
                                    props.debug_toggle_key = k_val
                    except Exception:
                        pass

        if parsed_any:
            loaded_source = "주입된 HLSL 및 .ini 파싱"

    if not loaded_source:
        return False, "모드 폴더에서 저장된 물리 설정 파일을 찾지 못했습니다."

    # 1. Map.buf 및 Position.buf 검색 (서브폴더 및 모드 루트)
    map_files = []
    for d in ([physics_dir] if physics_dir == mod_folder else [physics_dir, mod_folder]):
        for f in os.listdir(d):
            if f.lower().startswith("customphysicsmap_") and f.lower().endswith(".buf"):
                full_p = os.path.join(d, f)
                if full_p not in map_files:
                    map_files.append(full_p)

    pos_files = [
        os.path.join(mod_folder, f) for f in os.listdir(mod_folder)
        if f.lower().endswith("position.buf")
    ]

    # Map.buf와 Position.buf를 정점 개수 기준으로 1:1 매칭
    pairs = []
    for mp in map_files:
        m_vc = os.path.getsize(mp) // 16
        for pp in pos_files:
            if os.path.getsize(pp) // 40 == m_vc:
                pairs.append((mp, pp, m_vc))
                break

    # 2. 본 데이터가 JSON에 없는 경우 Map.buf와 Position.buf로부터 역추정
    if not bones_data and pairs:
        all_l_pts = []
        all_r_pts = []

        for mp, pp, map_v_count in pairs:
            with open(mp, 'rb') as f:
                map_data = f.read()
            with open(pp, 'rb') as f:
                pos_data = f.read()

            for i in range(map_v_count):
                wl, wr, _, _ = struct.unpack_from('<4f', map_data, i * 16)
                px, py, pz = struct.unpack_from('<3f', pos_data, i * 40)
                pt = Vector((px, py, pz))
                if wl > 0.15:
                    all_l_pts.append((pt, wl))
                if wr > 0.15:
                    all_r_pts.append((pt, wr))

        for side, pts, bname in (("L", all_l_pts, "Breast.L"), ("R", all_r_pts, "Breast.R")):
            if not pts:
                continue
            total_w = sum(w for _, w in pts)
            cent = sum((p * w for p, w in pts), Vector((0.0, 0.0, 0.0))) / total_w
            # 버퍼 좌표계에서 전방(가슴 돌출 방향)은 +Z 축입니다
            tip = max((p for p, _ in pts), key=lambda p: p.z)
            fwd = tip - cent
            if fwd.length < 0.01:
                fwd = Vector((0.0, 0.06, 0.0))
            else:
                fwd = fwd.normalized() * max(0.04, fwd.length * 1.2)
            head = cent - fwd * 0.4
            tail = cent + fwd * 0.6
            radius = max(0.03, (cent - tip).length * 0.75)
            bones_data[bname] = {
                "head": [head.x, head.y, head.z],
                "tail": [tail.x, tail.y, tail.z],
                "radius": radius,
            }

    # 3. XXMI_Breast_Rig 복원
    restored_rig = False
    rig = None
    if bones_data:
        from ..breast.operators import _get_or_create_breast_rig
        ref_obj = context.active_object or (bpy.data.objects[0] if bpy.data.objects else None)
        rig = _get_or_create_breast_rig(context, props, ref_obj)
        ensure_object_visible(context, rig)
        context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')

        ebones = rig.data.edit_bones
        for bname, bdata in bones_data.items():
            eb = ebones.get(bname) or ebones.new(bname)
            eb.head = Vector(bdata["head"])
            eb.tail = Vector(bdata["tail"])
            radius = float(bdata.get("radius", 0.05))
            eb.head_radius = radius
            eb.tail_radius = radius * 0.7
            eb.envelope_distance = radius * 1.4
            eb["xxmi_bone_type"] = "BREAST"
            eb["xxmi_side"] = "L" if ".L" in bname else "R"

        bpy.ops.object.mode_set(mode='OBJECT')
        props.anchor_obj = rig
        restored_rig = True

    # 4. KD-Tree 3D 정합 기반으로 블렌더 메쉬에 가슴 및 강체(XXMI_Role_Rigid) 정점 그룹 복원
    restored_mesh_names = []
    if pairs:
        pair_caches = []
        for mp, pp, n in pairs:
            with open(mp, 'rb') as f:
                map_data = f.read()
            with open(pp, 'rb') as f:
                pos_data = f.read()

            buf_positions = [Vector(struct.unpack_from('<3f', pos_data, i * 40)) for i in range(n)]
            kd = KDTree(n)
            for i, p in enumerate(buf_positions):
                kd.insert(p, i)
            kd.balance()

            pair_caches.append({
                "map_data": map_data,
                "kd": kd,
                "n": n,
                "pos_name": os.path.basename(pp),
            })

        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            mw = obj.matrix_world
            n_m = len(mesh.vertices)
            if n_m == 0:
                continue

            # 이 메쉬와 가장 일치하는 Position 버퍼 탐색
            best_pcache = None
            best_match_ratio = 0.0
            sample_size = min(30, n_m)
            step = (n_m - 1) / max(1, sample_size - 1)
            sample_pts = [mw @ mesh.vertices[int(k * step)].co for k in range(sample_size)]

            for pcache in pair_caches:
                kd = pcache["kd"]
                matched_s = sum(1 for p in sample_pts if kd.find(p)[2] < 0.05)
                ratio = matched_s / float(sample_size)
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_pcache = pcache

            if not best_pcache or best_match_ratio < 0.5:
                continue

            map_data = best_pcache["map_data"]
            kd = best_pcache["kd"]

            vg_l = obj.vertex_groups.get("XXMI_Breast_L") or obj.vertex_groups.new(name="XXMI_Breast_L")
            vg_r = obj.vertex_groups.get("XXMI_Breast_R") or obj.vertex_groups.new(name="XXMI_Breast_R")
            vg_rigid = obj.vertex_groups.get("XXMI_Role_Rigid") or obj.vertex_groups.new(name="XXMI_Role_Rigid")

            non_zero_breast = 0
            rigid_count = 0

            for vi, v in enumerate(mesh.vertices):
                p_w = mw @ v.co
                _, b_idx, dist = kd.find(p_w)
                if dist < 0.02:
                    wl, wr, vrigid, _ = struct.unpack_from('<4f', map_data, b_idx * 16)
                    if wl > 0.0001:
                        vg_l.add([vi], wl, 'REPLACE')
                        non_zero_breast += 1
                    if wr > 0.0001:
                        vg_r.add([vi], wr, 'REPLACE')
                        non_zero_breast += 1
                    if vrigid > 0.5:
                        vg_rigid.add([vi], 1.0, 'REPLACE')
                        rigid_count += 1

            if non_zero_breast > 0 or rigid_count > 0:
                restored_mesh_names.append(f"{obj.name} (가슴: {non_zero_breast}개, 강체: {rigid_count}개)")

    # 5. 복원 완료 후 본 위치 기반 가중치 재계산 동기화 (강체 그룹 보존)
    if restored_rig and rig:
        from ..breast.operators import _rebake_mesh_weights_from_rig
        _rebake_mesh_weights_from_rig(context, rig, props)

    rig_str = " (물리 본 및 가슴/강체 그룹 복원 완료)" if restored_rig else ""
    mesh_str = f" [복원 메쉬: {', '.join(restored_mesh_names)}]" if restored_mesh_names else ""
    summary = f"물리 설정 불러오기 완료{rig_str}{mesh_str} (강도: {props.strength:.2f}, 속도: {props.speed:.2f}, 탄성: {props.damping:.2f}, 관성: {props.inertia:.2f}, 숨쉬기: {props.idle_motion:.2f})"
    return True, summary
