import os
import re
import json
import math
import struct
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import ensure_object_visible
from ..bone.rig import get_or_create_breast_rig
from ..bone.weighting import rebake_mesh_weights_from_rig


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
            bone_items_data = cfg.get("bone_items", [])
            if bone_items_data and hasattr(props, "bone_items"):
                props.bone_items.clear()
                from ..bone.data import set_bone_data
                for bd in bone_items_data:
                    it = props.bone_items.add()
                    it.name = str(bd.get("name", "모핑 본"))
                    it.bone_type = str(bd.get("bone_type", "MORPH"))
                    it.chain_segments = int(bd.get("chain_segments", 4))
                    it.gravity_factor = float(bd.get("gravity_factor", 1.0))
                    it.stiffness = float(bd.get("stiffness", 0.5))
                    it.peak_weight = float(bd.get("peak_weight", getattr(props, "peak_weight", 0.40)))
                    it.falloff_power = float(bd.get("falloff_power", getattr(props, "falloff_power", 1.0)))
                    it.smooth_iterations = int(bd.get("smooth_iterations", 2))
                    it.lock_registered_verts = bool(bd.get("lock_registered_verts", True))
                    it.strength = float(bd.get("strength", getattr(props, "strength", 1.0)))
                    it.speed = float(bd.get("speed", getattr(props, "speed", 1.1)))
                    it.damping = float(bd.get("damping", getattr(props, "damping", 0.85)))
                    it.inertia = float(bd.get("inertia", getattr(props, "inertia", 1.2)))
                    it.idle_motion = float(bd.get("idle_motion", getattr(props, "idle_motion", 0.4)))
                    it.vert_count = int(bd.get("vert_count", 0))
                    if "data" in bd and isinstance(bd["data"], dict) and bd["data"]:
                        set_bone_data(it, bd["data"])
                        vg_name = f"XXMI_{it.name}"
                        for oname, vis in bd["data"].items():
                            obj = bpy.data.objects.get(oname)
                            if not obj or obj.type != 'MESH':
                                if context.active_object and context.active_object.type == 'MESH':
                                    obj = context.active_object
                            if obj and obj.type == 'MESH' and vis:
                                vg = obj.vertex_groups.get(vg_name) or obj.vertex_groups.new(name=vg_name)
                                valid_vis = [v for v in vis if v < len(obj.data.vertices)]
                                if valid_vis:
                                    vg.add(valid_vis, it.peak_weight, 'REPLACE')
                                if not props.target_obj:
                                    props.target_obj = obj

            rigid_items_data = cfg.get("rigid_items", [])
            if hasattr(props, "rigid_items"):
                props.rigid_items.clear()
                if rigid_items_data:
                    from ..rigid.data import set_rigid_data
                    for rd in rigid_items_data:
                        rit = props.rigid_items.add()
                        rit.name = str(rd.get("name", "강체 부품"))
                        rit.weight_scale = float(rd.get("weight_scale", 1.0))
                        rit.vert_count = int(rd.get("vert_count", 0))
                        if "data" in rd and isinstance(rd["data"], dict) and rd["data"]:
                            set_rigid_data(rit, rd["data"])
                            for oname, vis in rd["data"].items():
                                obj = bpy.data.objects.get(oname)
                                if not obj or obj.type != 'MESH':
                                    if context.active_object and context.active_object.type == 'MESH':
                                        obj = context.active_object
                                if obj and obj.type == 'MESH' and vis:
                                    vg = obj.vertex_groups.get("XXMI_Role_Rigid") or obj.vertex_groups.new(name="XXMI_Role_Rigid")
                                    valid_vis = [v for v in vis if v < len(obj.data.vertices)]
                                    if valid_vis:
                                        vg.add(valid_vis, 1.0, 'REPLACE')
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
        ref_obj = context.active_object or (bpy.data.objects[0] if bpy.data.objects else None)
        rig = get_or_create_breast_rig(context, props, ref_obj)
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
            b_item = next((it for it in props.bone_items if it.name == bname), None)
            eb["xxmi_bone_type"] = b_item.bone_type if b_item else "MORPH"
            eb["xxmi_side"] = "L" if (".L" in bname or "왼" in bname) else "R"

        bpy.ops.object.mode_set(mode='OBJECT')
        props.anchor_obj = rig
        restored_rig = True

    # 4. KD-Tree 3D 정합 기반으로 블렌더 메쉬에 물리 본 및 강체(XXMI_Role_Rigid) 정점 그룹 복원
    restored_mesh_names = []
    if pairs:
        pair_caches = []
        all_b0_verts = {}
        all_b1_verts = {}
        all_rigid_verts = {}
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

            b0 = props.bone_items[0] if len(props.bone_items) > 0 else None
            b1 = props.bone_items[1] if len(props.bone_items) > 1 else None
            vg_l_name = f"XXMI_{b0.name}" if b0 else "XXMI_Breast_L"
            vg_r_name = f"XXMI_{b1.name}" if b1 else "XXMI_Breast_R"

            vg_l = obj.vertex_groups.get(vg_l_name) or obj.vertex_groups.new(name=vg_l_name)
            vg_r = obj.vertex_groups.get(vg_r_name) or obj.vertex_groups.new(name=vg_r_name)
            vg_rigid = obj.vertex_groups.get("XXMI_Role_Rigid") or obj.vertex_groups.new(name="XXMI_Role_Rigid")

            non_zero_breast = 0
            rigid_count = 0
            b0_vis = []
            b1_vis = []
            rigid_vis = []

            for vi, v in enumerate(mesh.vertices):
                p_w = mw @ v.co
                _, b_idx, dist = kd.find(p_w)
                if dist < 0.02:
                    wl, wr, vrigid, _ = struct.unpack_from('<4f', map_data, b_idx * 16)
                    if wl > 0.0001:
                        vg_l.add([vi], wl, 'REPLACE')
                        b0_vis.append(vi)
                        non_zero_breast += 1
                    if wr > 0.0001:
                        vg_r.add([vi], wr, 'REPLACE')
                        b1_vis.append(vi)
                        non_zero_breast += 1
                    if vrigid > 0.5:
                        vg_rigid.add([vi], 1.0, 'REPLACE')
                        rigid_vis.append(vi)
                        rigid_count += 1

            if b0 and b0_vis:
                all_b0_verts[obj.name] = b0_vis
            if b1 and b1_vis:
                all_b1_verts[obj.name] = b1_vis
            if rigid_vis:
                all_rigid_verts[obj.name] = rigid_vis

            if non_zero_breast > 0 or rigid_count > 0:
                restored_mesh_names.append(f"{obj.name} (물리본: {non_zero_breast}개, 강체: {rigid_count}개)")

        if b0 and all_b0_verts and (not b0.data_json or b0.data_json == "{}"):
            from ..bone.data import set_bone_data
            set_bone_data(b0, all_b0_verts)
        if b1 and all_b1_verts and (not b1.data_json or b1.data_json == "{}"):
            from ..bone.data import set_bone_data
            set_bone_data(b1, all_b1_verts)
        if all_rigid_verts and len(props.rigid_items) == 0:
            from ..rigid.data import set_rigid_data
            r_item = props.rigid_items.add()
            r_item.name = "복원된 강체"
            set_rigid_data(r_item, all_rigid_verts)

    # 4.5. 메쉬 오브젝트에 이미 존재하는 버텍스 그룹을 기반으로 본 슬롯 정점 데이터(data_json) 자동 동기화
    from ..bone.data import get_bone_data, set_bone_data
    for it in props.bone_items:
        if not get_bone_data(it):
            cand_vgs = [f"XXMI_{it.name}"]
            if "왼" in it.name or ".L" in it.name.upper() or "_L" in it.name.upper():
                cand_vgs.extend(["XXMI_Breast_L", "XXMI_Breast.L"])
            elif "오" in it.name or ".R" in it.name.upper() or "_R" in it.name.upper():
                cand_vgs.extend(["XXMI_Breast_R", "XXMI_Breast.R"])

            recovered = {}
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    vg = None
                    for vgn in cand_vgs:
                        vg = obj.vertex_groups.get(vgn)
                        if vg:
                            break
                    if vg:
                        vis = []
                        for vi in range(len(obj.data.vertices)):
                            try:
                                if vg.weight(vi) > 0.0001:
                                    vis.append(vi)
                            except Exception:
                                pass
                        if vis:
                            recovered[obj.name] = vis
            if recovered:
                set_bone_data(it, recovered)

    # 강체 슬롯도 마찬가지로 정점 데이터 동기화
    from ..rigid.data import get_rigid_data, set_rigid_data
    for rit in props.rigid_items:
        if not get_rigid_data(rit):
            recovered_rigid = {}
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    vg = obj.vertex_groups.get("XXMI_Role_Rigid")
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
                set_rigid_data(rit, recovered_rigid)

    # 5. 복원 완료 후 본 위치 기반 가중치 재계산 동기화
    if restored_rig and rig:
        rebake_mesh_weights_from_rig(context, rig, props)

    rig_str = " (물리 본 및 가슴/강체 그룹 복원 완료)" if restored_rig else ""
    mesh_str = f" [복원 메쉬: {', '.join(restored_mesh_names)}]" if restored_mesh_names else ""
    summary = f"물리 설정 불러오기 완료{rig_str}{mesh_str} (강도: {props.strength:.2f}, 속도: {props.speed:.2f}, 탄성: {props.damping:.2f}, 관성: {props.inertia:.2f}, 숨쉬기: {props.idle_motion:.2f})"
    return True, summary

