import bpy
import struct
import os
import re
import math
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from ..utils import ensure_object_visible, get_local_axes
from ..simulation.operators import stop_physics_preview


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


def _read_buf_positions(buf_path):
    positions = []
    with open(buf_path, 'rb') as f:
        data = f.read()
    count = len(data) // 40
    for i in range(count):
        px, py, pz = struct.unpack_from('<3f', data, i * 40)
        positions.append(Vector((px, py, pz)))
    return positions


def _find_best_transform_for_buffer(sample_buf, kd_mesh, obj):
    """
    모드 익스포트 좌표계와 블렌더 메쉬 좌표계 간 최적 변환 행렬을 추정합니다.
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
        try:
            inv_mat = mat.inverted()
        except Exception:
            continue

        dists = []
        for bp in sample_buf:
            local_p = inv_mat @ bp
            _, _, dist = kd_mesh.find(local_p)
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


def _export_physics_state_buf(export_dir, idx):
    """
    물리 상태 버퍼(5 x float4 = 80 바이트)를 0으로 초기화하여 생성합니다.
    [0]: disp_l (xyz), speed_l (w)
    [1]: disp_r (xyz), speed_r (w)
    [2]: spring_vel_l (xyz), prev_time (w)
    [3]: spring_vel_r (xyz), init_flag (w)
    [4]: prev_root (xyz), 0 (w)
    """
    out_path = os.path.join(export_dir, f"CustomPhysicsState_{idx}.buf")
    with open(out_path, 'wb') as f:
        f.write(bytearray(5 * 16))


def _export_physics_motion_hlsl(export_dir, idx, root_vert_idx, props):
    strength = float(getattr(props, "strength", 1.0))
    speed = float(getattr(props, "speed", 1.1))
    damping = float(getattr(props, "damping", 0.85))
    inertia = float(getattr(props, "inertia", 1.2))
    idle_motion = float(getattr(props, "idle_motion", 0.8))
    use_debug = bool(getattr(props, "debug_mode", False))
    debug_boost = float(getattr(props, "debug_boost", 1.0)) if use_debug else 1.0

    omega_0 = 12.0 * speed
    stiff_k = omega_0 * omega_0
    damp_c = 2.0 * omega_0 * damping
    amp_scale = strength * debug_boost

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Torso Inertia Tracker (Feedback-Free ODE Physics)
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

StructuredBuffer<Vertex> LiveMesh : register(t50);
Texture1D<float4> IniParams : register(t120);

RWStructuredBuffer<float4> PhysState : register(u6);

static const uint ROOT_VERTEX_IDX = {root_vert_idx};

static const float STIFF_K = {stiff_k:.4f};
static const float DAMP_C = {damp_c:.4f};
static const float INERTIA_SCALE = {inertia:.4f};
static const float STRENGTH_SCALE = {amp_scale:.4f};
static const float IDLE_SCALE = {idle_motion:.4f};

[numthreads(1, 1, 1)]
void main() {{
    float time = IniParams.Load(int2(87, 0)).x;

    float4 s0 = PhysState[0];
    float4 s1 = PhysState[1];
    float4 s2 = PhysState[2];
    float4 s3 = PhysState[3];
    float4 s4 = PhysState[4];

    float3 prev_disp_l = s0.xyz;
    float3 prev_disp_r = s1.xyz;
    float3 spring_vel_l = s2.xyz;
    float  prev_time    = s2.w;
    float3 spring_vel_r = s3.xyz;
    float  init_flag    = s3.w;
    float3 prev_root    = s4.xyz;

    float3 curr_root = LiveMesh[ROOT_VERTEX_IDX].position;

    if (length(curr_root) < 1e-4) {{
        float breath_t = time * 2.5;
        float3 fb_disp = float3(0, sin(breath_t) * 0.025, cos(breath_t) * 0.015) * (IDLE_SCALE * STRENGTH_SCALE);
        PhysState[0] = float4(fb_disp, 0.5);
        PhysState[1] = float4(fb_disp, 0.5);
        return;
    }}

    if (init_flag < 0.5) {{
        prev_disp_l = float3(0, 0, 0);
        prev_disp_r = float3(0, 0, 0);
        spring_vel_l = float3(0, 0, 0);
        spring_vel_r = float3(0, 0, 0);
        prev_root = curr_root;
        prev_time = time;
        init_flag = 1.0;
    }} else {{
        float raw_dt = time - prev_time;
        if (abs(raw_dt) > 0.0005) {{
            float dt = clamp(raw_dt, 0.002, 0.04);

            float3 delta = curr_root - prev_root;

            if (length(delta) > 0.25 || raw_dt < 0.0 || raw_dt > 0.15) {{
                delta = float3(0, 0, 0);
                spring_vel_l = float3(0, 0, 0);
                spring_vel_r = float3(0, 0, 0);
            }}

            float3 anim_vel = delta / dt;
            float3 ext_force = -anim_vel * (16.0 * INERTIA_SCALE);

            if (IDLE_SCALE > 0.001) {{
                float breath_t = time * 2.5;
                float3 idle_f = float3(0, sin(breath_t) * 1.5, cos(breath_t) * 0.9) * IDLE_SCALE;
                ext_force += idle_f;
            }}

            float3 ext_force_l = ext_force;
            float3 ext_force_r = float3(-ext_force.x * 0.9, ext_force.y, ext_force.z);

            float3 accel_l = ext_force_l - STIFF_K * prev_disp_l - DAMP_C * spring_vel_l;
            float3 accel_r = ext_force_r - STIFF_K * prev_disp_r - DAMP_C * spring_vel_r;

            spring_vel_l = clamp(spring_vel_l + accel_l * dt, -4.0, 4.0);
            spring_vel_r = clamp(spring_vel_r + accel_r * dt, -4.0, 4.0);

            prev_disp_l = clamp(prev_disp_l + spring_vel_l * dt, -0.045, 0.045);
            prev_disp_r = clamp(prev_disp_r + spring_vel_r * dt, -0.045, 0.045);

            prev_root = curr_root;
            prev_time = time;
        }}
    }}

    float3 disp_l = prev_disp_l * STRENGTH_SCALE;
    float3 disp_r = prev_disp_r * STRENGTH_SCALE;

    PhysState[0] = float4(disp_l, length(spring_vel_l));
    PhysState[1] = float4(disp_r, length(spring_vel_r));
    PhysState[2] = float4(spring_vel_l, prev_time);
    PhysState[3] = float4(spring_vel_r, init_flag);
    PhysState[4] = float4(prev_root, 0);
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysicsMotion_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)


def _export_vertex_physics_hlsl(export_dir, idx, vertex_count, props, peak_weight=1.0):
    norm_factor = (1.0 / peak_weight) if peak_weight > 0.05 else 1.0

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Breast Skinning Compute Shader (Displacement Pass)
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

StructuredBuffer<Vertex> BaseMesh : register(t52);
StructuredBuffer<float4> BreastWeights : register(t53);
StructuredBuffer<float4> PhysState : register(t54);
RWStructuredBuffer<Vertex> Output : register(u5);

static const uint VERTEX_COUNT = {vertex_count};
static const float NORM_FACTOR = {norm_factor:.4f};

[numthreads(64, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {{
    uint i = threadID.x;
    if (i >= VERTEX_COUNT) return;

    Vertex v = BaseMesh[i];
    float4 bw = BreastWeights[i];
    float wl = bw.x;
    float wr = bw.y;
    float total_w = wl + wr;

    if (total_w > 0.0001) {{
        float3 disp_l = PhysState[0].xyz;
        float3 disp_r = PhysState[1].xyz;

        float3 total_disp = (disp_l * wl + disp_r * wr) * NORM_FACTOR;

        float max_disp = 0.050;
        float d_len = length(total_disp);
        if (d_len > max_disp && d_len > 1e-5) {{
            total_disp = (total_disp / d_len) * max_disp;
        }}

        v.position += total_disp;

        // 노멀 및 탄젠트 보정
        v.normal = normalize(v.normal + total_disp * 2.5);
    }}

    Output[i] = v;
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysics_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)


def _inject_physics_into_ini(ini_path, matched_buffers, props, master_idx=None):
    with open(ini_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    clean_lines, _ = _clean_injected_lines(raw_lines)

    use_debug = bool(getattr(props, "debug_mode", False))
    toggle_key = getattr(props, "debug_toggle_key", "h").strip() or "h"

    breast_entries = [e for e in matched_buffers if e[4]]
    if not breast_entries:
        return

    master_entry = max(breast_entries, key=lambda e: e[3])
    master_idx = master_entry[0]
    master_draw_sec = master_entry[8].lower()

    # blend lookup: res_name.lower() -> entry
    blend_res_map = {entry[1].lower(): entry for entry in breast_entries}

    has_constants = any(l.strip().lower() == "[constants]" for l in clean_lines)

    phys_globals_block = [
        "; [XXMI-PHYSICS-START]\n",
        "global persist $xxmi_phys_enable = 1\n",
        "; [XXMI-PHYSICS-END]\n",
    ]

    out_lines = []
    constants_injected = False
    current_sec = None
    motion_injected = False

    for line in clean_lines:
        l_strip = line.strip()
        ll = l_strip.lower()

        if l_strip.startswith('[') and l_strip.endswith(']'):
            current_sec = l_strip[1:-1].strip()
            motion_injected = False
            out_lines.append(line)

            if current_sec.lower() == "constants" and not constants_injected:
                out_lines.extend(phys_globals_block)
                constants_injected = True
            continue

        # Master Draw 섹션 내부에서 drawindexed / draw 호출 직전에 단 하나의 통합 모션 추적 패스 주입
        if current_sec and current_sec.lower() == master_draw_sec and not motion_injected:
            if ll.startswith("drawindexed") or ll.startswith("draw"):
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-START]\n")
                if use_debug:
                    out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                    out_lines.append(f"{indent}    ResourceLiveMesh_{master_idx} = copy vb0\n")
                    out_lines.append(f"{indent}    run = CustomShaderPhysicsMotion_{master_idx}\n")
                    out_lines.append(f"{indent}endif\n")
                else:
                    out_lines.append(f"{indent}ResourceLiveMesh_{master_idx} = copy vb0\n")
                    out_lines.append(f"{indent}run = CustomShaderPhysicsMotion_{master_idx}\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-END]\n")
                motion_injected = True

        # Blend 섹션의 vb0 교체
        if (ll.startswith("vb0 ") or ll.startswith("vb0=")) and "resourcephysicsanimated_" not in ll:
            res_val = _extract_res_name(line.split("=", 1)[1])
            if res_val.lower() in blend_res_map:
                entry = blend_res_map[res_val.lower()]
                idx = entry[0]
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-START]\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-ORIG] {l_strip}\n")
                if use_debug:
                    out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                    out_lines.append(f"{indent}    run = CustomShaderPhysics_{idx}\n")
                    out_lines.append(f"{indent}    vb0 = ResourcePhysicsAnimated_{idx}\n")
                    out_lines.append(f"{indent}else\n")
                    out_lines.append(f"{indent}    {l_strip}\n")
                    out_lines.append(f"{indent}endif\n")
                else:
                    out_lines.append(f"{indent}run = CustomShaderPhysics_{idx}\n")
                    out_lines.append(f"{indent}vb0 = ResourcePhysicsAnimated_{idx}\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-END]\n")
                continue

        out_lines.append(line)

    footer = "\n; [XXMI-PHYSICS-START]\n"
    if not has_constants:
        footer += "[Constants]\n" + "".join(phys_globals_block[1:-1]) + "\n"

    if use_debug:
        footer += f"""[KeyXXMIPhysToggle]
Key = {toggle_key}
type = cycle
$xxmi_phys_enable = 1, 0

"""

    # Master Motion Shader & Unified Physics State Buffer
    footer += f"""[CustomShaderPhysicsMotion_{master_idx}]
cs = CustomPhysics\\CustomPhysicsMotion_{master_idx}.hlsl
x87 = time
w85 = $xxmi_phys_enable
cs-t120 = IniParams
cs-t50 = ref ResourceLiveMesh_{master_idx}
cs-u6 = ResourcePhysicsState_{master_idx}
Dispatch = 1, 1, 1
cs-u6 = null
cs-t50 = null
cs-t120 = null

[ResourceLiveMesh_{master_idx}]
type = StructuredBuffer
stride = 40

[ResourcePhysicsState_{master_idx}]
type = RWStructuredBuffer
stride = 16
filename = CustomPhysics\\CustomPhysicsState_{master_idx}.buf

"""

    # 각 메쉬별 변위 적용 쉐이더 (모든 메쉬가 Master PhysicsState를 함께 참조하여 100% 동기화)
    for entry in breast_entries:
        idx, res_name, buf_filename, v_count, has_breast = entry[:5]
        dispatch_groups = math.ceil(v_count / 64.0)
        footer += f"""[CustomShaderPhysics_{idx}]
cs = CustomPhysics\\CustomPhysics_{idx}.hlsl
w85 = $xxmi_phys_enable
cs-t52 = ref ResourceCustomPhysicsExtracted_{idx}
cs-t53 = ref ResourcePhysicsMap_{idx}
cs-t54 = ref ResourcePhysicsState_{master_idx}
cs-u5 = copy ResourceCustomPhysicsExtracted_{idx}
ResourcePhysicsAnimated_{idx} = ref cs-u5
Dispatch = {dispatch_groups}, 1, 1
cs-u5 = null
cs-t52 = null
cs-t53 = null
cs-t54 = null

[ResourcePhysicsAnimated_{idx}]

[ResourcePhysicsMap_{idx}]
type = StructuredBuffer
stride = 16
filename = CustomPhysics\\CustomPhysicsMap_{idx}.buf

[ResourceCustomPhysicsExtracted_{idx}]
type = StructuredBuffer
stride = 40
filename = {buf_filename}

"""

    footer += "; [XXMI-PHYSICS-END]\n"
    out_lines.append(footer)

    with open(ini_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)


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

        if props.is_preview_running:
            stop_physics_preview(context)

        if not target_mod_input or not os.path.exists(target_mod_input):
            self.report({'ERROR'}, "타겟 모드 폴더(또는 .ini 파일) 경로를 올바르게 지정해주세요.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # XXMI_Breast_Rig 아머처가 존재하면 사용자가 수동 조정한 본 위치를 버텍스 가중치에 자동 반영
        rig = props.anchor_obj or bpy.data.objects.get("XXMI_Breast_Rig")
        if rig and rig.type == 'ARMATURE':
            from ..breast.operators import _rebake_mesh_weights_from_rig
            _rebake_mesh_weights_from_rig(context, rig, props)

        # 1. 가슴 웨이트를 가진 모든 메쉬 오브젝트 수집 및 KDTree/가중치 캐시 구축
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
            mesh = obj.data
            n_verts = len(mesh.vertices)
            vg_l_idx = obj.vertex_groups.find("XXMI_Breast_L")
            vg_r_idx = obj.vertex_groups.find("XXMI_Breast_R")
            vg_rigid_idx = obj.vertex_groups.find("XXMI_Role_Rigid")

            w_l = [0.0] * n_verts
            w_r = [0.0] * n_verts
            is_rigid = [0.0] * n_verts

            for v in mesh.vertices:
                for g in v.groups:
                    if g.group == vg_l_idx:
                        w_l[v.index] = g.weight
                    elif g.group == vg_r_idx:
                        w_r[v.index] = g.weight
                    elif g.group == vg_rigid_idx and g.weight > 0.5:
                        is_rigid[v.index] = 1.0

            kd = KDTree(n_verts)
            for i, v in enumerate(mesh.vertices):
                kd.insert(v.co, i)
            kd.balance()

            mesh_caches.append({
                "obj": obj,
                "kd": kd,
                "w_l": w_l,
                "w_r": w_r,
                "is_rigid": is_rigid,
            })

        # 타겟 폴더 내 유효 .ini 파일 수집
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
                buf_positions = _read_buf_positions(norm_path)
                if not buf_positions:
                    continue

                sample_size = min(300, len(buf_positions))
                sample_buf = [buf_positions[int(k * (len(buf_positions) - 1) / (sample_size - 1))] for k in range(sample_size)]

                best_match = None
                best_overall_score = 999.0

                for mcache in mesh_caches:
                    cand_obj = mcache["obj"]
                    b_mat, b_score = _find_best_transform_for_buffer(sample_buf, mcache["kd"], cand_obj)
                    if b_score < best_overall_score:
                        best_overall_score = b_score
                        best_match = (mcache, b_mat, b_score)

                if not best_match or best_overall_score > 0.05:
                    continue

                mcache, best_mat, best_score = best_match
                matched_obj = mcache["obj"]
                kd = mcache["kd"]
                w_l = mcache["w_l"]
                w_r = mcache["w_r"]
                is_rigid = mcache["is_rigid"]
                matched_obj_names.add(matched_obj.name)

                inv_mat = best_mat.inverted()

                # 버퍼 정점별 가중치 매핑: stride 16 (float4: wl, wr, is_rigid, 0.0)
                map_bytes = bytearray()
                max_wl = 0.0
                max_wr = 0.0
                peak_l_idx = 0
                peak_r_idx = 0

                for bp_idx, bp in enumerate(buf_positions):
                    local_p = inv_mat @ bp
                    _, v_idx, dist = kd.find(local_p)

                    if dist > 0.05:
                        vl = 0.0
                        vr = 0.0
                        vrigid = 0.0
                    else:
                        vl = w_l[v_idx]
                        vr = w_r[v_idx]
                        vrigid = is_rigid[v_idx]

                    if vl > max_wl:
                        max_wl = vl
                        peak_l_idx = bp_idx
                    if vr > max_wr:
                        max_wr = vr
                        peak_r_idx = bp_idx

                    map_bytes.extend(struct.pack('<4f', vl, vr, vrigid, 0.0))

                # 사용자가 직접 선택한 가슴 영역 가중치(XXMI_Breast_L / XXMI_Breast_R) 존재 여부 확인
                has_breast = (max_wl > 0.001 or max_wr > 0.001)

                if has_breast:
                    map_buf_path = os.path.join(custom_physics_dir, f"CustomPhysicsMap_{idx}.buf")
                    with open(map_buf_path, 'wb') as f:
                        f.write(map_bytes)

                    # 흉골/몸통 중심 정점(가슴 가중치가 0이면서 양 가슴 중심에 가장 가까운 기준점) 탐색
                    breast_pts = []
                    for bp_idx, bp in enumerate(buf_positions):
                        wl, wr, _, _ = struct.unpack_from('<4f', map_bytes, bp_idx * 16)
                        if wl > 0.1 or wr > 0.1:
                            breast_pts.append(bp)

                    best_root_idx = 0
                    if breast_pts:
                        center_breast = sum(breast_pts, Vector((0, 0, 0))) / len(breast_pts)
                        best_root_dist = 999.0
                        for bp_idx, bp in enumerate(buf_positions):
                            wl, wr, _, _ = struct.unpack_from('<4f', map_bytes, bp_idx * 16)
                            if wl < 0.01 and wr < 0.01:
                                dist = (bp - center_breast).length
                                if dist < best_root_dist:
                                    best_root_dist = dist
                                    best_root_idx = bp_idx

                    # 1. 런타임 물리 상태 버퍼 초기화 (.buf)
                    _export_physics_state_buf(custom_physics_dir, idx)
                    # 2. 순수 몸통 애니메이션 가속도 추적 ODE 물리 쉐이더 (.hlsl)
                    _export_physics_motion_hlsl(custom_physics_dir, idx, best_root_idx, props)
                    # 3. 계산된 변위를 정점에 적용하는 스키닝 쉐이더 (.hlsl)
                    _export_vertex_physics_hlsl(custom_physics_dir, idx, len(buf_positions), props, max(max_wl, max_wr))
                    total_injected_buffers += 1

                matched_in_ini.append((idx, res_name, raw_fname, len(buf_positions), has_breast, peak_l_idx, peak_r_idx, blend_sec, draw_sec))

            if matched_in_ini:
                _inject_physics_into_ini(ini_path, matched_in_ini, props)
                _export_physics_config_json(custom_physics_dir, props, context)

        if total_injected_buffers > 0:
            names_str = ", ".join(matched_obj_names)
            self.report({'INFO'}, f"성공: {total_injected_buffers}개 Position 버퍼에 움직임 기반 가슴 물리 주입 완료! (매칭된 오브젝트: {names_str})")
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
                   (fl == "xxmi_physics_config.json"):
                    try:
                        os.remove(os.path.join(mod_folder, fname))
                        removed_files += 1
                    except Exception:
                        pass

        self.report({'INFO'}, f"롤백 완료: {cleaned_count}개 .ini 복원, {removed_files}개 생성 파일 삭제됨.")
        return {'FINISHED'}


def _export_physics_config_json(export_dir, props, context):
    """모드 폴더에 현재 물리 파라미터 및 본 앵커 좌표를 담은 XXMI_Physics_Config.json을 저장합니다."""
    import json
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
        "idle_motion": float(getattr(props, "idle_motion", 0.8)),
        "falloff_power": float(getattr(props, "falloff_power", 1.6)),
        "auto_detect_rigid": bool(getattr(props, "auto_detect_rigid", True)),
        "rigid_max_verts": int(getattr(props, "rigid_max_verts", 120)),
        "debug_mode": bool(getattr(props, "debug_mode", False)),
        "debug_toggle_key": str(getattr(props, "debug_toggle_key", "h")),
        "debug_boost": float(getattr(props, "debug_boost", 1.0)),
        "bones": bones_data,
    }

    cfg_path = os.path.join(export_dir, "XXMI_Physics_Config.json")
    try:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


def _load_physics_from_mod(mod_folder, context, props):
    """
    모드 폴더(또는 CustomPhysics 서브폴더)의 XXMI_Physics_Config.json 또는 주입된 HLSL/INI/Map.buf를 읽어
    물리 시뮬레이션 파라미터, 가상 뼈대(XXMI_Breast_Rig), 가슴 및 강체 장신구(XXMI_Role_Rigid) 정점 그룹을 100% 복원합니다.
    """
    import json, re, math
    from mathutils.kdtree import KDTree

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
            props.auto_detect_rigid = bool(cfg.get("auto_detect_rigid", props.auto_detect_rigid))
            props.rigid_max_verts = int(cfg.get("rigid_max_verts", props.rigid_max_verts))
            props.debug_mode = bool(cfg.get("debug_mode", props.debug_mode))
            props.debug_toggle_key = str(cfg.get("debug_toggle_key", props.debug_toggle_key))
            props.debug_boost = float(cfg.get("debug_boost", props.debug_boost))
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
                                props.damping = round(c / (2.0 * omega_0), 2)
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
        return False, "모드 폴더에서 바스트모핑 설정 파일이나 쉐이더를 찾지 못했습니다."

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
            tip = max((p for p, _ in pts), key=lambda p: p.y)
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

    # 4. KD-Tree 3D 정합 기반으로 블렌더 메쉬에 가슴 및 강체 장신구(XXMI_Role_Rigid) 정점 그룹 복원
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

    # 5. 복원 완료 후 본 위치 기반 가중치 재계산 동기화 (강체 장신구 그룹 보존)
    if restored_rig and rig:
        from ..breast.operators import _rebake_mesh_weights_from_rig
        _rebake_mesh_weights_from_rig(context, rig, props)

    rig_str = " (가상 rig 및 가슴/강체 그룹 복원 완료)" if restored_rig else ""
    mesh_str = f" [복원 메쉬: {', '.join(restored_mesh_names)}]" if restored_mesh_names else ""
    summary = f"[{loaded_source}] 파라미터 로드 완료{rig_str}{mesh_str} - 강도: {props.strength:.2f}, 속도: {props.speed:.2f}, 감쇠: {props.damping:.2f}, 관성: {props.inertia:.2f}, 호흡: {props.idle_motion:.2f}"
    return True, summary


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
        # 폴더 내에 기존 물리 설정이 존재하면 자동으로 불러오기 시도
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
