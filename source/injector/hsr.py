import os
import re
import math
from .ini_parser import (
    _clean_injected_lines,
    _parse_ini_sections,
    _extract_res_name,
    backup_ini_file,
)
from ..bone.data import get_bone_params


def export_vertex_physics_hlsl_hsr(export_dir, idx, vertex_count, props, peak_weight=1.0, root_indices=None):
    """
    HSR (Honkai: Star Rail) 단일 패스 물리 HLSL 셰이더 생성.
    임의의 N개 물리 본을 지원하며, 4-Substep Symplectic Euler 물리 적분 및
    각 본별 독립 앵커 추적을 지원합니다.
    """
    active_bones = [it.name for it in getattr(props, "bone_items", []) if it.name.strip()]
    num_bones = max(1, len(active_bones))

    if root_indices is None or len(root_indices) == 0:
        root_indices = [0] * num_bones
    elif len(root_indices) < num_bones:
        root_indices = list(root_indices) + [root_indices[0]] * (num_bones - len(root_indices))

    bone_configs = []
    max_strength = 0.1

    for b_idx in range(num_bones):
        b_name = active_bones[b_idx] if b_idx < len(active_bones) else f"Bone_{b_idx}"
        params = get_bone_params(props, b_name)

        speed = max(0.2, float(params["speed"]))
        damping = max(0.1, float(params["damping"]))
        inertia = max(0.0, float(params["inertia"]))
        idle = max(0.0, float(params["idle_motion"]))
        strength = max(0.0, float(params["strength"]))
        peak_w = float(params["peak_weight"])
        norm = (1.0 / peak_w) if peak_w > 0.05 else 1.0
        scale = norm * strength

        omega = 12.0 * speed
        stiff_k = omega * omega
        damp_c = 2.0 * omega * (damping * 0.55)
        root_v = int(root_indices[b_idx]) if root_indices[b_idx] is not None else 0

        max_strength = max(max_strength, strength)
        bone_configs.append((stiff_k, damp_c, inertia, idle, speed, scale, root_v))

    bones_array_code = ""
    for i, (k, c, inr, idl, spd, sc, rv) in enumerate(bone_configs):
        comma = "," if i < len(bone_configs) - 1 else ""
        bones_array_code += f"        {{ {k:.4f}, {c:.4f}, {inr:.4f}, {idl:.4f}, {spd:.4f}, {sc:.4f}, {rv} }}{comma}\n"

    max_disp = 0.035 * max_strength

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - HSR Compute Shader (N-Bone Architecture)
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

struct WeightData {{
    float weights[{num_bones}];
    float is_rigid;
}};

struct BoneConfig {{
    float stiff_k;
    float damp_c;
    float inertia_scale;
    float idle_scale;
    float speed;
    float scale;
    uint root_idx;
}};

Texture1D<float4> IniParams : register(t120);
StructuredBuffer<Vertex> LiveMesh : register(t50);
StructuredBuffer<Vertex> BaseMesh : register(t52);
StructuredBuffer<WeightData> Weights : register(t53);
RWStructuredBuffer<Vertex> Output : register(u5);
RWStructuredBuffer<float4> PhysState : register(u6);

static const uint VERTEX_COUNT = {vertex_count};
static const uint NUM_BONES = {num_bones};
static const float MAX_DISP = {max_disp:.4f};

static const BoneConfig BONES[NUM_BONES] = {{
{bones_array_code}}};

[numthreads(64, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {{
    uint i = threadID.x;
    if (i >= VERTEX_COUNT) return;

    // 1. 스레드 0: 각 본별 독립 앵커 관성 및 스프링 물리 시뮬레이션 (프레임당 1회 갱신)
    if (i == 0) {{
        float time = IniParams.Load(int2(87, 0)).x;

        for (uint b = 0; b < NUM_BONES; b++) {{
            uint base_idx = b * 3;
            float4 s0 = PhysState[base_idx + 0];
            float4 s1 = PhysState[base_idx + 1];
            float4 s2 = PhysState[base_idx + 2];

            float3 prev_disp  = s0.xyz;
            float3 spring_vel = s1.xyz;
            float  prev_time  = s1.w;
            float3 prev_root  = s2.xyz;
            float  init_flag  = s2.w;

            BoneConfig cfg = BONES[b];
            float3 curr_root = LiveMesh[cfg.root_idx].position;
            float root_len = length(curr_root);
            if (!(root_len > 1e-4 && root_len < 1000.0)) {{
                curr_root = prev_root;
            }}

            if (init_flag < 0.5) {{
                prev_disp  = float3(0.0, 0.0, 0.0);
                spring_vel = float3(0.0, 0.0, 0.0);
                prev_root  = curr_root;
                prev_time  = time;
                init_flag  = 1.0;
            }} else {{
                float raw_dt = time - prev_time;
                if (abs(raw_dt) > 0.0005) {{
                    float dt = clamp(raw_dt, 0.002, 0.04);
                    float3 delta = curr_root - prev_root;

                    if (length(delta) > 0.25 || raw_dt < 0.0 || raw_dt > 0.15) {{
                        delta = float3(0.0, 0.0, 0.0);
                        spring_vel = float3(0.0, 0.0, 0.0);
                    }}

                    float3 raw_vel = delta / dt;
                    float3 anim_vel = raw_vel / (1.0 + length(raw_vel) * 0.35);
                    float3 ext_force = -anim_vel * (14.0 * cfg.inertia_scale);

                    if (cfg.idle_scale > 0.001) {{
                        float breath_t = time * (2.8 * cfg.speed) + float(b) * 0.785398;
                        ext_force += float3(0.0, sin(breath_t) * 1.5, cos(breath_t) * 1.0) * cfg.idle_scale;
                    }}

                    // 4-Substep Symplectic Euler
                    float sub_dt = dt * 0.25;
                    [unroll]
                    for (int step = 0; step < 4; step++) {{
                        float3 accel = ext_force - cfg.stiff_k * prev_disp - cfg.damp_c * spring_vel;
                        spring_vel = clamp(spring_vel + accel * sub_dt, -3.0, 3.0);
                        prev_disp  = clamp(prev_disp + spring_vel * sub_dt, -0.040, 0.040);
                    }}

                    prev_root = curr_root;
                    prev_time = time;
                }}
            }}

            PhysState[base_idx + 0] = float4(prev_disp, length(spring_vel));
            PhysState[base_idx + 1] = float4(spring_vel, prev_time);
            PhysState[base_idx + 2] = float4(prev_root, init_flag);
        }}
    }}

    // 2. 각 정점 변위 적용 (BaseMesh 원본 정점에 가중치 블렌딩)
    Vertex v = BaseMesh[i];
    WeightData wdata = Weights[i];

    float3 total_disp = float3(0.0, 0.0, 0.0);
    float sum_w = 0.0;
    float max_w = 0.0;

    [unroll]
    for (uint b = 0; b < NUM_BONES; b++) {{
        float w = wdata.weights[b];
        if (w > 0.0001) {{
            float3 disp_b = PhysState[b * 3 + 0].xyz * (w * BONES[b].scale);
            total_disp += disp_b;
            sum_w += w;
            max_w = max(max_w, w);
        }}
    }}

    if (sum_w > 0.0001) {{
        if (sum_w > max_w && sum_w > 1e-5) {{
            total_disp *= (max_w / sum_w);
        }}

        float d_len = length(total_disp);
        if (d_len > MAX_DISP && d_len > 1e-5) {{
            total_disp = (total_disp / d_len) * MAX_DISP;
        }}

        v.position += total_disp;

        if (wdata.is_rigid < 0.5) {{
            v.normal = normalize(v.normal + total_disp * 2.5);
        }}
    }}

    Output[i] = v;
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysics_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)
    return out_path


def inject_physics_hsr(ini_path, matched_buffers, props, master_idx=None, num_bones=1):
    """
    HSR .ini 파일에 커스텀 물리 셰이더 및 리소스 주입 (N-Bone 지원)
    """
    if not os.path.exists(ini_path):
        return

    with open(ini_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    clean_lines, _ = _clean_injected_lines(raw_lines)
    toggle_key = getattr(props, "debug_toggle_key", "h").strip() or "h"

    breast_entries = [e for e in matched_buffers if e[4]]
    if not breast_entries:
        return

    if master_idx is None:
        master_entry = max(breast_entries, key=lambda e: e[3])
        master_idx = master_entry[0]
    else:
        master_entry = next((e for e in breast_entries if e[0] == master_idx), breast_entries[0])

    master_buf_filename = master_entry[2]
    master_draw_sec = master_entry[8].strip().lower()

    # blend_sec -> entry 매핑 (대소문자 무시)
    blend_sec_map = {}
    for entry in breast_entries:
        blend_sec_name = entry[7].strip().lower()
        if blend_sec_name:
            blend_sec_map[blend_sec_name] = entry

    out_lines = []
    current_sec = None
    motion_injected = False

    for line in clean_lines:
        l_strip = line.strip()
        ll = l_strip.lower()

        if l_strip.startswith('[') and l_strip.endswith(']'):
            current_sec = l_strip[1:-1].strip()
            motion_injected = False
            out_lines.append(line)
            continue

        # 1. Blend 섹션 (DRAW_TYPE == 8 내부의 Resource\\SRMI\\PositionBuffer 교체)
        if current_sec and current_sec.lower() in blend_sec_map:
            entry = blend_sec_map[current_sec.lower()]
            idx = entry[0]

            if "resource\\srmi\\positionbuffer" in ll:
                parts = l_strip.split('=', 1)
                orig_rhs = parts[1].strip()
                indent = line[:len(line) - len(line.lstrip())]

                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-START]\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-ORIG] {l_strip}\n")
                out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                out_lines.append(f"{indent}    run = CustomShaderPhysics_{idx}\n")
                out_lines.append(f"{indent}    Resource\\SRMI\\PositionBuffer = ref ResourcePhysicsAnimated_{idx}\n")
                out_lines.append(f"{indent}else\n")
                out_lines.append(f"{indent}    {l_strip}\n")
                out_lines.append(f"{indent}endif\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-END]\n")
                continue

        # 2. DrawIndexed 섹션 (실시간 뼈대 애니메이션 관성 추적: LiveMesh = copy vb0)
        if current_sec and current_sec.lower() == master_draw_sec and not motion_injected:
            if ll.startswith("drawindexed") or ll.startswith("draw "):
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-START]\n")
                out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                out_lines.append(f"{indent}    ResourceLiveMesh_{master_idx} = copy vb0\n")
                out_lines.append(f"{indent}endif\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-END]\n")
                motion_injected = True

        out_lines.append(line)

    footer = "; [XXMI-PHYSICS-START]\n"
    if clean_lines and clean_lines[-1].strip():
        footer = "\n" + footer

    footer += f"""[Constants]
global persist $xxmi_phys_enable = 1

[KeyXXMIPhysToggle]
Key = {toggle_key}
type = cycle
$xxmi_phys_enable = 1, 0

[ResourceLiveMesh_{master_idx}]
type = StructuredBuffer
stride = 40
filename = {master_buf_filename}

[ResourcePhysicsState_{master_idx}]
type = RWStructuredBuffer
stride = 16
filename = CustomPhysics/CustomPhysicsState_{master_idx}.buf

"""

    map_stride = (num_bones + 1) * 4

    for entry in breast_entries:
        idx, res_name, buf_filename, v_count, has_breast = entry[:5]
        dispatch_groups = math.ceil(v_count / 64.0)
        orig_cs_res = f"{res_name}CS"

        footer += f"""[CustomShaderPhysics_{idx}]
cs = CustomPhysics/CustomPhysics_{idx}.hlsl
x87 = time
cs-t120 = IniParams
cs-t50 = ref ResourceLiveMesh_{master_idx}
cs-t52 = ref {orig_cs_res}
cs-t53 = ref ResourcePhysicsMap_{idx}
cs-u5 = ResourcePhysicsAnimated_{idx}
cs-u6 = ResourcePhysicsState_{master_idx}
Dispatch = {dispatch_groups}, 1, 1
cs-u5 = null
cs-u6 = null
cs-t50 = null
cs-t52 = null
cs-t53 = null

[ResourcePhysicsAnimated_{idx}]
type = RWStructuredBuffer
stride = 40
filename = {buf_filename}

[ResourcePhysicsMap_{idx}]
type = StructuredBuffer
stride = {map_stride}
filename = CustomPhysics/CustomPhysicsMap_{idx}.buf

"""

    footer += "; [XXMI-PHYSICS-END]\n"
    out_lines.append(footer)

    with open(ini_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
