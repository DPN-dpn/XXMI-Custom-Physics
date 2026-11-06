import os
import re
import math
from .ini_parser import (
    _clean_injected_lines,
    _parse_ini_sections,
    _extract_res_name,
    _find_motion_insert_index,
    backup_ini_file,
)
from ..bone.data import get_bone_params


def export_physics_motion_hlsl_zzz(export_dir, idx, root_vert_indices, props):
    active_bones = [it.name for it in getattr(props, "bone_items", []) if it.name.strip()]
    if not active_bones:
        active_bones = ["Physics_Bone_0"]
    num_bones = len(active_bones)

    bone_configs = []
    for b_i, bn in enumerate(active_bones):
        params = get_bone_params(props, bn)
        sp = max(0.2, float(params["speed"]))
        damp = max(0.1, float(params["damping"]))
        ine = max(0.0, float(params["inertia"]))
        idl = max(0.0, float(params["idle_motion"]))
        omg = 12.0 * sp
        stiff_k = omg * omg
        damp_c = 2.0 * omg * (damp * 0.55)
        rt = root_vert_indices[b_i] if (root_vert_indices and b_i < len(root_vert_indices) and root_vert_indices[b_i] is not None and root_vert_indices[b_i] > 0) else 0
        bone_configs.append((rt, stiff_k, damp_c, ine, idl, sp))

    bone_config_entries = []
    for rt, sk, dc, ine, idl, sp in bone_configs:
        bone_config_entries.append(
            f"    {{ {rt}, {sk:.4f}, {dc:.4f}, {ine:.4f}, {idl:.4f}, {sp:.4f} }}"
        )
    bone_configs_str = ",\n".join(bone_config_entries)

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Torso Inertia Tracker (Feedback-Free ODE Physics, N-Bones)
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

StructuredBuffer<Vertex> LiveMesh : register(t50);
Texture1D<float4> IniParams : register(t120);

RWStructuredBuffer<float4> PhysState : register(u6);

static const uint NUM_BONES = {num_bones};

struct BoneConfig {{
    uint root_idx;
    float stiff_k;
    float damp_c;
    float inertia_scale;
    float idle_scale;
    float speed;
}};

static const BoneConfig BONES[NUM_BONES] = {{
{bone_configs_str}
}};

[numthreads(1, 1, 1)]
void main() {{
    float time = IniParams.Load(int2(87, 0)).x;

    for (uint b = 0; b < NUM_BONES; b++) {{
        uint base_slot = b * 3;
        float4 s0 = PhysState[base_slot + 0];
        float4 s1 = PhysState[base_slot + 1];
        float4 s2 = PhysState[base_slot + 2];

        float3 prev_disp  = s0.xyz;
        float3 spring_vel = s1.xyz;
        float  prev_time  = s1.w;
        float3 prev_root  = s2.xyz;
        float  init_flag  = s2.w;

        BoneConfig cfg = BONES[b];
        float3 curr_root = LiveMesh[cfg.root_idx].position;
        if (isnan(curr_root.x) || isinf(curr_root.x) || length(curr_root) < 1e-4) {{
            curr_root = prev_root;
        }}

        if (init_flag < 0.5) {{
            prev_disp = float3(0, 0, 0);
            spring_vel = float3(0, 0, 0);
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
                    spring_vel = float3(0, 0, 0);
                }}

                float3 raw_vel = delta / dt;
                float3 anim_vel = raw_vel / (1.0 + length(raw_vel) * 0.5);
                float3 ext_force = -anim_vel * (12.0 * cfg.inertia_scale);

                if (cfg.idle_scale > 0.001) {{
                    float breath_t = time * (2.8 * cfg.speed) + float(b) * 0.785398;
                    ext_force += float3(0.0, sin(breath_t) * 1.5, cos(breath_t) * 1.0) * cfg.idle_scale;
                }}

                float sub_dt = dt * 0.25;
                [unroll]
                for (int step = 0; step < 4; step++) {{
                    float3 accel = ext_force - cfg.stiff_k * prev_disp - cfg.damp_c * spring_vel;
                    spring_vel = clamp(spring_vel + accel * sub_dt, -2.5, 2.5);
                    prev_disp = clamp(prev_disp + spring_vel * sub_dt, -0.035, 0.035);
                }}

                prev_root = curr_root;
                prev_time = time;
            }}
        }}

        PhysState[base_slot + 0] = float4(prev_disp, length(spring_vel));
        PhysState[base_slot + 1] = float4(spring_vel, prev_time);
        PhysState[base_slot + 2] = float4(prev_root, init_flag);
    }}
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysicsMotion_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)




def export_vertex_physics_hlsl_zzz(export_dir, idx, vertex_count, props, peak_weight=1.0):
    active_bones = [it.name for it in getattr(props, "bone_items", []) if it.name.strip()]
    if not active_bones:
        active_bones = ["Physics_Bone_0"]
    num_bones = len(active_bones)

    bone_scales = []
    max_str = 0.0
    for bn in active_bones:
        params = get_bone_params(props, bn)
        st = max(0.0, float(params["strength"]))
        pw = float(params["peak_weight"])
        norm = (1.0 / pw) if pw > 0.05 else 1.0
        bone_scales.append(norm * st)
        max_str = max(max_str, st)
    global_max_disp = 0.035 * max(0.1, max_str)
    scales_str = ", ".join(f"{s:.4f}" for s in bone_scales)

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Vertex Skinning Compute Shader (Displacement Pass, N-Bones)
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

StructuredBuffer<Vertex> BaseMesh : register(t52);
StructuredBuffer<WeightData> BoneWeights : register(t53);
StructuredBuffer<float4> PhysState : register(t54);
RWStructuredBuffer<Vertex> Output : register(u5);

static const uint VERTEX_COUNT = {vertex_count};
static const uint NUM_BONES = {num_bones};
static const float BONE_SCALE[NUM_BONES] = {{ {scales_str} }};
static const float GLOBAL_MAX_DISP = {global_max_disp:.4f};

[numthreads(64, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {{
    uint i = threadID.x;
    if (i >= VERTEX_COUNT) return;

    Vertex v = BaseMesh[i];
    WeightData wd = BoneWeights[i];

    float3 total_disp = float3(0, 0, 0);
    float sum_w = 0.0;
    float max_w = 0.0;

    for (uint b = 0; b < NUM_BONES; b++) {{
        float w = wd.weights[b];
        if (w > 0.0001) {{
            float3 b_disp = PhysState[b * 3 + 0].xyz * (w * BONE_SCALE[b]);
            total_disp += b_disp;
            sum_w += w;
            max_w = max(max_w, w);
        }}
    }}

    if (sum_w > 0.0001) {{
        float d_len = length(total_disp);
        if (d_len > GLOBAL_MAX_DISP && d_len > 1e-5) {{
            total_disp = (total_disp / d_len) * GLOBAL_MAX_DISP;
        }}

        v.position += total_disp;

        if (wd.is_rigid < 0.5) {{
            v.normal = normalize(v.normal + total_disp * 2.5);
        }}
    }}

    Output[i] = v;
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysics_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)




def inject_physics_zzz(ini_path, matched_buffers, props, master_idx=None, num_bones=1):
    backup_ini_file(ini_path)
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
    master_buf_filename = master_entry[2]
    master_draw_sec = master_entry[8].lower()

    # blend lookup: res_name.lower() -> entry
    blend_res_map = {entry[1].lower(): entry for entry in breast_entries}

    out_lines = []
    current_sec = None
    motion_injected = False

    motion_insert_idx = _find_motion_insert_index(clean_lines, master_draw_sec)

    for line_no, line in enumerate(clean_lines):
        l_strip = line.strip()
        ll = l_strip.lower()

        if l_strip.startswith('[') and l_strip.endswith(']'):
            current_sec = l_strip[1:-1].strip()
            motion_injected = False
            out_lines.append(line)
            continue

        # Master Draw 섹션에서 토글 분기와 무관하게 항상 실행되는 지점에 단 하나의 통합 모션 추적 패스 주입
        if line_no == motion_insert_idx and not motion_injected:
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

    footer = "; [XXMI-PHYSICS-START]\n"
    if clean_lines and clean_lines[-1].strip():
        footer = "\n" + footer

    if use_debug:
        footer += f"""[Constants]
global persist $xxmi_phys_enable = 1

[KeyXXMIPhysToggle]
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
filename = {master_buf_filename}

[ResourcePhysicsState_{master_idx}]
type = RWStructuredBuffer
stride = 16
filename = CustomPhysics\\CustomPhysicsState_{master_idx}.buf

"""

    active_bones = [it.name for it in getattr(props, "bone_items", []) if it.name.strip()]
    num_bones = max(1, len(active_bones))
    map_stride = (num_bones + 1) * 4

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
stride = {map_stride}
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



