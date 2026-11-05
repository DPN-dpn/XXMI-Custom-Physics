import os
import re
from datetime import datetime


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
    물리 주입 전 원본 .ini 파일의 백업본을 생성합니다.
    1. 최초 1회: 순수 바닐라 원본 보존용 'filename.ini.bak' 영구 생성 (있으면 유지)
    2. 매 주입 시: 변경 이력 보존용 'filename.ini.YYYYMMDD_HHMMSS.bak' 타임스탬프 백업 계속 누적 생성
    """
    if not ini_path or not os.path.exists(ini_path):
        return None

    try:
        with open(ini_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        clean_lines, modified = _clean_injected_lines(raw_lines)
        base_lines = clean_lines if modified else raw_lines

        # 1. 영구 원본 백업 (.bak)이 없으면 최초 생성
        pristine_bak = ini_path + ".bak"
        if not os.path.exists(pristine_bak):
            with open(pristine_bak, 'w', encoding='utf-8') as f:
                f.writelines(base_lines)

        # 2. 매 주입 시점의 타임스탬프 백업 생성 (.YYYYMMDD_HHMMSS.bak)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_bak = f"{ini_path}.{timestamp}.bak"
        counter = 1
        while os.path.exists(ts_bak):
            ts_bak = f"{ini_path}.{timestamp}_{counter}.bak"
            counter += 1

        with open(ts_bak, 'w', encoding='utf-8') as f:
            f.writelines(raw_lines)

        return ts_bak
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


def _find_position_buffers(target_dir, mod_ini_path=None):
    """
    모드 .ini 파일에서 Position.buf 버퍼와 Draw/Blend 섹션 매핑을 파싱하여 반환합니다.
    _find_position_buffers(ini_path) 및 _find_position_buffers(mod_folder, ini_path) 호출 방식을 모두 지원합니다.
    """
    if mod_ini_path is None:
        mod_ini_path = target_dir
        base_dir = os.path.dirname(mod_ini_path) if os.path.isfile(mod_ini_path) else mod_ini_path
    else:
        base_dir = target_dir or os.path.dirname(mod_ini_path)

    try:
        with open(mod_ini_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            ini_lines = f.readlines()
    except Exception:
        with open(mod_ini_path, 'r', encoding='utf-8', errors='ignore') as f:
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

        file_size = os.path.getsize(norm_path)
        if file_size % 40 == 0 and file_size > 0:
            seen_fnames.add(norm_path.lower())
            draw_sec = _find_draw_section(sections, blend_sec)
            matched_buffers.append((idx, actual_res, raw_fname, file_size // 40, norm_path, blend_sec, draw_sec))
            idx += 1

    return matched_buffers


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

