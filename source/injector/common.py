# Re-export all injector submodules for backward compatibility and clean architecture
from .ini_parser import (
    _extract_res_name,
    _clean_injected_lines,
    backup_ini_file,
    _parse_ini_sections,
    _find_draw_section,
    _find_position_buffers,
    _classify_ini_line,
    _find_motion_insert_index,
)
from .buffer_io import (
    _read_buf_positions_np,
    _read_buf_positions,
    _find_best_transform_for_object,
    _export_physics_state_buf,
    _export_physics_config_json,
)
from .loader import (
    _load_physics_from_mod,
)

__all__ = [
    "_extract_res_name",
    "_clean_injected_lines",
    "backup_ini_file",
    "_parse_ini_sections",
    "_find_draw_section",
    "_find_position_buffers",
    "_classify_ini_line",
    "_find_motion_insert_index",
    "_read_buf_positions_np",
    "_read_buf_positions",
    "_find_best_transform_for_object",
    "_export_physics_state_buf",
    "_export_physics_config_json",
    "_load_physics_from_mod",
]
