from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import struct


MAGIC = b"ACTDICT\x00"  # 8 bytes, 'ACTDICT' + NUL


@dataclass
class ActDictHeader:
    version: int
    fs: float
    length: int
    complex_mode: bool
    param_ranges: Dict[str, float]
    dict_size: int


def read_act_dict_header(path: str) -> Optional[ActDictHeader]:
    """Read the header of a saved ACT dictionary file written by ACT::save_dictionary.

    Returns None if the file is not a valid ACT dictionary.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            (fs,) = struct.unpack("<d", f.read(8))
            (length,) = struct.unpack("<i", f.read(4))
            (complex_u8,) = struct.unpack("<B", f.read(1))
            # Read ParameterRanges: 12 doubles
            pr_vals = struct.unpack("<12d", f.read(8 * 12))
            keys = [
                "tc_min", "tc_max", "tc_step",
                "fc_min", "fc_max", "fc_step",
                "logDt_min", "logDt_max", "logDt_step",
                "c_min", "c_max", "c_step",
            ]
            pr = {k: v for k, v in zip(keys, pr_vals)}
            (dict_size,) = struct.unpack("<i", f.read(4))
            return ActDictHeader(
                version=version,
                fs=fs,
                length=length,
                complex_mode=bool(complex_u8),
                param_ranges=pr,
                dict_size=dict_size,
            )
    except Exception:
        return None


def summarize_header(h: ActDictHeader) -> str:
    return (
        f"ACT Dictionary v{h.version}: fs={h.fs} Hz, length={h.length} samples, "
        f"complex={h.complex_mode}, dict_size={h.dict_size}\n"
        f"Ranges: tc[{h.param_ranges['tc_min']},{h.param_ranges['tc_max']}] step {h.param_ranges['tc_step']}; "
        f"fc[{h.param_ranges['fc_min']},{h.param_ranges['fc_max']}] step {h.param_ranges['fc_step']}; "
        f"logDt[{h.param_ranges['logDt_min']},{h.param_ranges['logDt_max']}] step {h.param_ranges['logDt_step']}; "
        f"c[{h.param_ranges['c_min']},{h.param_ranges['c_max']}] step {h.param_ranges['c_step']}"
    )
