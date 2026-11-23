"""Read the scalar metadata of a GGUF file without loading its tensors."""
from __future__ import annotations

import struct

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


def _read(source, size):
    data = source.read(size)
    if len(data) != size:
        raise ValueError("The GGUF header ends early.")
    return data


def _text(source):
    return _read(source, struct.unpack("<Q", _read(source, 8))[0]).decode("utf-8", "replace")


def _value(source, kind):
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, _read(source, struct.calcsize(fmt)))[0]
    if kind == _STRING:
        return _text(source)
    if kind == _ARRAY:
        inner, count = struct.unpack("<IQ", _read(source, 12))
        if inner in _SCALARS:
            # Arrays (token lists, scores) are skipped, never kept.
            _read(source, struct.calcsize(_SCALARS[inner]) * count)
        else:
            for _ in range(count):
                _value(source, inner)
        return None
    raise ValueError(f"Unknown GGUF value type {kind}.")


def metadata(path):
    """Return the scalar and string metadata keys. Array values are omitted."""
    found = {}
    with open(path, "rb") as source:
        if _read(source, 4) != b"GGUF":
            raise ValueError("Not a GGUF file.")
        version, _tensors, count = struct.unpack("<IQQ", _read(source, 20))
        if version < 2:
            raise ValueError("GGUF version 1 is not supported.")
        for _ in range(count):
            key = _text(source)
            value = _value(source, struct.unpack("<I", _read(source, 4))[0])
            if value is not None:
                found[key] = value
    return found


def architecture(path):
    value = metadata(path).get("general.architecture")
    if not isinstance(value, str):
        raise ValueError("The GGUF file names no architecture.")
    return value


def context_length(info):
    value = info.get(f"{info.get('general.architecture')}.context_length")
    return value if isinstance(value, int) and value > 0 else None
