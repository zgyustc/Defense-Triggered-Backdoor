"""Configuration bridge for the extracted experiment implementations."""
import json
import os
from dataclasses import asdict, is_dataclass


def configure(config):
    overrides = json.loads(os.environ.get("DTB_CONFIG_JSON", "{}"))
    values = config if isinstance(config, dict) else asdict(config)
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if isinstance(config, dict):
        config.update(overrides)
    elif is_dataclass(config):
        for key, value in overrides.items():
            setattr(config, key, value)
    else:
        raise TypeError("Expected a dictionary or dataclass configuration")
    print("[Resolved configuration]", json.dumps(
        config if isinstance(config, dict) else asdict(config),
        ensure_ascii=False, sort_keys=True, default=str,
    ))
    return config
