from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
