import os
import torch


# Config that serves all environment
GLOBAL_CONFIG = {
    "INIT_FROM": "gpt2-medium",
    "USE_CUDA_IF_AVAILABLE": True,
}


def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    config = GLOBAL_CONFIG.copy()
    config['DEVICE'] = 'cuda' if torch.cuda.is_available() and config['USE_CUDA_IF_AVAILABLE'] else 'cpu'

    return config

def update_config(config, key, value) -> dict:
    """
    Update config
    :return: dict of config
    """
    assert key in {'INIT_FROM', 'USE_CUDA_IF_AVAILABLE', 'DEVICE'}
    config = GLOBAL_CONFIG.copy()
    config[key] = value
    return config

# load config for import
CONFIG = get_config()

