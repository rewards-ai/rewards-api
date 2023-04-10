import os
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass(kw_only=True)
class GlobalConfig:
    USE_CUDE_IF_AVAILABLE: bool = False
    SCREEN_SIZE: Optional[Tuple] = (800, 700)
    ENABLE_WANDB: Optional[bool] = False
    IN_FEATURES: int = 5
    REWARDS_PARENT_CONFIG_DIR: str = ".rewards_ai"
    REWARDS_CONFIG_MODEL_FOLDER_NAME: str = "session_saved_models"
    REWARDS_CONFIG_METRIC_FOLDER_NAME: str = "session_metrics"
    
    WEBSOCKET_HOST : str = "127.0.0.1"
    WEBSOCKET_SECRET : str = "rewards-ai:v1.0.0:all-rights reserved"
    
    WEBSOCKET_VIDEO_STREAMING_PORT : int = 6520
    WEBSOCKET_VIDEO_STREAMING_ENDPOINT : str = "stream_video"

# Setup environment configuration

ENV_CONFIG = {
    "development": {"DEBUG": True},
    "staging": {"DEBUG": True},
    "production": {
        "DEBUG": False,
    },
}


def get_config(
    parent_config_dir: Optional[str] = ".rewards_ai",
    model_config_dir: Optional[str] = "session_saved_models",
    metric_config_dir: Optional[str] = "session_metrics",
    enable_wandb: Optional[bool] = False,
) -> Dict[str, Any]:
    """Returns the default Global and environment configurations. Some of the configurations
    can be overwritten by the user.

    NOTE:
    Configurations including:
        - USE_CUDA_IF_AVAILABLE,
        - IN_FEATURES,
        - SCREEN_SIZE
    must not be overridden

    The all default configurations looks like this:
    ```json
    {
        "CHECKPOINT_PARENT_PATH": "checkpoints",
        "USE_CUDE_IF_AVAILABLE": false,
        "SCREEN_SIZE": [
            800,
            700
        ],
        "ENABLE_WANDB": false,
        "IN_FEATURES": 5,
        "DEBUG": true,
        "ENV": "development",
        "DEVICE": "cpu"
    }
    ```

    Returns:
        Dict[str, Any]: Global and environment configurations
    """

    # determine the running configuration

    ENV = os.environ["PYTHON_ENV"] if "PYTHON_ENV" in os.environ else "development"
    ENV = ENV or "development"

    # raise error if environment for the unexpected
    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f"Config for environment {ENV} not found")

    global_config = GlobalConfig(
        ENABLE_WANDB=enable_wandb,
        REWARDS_PARENT_CONFIG_DIR=parent_config_dir,
        REWARDS_CONFIG_MODEL_FOLDER_NAME=model_config_dir,
        REWARDS_CONFIG_METRIC_FOLDER_NAME=metric_config_dir,
    ).__dict__.copy()

    global_config.update(ENV_CONFIG[ENV])
    global_config["ENV"] = ENV
    global_config["DEVICE"] = (
        "cuda" if torch.cuda.is_available() and global_config["USE_CUDE_IF_AVAILABLE"] else "cpu"
    )

    return global_config


# load this variable while loading the API directly

CONFIG = get_config()

if __name__ == "__main__":
    import json

    print(json.dumps(CONFIG, indent=4))
