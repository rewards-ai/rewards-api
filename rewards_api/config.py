import os 
from dataclasses import dataclass 
from typing import Optional

@dataclass(kw_only=True)
class GlobalConfig:
    DEVICE: str = "cpu"
    IN_FEATURES: int = 5
    TEST_TRIALS: int = 5
    MODEL_NAME: str = "model.pth"
    REWARDS_PARENT_CONFIG_DIR: str = ".rewards_ai"
    REWARDS_CONFIG_MODEL_FOLDER_NAME: str = "session_saved_models"
    REWARDS_CONFIG_METRIC_FOLDER_NAME: str = "session_metrics"
    
    TRAINING_METRICS_JSON_NAME : str = "training_metrics.json"
    EVALUATION_METRICS_JSON_NAME : str = "evaluation_metrics.json"
    MODEL_HISTORY_JSON_NAME : str = "model_history.json"

    EVALUATION_TOTAL_TIME_FOR_TRAINING_PATH : int = 30
    EVALUATION_TOTAL_TIME_FOR_EVALUATION_PATH : int = 60
    
    STREAMING_TEMP_JSON_PATH: str = "/home/anindya/workspace/RewardsHQ/RewardsSuit/training-platform/src/assets/temp.json"

CONFIG = GlobalConfig().__dict__.copy() 