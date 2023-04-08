import os
import ast
import time
import json
import pygame
import multiprocessing as mp
from typing import Dict, Any

import src.utils as utils
from src.config import CONFIG

from rewards import workflow, QTrainer, LinearQNet, CarGame


class RewardsStreamer:
    def __init__(self, session_id, response: Dict[str, Any]) -> None:
        """
        RewardsStreamer is the class which is responsible for streaming, saving
        metrics and also streaming frames over the network.
        """
        self.session_id = session_id
        self.config = workflow.WorkFlowConfigurations(
            # training parameters
            REWARD_FUNCTION=response["training_params"]["reward_function"], 
            ENABLE_WANDB=False,
            
            # environment parameters
            ENVIRONMENT_NAME=response["env_params"]["environment_name"],
            ENVIRONMENT_WORLD=int(response["env_params"]["environment_world"]),
            MODE=response["env_params"]["mode"],
            CAR_SPEED=response["env_params"]["car_speed"],
            
            # agent parameters
            LAYER_CONFIG=ast.literal_eval(response["agent_params"]["model_configuration"]),
            LR=response["agent_params"]["learning_rate"],
            LOSS=response["agent_params"]["loss_fn"],
            OPTIMIZER=response["agent_params"]["optimizer"],
            NUM_EPISODES=response["agent_params"]["num_episodes"],
            CHECKPOINT_FOLDER_PATH=os.path.join(
                utils.get_home_path(),
                CONFIG["REWARDS_PARENT_CONFIG_DIR"],
                f"{session_id}/{CONFIG['REWARDS_CONFIG_MODEL_FOLDER_NAME']}/",
            )
        )

        self.model = LinearQNet(self.config.LAYER_CONFIG)

        self.agent = QTrainer(
            lr=self.config.LR,
            gamma=self.config.GAMMA,
            epsilon=self.config.EPSILON,
            model=self.model,
            loss=self.config.LOSS,
            optimizer=self.config.OPTIMIZER,
            model_name=None,
            checkpoint_folder_path=self.config.CHECKPOINT_FOLDER_PATH,
        )

        self.reward_function = self._convert_str_func_to_exec(
            self.config.REWARD_FUNCTION, "reward_function"
        )

    def _convert_str_func_to_exec(self, str_function: str, function_name: str):
        globals_dict = {}
        exec(str_function, globals_dict)
        new_func = globals_dict[function_name]
        return new_func

    async def stream_episode(self, yield_response: bool = False):
        experiment = workflow.RLWorkFlow(self.config)
        for exp_results in experiment.run_episodes():
            utils.add_inside_session(
                self.session_id,
                config_name="metrics",
                rewrite=True,
                multi_config=True,
                episode_number=exp_results["episode_num"],
                episode_score=exp_results["episode score"], # requires changes in sdk 
                episode_mean_score=exp_results["mean score"],
            )

            if yield_response:
                yield json.dumps(exp_results) + "\n"

    def start_process(self):
        process = mp.Process(target=self.stream_episode)
        process.start()
        return process
