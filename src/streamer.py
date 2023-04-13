import os
import cv2 
import ast
import sys 
import time
import json
import pygame
import asyncio
import multiprocessing as mp
from typing import Dict, Any

import src.utils as utils
from src.config import CONFIG

from rewards import workflow, QTrainer, LinearQNet, CarGame

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class RewardsStreamer:
    def __init__(self, session_id, response: Dict[str, Any]) -> None:
        """
        RewardsStreamer is the class which is responsible for streaming, saving
        metrics and also streaming frames over the network.
        """
        self.session_id = session_id
        self.enable_wandb = False 
        self.device = "cpu"
        
        # environment parameters 
        self.env_name = response["env_params"]["environment_name"]
        self.env_world = int(response["env_params"]["environment_world"])
        self.mode = response["env_params"]["mode"]
        self.car_speed = response["env_params"]["car_speed"]
        
        # agent parameters
        self.layer_config = ast.literal_eval(response["agent_params"]["model_configuration"])
        self.lr = response["agent_params"]["learning_rate"]
        self.loss = response["agent_params"]["loss_fn"]
        self.optimizer = response["agent_params"]["optimizer"]
        self.gamma = response["agent_params"]["gamma"]
        self.epsilon = response["agent_params"]['epsilon']
        self.num_episodes = response["agent_params"]["num_episodes"]
        self.checkpoint_folder_path = os.path.join(
            utils.get_home_path(),
            CONFIG["REWARDS_PARENT_CONFIG_DIR"], 
            f"{session_id}/{CONFIG['REWARDS_CONFIG_MODEL_FOLDER_NAME']}/"
        )
        
        
        # reward function 
        self.reward_function = response["training_params"]["reward_function"]
        
        # make the model 
        self.model = LinearQNet(self.layer_config)

        # make the agent 
        self.agent = QTrainer(
            lr = self.lr, 
            gamma = self.gamma, 
            epsilon = self.epsilon, 
            model = self.model, 
            loss = self.loss, 
            optimizer = self.optimizer, 
            checkpoint_folder_path = self.checkpoint_folder_path, 
            model_name = "model.pth"
        )
        
        # build the screen and the game 
        self.game = CarGame(
            track_num=self.env_world, 
            mode = self.mode, 
            reward_function=self._convert_str_func_to_exec(
                self.reward_function, 
                function_name="reward_function"
            ), 
            display_type="surface", 
            screen_size=(800, 700) if self.mode == "training" else (1000, 700)
        )

    def _convert_str_func_to_exec(self, str_function: str, function_name: str):
        """Converts a string like function skeleton to a Callable <function>

        Args:
            str_function (str): The string function skeleton 
            function_name (str): The name of the function 

        Returns:
            Callable: Actual callable function of type <'function'>
        """
        globals_dict = {}
        exec(str_function, globals_dict)
        new_func = globals_dict[function_name]
        return new_func
    
    def stream_episode(self, yield_response: bool = False):
        for episode in range(1, self.num_episodes + 1):
            self.game.initialize() 
            self.game.FPS = self.car_speed
            total_score, record = 0, 0 
            done = False 
            
            
            while not done:
                _, done, score, pixel_data = self.agent.train_step(self.game)
                self.game.timeTicking()
            
            self.agent.n_games += 1 
            self.agent.train_long_memory() 
            
            if score > record:
                self.agent.model.save(
                    self.checkpoint_folder_path, 
                    'model.pth', 
                    device = self.device
                    )
            total_score += score 
            
            # stream all the metrics to a file 
            utils.add_inside_session(
                self.session_id, 
                config_name="metrics", 
                rewrite=True, multi_config=True, 
                episode_number = episode, 
                episode_score = score, 
                mean_score = total_score / self.agent.n_games
            )
                        
            response = {
                'episode_number' : episode, 
                'episode_score' : score,
                'episode_mean_score' : total_score / self.agent.n_games
            }
            
            yield json.dumps(response) + "\n"  