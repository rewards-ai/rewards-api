import os
import cv2
import pygame
import base64
import numpy as np
from typing import Dict, Any
import multiprocessing as mp

import src.utils as utils
from src.config import CONFIG
from dotenv import load_dotenv, set_key

from rewards import workflow, QTrainer, LinearQNet, CarGame

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class RewardsStreamer:
    def __init__(self, session_id, response: Dict[str, Any]) -> None:
        """
        RewardsStreamer is the class which is responsible for streaming, saving
        metrics and also streaming frames over the network.
        """
        print("called")
        self.session_id = session_id
        self.enable_wandb = False 
        self.device = "cpu"
        self.run_pygame_loop = False
        self.record = 0
        self.done = False
        
        # environment parameters 
        self.env_name = response["env_params"]["environment_name"]
        self.env_world = int(response["env_params"]["environment_world"])
        self.mode = response["env_params"]["mode"]
        self.car_speed = response["env_params"]["car_speed"]
        
        # agent parameters
        self.layer_config = eval(response["agent_params"]["model_configuration"])
        if type(self.layer_config) == str:
            self.layer_config = eval(self.layer_config)
    
        self.lr = response["agent_params"]["learning_rate"]
        self.loss = response["agent_params"]["loss_fn"]
        self.optimizer = response["agent_params"]["optimizer"]
        self.gamma = response["agent_params"]["gamma"]
        self.epsilon = response["agent_params"]['epsilon']
        self.num_episodes = response["agent_params"]["num_episodes"]
        
        self.reward_function = response["training_params"]["reward_function"]

    def _convert_str_func_to_exec(self, str_function: str, function_name: str):
        """
        Converts a string like function skeleton to a Callable <function>

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
        global lock
        
        self.checkpoint_folder_path = os.path.join(
            utils.get_home_path(),
            CONFIG["REWARDS_PARENT_CONFIG_DIR"], 
            f"{self.session_id}/{CONFIG['REWARDS_CONFIG_MODEL_FOLDER_NAME']}/"
        )
        
        self.model = LinearQNet(self.layer_config)

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
        self.game.FPS = self.car_speed
        
        self.record = 0
        self.done = False
        
        while True:
            print("in while")
            reward, self.done, self.score, pix = self.agent.train_step(self.game)
            self.game.timeTicking()

            if self.done:
                self.game.initialize()
                self.agent.n_games += 1
                self.agent.train_long_memory()
                if self.score > self.record:
                    self.record = self.score
                    # self.agent.model.save()
                print('Game', self.agent.n_games, 'Score', self.score, 'Record:', self.record)
            im_png = cv2.imencode(".png", pix)[1]
            # yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(im_png) + b'\r\n'
            yield {
                "status": 200,
                "frame": b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(im_png) + b'\r\n'
                }
        
        # while True:
        #     if self.num_episodes == self.agent.n_games:
        #         yield "done showing"
        #     else:
        #         _, self.done, self.score, self.pixel_data = self.agent.train_step(self.game)
        #         self.game.timeTicking()
                
        #         if self.done:
        #             self.game.initialize() 
        #             self.agent.n_games += 1 
        #             self.agent.train_long_memory() 
                    
        #             if self.score > self.record:
        #                 self.agent.model.save(
        #                     self.checkpoint_folder_path, 
        #                     'model.pth', 
        #                     device = self.device
        #                 )
        #             self.total_score += self.score 
                    
        #             utils.add_inside_session(
        #                 self.session_id, 
        #                 config_name="metrics", 
        #                 rewrite=True, multi_config=True, 
        #                 episode_number = self.agent.n_games, 
        #                 episode_score = self.score, 
        #                 mean_score = self.total_score / self.agent.n_games
        #             )
                    
                    # im_png = cv2.imencode(".png", self.pixel_data)[1]
                    # img_b64 = base64.b64encode(im_png).decode('utf-8')
                    # yield "data:image/png;base64," + img_b64
        # global lock
        # plot_scores = []
        # plot_mean_scores = []
        # total_score = 0
        # record = 0
        # linear_net = LinearQNet([[5, 9], [9, 3]])
        # agent = QTrainer(
        #     model=linear_net,
        #     lr=0.001, gamma=0.9,
        #     epsilon=0.2, loss='mse',
        #     optimizer="adam", 
        #     checkpoint_folder_path=self.checkpoint_folder_path,
        #     model_name = "model.pth")
        # game = CarGame(
        #     track_num=self.env_world, 
        #     reward_function=self._convert_str_func_to_exec(
        #         self.reward_function, 
        #         function_name="reward_function"
        #     ),
        #     mode = self.mode,
        #     display_type="surface", 
        #     screen_size=(800, 700) if self.mode == "training" else (1000, 700)
        # )
        # while True:
        #     # pygame.display.update()
        #     _, done, score, pix = agent.train_step(game)
        #     game.timeTicking()

        #     if done:
        #         game.initialize()
        #         agent.n_games += 1
        #         agent.train_long_memory()
        #         if score > record:
        #             record = score
        #             agent.model.save()
        #         print('Game', agent.n_games, 'Score', score, 'Record:', record)
            # im_png = cv2.imencode(".png", pix)[1]
            # img_b64 = base64.b64encode(im_png).decode('utf-8')
            # yield "data:image/png;base64," + img_b64
        
            
    def stream(self):
        yield "sending images"