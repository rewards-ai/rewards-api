import os 
import cv2 
import time 
import json 
import pygame 
import shutil 
import numpy as np 
from datetime import datetime
from .config import CONFIG 
from typing import Optional, Any, Dict, Union 


# import the latest rewards package 
from rewards_experimental import (
    QTrainer, 
    LinearQNet
)

from rewards_envs import CarConfig, CarGame
from rewards_api.utils import *


def def_reward_function(params):
    reward = 0
    if params["is_alive"]:
        reward = 1
    return reward

class RewardsStreamer:
    """This streamer is very much specific to the Car Game Environment and not for a general purpose
    """
    def __init__(self, session_id):
        self.session_id = session_id
        self.incoming_request = get_session_files(session_id=session_id)
        
        # setting up environment parameters 
        self.env_world = int(self.incoming_request["env_params"]["environment_world"])
        self.car_speed = self.incoming_request["env_params"]["car_speed"]
        
        # agent parameters
        self.layer_config = eval(self.incoming_request["agent_params"]["model_configuration"])
        if type(self.layer_config) == str: self.layer_config = eval(self.layer_config)
        
        # training parameters
        self.lr = self.incoming_request["agent_params"]["learning_rate"]
        self.loss = self.incoming_request["agent_params"]["loss_fn"]
        self.optimizer = self.incoming_request["agent_params"]["optimizer"]
        
        self.gamma = self.incoming_request["agent_params"]["gamma"] 
        self.epsilon = self.incoming_request["agent_params"]['epsilon']
        
        self.num_episodes = self.incoming_request["agent_params"]["num_episodes"]
        self.reward_function = self.incoming_request["training_params"]["reward_function"]
        
        if self.gamma > 1 and self.gamma < 100: self.gamma = self.gamma / 100 
        if self.epsilon < 1: self.epsilon = self.epsilon * 100

        # model checkpointing 
        
        self.checkpoint_folder_path = os.path.join(
            get_home_path(),
            CONFIG["REWARDS_PARENT_CONFIG_DIR"], 
            f"{session_id}/{CONFIG['REWARDS_CONFIG_MODEL_FOLDER_NAME']}/"
        )
        
        # setup the model 
        self.model = LinearQNet(layers_conf=self.layer_config)
        self.model_name = CONFIG['MODEL_NAME'] 
        
        self._model_history_path = os.path.join(
            get_home_path(), 
            CONFIG['REWARDS_PARENT_CONFIG_DIR'], 
            self.session_id, 
            CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"],
            CONFIG['MODEL_HISTORY_JSON_NAME']
        )
        self.model_history = json.load(open(self._model_history_path))
    
    def _write_model_json(self, record, plot_scores, plot_mean_scores):
        creation_init = datetime.now() 
        creation_date, creation_time = creation_init.__str__().split(' ')
        self.model_history['last_trained'] = {
            'date' : creation_date, 
            'time' : creation_time, 
            'record' : record, 
            'scores' : plot_scores, 
            'mean_scores' : plot_mean_scores
        } 
        
        #print(json.dumps(self.model_history, indent=4))
        # save this 
        with open(self._model_history_path, "w") as model_history_json:
            json.dump(self.model_history, model_history_json)
        
    def train(self):
        screen_size = (800, 700)
        train_env_config = CarConfig(
            car_fps = self.car_speed, 
            render_fps=self.car_speed, 
            render_mode="rgb_array", 
            screen_size=screen_size
        )
        
        game = CarGame(
            mode = "training", 
            track_num = self.env_world, 
            reward_function= convert_str_func_to_exec(
                self.reward_function, function_name="reward_function"
            ), 
            config = train_env_config
        )
        
        agent = QTrainer(
            lr = self.lr, 
            gamma = self.gamma, 
            epsilon = self.epsilon, 
            model = self.model, 
            loss = self.loss, 
            optimizer = self.optimizer, 
            checkpoint_folder_path = self.checkpoint_folder_path, 
            model_name = self.model_name
        )
        
        done = False, 
        total_score, record = 0, 0 
        plot_scores, plot_mean_scores = [], [] 
        
        while True:
            if agent.n_games == self.num_episodes:
                pygame.quit() 
                self._write_model_json(
                    record=record, 
                    plot_scores=plot_scores, 
                    plot_mean_scores=plot_mean_scores
                )
                return {
                    "status" : 204
                } 
            _, done, score, pixel_data = agent.train_step(game)
            game.timeTicking()
            
            if done:
                game.initialize()
                agent.n_games += 1 
                agent.train_long_memory() 
                
                if score > record:
                    record = score 
                    agent.model.save(
                        self.checkpoint_folder_path, 
                        self.model_name, 
                        device = CONFIG['DEVICE'], 
                    )
                    
                print('=> Session:', self.session_id, 'Game', agent.n_games, 'Score', score, 'Record:', record)
                plot_scores.append(score)
                total_score += score 
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                update_graphing_file(
                    session_id = self.session_id, 
                    data = {"plot_scores": plot_scores, "plot_mean_scores": plot_mean_scores}
                )
                
                # writing the latest changes after every done. So that closing will not loose
                # latest information 
                
                self._write_model_json(
                    record=record, 
                    plot_scores=plot_scores, 
                    plot_mean_scores=plot_mean_scores
                )
                
            img = np.fliplr(pixel_data)
            img = np.rot90(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.imencode(".png", img)[1]
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n')
    
    
    def evaluate(self, mode = "training", track_num : int  = 1):
        
        # mode : train and test
        # First we evaluate for all the training environments
        # then we will evaluate for the test environments
        # then club down all the results 
        # all of the things will be sent at eval.json 
        
        metrics = {}
        eval_metrics_json_path = os.path.join(get_home_path(), CONFIG['REWARDS_PARENT_CONFIG_DIR'], self.session_id, CONFIG['EVALUATION_METRICS_JSON_NAME'])
        eval_metrics_json = json.load(open(eval_metrics_json_path))
        
        env_config = CarConfig(
            mode = mode, 
            car_fps = self.car_speed, 
            render_fps=self.car_speed, 
            render_mode="rgb_array", 
            screen_size = (800, 700) if mode == "training" else (1000, 700)
        )
        
        agent = QTrainer(
            lr = self.lr, 
            gamma = self.gamma, 
            epsilon = self.epsilon,
            model = self.model, 
            loss = self.loss, 
            optimizer = self.optimizer,
            checkpoint_folder_path = self.checkpoint_folder_path, 
            model_name = self.model_name 
        )
        
        game = CarGame(
            mode = mode, 
            track_num = track_num if mode == "training" else 1, 
            reward_function= def_reward_function, 
            config = env_config
        )
        elapsed_time = CONFIG['EVALUATION_TOTAL_TIME_FOR_TRAINING_PATH'] if mode == "training" else CONFIG['EVALUATION_TOTAL_TIME_FOR_EVALUATION_PATH']
        
        # also calculate the total number of 'done'.
        
        start_time = time.time() 
        record = 0 
        num_trials = 1
        
        while True:
            _, done, score, pixel_data = agent.evaluate(game)
            game.timeTicking() 
            
            current_time = time.time() 
            if current_time - start_time > elapsed_time:
                eval_metrics_json['session_id'] = self.session_id
                eval_metrics_json[mode]['total_elapsed_time'] = elapsed_time
                eval_metrics_json[mode][str(track_num)] = {
                    "num_trials" : num_trials, 
                    "record" : record 
                }
                
                with open(eval_metrics_json_path, "w") as json_file:
                    json.dump(eval_metrics_json, json_file)
                print(f"=> Finished for evaluating on training environment for session {self.session_id}")
                
                pygame.quit() 
                return {
                    "status" : 204
                }  
            
            if done:
                num_trials += 1 
                game.initialize() 
                agent.n_games += 1 
                if score > record:
                    record = score 
                    
                print(f"Evaluating on training track: {track_num} for session : {self.session_id}, score: {score}")
                
            img = np.fliplr(pixel_data)
            img = np.rot90(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.imencode(".png", img)[1]
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n')