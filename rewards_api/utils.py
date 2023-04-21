import os 
import cv2  
import json 
import pygame 
import shutil 
import numpy as np 
from .config import CONFIG 
from datetime import datetime 
from typing import Optional, Any, Dict, Union 


# import the latest rewards package 
from rewards_experimental import (
    Agent, 
    QTrainer, 
    LinearQNet
)

from rewards_envs import CarConfig, CarGame


def get_home_path():
    # get the home directory using os
    return os.path.expanduser("~")

def update_graphing_file(session_id: str, data: dict, dir: Optional[str] = None, name: Optional[str] = None) -> None:
    f = open(CONFIG['STREAMING_TEMP_JSON_PATH'], "w")
    f.write(json.dumps(data))
    

def create_folder_struct(dir: Optional[str] = None, name: Optional[str] = None) -> None:
    """Creates a root folder to store all the session configuration

    Args:
        dir (Optional[str], optional): _description_. Defaults to None.
        name (Optional[str], optional): _description_. Defaults to None.
    """
    dir = get_home_path() if dir is None else dir
    name = ".rewards_ai" if name is None else name

    if not os.path.exists(os.path.join(dir, name)):
        os.mkdir(os.path.join(dir, name))

    else:
        print("=> Folder already exists")


def create_session(session_name: str, session_root_dir: Optional[str] = None) -> Union[str, bool]:
    """Creates a folder inside the root folder to store all the configuration for a particular session

    Args:
        session_dir (Optional[str], optional): _description_. Defaults to None.
        session_name (Optional[str], optional): _description_. Defaults to None.
    """
    # TODO : Make a proper return value to detect error

    session_root_dir = (
        os.path.join(get_home_path(), CONFIG["REWARDS_PARENT_CONFIG_DIR"])
        if session_root_dir is None
        else session_root_dir
    )
    session_dir = os.path.join(session_root_dir, session_name)
    creation_init = datetime.now() 
    creation_date, creation_time = creation_init.__str__().split(' ')
    
    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
        os.mkdir(os.path.join(session_dir, CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"]))
        
        session_training_metrics = {
            'plot_scores' : [], 
            'plot_mean_scores' : []
        }
        
        # NOTE: This is very much specific to the rewards Car Environment 
        session_evaluation_metrics = {
            'training' : {
                "1" : {}, # this will contain the trials results 
                "2" : {}, 
                "3" : {}
            }, 
            
            'evaluation': {
                "1" : {}
            }
        }
        
        with open(os.path.join(session_dir, CONFIG['TRAINING_METRICS_JSON_NAME']), "w") as json_file:
            json.dump(session_training_metrics, json_file)
        print(f"=> Created session_training_metrics.json for session: {session_name}")
        
        with open(os.path.join(session_dir, CONFIG['EVALUATION_METRICS_JSON_NAME']), "w") as json_file:
            json.dump(session_evaluation_metrics, json_file)
        print(f"=> Created session_evaluation_metrics.json for session: {session_name}")

        # also create model_hisory.json 
        initial_model_hisory = {
            'last_created' : {}, 
            'last_trained' : {}
        }
        with open(os.path.join(
            session_dir, 
            CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"],
            CONFIG['MODEL_HISTORY_JSON_NAME']), "w") as model_hisory:
            json.dump(initial_model_hisory, model_hisory)
        
        response_body = {
            'status' : 200, 
            'session_id' : session_name, 
            'session_creation_date' : creation_date, 
            'session_creation_time' : creation_time,
            'dir_paths' : {
                'session_root_dir' : session_root_dir,
                'saved_models_dir' : os.path.join(session_dir, CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"]), 
            },
            'json_paths' : {
                'training_metrics' : os.path.join(session_dir, CONFIG['TRAINING_METRICS_JSON_NAME']),
                'evaluation_metrics' : os.path.join(session_dir, CONFIG['EVALUATION_METRICS_JSON_NAME']), 
                'model_history' : os.path.join(
                    session_dir, 
                    CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"],
                    CONFIG['MODEL_HISTORY_JSON_NAME'])
            }
        }
        
        
        # TODO: Remove the print when putting to final production 
        print(json.dumps(response_body, indent=4))
        
        return response_body
    
    else:
        return {
            'status' : 204, 
            'message' : 'Folder already exists'
        }

def delete_session(session_id : str, session_root_dir : Optional[str] = None):
    """Deletes a particulart session 

    Args:
        session_id (str): The name of the session sub folder inside the root directory
        session_root_dir (Optional[str], optional): The directory where all the session config are been kept. Defaults to None.

    """
    session_root_dir = (
        os.path.join(get_home_path(), ".rewards_ai")
        if session_root_dir is None
        else session_root_dir
    )
    session_dir = os.path.join(session_root_dir, session_id)
    try:
        shutil.rmtree(session_dir)
        return {
            'status' : 200, 
            'message': f'Session {session_dir} deleted succesfully'
        }
    except Exception as e:
        return {
            'status' : 500, 
            'message': f'Internal server error {e}'
        }

def add_inside_session(
    session_id: str, config_name: str, rewrite: bool = False, multi_config: bool = False, **kwargs
):
    """Add configuration inside a session folder

    Args:
        session_id (str): _description_
        config_name (str): _description_
        rewrite (bool) : True if it needs to rewrite the existing json,
        multi_config (bool) : True if it has a multiple json configuration, then
        the configuration will be in the form of: [{}, {}, .. ]
    """
    session_root_dir = os.path.join(get_home_path(), ".rewards_ai")

    configuration = [dict(kwargs)] if multi_config else dict(kwargs)
    save_path = os.path.join(session_root_dir, session_id, f"{config_name}.json")

    if rewrite and os.path.exists(save_path):
        existing_config = json.load(open(save_path, "r"))
        if multi_config:
            existing_config.append(configuration)
        else:
            existing_config[list(configuration.keys())[0]] = list(configuration.values())[0]
        configuration = existing_config

    with open(save_path, "w") as json_file:
        json.dump(configuration, json_file)


def get_session_files(session_id: str, session_root_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get all the files inside a session configuration folder.

    Args:
        session_id (str): The name of the session sub folder inside the root directory
        session_root_dir (Optional[str], optional): The directory where all the session config are been kept. Defaults to None.

    Returns:
        Dict[str, Any]: All the configurations for a particular session
    """
    session_root_dir = (
        os.path.join(get_home_path(), ".rewards_ai")
        if session_root_dir is None
        else session_root_dir
    )

    session_dir = os.path.join(session_root_dir, session_id)
    all_configs = {}

    for json_file_name in os.listdir(session_dir):
        if json_file_name.endswith(".json") and not json_file_name.startswith("metrics"):
            json_dict = json.load(open(os.path.join(session_dir, json_file_name), "r"))
            all_configs[json_file_name.split(".json")[0]] = json_dict

    # TODO: Show the metrics and model configurations so that it can be displayed in fronend
    return all_configs


def get_all_sessions_info(session_root_dir : Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get all the sessions information present for a particular user. 
    
    Args:
        session_root_dir : Optional[str] -> The root directory where all the sessions are stored
    
    Returns:
        Dict[str, Dict[str, Any]]: All the configurations for all the sessions
    """
    all_session_infos = {}
    session_root_dir = (
        os.path.join(get_home_path(), ".rewards_ai")
        if session_root_dir is None
        else session_root_dir
    )
    
    session_id_paths = [
        os.path.join(
            session_root_dir, session_id
        ) for session_id in os.listdir(session_root_dir)]
    
    for session_id, session_id_path in zip(os.listdir(session_root_dir), session_id_paths):
        all_session_infos[session_id] = get_session_files(session_id=session_id, session_root_dir=session_root_dir)
    
    return all_session_infos



# Going to be deprecated soon. Must be replaced with some good functions 

def get_all_envs() -> Dict[str, Dict[str, Any]]:
    # TODO: Changhe from static config to a dynamic config 
    #   - Data is static, has to come from SDK
    data = [
        {
            "name": "Car Racer",
            "id": "car_racer", 
            "description": "2D racing environment, where a car tries to complete a given track using it's 5 radars.",
            "isReleased": True
        },
        {
            "name": "Snake Game",
            "id": "snake_game", 
            "description": "2D snake environment, where mutiple snakes controlled by multiple models try to get food as fast as possible",
            "isReleased": False
        },
        {
            "name": "Street Fighter",
            "id": "street_fighter", 
            "description": "2D fighting arena where 2 fighters are trained to fight and get maximum socre.",
            "isReleased": False
        },
        {
            "name": "Bi-Pedal",
            "id": "bi-pedal", 
            "description": "2D rough road, where a bi-pedal humanoid learns to walk as fast as possible",
            "isReleased": False
        }
    ]
    
    return data

def get_all_tracks(environment_name) -> Dict[str, Dict[str, Any]]:
    # TODO:
    #   - Data is static, has to come from SDK
    
    data = []
    
    if environment_name == "car-racer":
        data = ["track-1.png", 'track-2.png', 'track-3.png']
    
    return data



# Newer additions 

def convert_str_func_to_exec(str_function: str, function_name: str):
    globals_dict = {}
    exec(str_function, globals_dict)
    new_func = globals_dict[function_name]
    return new_func

def generate(session_id, model_name : Optional[str] = None):
    r = get_session_files(session_id)
    print(session_id)
    print("check" ,session_id)
    record = 0
    done = False
    #enableStreaming()
    
    # environment parameters 
    env_world = int(r["env_params"]["environment_world"])
    mode = "training"
    car_speed = r["env_params"]["car_speed"]
    
    # agent parameters
    layer_config = eval(r["agent_params"]["model_configuration"])
    if type(layer_config) == str:
        layer_config = eval(layer_config)
    
    # training parameters 
    
    lr = r["agent_params"]["learning_rate"]
    loss = r["agent_params"]["loss_fn"]
    optimizer = r["agent_params"]["optimizer"]
    gamma = r["agent_params"]["gamma"] / 100 # have to change it later
    epsilon = r["agent_params"]['epsilon']
    num_episodes = r["agent_params"]["num_episodes"]
    reward_function = r["training_params"]["reward_function"]
    
    checkpoint_folder_path = os.path.join(
        get_home_path(),
        CONFIG["REWARDS_PARENT_CONFIG_DIR"], 
        f"{session_id}/{CONFIG['REWARDS_CONFIG_MODEL_FOLDER_NAME']}/"
    )
    
    screen_size = (800, 700)
    model = LinearQNet(layer_config)
    env_config = CarConfig(
        car_fps = car_speed,
        render_mode = "rgb_array",
        render_fps=car_speed, 
        screen_size=screen_size, 
    )
    
    game = CarGame(
        mode = mode, 
        track_num = env_world, 
        reward_function=convert_str_func_to_exec(
            reward_function, function_name="reward_function"
        ), 
        config = env_config
    )
    
    model_name = CONFIG['MODEL_NAME'] if model_name is None else model_name 
    agent = QTrainer(
        lr = lr, 
        gamma = gamma, 
        epsilon = epsilon, 
        model = model, 
        loss = loss, 
        optimizer = optimizer, 
        checkpoint_folder_path = checkpoint_folder_path, 
        model_name = model_name 
    )
    
    plot_scores, plot_mean_scores = [], []
    total_score, record = 0, 0
    
    while True:
        if agent.n_games == num_episodes:
            pygame.quit() 
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
                    checkpoint_folder_path, 
                    model_name, 
                    device = CONFIG['DEVICE'], 
                )
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            update_graphing_file(
                session_id=session_id, 
                data = {"plot_scores": plot_scores, "plot_mean_scores": plot_mean_scores}
            )
        img = np.fliplr(pixel_data)
        img = np.rot90(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imencode(".png", img)[1]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n')
        