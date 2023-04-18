from typing import Optional, Union, Any, Dict
from src.config import CONFIG
import shutil 
import json
import os


def get_home_path():
    # get the home directory using os
    return os.path.expanduser("~")

def update_graphing_file(session_id: str, data: dict, dir: Optional[str] = None, name: Optional[str] = None) -> None:
    f = open("D:/Prototypes/rewards.ai/training-platform\src/assets/temp.json", "w")
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

    if not os.path.exists(session_dir):
        os.mkdir(session_dir)
        os.mkdir(os.path.join(session_dir, CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"]))

        # create a metrics.json
        null_metrics = []
        with open(os.path.join(session_dir, "metrics.json"), "w") as metrics_file:
            json.dump(null_metrics, metrics_file)


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
        
def get_all_envs() -> Dict[str, Dict[str, Any]]:
    # TODO:
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