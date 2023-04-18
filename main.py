from rewards import QTrainer, LinearQNet, CarGame
from werkzeug.exceptions import HTTPException
from flask import Flask, Response, request, Request
from werkzeug.exceptions import NotFound
import matplotlib.pyplot as plt
from src.config import CONFIG
from flask_cors import CORS
import src.utils as utils
from flasgger import Swagger
import numpy as np
import json
import cv2
import os

app = Flask(__name__)
CORS(app)
Swagger(app)

@app.before_first_request
def startup_event():
    utils.create_folder_struct()
    app.logger.info("Folder Created")
    app.logger.info("Starting up")
    
@app.post('/api/v1/create_session')
def create_new_session():
    '''
    Create a new session in the rewards-platform where user can now initialize with different environment, agent configuration.

    ---
    parameters:
    - name: session_id
        in: query
        type: string
        required: true
        description: The session ID is a unique string with a format of <DATE><USERID><TIME>. In each of the session, a unique type of model can be made (which will remain unchanged). However, during a session, some parameters including environment, agent, and some training can be changed.
    '''
    session_id = request.args.get("session_id")
    utils.create_session(session_name=session_id)
    return {
        'status' : 200, 
        'response' : f'Session {session_id} created successfully'
    } 


@app.post('/api/v1/delete_session')
def delete_session():
    """
    Deletes an existing session. Mainly done when there is no need of that session. 
    """
    session_id = request.args.get("session_id")
    return utils.delete_session(session_id = session_id)

@app.post('/api/v1/write_env_params')
def push_env_parameters():
    """
    Create and save environment parameters for the given environment. 
    List of environment ---
    parameters:
    - environment_name : The name of the environment defaults to 'car-race'
    - environment_world : The environment map to choose options : 0/1/2
    - mode : training/validation mode 
    - car_speed : the speed of the car (agent)
    
    ---
    parameters:
    
    - request (Request): Incoming request headers 
    - `body (EnvironmentConfigurations)`: Request body
    """
    body = request.json
    utils.add_inside_session(
        session_id = body["session_id"], config_name="env_params", 
        environment_name = body["environment_name"],
        environment_world = body["environment_world"],
        mode = body["mode"], 
        car_speed = body["car_speed"] 
    )
    
    return {
        'status' : 200, 
        'response' : 'saved all the environment configurations successfully'
    } 


@app.post("/api/v1/write_agent_params")
def push_agent_parameters():
    """
    Create and save agent parameters for the given session
    List of the agent ---
    parameters:
    - model_configuration : example: '[[5, 128], [128, 64], [64, 3]]' 
    - learning_rate : example : 0.01 
    - loss_fn : example : mse 
    - optimizer : example : adam 
    - num_episodes : The number of episodes to train the agent. example : 100
    
    ---
    parameters:
    
    - request (Request): Incoming request headers 
    - body (EnvironmentConfigurations): Request body
    """
    body = request.json
    utils.add_inside_session(
        session_id=body["session_id"], config_name="agent_params",
        model_configuration = body["model_configuration"], 
        learning_rate = body["learning_rate"], 
        loss_fn = body["loss_fn"], 
        optimizer = body["optimizer"], 
        gamma = body["gamma"], 
        epsilon = body["epsilon"],
        num_episodes = body["num_episodes"]
    )
    
    return {
        'status' : 200, 
        'response' : 'Agent configurations saved sucessfully',
        'test': body["model_configuration"]
    } 

@app.post("/api/v1/write_training_params")
def push_training_parameters():
    """
    Create and save training parameters for the given session
    List of the training ---
    parameters:
    - learning_algorithm : example : 0.01 
    - enable_wandb : example : mse 
    - reward_function : example : Callable a reward function looks like this: 
    
    ```python
    def reward_func(props):
        reward = 0
        if props["isAlive"]:
            reward = 1
        obs = props["obs"]
        if obs[0] < obs[-1] and props["dir"] == -1:
            reward += 1
        elif obs[0] > obs[-1] and props["dir"] == 1:
            reward += 1
        else:
            reward += 0
        return reward
    ``` 
    
    ---
    parameters:
    
    - request (Request): Incoming request headers 
    - body (EnvironmentConfigurations): Request body
    """
    body = request.json
    utils.add_inside_session(
        session_id=body["session_id"], config_name = "training_params",
        learning_algorithm = body["learning_algorithm"], 
        enable_wandb = body["enable_wandb"] == 1, 
        reward_function = body["reward_function"]
    )
    
    return {
        'status' : 200, 
        'response' : 'ok'
    } 


@app.post("/api/v1/write_reward_fn")
def write_reward_function():
    """
    Rewriting the reward function during the time of experimentation
    
    ---
    parameters:
        
    - request (Request): Incoming request headers 
    - body (EnvironmentConfigurations): Request body
    """
    body = request.json
    utils.add_inside_session(
        session_id=body["session_id"], config_name="training_params", 
        rewrite=True, 
        reward_function = body["reward_function"]
    )


@app.get('/api/v1/get_all_params')
async def get_all_parameters():
    """
    Listing all the parameters (environment, agent and training) parameters 
    as one single json response. 
    
    ---
    parameters:
    
    - session_id (str): The session ID which was used in the start. 
    """
    session_id = request.args.get("session_id")
    file_response = utils.get_session_files(session_id)
    file_response['status'] = 200 
    return file_response

@app.get('/api/v1/get_all_sessions')
def get_all_sessions():
    """
    Fetches all the session infors and parameters
    
    TODO
    - Add a proper Response model for this endpoint and other endpoints 
      in the coming versions. 
    """
    return utils.get_all_sessions_info()


@app.post("/api/v1/validate_exp")
def validate_latest_expriment():
    # Steps:
    # First it will check if training got stopped or not. If not it will stop the training 
    # Then it will save the model 
    # Load the model and run one episode in an validation environment 
    # Then it will put a response status of 200 of returning status 
    # {
    #   status : 200,
    #   reward : ... 
    #}
    return {
        'status' : 200, 
        'response' : 'ok'
    }  


@app.post("/api/v1/push_model")
def push_model():
    # In the frontend we will show the list of available model agents and their infos like 
    # How much they were trained 
    # their total reward 
    # their average reward 
    # other agent and env configs stored 
    # user will select a model to push 
    # that name of the model will be the parameter 
    # then it will get pushed into S3 bucket 
    return {
        'status' : 200, 
        'response' : 'ok'
    }   
    
@app.post("/api/v1/get_all_envs")
def get_all_envs():
    # Returns data of all the environments
    return utils.get_all_envs()

@app.post("/api/v1/get_all_tracks")
def get_all_tracks():
    return utils.get_all_tracks("car-racer")

def enableStreaming():
    global stop_streaming
    stop_streaming = False
    
def convert_str_func_to_exec(str_function: str, function_name: str):
    globals_dict = {}
    exec(str_function, globals_dict)
    new_func = globals_dict[function_name]
    return new_func

def generate(session_id):
    r = utils.get_session_files(session_id)
    print(session_id)
    print("check" ,session_id)
    record = 0
    done = False
    enableStreaming()
    
    # environment parameters 
    env_name = r["env_params"]["environment_name"]
    env_world = int(r["env_params"]["environment_world"])
    mode = r["env_params"]["mode"]
    car_speed = r["env_params"]["car_speed"]
    
    # agent parameters
    layer_config = eval(r["agent_params"]["model_configuration"])
    if type(layer_config) == str:
        layer_config = eval(layer_config)

    lr = r["agent_params"]["learning_rate"]
    loss = r["agent_params"]["loss_fn"]
    optimizer = r["agent_params"]["optimizer"]
    gamma = r["agent_params"]["gamma"]
    epsilon = r["agent_params"]['epsilon']
    num_episodes = r["agent_params"]["num_episodes"]
    
    reward_function = r["training_params"]["reward_function"]
    
    global lock
        
    checkpoint_folder_path = os.path.join(
        utils.get_home_path(),
        CONFIG["REWARDS_PARENT_CONFIG_DIR"], 
        f"{session_id}/{CONFIG['REWARDS_CONFIG_MODEL_FOLDER_NAME']}/"
    )
    
    model = LinearQNet(layer_config)

    agent = QTrainer(
        lr = lr, 
        gamma = gamma, 
        epsilon = epsilon, 
        model = model, 
        loss = loss, 
        optimizer = optimizer, 
        checkpoint_folder_path = checkpoint_folder_path, 
        model_name = "model.pth"
    )
    
    game = CarGame(
        track_num=env_world, 
        mode = mode, 
        reward_function=convert_str_func_to_exec(
            reward_function, 
            function_name="reward_function"
        ), 
        display_type="surface", 
        screen_size=(800, 700)
    )        
    game.FPS = car_speed
    
    record = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    done = False
    
    while True:
        global stop_streaming
        if stop_streaming or agent.n_games == num_episodes:
            return {"status": 204}
        reward, done, score, pix = agent.train_step(game)
        game.timeTicking()

        if done:
            game.initialize()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save(
                    checkpoint_folder_path, 
                    'model.pth', 
                    device = "cpu"
                )
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            utils.update_graphing_file(session_id, {"plot_scores": plot_scores, "plot_mean_scores": plot_mean_scores})
        img = np.fliplr(pix)
        img = np.rot90(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imencode(".png", img)[1]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n')

@app.route('/api/v1/stream', methods = ['GET'])
def stream():
    session_id = request.args.get('session_id')
    print(session_id)
    return Response(generate(session_id), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/api/v1/stop')
def stop():
    global stop_streaming
    stop_streaming = True
    return {"status": 204}
    
if __name__ == '__main__':
   host = "127.0.0.1"
   port = 8000
   debug = True
   options = None
   app.run(host, port, debug, options)
