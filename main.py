import os 
import json 
from datetime import datetime
from flask_cors import CORS 
from flasgger import Swagger
from flask import Flask, Response, request, Request

from rewards_api import utils 
from rewards_api.config import CONFIG 
from rewards_api.streamer import RewardsStreamer

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
    """
    Create a new session in the rewards-platform where user can now initialize 
    with different environment, agent configuration
    """ 
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
    
    model_history_path = os.path.join(
        utils.get_home_path(), 
        CONFIG['REWARDS_PARENT_CONFIG_DIR'], 
        body["session_id"], 
        CONFIG["REWARDS_CONFIG_MODEL_FOLDER_NAME"],
        CONFIG['MODEL_HISTORY_JSON_NAME']
    )
    
    creation_init = datetime.now() 
    creation_date, creation_time = creation_init.__str__().split(' ')
    
    model_history = json.load(open(model_history_path))
    model_history['last_created'] = {
        'date' : creation_date, 
        'time' : creation_time, 
        'loss' : body["loss_fn"],
        'optimizer' : body["optimizer"], 
        'gamma' : body["gamma"], 
        'epsilon' : body["epsilon"],
        'model_config' : body["model_configuration"],
    }
    
    # TODO: Printing must be deprecated before all deployement 
    print(json.dumps(model_history, indent=4))
    
    with open(model_history_path, "w") as model_history_json:
        json.dump(model_history, model_history_json)
            
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

@app.post("/api/v1/get_all_envs")
def get_all_envs():
    # Returns data of all the environments
    return utils.get_all_envs()

@app.post("/api/v1/get_all_tracks")
def get_all_tracks():
    return utils.get_all_tracks("car-racer")

# there will be multiple API requests 
# for all the different types of the environments
# such that it will do the evaluation for all the environments
# and also we will track the total time spent 
# and how many "done" are there 


# During evaluation the metrics must not be dependent on the score 
# Number of dones 

@app.route('/api/v1/stream', methods = ['GET'])
def train_stream():
    print("Streaming started")
    session_id = request.args.get('session_id')
    rewards_streamer = RewardsStreamer(
        session_id=session_id
    )
    
    return Response(
        rewards_streamer.train(), 
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )


@app.route('/api/v1/evaluate/', methods = ['POST'])
def evaluate_stream():
    body = request.json 
    session_id = body['session_id']
    mode = body['mode']
    track_num = int(body['track_num']) 
    rewards_streamer = RewardsStreamer(
        session_id=session_id
    )
    
    return Response(
        rewards_streamer.evaluate(
            mode = mode, 
            track_num = track_num
        ), 
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )


# Will be deprecated soon 
@app.route('/api/v1/stop')
def stop():
    # global stop_streaming 
    # stop_streaming = True 
    return {"status": 204}

if __name__ == '__main__':
   host = "127.0.0.1"
   port = 8000
   debug = True
   options = None
   app.run(host, port, debug, options)
