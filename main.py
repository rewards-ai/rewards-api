import os 
import json 
import pygame 
from fastapi.logger import logger 
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

# configs and import from other modules

from src.config import CONFIG 
from src.schemas import (
    AgentConfiguration, 
    TrainingConfigurations,
    EnvironmentConfigurations, 
    RewardFunction
)
from src.exceptions import (
    validation_exception_handler, 
    python_exception_handler
)

import src.utils as utils 
from src.streamer import RewardsStreamer


# TODO: 
# -----
# Add a response model for returning all the configuration in structured format 
# Also add a error response model 


app = FastAPI(
    title="RewardsAI API for interacting with rewards-platform", 
    version="1.0.0", 
    description="""
    rewards-api is the easy to use API for interacting with agents and environments.
    It enables users to easily create experiments and manage each of the experiments
    by changing different types of parameters and reward function and also pushing the 
    model to other location while competing. 
    """
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

@app.on_event('startup')
async def startup_event():
    utils.create_folder_struct()
    logger.info("Folder Created")
    logger.info("Starting up")
    
    
@app.post('/api/v1/create_session/{session_id}')
def create_new_session(session_id : str):
    """
    This create a new session in the rewards-platform where 
    user can now initialize with different environment, agent configuration. 

    Args:
    
    - `session_id (str)`:The session ID is a unique string with a format of <DATE>_<USERID>_<TIME>. 
    In each of the session, a unique type of model can be made (which will be remain unchanged)
    Howevar during a session some parameters including environment, agent and some training 
    can be changed.
    """
    utils.create_session(session_name=session_id)
    return {
        'status' : 200, 
        'response' : f'Session {session_id} created successfully'
    } 


@app.post('/api/v1/delete_session/{session_id}')
def delete_session(session_id : str):
    """
    Deletes an existing session. Mainly done when there is no need of that session. 
    """
    return utils.delete_session(session_id = session_id)

@app.post('/api/v1/write_env_params')
def push_env_parameters(request : Request, body : EnvironmentConfigurations):
    """
    Create and save environment parameters for the given environment. 
    List of environment parameters:
    - environment_name : The name of the environment defaults to 'car-race'
    - environment_world : The environment map to choose options : 0/1/2
    - mode : training/validation mode 
    - car_speed : the speed of the car (agent)
    
    Args:
    
    - `request (Request)`: Incoming request headers 
    - `body (EnvironmentConfigurations)`: Request body
    """
    utils.add_inside_session(
        session_id = body.session_id, config_name="env_params", 
        environment_name = body.environment_name,
        environment_world = body.environment_world,
        mode = body.mode, 
        car_speed = body.car_speed 
    )
    
    return {
        'status' : 200, 
        'response' : 'saved all the environment configurations successfully'
    } 


@app.post("/api/v1/write_agent_params")
def push_agent_parameters(request : Request, body : AgentConfiguration):
    """
    Create and save agent parameters for the given session
    List of the agent parameters:
    - model_configuration : example: '[[5, 128], [128, 64], [64, 3]]' 
    - learning_rate : example : 0.01 
    - loss_fn : example : mse 
    - optimizer : example : adam 
    - num_episodes : The number of episodes to train the agent. example : 100
    
    Args:
    
    - `request (Request)`: Incoming request headers 
    - `body (EnvironmentConfigurations)`: Request body
    """
    utils.add_inside_session(
        session_id=body.session_id, config_name="agent_params",
        model_configuration = body.model_configuration, 
        learning_rate = body.learning_rate, 
        loss_fn = body.loss_fn, 
        optimizer = body.optimizer, 
        gamma = body.gamma, 
        epsilon = body.epsilon,
        num_episodes = body.num_episodes
    )
    
    return {
        'status' : 200, 
        'response' : 'Agent configurations saved sucessfully'
    } 

@app.post("/api/v1/write_training_params")
def push_training_parameters(request : Request, body : TrainingConfigurations):
    """
    Create and save training parameters for the given session
    List of the training parameters:
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
    
    Args:
    
    - `request (Request)`: Incoming request headers 
    - `body (EnvironmentConfigurations)`: Request body
    """
    utils.add_inside_session(
        session_id=body.session_id, config_name = "training_params",
        learning_algorithm = body.learning_algorithm, 
        enable_wandb = body.enable_wandb, 
        reward_function = body.reward_function
    )
    
    return {
        'status' : 200, 
        'response' : 'ok'
    } 


@app.post("/api/v1/write_reward_fn")
def write_reward_function(request : Request, body : RewardFunction):
    """
    Rewriting the reward function during the time of experimentation
    
    Args:
        
    - `request (Request)`: Incoming request headers 
    - `body (EnvironmentConfigurations)`: Request body
    """
    utils.add_inside_session(
        session_id=body.session_id, config_name="training_params", 
        rewrite=True, 
        reward_function = body.reward_function
    )


@app.get('/api/v1/get_all_params/{session_id}')
async def get_all_parameters(session_id : str):
    """
    Listing all the parameters (environment, agent and training) parameters 
    as one single json response. 
    
    Args:
    
    - `session_id (str)`: The session ID which was used in the start. 
    """
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

# make streamer as the generator 
# make this endpoint as the client 
# so it will be back and forth connections between the client and the server 



@app.get('/api/v1/start_training/{session_id}')
async def start_training(session_id : str):
    """/start_training is the endpoint to start training the agent
    These are the sets of events that will happen during this session while triggering this endpoint
    
    - Validation of all the parameters (TODO)
    - Loading the model and the agent 
    - Start loading the game and streaming the results 

    Args:
        session_id (str): The session. 
        Using this session id we can train any of the experiment 
    """
    rewards_response = utils.get_session_files(session_id)
    streamer = RewardsStreamer(session_id = session_id, response = rewards_response)
    return StreamingResponse(streamer.stream_episode())
    
     

@app.get('/api/v1/stop_training/')
def stop_training():
    # NOTE: (TODO)
    # Stop training does not work for now. 
    # One main reason is to make it into a different threading. This might introduce
    # more bugs and problems. One can stop training by just clicking the cross button. 
    # Howevar it will automatically close once episode gets finished
    
    pygame.quit() 
    return {
        "status" : 200, 
        "message" : "Stopped training successfully"
    }


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
def push_model(model_name : str):
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
