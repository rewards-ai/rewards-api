import os
import cv2
import pygame
import threading
import flask_cors
import src.utils as utils
import matplotlib.pyplot as plt
from src.config import CONFIG
from flask import Flask, Response, request, stream_with_context
from rewards import QTrainer, LinearQNet, CarGame

app = Flask(__name__)
flask_cors.CORS(app)
lock = threading.Lock()

def _convert_str_func_to_exec(str_function: str, function_name: str):
    globals_dict = {}
    exec(str_function, globals_dict)
    new_func = globals_dict[function_name]
    return new_func

def generate(session_id):
    r = utils.get_session_files(session_id)
    print(session_id)
    print("check" ,session_id)
    enable_wandb = False 
    device = "cpu"
    run_pygame_loop = False
    record = 0
    done = False
    
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
        reward_function=_convert_str_func_to_exec(
            reward_function, 
            function_name="reward_function"
        ), 
        display_type="surface", 
        screen_size=(800, 700)
    )        
    game.FPS = car_speed
    
    record = 0
    done = False
    
    while True:
        if agent.n_games == num_episodes:
            return {
            "status": 204,
            "frame": ""
        }
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
        im_png = cv2.imencode(".png", pix)[1]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(im_png) + b'\r\n')

@app.route('/stream', methods = ['GET'])
def stream():
    session_id = request.args.get('id')
    print(session_id)
    return Response(generate(session_id), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
   host = "127.0.0.1"
   port = 8005
   debug = False
   options = None
   app.run(host, port, debug, options)