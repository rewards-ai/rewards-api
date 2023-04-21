## **Rewards API:latest**

This latest version of rewards api has some very major changes. The main changes has been introduced in the areas of streaming and endpoint response body formats and intodcuing newer endpoints and deprecating old ones. 



### **How to install and run this repo**

This latest API is dependent on **`rewards_experimental`** and **`rewards_envs`** packages. So first please install the following two packages by typing: 

```bash
pip install rewards-experimental
pip install rewards-envs
```

After this clone the repository and go inside the repository.

```bash
git clone -b latest https://github.com/rewards-ai/rewards-api.git
```

This will clone the latest branch of the API. Please make sure you have `flask` and some other flask related third-party library installed. If not you can install them by: 

```bash
pip install flask flask_cors flasgger
```

After this you can run the start the server by typing:

```bash
PYTHONPATH=. python3 rewards_api/main.py
```

### **API endpoints requests and responses**

Here we explain the different API endpoints that rewards-api provides and the expected work it does under the hood and the responses body it provides. 

#### **`/api/v1/create_session`**

Creates a new session for the user to carry out different experiments. It expects the following request parameters

- `session_id` : The name of the session to be created (must be unique)

Response body looks like this

```json
{
    "status": 200,
    "session_id": "some-session-name",
    "session_creation_date": "2023-04-21",
    "session_creation_time": "10:21:10.414299",
    "dir_paths": {
        "session_root_dir": "path for.rewards_ai",
        "saved_models_dir": "path for session_saved_models"
    },
    "json_paths": {
        "training_metrics": "path for training_metrics.json",
        "evaluation_metrics": "path for evaluation_metrics.json",
        "model_history": "path for model_history.json"
    }
}
```

#### **`/api/v1/delete_session`**

Deletes the mentioned session. It expects the following request parameters 

- `session_id` : The name of the session 

#### **`/api/v1/write_env_params`**

Writes the environment parameters. It writes `env_params.json` which looks something like this. 

```json
{
  "environment_name": "car-race",
  "environment_world": 2,
  "mode": "training",
  "car_speed": 20
}
```

#### **`/api/v1/write_agent_parameters`**

After creating the first model it will make provide a response which looks something like this. 

```json
{
    "last_created": {
        "date": "2023-04-21",
        "time": "10:21:10.431062",
        "loss": "mse",
        "optimizer": "adam",
        "gamma": 90,
        "epsilon": 20,
        "model_config": "[[5,49],[49,3]]"
    },
    "last_trained": {}
}
```

This writes the agent parameters inside the required default path. This looks something like this

```bash
{
  "model_configuration": "[[5,49],[49,3]]",
  "learning_rate": 60,
  "loss_fn": "mse",
  "optimizer": "adam",
  "gamma": 90,
  "epsilon": 20,
  "num_episodes": 700
}

```

#### **`/api/v1/stream`**

This starts the training streaming process. After the training fininshes or after the training is exited, it will write the `model_history.json` something like this, so that it can be accesed

```json
{
  "last_created": {
    "date": "2023-04-21",
    "time": "09:51:10.808311",
    "loss": "mse",
    "optimizer": "adam",
    "gamma": 90,
    "epsilon": 20,
    "model_config": "[[5,62],[62,3]]"
  },
  "last_trained": {
    "date": "2023-04-21",
    "time": "09:51:37.416074",
    "record": 9,
    "scores": [
      9,
      8
    ],
    "mean_scores": [
      9,
      8.5
    ]
  }
}

```

#### **Some important TODOs**

- [ ] Creat the evaluation stream.
  
  - evaluation endpoint must recieve the following request arguments:
    
    - `session_id` The id of the session 
    
    - `mode` It can be `training` or `evaluation`
    
    - `track_num` Which track environment to evaluate on 
    
    This should write the following json on the `evaluation_metrics.json`. :
    
    ```json
    {
        "session_id" : "session_id",
        "training": { 
            "total_elapsed_time" : "total_elapsed_time", 
            "1": {
                "num_trials" : 5, 
                "record" : 120
            }, 
            "2": {
                "num_trials" : 5, 
                "record" : 120
            }, 
            "3": {
                "num_trials" : 5, 
                "record" : 120
            }
        }, 
        "evaluation": {
            "total_elapsed_time" : "total_elapsed_time", 
            "1": {
                "num_trials" : 5, 
                "record" : 120
            } 
        } 
    }
    ```

           This endpoint is already created. We need to integrate this to react. 



- [ ] Need to show all the information about the training. In the `Your models` section we will need to show the following things
  
  - The date and time of creation and latest train
  
  - The model configurations 
  
  - The latest record the model made
  
  - A very small graph of the training (optional)

- [ ] UI Changes for integrating other inputs too 

- [ ] Support for re-training the same model in different environments 

- [ ] Support for pushing the model to AWS 

- [ ] Leaderboard 

- [ ] Documentation support (optional)

- [ ] Change the `temp.json` path to the `training_metrics.json`. The path will also be given when the session is created. So there will be no issue for react to create that inside assets. 

- [x] Authentication
