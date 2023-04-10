import inspect
from typing import Union
from pydantic import BaseModel, Field


def reward_function(props):
    if props["isAlive"]:
        return 1
    return 0


class EnvironmentConfigurations(BaseModel):
    """
    All the environment configurations for the agent
    Once user logs in the user will select the environment, the environment world and
    different parameters of the car.

    TODO:
    -----
    Some of the further parameters that we can introduce:
    - Number of the radars
    - Night environment (might be possible for next version)

    """

    session_id: str = Field(
        ...,
        description="""
        The session ID is a unique string with a format of <DATE>_<USERID>_<TIME>. 
        In each of the session, a unique type of model can be made (which will be remain unchanged)
        Howevar during a session some parameters including environment, agent and some training 
        can be changed.
        """,
    )

    environment_name: str = Field(
        ...,
        example="car-race",
        description="""
        Name of the environment choosen for trainig. rewards v1 only supports one environment 
        that is the car-race environment. Howevar we will be providing support for other battle 
        environment and custom experimental environments in the comming versions. 
        """,
    )

    environment_world: Union[str, int] = Field(
        ...,
        gt=-1,
        lt=3,
        example=1,
        description="""
        Which world of the environment is been choosen. There are three options to
        excepted: 0/1/2. In the version of rewards v1, there is only the support for these 
        three training environments. Although in the coming versions support for adding similar 
        custom environment will be added. 
        """,
    )

    mode: str = Field(
        ...,
        example="training",
        description="""
        There are two different modes of the agent, training and evaluation.
        `training`: In this mode, the model connfigurations will be used for training 
        with optimizations enabled. Here the agent can be trained for multiple episodes. 
        
        `validation`: In this mode, the model is in model.eval() mode. Here the agent can be 
        trained for only 1 episode. And no further optimizations will be applied.
        """,
    )

    car_speed: Union[int, float] = Field(
        ...,
        example=10,
        lt=200,
        description="""
        The speed of the car. There is no variable speed supported for now. However, user can train
        the same agent with different car speed. 
        """,
    )


class AgentConfiguration(BaseModel):
    """
    All the model configurations and model hyperparameters.
    """

    session_id: str = Field(
        ...,
        description="""
        The session ID is a unique string with a format of <DATE>_<USERID>_<TIME>. 
        In each of the session, a unique type of model can be made (which will be remain unchanged)
        Howevar during a session some parameters including environment, agent and some training 
        can be changed.
        """,
    )

    model_configuration: str = Field(
        ...,
        example="[[5, 128], [128, 64], [64, 3]]",
        description="""
        model_configuration is the automatic model LinearQNet model maker of rewards v1. All the user
        needs to do is to create a string of list of list, as shown in the example above.  
        NOTE: Do not change the first and the last element of the lists. So the list must be in the format:
        [[5, ..], ...., [.., 3]] Where 5 is the number of input features and 3 is the number of output layers.
        
        rewards v1 only supports a string of list of list as the input and a very simple LinearQNet model
        which follows a DeepQ leanrning algorithm. In the coming version, we are gonna support more 
        customizations with non linearity and also custom pytorch model upload. 
        """,
    )

    learning_rate: float = Field(
        ...,
        gt=0.0000001,
        example=0.01,
        description="""
        Learning rate for weight updation. Some of the best learning rates are: [0.01, 0.001, 0.0001, 0.015]
        The more less the learing rate will be, slower the training the model will be. So please choose the 
        learning rate wisely.
        """,
    )

    loss_fn: str = Field(
        ...,
        example="mse",
        description="""
        The loss function for agent training. Available options: mse/mae/rmse. 
        In the coming version, we will support more loss functions and also custom loss function. 
        """,
    )

    optimizer: str = Field(
        ...,
        example="adam",
        description="""
        The optimizer for agent training. Available options: adam/sgd/rmsprop.
        In the coming version, we will support more loss functions and also custom optimizer (if possible)
        """,
    )
    
    gamma : float = Field(
        ..., 
        example = 0.99, 
        description="""
        In reinforcement learning, gamma (γ) is a discount factor used to balance immediate and future rewards
        Optimal values of gamma is ranges between 0.9 to 0.99 
        """
    )
    
    epsilon : float = Field(
        ..., 
        example = 0.99, 
        description="""
        In reinforcement learning, epsilon (ε) is an exploration-exploitation parameter used to 
        determine the agent's probability of taking a random action versus the optimal action
        """
    )

    num_episodes: int = Field(
        ...,
        example=100,
        description="""
        The number of the episodes required for the agent to train. In each episode the agent will load 
        it's model configurations and other training configurations and then it will train it self to 
        gain maximum reward based on provided reward function. 
        """,
    )


class TrainingConfigurations(BaseModel):
    """
    All the configuration required for start the training process
    """

    session_id: str = Field(
        ...,
        description="""
        The session ID is a unique string with a format of <DATE>_<USERID>_<TIME>. 
        In each of the session, a unique type of model can be made (which will be remain unchanged)
        Howevar during a session some parameters including environment, agent and some training 
        can be changed.
        """,
    )

    # TODO: To be implemented
    learning_algorithm: str = Field(
        ...,
        example="ppo",
        description="""
        The RL algorithm that will be used for training. Although this is not implemented. 
        For now the default is already DeepQ but we also want to provide support for other algorithm like 
        (Actor-Critc, PPO, Sarsa, DeepQ)
        """,
    )

    enable_wandb: bool = Field(
        ...,
        example=False,
        description="""
        wandb or Weights and Biases is a experimenting and monitoring service that helps users to 
        monitor their experimentation. Enabling this would require to install wandb using:
        
        ```bash
        pip install wandb 
        ```
        Create a weigts and bias account at https://wandb.ai/site After this, for linux user in the terminal 
        add the command: 
        ```bash
        wandb login
        ```
        For windows user you have to do that by:
        ```cmd
        python -m wandb login <API-KEY>
        ```
        """,
    )

    reward_function: str = Field(
        ...,
        example=inspect.getsource(reward_function),
        description="""
        A reward function is a mathematical representation of the goal or objective that an agent is 
        trying to achieve in a given task or environment. In reinforcement learning, an agent interacts 
        with an environment by taking actions and receiving feedback in the form of rewards. The goal 
        of the agent is to learn a policy that maximizes its cumulative reward over time.
        
        The reward function is used to provide a scalar value to the agent after each action it takes, 
        which indicates how well the agent is performing in achieving its goal. The reward signal serves 
        as the basis for the agent's learning process, as it guides the agent towards actions that lead 
        to a higher cumulative reward.
        
        Designing an appropriate reward function is crucial for achieving good performance 
        in a reinforcement learning task. A poorly designed reward function can lead to 
        undesirable behavior by the agent, such as exploiting loopholes in the environment or 
        focusing on short-term gains at the expense of long-term objectives. Therefore, a well-designed 
        reward function should be carefully crafted to incentivize the desired behavior of the agent.
        """
    )


class RewardFunction(BaseModel):
    session_id: str = Field(
        ...,
        description="""
        The session ID is a unique string with a format of <DATE>_<USERID>_<TIME>. 
        In each of the session, a unique type of model can be made (which will be remain unchanged)
        Howevar during a session some parameters including environment, agent and some training 
        can be changed.
        """,
    )
    
    reward_function: str = Field(
        ...,
        example=inspect.getsource(reward_function),
        description="""
        A reward function is a mathematical representation of the goal or objective that an agent is 
        trying to achieve in a given task or environment. In reinforcement learning, an agent interacts 
        with an environment by taking actions and receiving feedback in the form of rewards. The goal 
        of the agent is to learn a policy that maximizes its cumulative reward over time.
        
        The reward function is used to provide a scalar value to the agent after each action it takes, 
        which indicates how well the agent is performing in achieving its goal. The reward signal serves 
        as the basis for the agent's learning process, as it guides the agent towards actions that lead 
        to a higher cumulative reward.
        
        Designing an appropriate reward function is crucial for achieving good performance 
        in a reinforcement learning task. A poorly designed reward function can lead to 
        undesirable behavior by the agent, such as exploiting loopholes in the environment or 
        focusing on short-term gains at the expense of long-term objectives. Therefore, a well-designed 
        reward function should be carefully crafted to incentivize the desired behavior of the agent.
        """
    )
