# Model-Based-RL-with-Probabilistic-Ensemble-and-Trajectory-Sampling
Model-Based Reinforcement Learning with Probabilistic Ensemble and Trajectory Sampling (PETS) 

Implementation of model-based RL: <br>
(1) Cross-Entropy Method (CEM) using Ground-Truth Dynamics <br>
(2) Cross-Entropy Method (CEM) using Probabilistic Ensemble and Trajectory Sampling (PETS) <br>
<br>
<br>
Experimental Results:
### (1) CEM with Ground-Truth Dynamics versus CEM with PETS
<img src="https://i.ibb.co/KLzx91t/PETS.png" width="45%" height="45%"> <br>


### (2) PETS Loss
<img src="https://i.ibb.co/j3h84xd/PETS-loss.png" width="40%" height="40%">

# File structure
```
|-- envs
    |-- 2Dpusher_env.py # Implementation of the pusher environment 
    |-- __init__.py # Register two pusher environmnets
|-- run.py # Entry of all the experiments.
|-- mpc.py # MPC to optimize model; (Includes CEM)
|-- model.py # Creat and train the ensemble dynamics model
|-- agent.py # An agent that can be used to interact with the environment
             # and a random policy
|-- util.py # Some utility functions.
|-- opencv_draw.py # Contains features for rendering the Box2D environments.
```

# Prerequisite
Run `pip install opencv-python`
