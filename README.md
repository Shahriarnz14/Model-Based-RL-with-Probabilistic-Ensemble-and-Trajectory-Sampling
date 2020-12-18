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