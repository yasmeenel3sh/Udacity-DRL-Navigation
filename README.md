# Udacity-DRL-Navigation
This repo is an implementation to the first project, called Navigation, in the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
<p align="center"><img src=banana.gif></p>

## Project Description 
In this project, it is required to train an agent to be able to navigate and collect as much yellow bananas it can without hitting blue bananas. It is an episodic task and the training ends when the agent manages to have a score of +13 or above for 100 consecutive episodes.

### State Space
The state space size is 37 which contains the agent's velocity and other need values for moving.

### Action space
The agent can do 4 actions.
- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

### Rewards
The agent receive a reward of +1 if it runs into a yellow banana, and -1 if it runs into a blue banana.

### Project Environment
#### Step 1: Clone the [DRLND repo](https://github.com/udacity/deep-reinforcement-learning). Follow the instructions in the readme file to configure the python (3.6 was used in this project) environment.
#### Step 2: Download the Unity Environment according to your operating system:
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  
Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

#### Step 3: Run
In the terminal run the following command (make sure you are in the correct directory):
```shell
$ jupyter notebook
```
This will open jupyter notebook in the browser, in which you can add the files from this repository to be able to run the code implemented here. 
- Navigation.ipynb contains the code that you should run to train the agents.
- model.py contains the Vanille DQN and the Dueling DQN networks.
- dqn_agent.py contains the implementation of the agent step, act and learn, as well as the replay buffer.
- checkpoint files contain the trained models that can be loaded and used directly.

