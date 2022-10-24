# Learning Visual Feedback Control for Dynamic Cloth Folding (IROS 2022)

<p align="center">
  <img src="https://user-images.githubusercontent.com/4254623/197352162-4ef75923-7618-43c4-8f87-2796acead602.gif" alt="The above gif has been slowed down to highlight the cloth dynamics, full speed videos can be found [here](https://sites.google.com/view/dynamic-cloth-folding/home)">
  <h5 align="center">The above gif has been slowed down to highlight the cloth dynamics, full speed videos can be found at https://sites.google.com/view/dynamic-cloth-folding/home</h6>
</p>

## Introduction
This repository contains code and instructions on how to run the simulation training presented in our paper [Learning Visual Feedback Control for Dynamic Cloth Folding](https://sites.google.com/view/dynamic-cloth-folding/home) published at 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2022).

The setup has been tested on Ubuntu 22.04.1 LTS with Asus ROG Strix G513IE_G513IE  (GeForce RTX 3050 Ti Mobile, Driver Version: 510.85.02, CUDA Version: 11.6).

Please contact the authors for instructions on how to run trained policies in the real world (our real-world setup included the Franka Emika Panda and Intel Realsense D435, but other robots and sensors can also be used).

## Installation

* Run `conda create -n dynamic-cloth-folding python=3.8.13 && conda activate dynamic-cloth-folding` to create and activate a new python environment (conda is not a strict requirement) 
* Run `git clone git@github.com:hietalajulius/dynamic-cloth-folding.git --recursive` to clone the repository and necessary submodules
* Install [MuJoCo](https://mujoco.org/) using these [instructions](https://github.com/hietalajulius/mujoco-py/tree/8131d34070e684705990ef25e5b3f211e218e2e4#install-mujoco) (i.e. extract the downloaded `mujoco210` directory into ~/.mujoco/mujoco210)
* Run `cd dynamic-cloth-folding && ./install-dependencies.sh` to install all required dependencies (assumes CUDA 11.6 and a compatible Nvidia GPU)

## Training a model

<p align="center">
  <img src="https://user-images.githubusercontent.com/4254623/197414575-c791c1ba-e4b9-453a-a020-5e9832c64048.gif" >
  <h5 align="center">The gif shows the RL simulation environment. It is implemented using MuJoCo and OpenAI Gym</h6>
</p>

* Run `source env.sh` before each training to set proper environment variables
* Run `python train.py <--kwarg value>` for a full training

Each training run will create a new directory in the `trainings` directory that contains
* The current neural network weights
* The settings used for the run
* The git commit hashes of the submodules
* The compiled MuJoCo models used during training
* The evaluation policy and images captured during evaluation for each epoch

Training progress and metrics can be visualized using tensorboard by running `tensorboard --logdir tblogs` in the root of the repository and opening `http://localhost:6006/` in your browser.

### Settings
The different settings are passed to the training script as keyword arguments. The available settings are:

#### General
| Kwarg | Type | Default value | Description |
| --- | --- | --- | --- |
| `--title` | String | `default` | Give a title to the training run, used in filenames when saving progress. | 
| `--run` | Int | `0` | Run number, used to generate random seeds. Useful when running multiple experiments with the same setup. | 
| `--num-processes` | Int | `1` | How many python multiprocesses/parallel RL environments to use when collecting experience for the training.  |

#### Environment
The environment consists of a Franka Emika Panda robot (agent) and a cloth model lying on a flat surface. Visual feedback is captured using a camera whose settings are tuned to approximately match the Realsense D435 camera.
| Kwarg | Type | Default value | Description |
| --- | --- | --- | --- |
| `--cloth-size` | Float | `0.2` | The cloth size in meters | 
| `--robot-observation` | Choice[`ee`, `ctrl`, `none`] | `ctrl` | Whether the policy/value functions should observe the true end effector position (`ee`) or the current desired position of the controller (`ctrl`) or `none`  | 
| `--filter` | Float | `0.03` | The filter value to use in the convex combination interpolation of the controller desired position between time steps  | 
| `--output-max` | Float | `0.03` | The maximum Cartesian displacement in any direction of the previous controller desired position between time steps i.e. the maximum action from the policy  | 
| `--damping-ratio` | Float | `1.0` | The damping ratio of the OSC controller  | 
| `--kp` | Float | `1000.0` | Controller position gain  |
| `--success-distance` | Float | `0.05` | The minimum distance within which the considered cloth points should be from their goals for the task to be considered successful  |
| `--frame-stack-size` | Int | `1` | How many consecutive frames should be stacked together in the observation |
| `--sparse-dense` | Int (`1=true`, `0=false`) | `1` | Whether the reward should increase linearly with the error if the task is considered successful |
| `--success-reward` | Int | `0` | The reward at a time step when the task is considered successful |
| `--fail-reward` | Int | `-1` | The reward at a time step when the task is considered unsuccessful |
| `--extra-reward` | Int | `1` | When using a sparse dense reward, the maximum extra reward that can be achieved |
| `--timestep` | Float | `0.01` | Simulation timestep length |
| `--control-frequency` | Int | `10` | Control frequency (Hz) |
| `--camera-type` | Choice[`up`, `side`, `front`, `all`] | `side` | The camera angle to use during training |
| `--camera-config` | Choice[`small`, `large`] | `small` | Defines the camera fovy range to use. `small` corresponds to a 848x100 image on the real camera whereas `large` corresponds to a 848x480 image on the real camera. |

#### Domain randomization

| Kwarg | Type | Default value | Description |
| --- | --- | --- | --- |
| `--goal-noise` | Float | `0.03` | The range within which the goal positions of the cloth points should be randomized |
| `--image-obs-noise-mean` | Float | `0.5` | The mean of the gaussian defining what the delay is between taking an action and observing an image within a `[0,1]` timespan between actions | 
| `--image-obs-noise-std` | Float | `0.5` | The std of the gaussian defining what the delay is between taking an action and observing an image within a `[0,1]` timespan between actions | 
| `--lights-randomization` | Int (`1=true`, `0=false`)  | `1` | Whether lights in the environment should be randomized | 
| `--materials-randomization` | Int (`1=true`, `0=false`)  | `1` | Whether the visual cloth material properties in the environment should be randomized | 
| `--camera-position-randomization` | Int (`1=true`, `0=false`)  | `1` | Whether the camera position in the environment should be randomized | 
| `--lookat-position-randomization-radius` | Float  | `0.03` | Within what radius in meters should the camera lookat position be randomized from the center of the cloth| 
| `--lookat-position-randomization` | Int (`1=true`, `0=false`)  | `1` | Whether the lookat position should be randomized | 
| `--albumentations-randomization` | Int (`1=true`, `0=false`)  | `1` | Whether the albumentations library should be used to randomly blur and adjust colors of the image observations |
| `--dynamics-randomization` | Int (`1=true`, `0=false`)  | `1` | Whether the cloth dynamics should be randomized | 
| `--fovy-perturbation-size` | Float  | `0.05` | The max percentage that the camera fovy should be randomized | 
| `--rotation-perturbation-size` | Float  | `0.75` | The max percentage that the camera rotation should be randomized | 
| `--position-perturbation-size` | Float  | `0.2` | The max percentage that the camera position should be randomized |


#### Training
The training is structured as follows: one `epoch` consists of `n_cycles` cycles. A `cycle` consists of `n_steps`. A `step` means a single step in the RL environment, but also a single gradient update of the model.
| Kwarg | Type | Default value | Description |
| --- | --- | --- | --- |
| `--train-steps` | Int | `1000` | How many `steps` should be performed per a single `cycle` | 
| `--num-cycles` | Int | `20` | How many `cycles` should be performed per a single `epoch` | 
| `--num-epochs` | Int | `100` | How many `epochs` to run for |
| `--save-policy-every-epoch` | Int | `1` | How often should the policy be saved during training |
| `--num-eval-rollouts` | Int | `20` | How many evaluation rollouts (until success or `--max-path-length` environment steps per rollout) to perform after each `epoch` |
| `--batch-size` | Int | `256` | The batch size to use during training the policy network |
| `--discount` | Float | `0.99` | The discount factor in the RL problem |
| `--corner-prediction-loss-coef` | Float | `0.001` | The weight of the cloth corner predction loss when updating gradients |
| `--save-images-every-epoch` | Int | `10` | How often should evaluation images from the RL environment be saveed during training |
| `--fc-layer-size` | Int | `512` | The fully-connected layer size to use in the policy and value networks |
| `--fc-layer-depth` | Int | `5` | The fully-connected layer depth to use in the policy and value networks |
| `--her-percent` | Float | `0.8` | Percentage of training samples whose goal should be recalculated using [HER](https://arxiv.org/abs/1707.01495) |
| `--buffer-size` | Int | `100000` | The maximum number of steps to store in the replay buffer |
| `--num-demoers` | Int | `0` | How many of the parallel RL environments should use an agent only executing a demonstration with added noise |
| `--max-path-length` | Int | `50` | The maximum number of steps the agent can take in the environment before it resets |
| `--max-close-steps` | Int | `10` | The maximum number of steps the task can be considered successful before the environment resets |


## Misc
* The `./data` folder contains a file `demos.csv` that is the real-world demonstration that is used along with random noise during training
* The `./data` folder also contains a file `model_params.csv` that contains the cloth dynamics parameters that are used during dynamics randomization
