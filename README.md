# clothmanip

## Setup:
* Create a fresh directory e.g. `robotics` and clone the `clothmanip` repository there
* Add `export ROBOTICS_PATH="<PATH_TO_NEW_DIR_HERE>"`to the newly created folder into `.bashrc` or similar startup script
* Create and avtivate a fresh python environment (preferrably conda), dependencies have been dumped into `requirements.txt`, although there is some redundancy (do not install directly, only for reference if something is missing)
* Create following folder structure into `ROBOTICS_PATH`: `/osc_ws/src` and clone `mujoco_ros_control_OSC` with `git clone --recursive https://version.aalto.fi/gitlab/hietalj4/mujoco_ros_control_osc` into `/osc_ws/src`
* Clone `osc_binding` into `ROBOTICS_PATH` with `git clone --recursive git@github.com:hietalajulius/osc_binding.git`
* In `ROBOTICS_PATH/osc_binding` run `pip install .` (requires cmake to work)
* Clone `robosuite` with `git clone git@github.com:hietalajulius/robosuite.git` and run `pip install -e .` in `ROBOTICS_PATH/robosuite`
* Clone `rlkit` with `git clone git@github.com:hietalajulius/rlkit.git` and run `pip install -e .` in `ROBOTICS_PATH/rlkit`
* Clone `mujoco-py` with `git clone git@github.com:hietalajulius/mujoco-py.git` and run `pip install -e .` in `ROBOTICS_PATH/mujoco-py`

## Test
* `cd ROBOTICS_PATH/clothmanip/examples && python create_cloth_env.py`
