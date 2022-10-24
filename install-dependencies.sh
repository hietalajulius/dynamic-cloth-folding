#!/bin/bash

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
pip install -e .

pushd ./submodules/mujoco-py
pip install -r requirements.txt
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin pip install .
popd

pushd ./submodules/robosuite
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install -e .
popd

pushd ./submodules/rlkit
pip install -e .
popd

pushd ./osc-controller-binding
pip install -e .
popd