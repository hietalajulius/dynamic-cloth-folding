# Learning Visual Feedback Control for Dynamic Cloth Folding

![ezgif com-gif-maker](https://user-images.githubusercontent.com/4254623/197352162-4ef75923-7618-43c4-8f87-2796acead602.gif)


Code for our paper: [Learning Visual Feedback Control for Dynamic Cloth Folding](https://sites.google.com/view/dynamic-cloth-folding/home)
published at 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2022)

## Introduction
This repository contains code and instructions on how to run the simulation training presented in our paper [Learning Visual Feedback Control for Dynamic Cloth Folding](https://sites.google.com/view/dynamic-cloth-folding/home).

The setup has been tested on Ubuntu 22.04.1 LTS with Asus ROG Strix G513IE_G513IE  (GeForce RTX 3050 Ti Mobile, Driver Version: 510.85.02, CUDA Version: 11.6).

Please contact the authors for instructions on how to run trained policies in the real world (our real-world setup included the Franka Emika Panda and Intel Realsense D435, but other robots and sensors could also be used).

## Installation

* Run `conda create -n dynamic-cloth-folding python=3.8.13 && conda activate dynamic-cloth-folding` to create and activate a new python environment (conda is not a strict requirement) 
* Run `git clone git@github.com:hietalajulius/dynamic-cloth-folding.git --recursive` to clone the repository and necessary submodules
* Install [MuJoCo](https://mujoco.org/) using these [instructions](https://github.com/hietalajulius/mujoco-py/tree/8131d34070e684705990ef25e5b3f211e218e2e4#install-mujoco) (i.e. extract the downloaded `mujoco210` directory into ~/.mujoco/mujoco210)
* Run `cd dynamic-cloth-folding && ./install-dependencies.sh` to install all required dependencies (assumes CUDA 11.6 and a compatible Nvidia GPU)

## Training a model
* Run `source source.sh` before each training to set proper environment variables
* Run `python train.py` for a full training
