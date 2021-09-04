import cv2
import numpy as np


im_cv = cv2.imread('/home/julius/robotics/clothmanip/clothmanip/envs/mujoco_templates/textures/white.png')

im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

cv2.imwrite('/home/julius/robotics/clothmanip/clothmanip/envs/mujoco_templates/textures/white.png', im_cv)