from envs.cloth import ClothEnv
import utils
import cv2
import numpy as np


args = utils.argsparser()
variant = utils.get_variant(args)
env = ClothEnv(**variant['env_kwargs'])
while True:
    env.reset()
    for s in range(50):
        print("Step", s)
        w, h = 1000, 1000
        env.viewer.render(w, h)
        data = env.viewer.read_pixels(w, h, depth=False)
        data = data[::-1, :, :]
        cv2.imshow('RGB', data)
        cv2.waitKey(10)
        env.step(np.array([0,0,0.3]))