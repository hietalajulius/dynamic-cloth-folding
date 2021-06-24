from rlkit.torch.sac.policies import ScriptPolicy
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from collections import deque
import cv2
import albumentations as A
import copy
from train_corner_prediction import validate, CornerDataset
from clothmanip.envs.template_renderer import TemplateRenderer
from clothmanip.utils import mujoco_model_kwargs
import mujoco_py
import random
import cv2

class MujocoDataset(Dataset):
    def __init__(self, folder, transformation_pipelines=None):
        self.folder = folder
        self.transformation_pipelines = transformation_pipelines

        image_files = []
        files = os.listdir(self.folder)
        for file in files:
            if file.split(".")[-1] == "png":
                image_files.append(file)

        self.df = pd.DataFrame(image_files)

        


    def __len__(self):
        return self.df.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transformation_pipeline = None
        if self.transformation_pipelines is not None:
            transformation_pipeline = np.random.choice(self.transformation_pipelines)
        
        image_file = self.df.iloc[idx].iloc[0]
        corners = []
        corner_split = image_file.split("_")
        for i in range(1,9):
            corners.append(corner_split[i])
        image = cv2.imread(os.path.join(self.folder, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transformation_pipeline(image=image)["image"]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        corners_array = np.array(corners, dtype=np.float32)
        image_array = image.flatten().astype("float32")/255.0

        return torch.from_numpy(image_array), torch.from_numpy(corners_array)


def main(folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transforms = [A.Compose(
        [
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Blur (blur_limit=7, always_apply=False, p=0.5),
            A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
        ]
    )]
    train_dataset = MujocoDataset(os.path.join(folder, "labeled_images"), transformation_pipelines=train_transforms)
    val_dataset = CornerDataset(os.path.join(folder, "eval"), 1, transformation_pipelines=None)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    model = ScriptPolicy(100,
                        100,
                        1,
                        3,
                        [3, 3, 3, 3],
                        [32, 32, 32, 32],
                        [2, 2, 2, 2],
                        [0, 0, 0, 0],
                        aux_output_size=8,
                        hidden_sizes_aux=[256, 8],
                        hidden_sizes_main=[256, 256, 256, 256],
                        )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3E-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(200000):  # loop over the dataset multiple times
        save_images = False
        if epoch % 20 == 0:
            save_images = True
            preds_path = f"{folder}/train_images/{epoch}"
            try:
                os.makedirs(preds_path)
            except:
                print("folders existed already")

        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            images, corners = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, _, outputs, _ = model(images)
            loss = criterion(outputs, corners)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

            if save_images and i == 0:
                for image_idx, (image_to_save, image_to_save_corners) in enumerate(zip(images, outputs.detach())):
                    image_detach = image_to_save.cpu().numpy().reshape((-1,100,100))[-1]*255
                    image_corners = image_to_save_corners.cpu().numpy()
                    for aux_idx in range(int(image_corners.shape[0]/2)):
                        aux_u = int(image_corners[aux_idx*2]*100)
                        aux_v = int(image_corners[aux_idx*2+1]*100)
                        cv2.circle(image_detach, (aux_u, aux_v), 2, (0, 255, 0), -1)
                    cv2.imwrite(f'{folder}/train_images/{epoch}/{image_idx}.png', image_detach)
        print(epoch, "Train Epoch loss", epoch_loss)
        validate(val_loader, model, criterion, epoch, device, folder)
        print("\n")

    print('Finished Training')











if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('folder', type=str)

    args = parser.parse_args()
    main(args.folder)