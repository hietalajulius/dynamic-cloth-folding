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
    train_dataset = CornerDataset(os.path.join(folder, "train"), 3, transformation_pipelines=train_transforms)
    val_dataset = CornerDataset(os.path.join(folder, "eval"), 3, transformation_pipelines=None)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    model = ScriptPolicy(100,
                        100,
                        3,
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
        if epoch % 100 == 0:
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