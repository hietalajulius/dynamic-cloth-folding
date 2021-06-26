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
from clothmanip.envs.template_renderer import TemplateRenderer
from clothmanip.utils import mujoco_model_kwargs
import mujoco_py
import random
import cv2
import re

def validate(val_loader, model, criterion, epoch, device, folder, frame_stack_size, save_images_every_epoch):
    model.eval()
    eval_loss = 0
    save_images = False

    if epoch % save_images_every_epoch == 0:
        saved_model = copy.deepcopy(model)
        sm = torch.jit.script(saved_model).cpu()
        torch.jit.save(sm, f'{folder}/{epoch}_policy.pt')

        save_images = True
        preds_path = f"{folder}/eval_images/{epoch}"
        try:
            os.makedirs(preds_path)
        except:
            print("folders existed already")

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, corners = data[0].to(device), data[1].to(device)
            _, _, outputs, _ = model(inputs)
            loss = criterion(outputs, corners)
            eval_loss += loss.item()

            if save_images:
                image = inputs[0].cpu().numpy()[:int(frame_stack_size*10000)].reshape((-1,100,100))[-1]*255
                corners = outputs[0].cpu().numpy()
                for aux_idx in range(int(corners.shape[0]/2)):
                    aux_u = int(corners[aux_idx*2]*100)
                    aux_v = int(corners[aux_idx*2+1]*100)
                    cv2.circle(image, (aux_u, aux_v), 2, (0, 255, 0), -1)
                cv2.imwrite(f'{folder}/eval_images/{epoch}/{i}.png', image)
    print(epoch, "Eval loss", eval_loss)

class CornerDataset(Dataset):
    def __init__(self, folder, frame_stack_size, transformation_pipelines=None):
        self.folder = folder
        self.frame_stack_size = frame_stack_size
        self.transformation_pipelines = transformation_pipelines
        total_images = 0
        self.file_index = dict()
        for image_folder in os.listdir(self.folder):
            image_dir_path = os.path.join(self.folder, image_folder)
            for file in os.listdir(image_dir_path):
                if file.split(".")[1] in ["png"]:
                    self.file_index[str(total_images)] = dict(folder=image_dir_path, file=file)
                    total_images += 1
        self.total_images = total_images

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder = self.file_index[str(idx)]['folder']
        image_file = self.file_index[str(idx)]['file']
        image_index = int(image_file.split(".")[0])

        labels = pd.read_csv(f"{folder}/labels.csv", names=["corner", "u", "v", "file", "w", "h"])
        c0 = labels[(labels["corner"] == 0) & (labels["file"] == f'{image_index}.png')]
        c1 = labels[(labels["corner"] == 1) & (labels["file"] == f'{image_index}.png')]
        c2 = labels[(labels["corner"] == 2) & (labels["file"] == f'{image_index}.png')]
        c3 = labels[(labels["corner"] == 3) & (labels["file"] == f'{image_index}.png')]

        corners = [c0, c1, c2, c3]
        corners_values = []
        for c in corners:
            corner_value = np.array([c['u']/100, c['v']/100]).flatten()
            if corner_value.shape[0] == 2:
                corners_values.append(corner_value)
            else:
                corners_values.append(np.zeros(2))
        
        corners_array = np.array(corners_values, dtype=np.float32).flatten() 

        frame_stack = deque([], maxlen = self.frame_stack_size)
        transformation_pipeline = None
        if self.transformation_pipelines is not None:
            transformation_pipeline = np.random.choice(self.transformation_pipelines)

        for i in range(-self.frame_stack_size+1, 1):
            stack_idx = max(1, image_index+i)
            stack_image_file = os.path.join(folder, f"{stack_idx}.png")
            stack_image = cv2.imread(stack_image_file)
            stack_image = cv2.cvtColor(stack_image, cv2.COLOR_BGR2RGB)
            if transformation_pipeline is not None:
                stack_image = transformation_pipeline(image=stack_image)["image"]
            stack_image = cv2.cvtColor(stack_image, cv2.COLOR_RGB2GRAY)            
            frame_stack.append(stack_image.flatten()/255)
        images = np.array([image for image in frame_stack], dtype=np.float32).flatten()
        return torch.cat((torch.from_numpy(images), torch.zeros(27))), torch.from_numpy(corners_array)

class MujocoDataset(Dataset):
    def __init__(self, folder, frame_stack_size, transformation_pipelines=None):
        self.folder = folder
        self.transformation_pipelines = transformation_pipelines
        self.frame_stack_size = frame_stack_size
        image_files = []
        files = os.listdir(self.folder)
        for file in sorted(files[:10]):
            ext_split = file.split(".")
            if ext_split[-1] == "png": #TODO: BUGGY
                file_split = ext_split[0].split("_")
                rollout_step = int(file_split[0])
                step_number = int(file_split[1])
                process_number = int(file_split[2])
                frame_stack_files = []

                skip = False
                for stack_idx in range(-self.frame_stack_size+1, 1):
                    actual_stack_idx = max(0,step_number+stack_idx)
                    actual_rollout_step = max(0,rollout_step+stack_idx)
                    matches = [match for match in files if match.startswith(f"{actual_rollout_step}_{actual_stack_idx}_{process_number}")]
                    if len(matches) == 0:
                        skip = True
                    else:
                        frame_stack_files.append(matches[0])
                if not skip:
                    image_files.append(frame_stack_files)
        self.df = pd.DataFrame(image_files)

        


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transformation_pipeline = None
        if self.transformation_pipelines is not None:
            transformation_pipeline = np.random.choice(self.transformation_pipelines)
        
        
        corners = []
        last_frame_corners = self.df.iloc[idx].iloc[-1].split("_")
        for c_idx in range(3,11):
            corner = last_frame_corners[c_idx].split(".png")[0]
            corners.append(corner)

        frame_stack = []

        for img_file in self.df.iloc[idx]:
            image = cv2.imread(os.path.join(self.folder, img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transformation_pipeline(image=image)["image"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            frame_stack.append(image.flatten())


        corners_array = np.array(corners, dtype=np.float32)
        image_array = np.array([image for image in frame_stack]).flatten().astype("float32")/255.0

        return torch.cat((torch.from_numpy(image_array), torch.zeros(27))), torch.from_numpy(corners_array)


def main(args):
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
    train_dataset = MujocoDataset(os.path.join(args.folder, "labeled_images"), args.frame_stack_size, transformation_pipelines=train_transforms)
    val_dataset = CornerDataset(os.path.join(args.folder, "eval"), args.frame_stack_size, transformation_pipelines=None)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    model = ScriptPolicy(100,
                        100,
                        args.frame_stack_size,
                        3,
                        [3, 3, 3, 3],
                        [32, 32, 32, 32],
                        [2, 2, 2, 2],
                        [0, 0, 0, 0],
                        added_fc_input_size=27,
                        aux_output_size=8,
                        hidden_sizes_aux=[256, 8],
                        hidden_sizes_main=[256, 256, 256, 256],
                        )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3E-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(200000):  # loop over the dataset multiple times
        save_images = False
        if epoch % args.save_every_epoch == 0:
            save_images = True
            preds_path = f"{args.folder}/train_images/{epoch}"
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
                    image_detach = image_to_save.cpu().numpy()[:10000].reshape((-1,100,100))[-1]*255
                    image_corners = image_to_save_corners.cpu().numpy()
                    for aux_idx in range(int(image_corners.shape[0]/2)):
                        aux_u = int(image_corners[aux_idx*2]*100)
                        aux_v = int(image_corners[aux_idx*2+1]*100)
                        cv2.circle(image_detach, (aux_u, aux_v), 2, (0, 255, 0), -1)
                    cv2.imwrite(f'{args.folder}/train_images/{epoch}/{image_idx}.png', image_detach)
        print(epoch, "Train Epoch loss", epoch_loss)
        validate(val_loader, model, criterion, epoch, device, args.folder, args.frame_stack_size, args.save_every_epoch)
        print("\n")

    print('Finished Training')











if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('folder', type=str)
    parser.add_argument('frame_stack_size', type=int)
    parser.add_argument('save_every_epoch', type=int)

    args = parser.parse_args()
    main(args)