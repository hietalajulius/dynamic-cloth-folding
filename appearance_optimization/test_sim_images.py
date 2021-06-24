import torch
import argparse
from collections import deque
import cv2
import os
import numpy as np
import albumentations as A

transformation = A.Compose(
        [
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Blur (blur_limit=7, always_apply=False, p=0.5),
            A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
        ]
    )

def main(folder):
    model = torch.jit.load(f"{folder}/600_policy.pt")
    model.eval()

    frame_stack = deque([], maxlen = 3)
    images_file_path = os.path.join(folder, "sim_eval_images")
    first_image_file_path = os.path.join(images_file_path, "1.png")
    first_image = cv2.imread(first_image_file_path)
    for _ in range(3):
        trans_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        trans_image = transformation(image=trans_image)["image"]
        trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2GRAY)
        frame_stack.append(trans_image.flatten()/255)

    for image_file in sorted(os.listdir(images_file_path)):
        file, suffix = image_file.split(".")
        if suffix == "png":
            image_file_path = os.path.join(images_file_path, image_file)
            image = cv2.imread(image_file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transformation(image=image)["image"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            frame_stack.append(image.flatten()/255)

            input = np.array([image for image in frame_stack], dtype=np.float32).flatten()
            input_tensor = torch.reshape(torch.from_numpy(input), (1,-1))

            print("input shape", input_tensor.shape)

            _, _, preds, _ = model(input_tensor)
            corners = preds[0].detach().cpu().numpy()
            dotted_image = image.copy()
            for aux_idx in range(int(corners.shape[0]/2)):
                aux_u = int(corners[aux_idx*2]*100)
                aux_v = int(corners[aux_idx*2+1]*100)
                cv2.circle(dotted_image, (aux_u, aux_v), 2, (0, 255, 0), -1)
            cv2.imwrite(f'{folder}/sim_eval_predictions/{file}.png', dotted_image)
            print("Preds", preds)




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('folder', type=str)

    args = parser.parse_args()
    main(args.folder)