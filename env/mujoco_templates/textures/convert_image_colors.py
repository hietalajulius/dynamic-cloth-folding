import os
import cv2








def convert_colors():
    files_dir = os.path.dirname(os.path.abspath(__file__))

    for file in os.listdir(files_dir):
        filename, extension = file.split(".")
        print(filename, extension)
        if extension == "jpg":
            image_file_path = os.path.join(files_dir, file)
            image = cv2.imread(image_file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{files_dir}/{filename}.png', image)
            print("Saved")







if __name__ == "__main__":
    convert_colors()