from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from tqdm import tqdm
from imageio import imread

def files_to_data(path):
    path_to_files = path
    files = os.listdir(path=path_to_files)
    files = list(filter(lambda p: os.path.isfile(os.path.join(path_to_files, p)) and p.endswith(".ppm"), files))
    images1 = []
    for i in range(len(files)):
        images1 += [imread(os.path.join(path_to_files, files[i]))]

    return(images1)


def jpg_image_to_array(image, size):

    im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((size[0], size[1], 3))                                   
    return im_arr

#data = np.load(r"C:\Users\Mike\Desktop\Studium\TUHH\car-firmware-master\Projekt\GTSRB\images.npy")
#np.save(r"C:\Users\Mike\Desktop\Studium\TUHH\car-firmware-master\Projekt\GTSRB\newdata.npy", newdata)


def resize_image(data, size):
   newdata = []

   for i in range(len(data)):
      image = Image.fromarray(data[i], 'RGB')
      image = image.resize(size)
      newdata.append(jpg_image_to_array(image, size))
   return(newdata)











