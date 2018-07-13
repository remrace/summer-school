import os
from skimage import io
from skimage.util import random_noise
from skimage import filters
for filename in os.listdir('./synimage/original'):
    if filename.endswith(".bmp"):
        image = io.imread(os.path.join('./synimage/original', filename))
        for i in range(3):
            image = random_noise(image, mode='gaussian')
            image = filters.gaussian(image, sigma=0.7)
        root, ext = os.path.splitext(filename)
        name = os.path.join('./synimage/noise/', root+'.png')
        io.imsave(fname=name ,arr=image)