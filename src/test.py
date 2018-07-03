from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
im = Image.open('image.jpg').convert("L").crop((50,50,100,100))
op = ImageOps.autocontrast(im, cutoff = 30)

plt.imshow(mpimg.pil_to_array(op), cmap='gray')
plt.show()