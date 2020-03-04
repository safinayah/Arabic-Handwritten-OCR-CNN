import cv2
import numpy as np

from matplotlib import pyplot as plt

cleanImage = cv2.imread('/home/ayah/Desktop/bab.jpg')

dst = cv2.fastNlMeansDenoisingColored(cleanImage, None, 10, 10, 7, 21)

plt.subplot(121), plt.imshow(cleanImage)
plt.subplot(122), plt.imshow(dst)
plt.show()