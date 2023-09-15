import cv2
import numpy as np
import dlib
import bz2
import matplotlib.pyplot as plt


if __name__== "__main__":
    
    img = cv2.imread("zoom_pic.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    