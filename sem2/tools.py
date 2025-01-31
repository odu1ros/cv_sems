import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

def display(window, img):
    cv2.imshow(window, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def display_inline(img, grey=False):
    fig = plt.figure(figsize=(10, 6))
    if grey:
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()

def display_hist(hist):
    plt.figure(figsize=(10, 6))
    plt.hist(hist, bins=100)