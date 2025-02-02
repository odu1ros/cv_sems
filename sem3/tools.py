import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

def display(window:str, img:np.ndarray):
    """
    Display image in a separate window

    :param window: window name
    :param img: image array
    """

    cv2.imshow(window, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def display_inline(img:np.ndarray, grey:bool=False, show:bool=True, figsize:tuple=(10, 6)):
    """
    Display image inline with code

    :param img: image array
    :param grey: whether the image is greyscale
    :param show: whether to display an image (False useful for loops)
    :param figsize: width and height of figure
    """

    fig = plt.figure(figsize=figsize)
    if grey:
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

    plt.axis('off')

    if show:
        plt.show()

def display_hist(hist):
    """
    Display pixels distribution histogram
    """

    plt.figure(figsize=(10, 6))
    plt.hist(hist, bins=100)

def scaler(img:np.ndarray, scale:float) -> np.ndarray:
    """
    Shrink an image by n times

    :param img: image array
    :param scale: n times squeezing an image in both dimensions

    :return: image of less dimensinality
    """

    w, h, c = img.shape
    return cv2.resize(img, (h // scale, w // scale))


def display_multiple_inline(images:np.ndarray, cols:int=3, title:str='', img_titles:list=[], grey:bool=False, base_sizes:tuple=(5,4)):
    """
    Display multiple images on the same plot
    
    :param images: array of image arrays
    :param cols: number of columns in the figure grid to be displayed
    :param title: figure title
    :param img_titles: titles of each separate image
    :param grey: whether the images are greyscale
    :param base_sizes: width and height of each subplot
    """

    num_images = len(images)

    rows = np.ceil(num_images / cols).astype(int)
    
    col_size, row_size = base_sizes
    plt.figure(figsize=(cols * col_size, rows * row_size))
    plt.suptitle(title)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if grey:
            plt.imshow(img, cmap='gray')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        
        if len(img_titles) == num_images:
            plt.title(img_titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_noised_image(img:np.ndarray, threshold:float=0.2, sp:bool=False) -> np.ndarray:
    """
    Add noise to an image

    :param img: array of image arrays
    :param threshold: a share of noised pixels
    :param sp: whether to drop/pump intensity of whole pixels (True) or separate channels (False) #salt-pepper

    :return: image with noise applied
    """
    
    h, w, c = img.shape

    if sp:
        random_values = np.random.rand(h, w)

        random_mask = np.where(random_values > 1 - threshold, 255, np.where(random_values < threshold, 0, 1))

        random_pixels = np.where(random_mask[:,:,None] == 255, [255,255,255], \
                                 np.where(random_mask[:,:,None] == 0, [0,0,0], \
                                          1))
    else:
        random_values = np.random.rand(h, w, c)
        
        random_pixels = np.where(random_values > 1 - threshold, 255, np.where(random_values < threshold, 0, 1))

    img_noised = np.where(random_pixels == 1, img, random_pixels)

    return img_noised.astype('uint8')

def iterative_filter(img:np.ndarray, params:list[dict], function_call:callable) -> np.ndarray:
    """
    Applies the same function to an image with different parameters

    :param img: image array
    :param params: list of dictionaries, each containing parameters for one iteration
    :function call: function that takes for input an image and params (as **kwargs), returning processed image

    :return: processed images
    """
    num_images = len(params)
    images = np.empty((num_images, *img.shape), dtype=np.float32)

    for i, param_dict in enumerate(params):
        cur_img = function_call(img, **param_dict)
        cur_img = (cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())
        images[i] = cur_img

    return images