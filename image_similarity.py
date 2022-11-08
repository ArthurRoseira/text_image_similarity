import numpy as np
from skimage.metrics import structural_similarity,normalized_root_mse
import cv2
from PIL import Image
import requests
from io import BytesIO



def rgb_similarity(img_a,img_b):
    img_a = Image.open(img_a)
    img_b  = Image.open(img_b)
    img_a = img_a.resize((500, 500))
    img_b = img_b.resize((500, 500))
    img1 = img_a.getcolors(maxcolors=img_a.size[0]*img_a.size[1])
    img2 = img_b.getcolors(maxcolors=img_b.size[0]*img_b.size[1])
    img1= dict(map(lambda x: (x[1],x[0]),img1))
    img2 = dict(map(lambda x: (x[1],x[0]),img2))
    results = []
    anchor_list = list(set(list(img1.keys()) + list(img2.keys())))
    for color in anchor_list:
        if color in img1.keys() and color in img2.keys():
            colorpixels1 = img1[color]/(500*500)
            colorpixels2 = img2[color]/(500*500)
            delta = abs(colorpixels1-colorpixels2)/(colorpixels1+colorpixels2)
            results.append(delta)
        else:
            results.append(1) 
    return 1 - np.mean(results)

def gray_scale_metrics(img_a,img_b):
    try:
        imageA = cv2.imread(img_a)
        imageB = cv2.imread(img_b)
        grayA = cv2.resize(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY),(500,500))
        grayB = cv2.resize(cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY),(500,500))  
    except:
        grayA = cv2.resize(cv2.cvtColor(np.array(Image.open(img_a)), cv2.COLOR_BGR2GRAY),(500,500))
        grayB = cv2.resize(cv2.cvtColor(np.array(Image.open(img_b)), cv2.COLOR_BGR2GRAY),(500,500))
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    return score,1-normalized_root_mse(grayA, grayB)


def image_similarity(img_a,img_b):
    a = rgb_similarity(img_a,img_b)
    b,c = gray_scale_metrics(img_a,img_b)
    print(a,b,c)
    return (a+0.7*b+0.3*c)/2


def get_image_url(url:str):
    image_data = requests.get(url)
    img = BytesIO(image_data.content)
    return img

