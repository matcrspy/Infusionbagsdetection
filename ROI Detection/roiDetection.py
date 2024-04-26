# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:59:20 2022

@author: Matth
"""
import cv2
import numpy as np
import glob
import sys
import os

from matplotlib import pyplot as plt
from PIL import Image 
import PIL 

#判断黑色图片或者白色图片
def isDarkImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.imshow(binary)
    plt.show()
    mean = np.mean(binary)
    print(mean)
    if(mean > 50):
        return 0
    else:
        return 1
    
#模板匹配画轮廓
def generateROI(processed_img, methods, template):
    folder = 'testset'
    pattern = folder + '\\*\\*.bmp'
    count = 0
    
    img_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    h, w = template.shape[:2] 
    for meth in methods:
        img2 = img.copy()
        method = eval(meth)
        res = cv2.matchTemplate(img_gray, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.circle(res, top_left, 10, 0, 2)
            
        cv2.rectangle(img2, top_left, bottom_right, (0, 255, 0), 2)
        cv2.rectangle(img2, (bottom_right[0] - 100, 250), (2000,1750), (0, 255, 0), 2)
        print(meth)
        print(top_left)
        plt.imshow(img2)
        plt.show()
        status = cv2.imwrite("resultbag2.jpg", img2)
        print(status)
        return((top_left, bottom_right))
    
if __name__ == "__main__":
    image = cv2.imread("test.bmp")
    if isDarkImage(image) == 1:
        print("dark")
        template = cv2.imread('portgoal3.bmp', 0)
        methods = ['cv2.TM_CCORR_NORMED']
        box = generateROI(image, methods, template)
    else:
        print("white")
        template = cv2.imread('portgoal.bmp', 0)
        methods = ['cv2.TM_CCOEFF_NORMED']
        box = generateROI(image, methods, template)
    print("the to points for the bounding box for the port are:")
    print(box)
        
    
    
    