# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:59:20 2022

@author: Matth, Jiaqi Lu
"""
from calendar import c
from concurrent.futures import process
from multiprocessing import dummy
import cv2
import numpy as np
import glob
import sys
import os

from matplotlib import pyplot as plt
from PIL import Image 
import PIL 
from pickle import FALSE, TRUE
import imutils

filter_con_min = 0.25
filter_con_max = 0.6

# read images of .bmp format
def load_images_bmp(fold):
    pattern = fold + '\\*\\*.bmp'
    i_lit = []
    n_lst = []
    for path in glob.glob(pattern):
        print(path)
        n_lst.append(path)
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        i_lit.append(image)
    return i_lit, n_lst

# read images of .bmp format, but not for nested folders
def load_images_bmp_single(fold):
    pattern = fold + '\\*.bmp'
    i_lit = []
    n_lst = []
    for path in glob.glob(pattern):
        print(path)
        n_lst.append(path)
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        i_lit.append(image)
    return i_lit, n_lst

# read images of .jpg format
def load_images_jpg(fold):
    pattern = fold + '\\*.jpg'
    i_lst = []
    n_lst = []
    for path in glob.glob(pattern):
        print(path)
        n_lst.append(path)
        i_lst.append(cv2.imread(path))
    return i_lst, n_lst

#判断黑色图片或者白色图片
def isDarkImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #plt.imshow(binary)
    #plt.show()
    mean = np.mean(binary)
    #print(mean)
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
        bag_top_left = (bottom_right[0]-100,250)
        bag_bottom_right=(2000,1750)
        cv2.rectangle(img2, (bottom_right[0] - 100, 250), (2000,1750), (0, 255, 0), 2)
        #print(meth)
        #print(top_left)
        #plt.imshow(img2)
        #plt.show()
        status = cv2.imwrite("resultbag2.jpg", img2)
        #print(status)
        return((top_left, bottom_right),(bag_top_left,bag_bottom_right))

# Cropping out interested area for processing
def extraction(src, top_left, bottom_right):
    original = src
    blank = np.zeros(src.shape[:2], dtype='uint8')
    mask = cv2.rectangle(blank,top_left,bottom_right,255,-1)
    masked = cv2.bitwise_and(src,src,mask=mask)
    #cv2.imshow("Cropped",masked)
    return original,masked

# Cropping out but using white background
def extraction_white(src, top_left, bottom_right):
    original = src
    blank = np.zeros(src.shape[:2], dtype='uint8')
    mask = cv2.rectangle(blank,top_left,bottom_right,150,-1)
    masked = cv2.bitwise_and(src,src,mask=mask)
    #cv2.imshow("Cropped",masked)
    return original,masked

# Apply Threshold to the image
def thresholding(src,mode=1):
    #im = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,2)
    _,im = cv2.threshold(src,247,255,cv2.THRESH_BINARY)
    #cv2.imshow("Threshold",im)
    #cv2.waitKey(0)

    return im

# Apply Blur to the image
def blur(src, mode=1):
    im = cv2.GaussianBlur(src,(13,13),0)

    #cv2.imshow("Blurred",im)
    #cv2.waitKey(0)

    return im

# Crop Image
def crop(src,top_left,bottom_right):
    cropped = src[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    return cropped


# Show Result
def show_result(name,src):
    width = int(src.shape[1]*0.5)
    height = int(src.shape[0]*0.5)
    dim=(width,height)
    resized = cv2.resize(src,dim,interpolation=cv2.INTER_AREA)
    cv2.imshow(name,resized)
    return src


# Pre-Process Image
def preprocess(src, show=FALSE):
    # for image in image_list:
    rows, cols, _channels = map(int, src.shape)
    # right now try to prevent down sampling to preserve original image size
    #scaled_img = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))  # down sampling
    scaled_img = src
    # 
    show = False
    if show:
        cv2.imshow("original", scaled_img)
    # image enhancing
    # converting to LAB color space
    lab = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe1 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    cl = clahe1.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if show:
        cv2.imshow('enhance1: CLAHE', im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    if show:
        cv2.imshow("grey", im)
    im = cv2.equalizeHist(im)  # 直方图均衡化
    if show:
        cv2.imshow("enhance2:", im)
    im = blur(im, 2)
    if show:
        cv2.imshow("blur", im)
    # th = thresholding(im, 2)  # binary
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im, scaled_img

# Pre-Process for dark backgrounds
def preprocess_dark(src, show=FALSE):
    # for image in image_list:
    rows, cols, _channels = map(int, src.shape)
    # right now try to prevent down sampling to preserve original image size
    #scaled_img = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))  # down sampling
    scaled_img = src
    # 
    show = False
    if show:
        cv2.imshow("original", scaled_img)
    # image enhancing
    # converting to LAB color space
    lab = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe1 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    cl = clahe1.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if show:
        cv2.imshow('enhance1: CLAHE', im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    if show:
        cv2.imshow("grey", im)
    im = cv2.equalizeHist(im)  # 直方图均衡化
    if show:
        cv2.imshow("enhance2:", im)
    im = bag_blur(im, 2)
    if show:
        cv2.imshow("blur", im)
    # th = thresholding(im, 2)  # binary
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im, scaled_img

def bag_blur(new_image, mode=1):
    if mode == 1:
        new_image = cv2.GaussianBlur(new_image, (13, 13), 0)
        # new_image = cv2.GaussianBlur(new_image, (0, 0), sigmaX = 2.5, sigmaY = 2.5)
    elif mode == 2:
        new_image = cv2.medianBlur(new_image, 5)
    elif mode == 3:
        new_image = cv2.bilateralFilter(new_image, 9, 75, 75)
    return new_image

# Filter small contours
def filter_small_contour(src):
    contours_list, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ImageArea = image.shape[0] * image.shape[1]
    filtered_contours =  []
    for i in contours_list:
        # criteria for filtering
        area = cv2.contourArea(i)
        #print(area)
        if area>=(35) and area<=160:
            filtered_contours.append(i)
    return filtered_contours

# Filter small contours (2nd time)
def filter_small_contour_second(src):
    filtered_contours =  []
    for i in src:
        # criteria for filtering
        area = cv2.contourArea(i)
        #print(area)
        if area>=(35):
            filtered_contours.append(i)
    return filtered_contours

# Find biggest Contour
def find_biggest_contour(src):
    biggest = src[0]
    for i in src:
        area = cv2.contourArea(i)
        biggest_area = cv2.contourArea(biggest)
        if(area>biggest_area):
            biggest=i
    return biggest




# Detect Box Contour
def locate_contour(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(src, kernel)
    edges = imutils.auto_canny(dilate)
    #show_result("Canny",edges)
    #cv2.waitKey(0)

    ini_contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    dummy_edge = ini_contours[0]

    # get rid of small contours
    good_contours = filter_small_contour(edges)
    mask = np.ones(edges.shape[:2],dtype="uint8")*255
    contourImg = cv2.drawContours(mask, good_contours, -1, (0,255,0), 2)
    contourImg = cv2.bitwise_not(contourImg)
    cv2.imshow("Contours", contourImg)
    cv2.waitKey(0)

    edges = contourImg
   
    for cnt in good_contours:
        size = cv2.contourArea(cnt)
        #print("printing cnt size")
        #print(size)

    edges = blur(edges,1)
    #good_contours=filter_small_contour(edges)
    new_good_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if(len(new_good_contours)==0):
        return edges,dummy_edge

    biggest = find_biggest_contour(new_good_contours)
    #new_good_contours = filter_small_contour_second (new_good_contours)
    #for cnt in new_good_contours:
    #    size = cv2.contourArea(cnt)
    #    print("printing cnt size")
    #    print(size)

    return edges,biggest

# Draw Box
def locate_box(src,gc):
    #print("printing gc ")
    #print(gc)
    #print(len(gc))
    #if(len(gc)==0):
    #    return src,gc
    cnt=gc
    #if (len(gc)>1):
    #    cnt=np.vstack([gc[0],gc[1]])
    boundRect = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    mask = src
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    #print(box)
    mask = cv2.drawContours(mask,[box],-1,(0,0,255),4 )
    mask = cv2.rectangle(mask, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,0), 2)

    #show_result("box",mask)
    #cv2.waitKey(0)
    return mask,box,[int(boundRect[0]), int(boundRect[1])],[int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])]


# Adapting from Yilin's Code for bag location
def bag_blur(new_image, mode=1):
    if mode == 1:
        new_image = cv2.GaussianBlur(new_image, (13, 13), 0)
        # new_image = cv2.GaussianBlur(new_image, (0, 0), sigmaX = 2.5, sigmaY = 2.5)
    elif mode == 2:
        new_image = cv2.medianBlur(new_image, 5)
    elif mode == 3:
        new_image = cv2.bilateralFilter(new_image, 9, 75, 75)
    return new_image

def bag_thresholding(corrected,mode=1):
    if mode == 1:
        th = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 13, 2)  # 自适应二值化
    elif mode == 2:
        th = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 13, 2)  # 自适应二值化
    elif mode == 3:
        _, th = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = corrected
    return th

def filter_bag_contours(image, contour_list):
    filtered_contours = []
    ImageArea = image.shape[0] * image.shape[1]
    largest = 0
    for i in contour_list:
        # criteria for filtering
        area = cv2.contourArea(i)
        if (area > largest):
            largest = area
        if (area >= filter_con_min * ImageArea) and (area <= filter_con_max * ImageArea):
            filtered_contours.append(i)
    print("The image area is")
    print (ImageArea)
    print ("The largest contour area is")
    print (largest)
    return filtered_contours

def bag_border_extraction(im, mode=2):
    if mode == 1:
        # for image in image_list:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate = cv2.dilate(im, kernel)
        erode = cv2.erode(im, kernel)
        result = cv2.absdiff(dilate, erode)
        cv2.imwrite("dilate.jpg", dilate)
        cv2.imwrite("erode.jpg", erode)
        cv2.imwrite("result.jpg", result)
        return result
    elif mode == 2:
        # for image in image_list:
        result = cv2.Canny(im, 100, 200)
        return result

def locate_bags(th, scaled_im):
    # for th, new_image in zip(edge_images, scaled_im):
    contours_list, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours num:{}".format(len(contours_list)))
    filtered_con = filter_bag_contours(th, contours_list)
    print("Filtered Contours num:{}".format(len(filtered_con)))
    result_im = np.copy(scaled_im)
    try:
        contour = filtered_con[0]
        # new_image = cv2.rectangle(new_image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
    except IndexError:
        print("Index Out of Range For Data: No contour Detected or contours are all filtered")
        filtered_con = None

    return result_im, filtered_con

def locate_bag_box(src,gc):
    if gc is not None:
        bag_contour = gc[0]
        # Get absolute position rectangle for bounding box
        boundRect = cv2.boundingRect(bag_contour)

        # Get rotate rectangle for bounding box
        rect = cv2.minAreaRect(bag_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mask = src
        #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        #print(box)
        mask = cv2.drawContours(mask,[box],-1,(0,0,255),4 )
        mask = cv2.rectangle(mask, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,0), 2)

    #show_result("box",mask)
    cv2.waitKey(0)
    return mask,box,[int(boundRect[0]), int(boundRect[1])],[int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])]

def bag_preprocess(src, show=True):
    # for image in image_list:
    rows, cols, _channels = map(int, src.shape)
    scaled_img = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))  # down sampling

    show = False
    if show:
        cv2.imshow("original", scaled_img)
    # image enhancing
    # converting to LAB color space
    lab = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe1 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl = clahe1.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if show:
        cv2.imshow('enhance1: CLAHE', im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    if show:
        cv2.imshow("grey", im)
    im = cv2.equalizeHist(im)  # 直方图均衡化
    if show:
        cv2.imshow("enhance2:", im)
    im = bag_blur(im, 2)
    if show:
        cv2.imshow("blur", im)
    th = bag_thresholding(im, 2)  # binary
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return th, scaled_img


    
if __name__ == "__main__":

    # Use these two for overall detection
    folder="Mal_Functioning"
    images, names = load_images_bmp_single(folder)

    # Use these two for checking mal_detection images
    #folder = "img"
    #images, names = load_images_bmp(folder)
    count = 0
    for (image, name) in zip(images, names):
        init_image = image.copy()
        if isDarkImage(image) == 1:
            #print("dark")
            template = cv2.imread('portgoal3.bmp', 0)
            methods = ['cv2.TM_CCORR_NORMED']
            box,bag_box = generateROI(image, methods, template)
        else:
            #print("white")
            template = cv2.imread('portgoal.bmp', 0)
            methods = ['cv2.TM_CCOEFF_NORMED']
            box,bag_box = generateROI(image, methods, template)
        #print("the two points for the bounding box for the port are:")
        #print(box)
        #print("the two points for the bounding box for the bag are: ")
        #print(bag_box)
        original,masked = extraction(image,box[0],box[1])
        #cv2.waitKey(0)
        preprocessed,cropped = preprocess(masked,FALSE)

        if(isDarkImage(image)==1):
            preprocessed, cropped = preprocess_dark(masked,FALSE)
        else:
            preprocessed,cropped = preprocess(masked,FALSE)
        #show_result("After",preprocessed)
        #show_result("Cropped",cropped)

        # change parameters to get rid of the border
        newbox0 = list(box[0])
        newbox1 = list(box[1])
        for i in range(2):
            newbox0[i]+=37
            newbox1[i]-=30
        nbox0=tuple(newbox0)
        nbox1=tuple(newbox1)
        nbox = (nbox0,nbox1)
        #print(nbox)

        _,secondcrop = extraction_white(preprocessed,nbox[0],nbox[1])
        #show_result("SecondCrop",secondcrop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # take only the interested area
        nc = secondcrop[nbox0[1]:nbox1[1],nbox0[0]:nbox1[0]]
        #show_result("only port",nc)
        #cv2.waitKey(0)
        # start processing the image to get contour
        # first, enhancement

        # histogram
        #hist = cv2.calcHist(nc,[0],None,[256],[0,256])
        #plt.hist(nc.ravel(),256,[0,256])
        #plt.title("Histogram")
        #plt.show()
        #cv2.waitKey(0)

        thnc = thresholding(nc,1)
        #show_result("Threshold", thnc)
        #cv2.waitKey(0)

        contour_edge,gc = locate_contour(thnc)
        # time to get the bounding box for this contour

        boxim,box,no_turn_top_left,no_turn_bottom_right = locate_box(contour_edge,gc)

        if(len(box)!=0):
        
            # process box coordinate so to fit original picture
            box[0][0]+=newbox0[0]
            no_turn_top_left[0]+=newbox0[0]
            no_turn_bottom_right[0]+=newbox0[0]
            box[0][1]+=newbox0[1]
            no_turn_top_left[1]+=newbox0[1]
            no_turn_bottom_right[1]+=newbox0[1]
            box[1][0]+=newbox0[0]
            box[1][1]+=newbox0[1]
            box[2][0]+=newbox0[0]
            box[2][1]+=newbox0[1]
            box[3][0]+=newbox0[0]
            box[3][1]+=newbox0[1]

            boxed_original = image
            boxed_original = cv2.drawContours(image,[box],-1,(0,0,255),2)
            boxed_original = cv2.rectangle(boxed_original,(no_turn_top_left[0],no_turn_top_left[1]),(no_turn_bottom_right[0],no_turn_bottom_right[1]),(0,255,0),2)
        #cv2.imshow("boxed",boxed_original)

        

        #only need x coordinate of port box for big box
        bag_box_left_extreme = no_turn_bottom_right[0]


        #From here, bag detection adapted from Yilin
        #original, cropped_softbag = extraction(init_image,bag_box[0],bag_box[1])

        # change parameters to get rid of the border
        #newbbox0 = list(bag_box[0])
        #newbbox1 = list(bag_box[1])
        #for i in range(2):
        #    newbbox0[i]+=30
        #    newbox1[i]-=30
        #nbbox0=tuple(newbbox0)
        #nbbox1=tuple(newbbox1)
        #nbbox = (nbbox0,nbbox1)

        #cropped_softbag = cropped_softbag[nbbox0[1]:nbbox1[1],0:1944]
        #cv2.imshow("cropped",cropped_softbag)
        #cv2.waitKey(0)
        #processed_bag, scaled_bag = bag_preprocess(cropped_softbag,FALSE)
        #border_images = bag_border_extraction(processed_bag, 1)
        #located, contour = locate_bags(border_images, scaled_bag)

        #if contour is None:
        #    print("Could not locate infusion bags in data: ".format(name))
        #else:
        #    boxim,box,no_turn_top_left,no_turn_bottom_right = locate_bag_box(located,contour)
            

            # process box coordinate so to fit original picture
        #    box[0][0]*=2
        #    no_turn_top_left[0]*=2
        #    no_turn_bottom_right[0]*=2
        #    box[0][1]*=2
        #    no_turn_top_left[1]*=2
        #    no_turn_bottom_right[1]*=2
        #    box[1][0]*=2
        #    box[1][1]*=2
        #    box[2][0]*=2
        #    box[2][1]*=2
        #    box[3][0]*=2
        #    box[3][1]*=2


        #    boxed_original = cv2.drawContours(boxed_original,[box],-1,(255,0,0),2)
        #    boxed_original = cv2.rectangle(boxed_original,(no_turn_top_left[0],no_turn_top_left[1]),(no_turn_bottom_right[0],no_turn_bottom_right[1]),(0,255,0),2)
        #    cv2.imshow("boxed",boxed_original)
        #    cv2.waitKey(0)


        # From here, overall detection by Yilin
        processed_bag, scaled_bag = bag_preprocess(init_image,FALSE)
        border_images = bag_border_extraction(processed_bag, 1)
        located, contour = locate_bags(border_images, scaled_bag)

        if contour is None:
            print("Could not locate infusion bags in data: ".format(name))
        else:
            boxim,box,no_turn_top_left,no_turn_bottom_right = locate_bag_box(located,contour)
            

            # process box coordinate so to fit original picture
            box[0][0]*=2
            no_turn_top_left[0]*=2
            no_turn_bottom_right[0]*=2
            box[0][1]*=2
            no_turn_top_left[1]*=2
            no_turn_bottom_right[1]*=2
            box[1][0]*=2
            box[1][1]*=2
            box[2][0]*=2
            box[2][1]*=2
            box[3][0]*=2
            box[3][1]*=2

            print(box)
            print(bag_box[0])
            #if(box[1][0]<bag_box_left_extreme):
            #    box[1][0]=bag_box_left_extreme
            #    box[2][0]=bag_box_left_extreme
            #    no_turn_top_left[0]=bag_box_left_extreme



            boxed_original = cv2.drawContours(boxed_original,[box],-1,(0,0,255),2)
            boxed_original = cv2.rectangle(boxed_original,(no_turn_top_left[0],no_turn_top_left[1]),(no_turn_bottom_right[0],no_turn_bottom_right[1]),(0,255,0),2)
            #cv2.imshow("boxed",boxed_original)
            #cv2.waitKey(0)

        # From here, axis



        name = "boxed"+str(count)+".jpg"
        
        # Use this path for overall output
        Path="./result_roi/"+name

        # Use this path for 
        cv2.imwrite(Path,boxed_original)
        count+=1

