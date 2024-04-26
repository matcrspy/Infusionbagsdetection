
from tkinter import *
from tkinter import messagebox, filedialog

import os
import cv2
import numpy as np
import matplotlib as mpl
from copy import deepcopy
from PIL import Image, ImageTk

from calendar import c
from concurrent.futures import process
from multiprocessing import dummy
import glob
import sys

from matplotlib import pyplot as plt
from PIL import Image 
import PIL 
from pickle import FALSE, TRUE
import imutils

filter_con_min = 0.25
filter_con_max = 0.6

# Center the tkinter window
def center_window(tk_window, w, h):
    global ws, hs
    # Obtain the screen size and calculate the center coordinates
    ws = tk_window.winfo_screenwidth()
    hs = tk_window.winfo_screenheight()
    x = (ws//2) - (w//2)
    y = (hs//2) - (h//2)
    tk_window.geometry('%dx%d+%d+%d' % (w, h, x, y))

# Center the opencv window
def cv2_window(window_name, image):
    global ws, hs
    
    if len(image.shape) == 3:
        h, w, c = image.shape
    elif len(image.shape) == 2:
        h, w = image.shape
    
    x = ws//2 - w//2
    y = hs//2 - h//2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, x, y)

# Resize a PIL image while retaining the aspect ratio
def resize(pil_image, w, h):
    global root
    # Define size of the display box 
    w_box = root.winfo_width() * 2/3
    h_box = root.winfo_height() * 2/3
    # Factor to retain the aspect ratio
    f1 = w_box / w
    f2 = h_box / h
    factor = min([f1, f2])
    # Best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    
    return pil_image.resize((width, height), Image.ANTIALIAS)

# Import an image from file 
def select_image():
    global path
    global panelA
    global img, roi, out_img, temp
    # Open a file chooser dialog 
    # Ensure a file path was selected
    path = filedialog.askopenfilename()
    if len(path) > 0:
        # Open as a PIL image object
        pil_image = Image.open(path) #RGB
        w, h = pil_image.size
        # Resize to fit
        image_resized = resize(pil_image, w, h)
        tk_image = ImageTk.PhotoImage(image_resized)
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR) # BGR 
        roi = img # BGR
        # Make sure the output image is RGB
        out_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Use as a placeholder
        temp = out_img
        # Assign images to panels; If the image panel is None, initialize it
        if panelA is None:
            panelA = Label(frame3, image = tk_image, bg = 'white')
            panelA.image = tk_image
            panelA.pack(side = 'top', expand ='yes')
        # otherwise, update the image panel
        else:
            panelA.configure(image = tk_image)
            panelA.image = tk_image
            
    return path

# Mouse Events to manually select roi
def on_mouse(event, x, y, flags, param):
    global img, roi
    global pt1, pt2
    
    copy = img.copy()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x, y)
        cv2.circle(copy, pt1, 10, (255, 0, 0), thickness = 2)
        cv2_window('ROI Selector', copy)
        cv2.imshow('ROI Selector', copy)
        
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(copy, pt1, (x,  y), (255, 0, 0), thickness = 2)
        cv2_window('ROI Selector', copy)
        cv2.imshow('ROI Selector', copy)
        
    elif event == cv2.EVENT_LBUTTONUP:
        pt2 = (x, y)
        cv2.rectangle(copy, pt1, pt2, (0, 0, 255), thickness = 2)
        cv2_window('ROI Selector', copy)
        cv2.imshow('ROI Selector', copy)
        
        minx = min(pt1[0], pt2[0])
        miny = min(pt1[1], pt2[1])
        w = abs(pt1[0] - pt2[0])
        h = abs(pt1[1] - pt2[1])
      
        roi = img[miny : miny + h, minx : minx + w] # BGR
        
# ROI Selector
def select_roi():
    global path
    global panelA
    global img, roi, out_img, temp

    if len(path) > 0:
        cv2_window('ROI Selector', img)
        cv2.setMouseCallback('ROI Selector', on_mouse)
        cv2.imshow('ROI Selector', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Convert the ROI to PIL format 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # RGB
        pil_roi = Image.fromarray(roi)
        # Resize PIL ROI to fit the display window
        w,h = pil_roi.size
        pil_roi_resized = resize(pil_roi, w, h)
        # ...to Tkimage format
        tk_roi = ImageTk.PhotoImage(pil_roi_resized)
        # Update the image panel      
        panelA.configure(image = tk_roi)
        panelA.image = tk_roi
        # Update the output image
        out_img = roi # RGB
        temp = roi
    # Make sure there is an image to process
    else:
        messagebox.showwarning('No Image', 
                               'Please first select an image')

# Grayscale
def grayscale():
    global path
    global panelA
    global img, roi, out_img, temp

    if len(path) > 0:
        copy = out_img.copy()
        # To check if the image is 2-channel or 3-channel
        if len(copy.shape) == 2:
            gray = copy
        elif len(copy.shape) ==3:
            gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
        # Convert the ROI to PIL format 
        pil_gray = Image.fromarray(gray)
        # Resize PIL ROI to fit the display window
        w, h = pil_gray.size
        pil_gray_resized = resize(pil_gray, w, h)
        # ...to Tkimage format
        tk_gray = ImageTk.PhotoImage(pil_gray_resized)
        # Update the image panel      
        panelA.configure(image = tk_gray)
        panelA.image = tk_gray
        # Update the output image and roi
        out_img = gray # RGB
        temp = gray
    # Make sure there is an image to process
    else:
        messagebox.showwarning('No Image', 
                               'Please first select an image')

def revert():
    global path
    global panelA
    global img, roi, out_img, temp
    
    if len(path) > 0:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # RGB
        pil_roi = Image.fromarray(roi)
        # Resize PIL ROI to fit the display window
        w,h = pil_roi.size
        pil_roi_resized = resize(pil_roi, w, h)
        # ...to Tkimage format
        tk_roi = ImageTk.PhotoImage(pil_roi_resized)
        # Update the image panel      
        panelA.configure(image = tk_roi)
        panelA.image = tk_roi
        
        temp = roi
        out_img = roi
    else:
        messagebox.showwarning('No Image', 
                               'Please first select an image')

# Gaussian Filter
def gaussian_blur():
    
    # Get kernel size entry
    def get_param():
        global path
        global panelA
        global img, roi, out_img, temp
        
        try:
            # Check the entry
            value = int(v.get().strip())
            if (value % 2) == 0:
                messagebox.showwarning('Invalid Input', 
                                       'Entry should be a positive odd number')
            else:
                kernelSize = value
                # Import as RGB
                blur = out_img.copy() 
                # Convert to opencv BGR format
                blur = cv2.cvtColor(blur, cv2.COLOR_RGB2BGR)
                blur = cv2.GaussianBlur(blur, (kernelSize, kernelSize), 0)
                # Convert to PIL RGB format and resize
                blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB) #RGB
                pil_blur = Image.fromarray(blur)
                w, h = pil_blur.size
                pil_blur_resized = resize(pil_blur, w, h)
                # Update image panel with the resized blurred image
                tk_blur = ImageTk.PhotoImage(pil_blur_resized)
                panelA.configure(image = tk_blur)
                panelA.image = tk_blur
                # Update output image and roi
                out_img = blur #RGB
                temp = blur
        except:
            messagebox.showwarning('Invalid Input', 
                                       'Please enter a valid value')          
       
    # Make sure an image is imported
    if len(path)>0:
        # Pop-up parameter setting window
        window = Toplevel(root)
        window.configure(bg = "white")
        window.title('Parameter Setting')
        center_window(window, 300, 100)
        v = StringVar()
        
        # layout
        label1 = Label(window, 
                       text = 'Gaussian Filtering', 
                       bg = '#00C1D3', 
                       fg = 'white', 
                       font = ('Helvetica', 18, 'bold'))
        label2 = Label(window, 
                       text = 'Kernel Size: ', 
                       bg = 'white', 
                       fg = 'black', 
                       font = ('Helvetica', 14))
        entry = Entry(window, 
                      textvariable = v,
                      bg = 'linen', 
                      width = 16, 
                      font = ('Helvetica', 14),
                      justify = CENTER)
        btn = Button(window, 
                     text = 'Confirm', 
                     command = get_param, 
                     bg = '#00C1D3', 
                     fg = 'white', 
                     font = ('Helvetica', 14))
    
        label1.grid(row = 0, column = 0, columnspan = 2, sticky = W+E)
        label2.grid(row = 1, column = 0, sticky = W)
        entry.grid(row = 1, column = 1, stick = E)
        btn.grid(row = 2, column = 1, sticky = E)
    
    else:
        messagebox.showwarning('No Image', 
                               'Please first select an image')
    return

# Median Filter
def median_blur():
    # Get kernel size entry
    def get_param():
        global path
        global panelA
        global img, roi, out_img, temp
        
        try:
            # Check the entry
            value = int(v.get().strip())
            if (value % 2) == 0:
                messagebox.showwarning('Invalid Input', 
                                       'Entry should be a positive odd number')
            else:  
                kernelSize = value
                #Import as RGB
                blur = out_img.copy() 
                # Convert to opencv BGR format
                blur = cv2.cvtColor(blur, cv2.COLOR_RGB2BGR)
                blur = cv2.medianBlur(blur, kernelSize)
                # Convert to PIL RGB format and resize
                blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB) #RGB
                pil_blur = Image.fromarray(blur)
                w, h = pil_blur.size
                pil_blur_resized = resize(pil_blur, w, h)
                # Update image panel with the resized blurred image
                tk_blur = ImageTk.PhotoImage(pil_blur_resized)
                panelA.configure(image = tk_blur)
                panelA.image = tk_blur
                # Update output image
                out_img = blur #RGB
                temp = blur
        except:
            messagebox.showwarning('Invalid Input', 
                                   'Please enter a valid value')       
        
    # Make sure an image is imported    
    if len(path) > 0:
        #Pop-up parameter setting window
        window = Toplevel(root)
        window.configure(bg = "white")
        window.title('Parameter Setting')
        center_window(window, 300, 100)
    
        v = StringVar()
        label1 = Label(window, 
                       text = 'Median Filtering', 
                       bg = '#00C1D3', 
                       fg = 'white', 
                       font = ('Helvetica', 18, 'bold'))
        label2 = Label(window, 
                       text = 'Kernel Size: ', 
                       bg = 'white', 
                       fg = 'black', 
                       font = ('Helvetica', 14))
        entry = Entry(window, 
                      textvariable = v, 
                      bg = 'linen', 
                      width = 16, 
                      font = ('Helvetica', 14),
                      justify = CENTER)
        btn = Button(window, 
                     text = 'Confirm', 
                     command = get_param, 
                     bg = '#00C1D3', 
                     fg = 'white', 
                     font = ('Helvetica', 14))
    
        label1.grid(row = 0, column = 0, columnspan = 2, sticky = W+E)
        label2.grid(row = 1, column = 0, sticky = W)
        entry.grid(row = 1, column = 1, stick = E)
        btn.grid(row = 2, column = 1, sticky = E)
    
    else:
        messagebox.showwarning('No Image', 
                               'Please first select an image')
   
    return


# Get threshold value for simple thresholding use
def get_ctr_thres():
    global input1
    global ctr_thres
    
    ctr_thres = 0
    input1 = v1.get()

    try:
        input1 = int(v1.get().strip())
        if input1 == 0 or input1 > 255:
            messagebox.showwarning('Invalid Input', 'Value out of range 0 ~ 1')
        else:
            ctr_thres = input1
        
    except:
        messagebox.showwarning('Invalid input', 
                                'Please enter a valid threshold value')

# Contour detection method
def contour_detection():
    global path
    global panelA
    global img, roi, out_img, temp
    global ctr_thres
    global contours
    
    if len(path) > 0:
        thres_value = get_ctr_thres()
        # Make sure there is an entry (0 is the default value)
        if thres_value != 0:
            if len(temp.shape) == 3:
                ctr = temp.copy()
                gray = cv2.cvtColor(ctr, cv2.COLOR_RGB2GRAY)
            elif len(temp.shape) == 2:
                ctr = temp.copy()
                gray = temp.copy()
            
            ret, thres = cv2.threshold(gray, ctr_thres, 255, cv2.THRESH_BINARY)
            
            contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            copy = cv2.drawContours(ctr, contours, -1, (255, 0, 0), 2)
            
            pil_image = Image.fromarray(copy)
            tk_image = ImageTk.PhotoImage(pil_image)
        
            panelA.configure(image = tk_image)
            panelA.image = tk_image
            out_img = copy
        
        else:
            messagebox.showwarning('Invalid Input', 
                                   'Please enter a valid threshold value') 
    else:
        messagebox.showwarning('No Image', 'Please first select an image')

# Filter contour results after detection
def filter_contour():
        
    def get_param():
        global path
        global panelA
        global img, roi, out_img, temp
        global contours
        
        if len(path) > 0:
            
            try:
                lower = int(v1.get().strip())
                upper = int(v2.get().strip())
                if (lower > 100) or (lower < 0) or (upper > 100) or (upper < 0):
                    messagebox.showwarning('Invalid Input', 
                                            'Entry should be between 0 and 100')
                    
                else:
                    copy = temp.copy()
                    imageArea = copy.shape[0] * copy.shape[1]
                    
                    filtered_contours = []
                    for i in contours:
                        area = cv2.contourArea(i)
                        if (area >= lower/100*imageArea) and (area <= upper/100*imageArea):
                            filtered_contours.append(i)
                    
                    copy = cv2.drawContours(copy, filtered_contours, -1, (255, 0, 0), 2)
                    pil_image = Image.fromarray(copy)
                    tk_image = ImageTk.PhotoImage(pil_image)
        
                    panelA.configure(image = tk_image)
                    panelA.image = tk_image
                    out_img = copy

            except:
                messagebox.showwarning('Invalid Input', 
                                       'Please enter a valid value')          
        else:
            messagebox.showwarning('No Image', 
                                   'Please first select an image')
    
    # Pop-up parameter setting window
    window = Toplevel(root)
    window.configure(bg = "white")
    window.title('Parameter Setting')
    center_window(window, 320, 200)
    # layout
    v1= StringVar()
    v2 = StringVar()
    label1 = Label(window, 
                   text = 'Contour Filtering', 
                   bg = '#00C1D3', 
                   fg = 'white', 
                   font = ('Helvetica', 18, 'bold'))
    label2 = Label(window, 
                   text = 'Range: ', 
                   bg = 'white', 
                   fg = 'black', 
                   font = ('Helvetica', 14),
                   anchor = W)
    label3 = Label(window, 
                   text = '%', 
                   bg = 'white', 
                   fg = 'black', 
                   font = ('Helvetica', 14))
    label4 = Label(window, 
                   text = ' ~ ', 
                   bg = 'white', 
                   fg = 'black', 
                   font = ('Helvetica', 18, 'bold'))
    label5 = Label(window, 
                   text = '%', 
                   bg = 'white', 
                   fg = 'black', 
                   font = ('Helvetica', 14))
    label6 = Label(window, 
                   bg = '#00C1D3', 
                   fg = 'white', 
                   font = ('Helvetica', 18, 'bold'))
    entry1 = Entry(window, 
                  textvariable = v1,
                  bg = 'linen', 
                  width = 10, 
                  font = ('Helvetica', 14),
                  justify = CENTER)
    entry2 = Entry(window, 
                  textvariable = v2,
                  bg = 'linen', 
                  width = 10, 
                  font = ('Helvetica', 14),
                  justify = CENTER)
    btn1 = Button(window, 
                 text = 'Confirm', 
                 command = get_param, 
                 bg = '#00C1D3', 
                 fg = 'white', 
                 font = ('Helvetica', 14))
   
    label1.pack(side = 'top', fill = X, ipady = 5)
    label2.pack(side = 'top', fill = X, anchor = W, padx = 3, pady = 5)
    label6.pack(side = 'bottom', fill = BOTH)
    btn1.pack(side = 'bottom', anchor = E, padx = 10, pady = 5)
    entry1.pack(side = 'left', anchor = NW, fill = X, padx = 5)
    label3.pack(side = 'left', anchor = NW, fill = X)
    label4.pack(side = 'left', anchor = NW, fill = X, expand = 'YES')
    entry2.pack(side = 'left', anchor = NE, fill = X)
    label5.pack(side = 'left', anchor = NE, fill = X, padx= 5)
   
    return




#generate roi
def generate_roi():
    global path
    global panalA
    print("yes")
    image = cv2.imread(path)
    print(path)
    #cv2.imshow("merge", image)
    
    # Use these two for overall detection
    #folder="Mal_Functioning"
    #images, names = load_images_bmp_single(folder)

    # Use these two for checking mal_detection images
    #folder = "img"
    #images, names = load_images_bmp(folder)
    count = 0
    
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
    ##Path="./result_roi/"+name

    # Use this path for 
    ##cv2.imwrite(Path,boxed_original)
    count+=1
    
    
    
    img_gray = boxed_original
    #plt.imshow(img_gray)
    #plt.show()
    pil_image2 = cv2.resize(img_gray, (648, 486))
    #pil_image = Image.fromarray(img_gray)
    pil_image = Image.fromarray(pil_image2)
    
    tk_image = ImageTk.PhotoImage(pil_image)
        
    panelA.configure(image = tk_image)
    panelA.image = tk_image
    

# port detection:
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

#decide if the images is dark
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
    
#find the inititial roi of the port and bag
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
    ImageArea = src.shape[0] * src.shape[1]
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
    #cv2.imshow("Contours", contourImg)
    #cv2.waitKey(0)

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


# Save image
def save_image():
    global out_img
    dest = filedialog.asksaveasfilename(filetypes = [("PNG", ".png")])
    pil_out_image = Image.fromarray(out_img)
    pil_out_image.save(str(dest) + '.png', 'PNG')

# Empty entry inputs
def emptyval():
    global input1, input2
    v1.set('')
    v2.set('')
    input1 = None
    input2 = None

# Empty image display window
def clearFrame():
    for widget in frame3.winfo_children():
        widget.destroy()

# Help menu
def contact():
    messagebox.showinfo('Help', 'Welcome to the VNX-0-2 Toolkit Interface!' + 
                        '\nPlease contact support@visionx.org for any questions.' + 
                        '\nVisionX LLC. Copyright')

# Quit
def close():
    if messagebox.askyesno(title = 'Confirm to Quit', 
                           message = 'Are you sure you want to quit?'):
        root.destroy()


root = Tk()
root.configure(bg = 'white')
root.title('VNX-0-2')

# Initialize
panelA = None
panelB = None
path = ''
v1 = StringVar()
v2 = StringVar()

# Import images
img_path = os.getcwd() + "/icons/"
img1 = ImageTk.PhotoImage(Image.open(img_path + 'vnx1.png'))
img2 = ImageTk.PhotoImage(Image.open(img_path + 'vnx2.png'))

# Title bar
frame1 = Frame(root, bg = '#00C1D3')
# Bottom bar
frame2 = Frame(root, bg = '#00C1D3')
# Image Display Window
frame3 = Frame(root, bg = 'white', relief = GROOVE, bd = 3)
# Function bar
frame4 = Frame(root, bg='white')
# Console bar
frame5 = LabelFrame(root, bg = 'white', relief = GROOVE, bd = 3, 
                    text = 'Console', font = ('Helvetica', 14))

frame1.pack(side = 'top', fill = X)
frame2.pack(side = 'bottom', fill = X) 
frame3.pack(side = 'left', fill = BOTH, padx = 3, pady = 3,expand = 'Yes')
frame4.pack(side = 'top', anchor = N, padx = 3, pady = 3) 
frame5.pack(side = 'bottom', fill = BOTH, anchor = N, 
            padx = 3, pady = 3, expand = 'Yes') 

# Widgets 
label1 = Label(frame1, image = img1, bd = 0)
label2 = Label(frame2, image = img2, bd = 0)
label1.pack()
label2.pack()

label3 = Label(frame4, 
               text = 'Parameter \nSetting', 
               bg = '#00C1D3', 
               fg = 'white', 
               width = 20, 
               font = ('Helvetica', 18, 'bold'))

label5 = Label(frame4, 
               text = 'Run:', 
               bg = 'white', 
               fg = 'black', 
               font = ('Helvetica', 16), 
               anchor = W, 
               justify = LEFT)

button3 = Button(frame4, 
                 text = 'Generate Bounding Box', 
                 command = generate_roi, 
                 bg = '#00C1D3', 
                 fg = 'white', 
                 width = 18, 
                 font = ('Helvetica', 14, 'bold'))

label3.grid(row = 0, column = 0, columnspan = 3)
label5.grid(row = 2, column = 0, sticky = W)

button3.grid(row =3, column = 1, columnspan = 2, sticky = W+E)

txt = Text(frame5, 
           width = 30,
           height = 20, 
           spacing1 = 10, 
           wrap = WORD, 
           font = ('Courier', 12))
txt.pack(padx = 5, pady = 5)
txt.insert(1.0, "VNX-0-2")

# Icons
ico_path = os.getcwd() + "/icons/"
ico_open = ImageTk.PhotoImage(Image.open(ico_path + 'open.png'))
ico_file = ImageTk.PhotoImage(Image.open(ico_path + 'file.png'))
ico_save = ImageTk.PhotoImage(Image.open(ico_path + 'save.png'))
ico_quit = ImageTk.PhotoImage(Image.open(ico_path + 'quit.png'))
ico_new = ImageTk.PhotoImage(Image.open(ico_path + 'new.png'))
ico_edit = ImageTk.PhotoImage(Image.open(ico_path + 'edit.png'))
ico_help = ImageTk.PhotoImage(Image.open(ico_path + 'help.png'))

# Main menu initialization
menubar = Menu(root)
filemenu = Menu(menubar, tearoff = False)
editmenu = Menu(menubar, tearoff = False)
runmenu = Menu(menubar, tearoff = False)    
helpmenu = Menu(menubar, tearoff = False)

# Sub menu initialization
selectmenu = Menu(filemenu, tearoff = False)
colormenu = Menu(editmenu, tearoff = False)
blurmenu = Menu(editmenu, tearoff = False)
fncmenu1 = Menu(runmenu, tearoff = False)
fncmenu2 = Menu(runmenu, tearoff = False)

# Select file (File menu) 
menubar.add_cascade(label = 'File', menu = filemenu)

filemenu.add_cascade(label = ' New File', 
                     menu = selectmenu, 
                     image = ico_open, 
                     compound = 'left', 
                     font = ('Helvetica', 10))
filemenu.add_command(label = ' Save', 
                     command = save_image, 
                     image = ico_save, 
                     compound = 'left', 
                     font = ('Helvetica', 10))
filemenu.add_separator()
filemenu.add_command(label = ' Quit', 
                     command = close, 
                     image = ico_quit, 
                     compound = 'left', 
                     font = ('Helvetica', 10))

selectmenu.add_command(label = ' From File', 
                       command = select_image, 
                       image = ico_file, 
                       compound = 'left', 
                       font = ('Helvetica', 10))

# Pre-processing (Edit menu)
menubar.add_cascade(label = 'Edit', menu = editmenu)

editmenu.add_command(label = 'ROI Selection', 
                     command = select_roi, 
                     image = ico_new, 
                     compound = 'left', 
                     font = ('Helvetica', 10))
editmenu.add_separator()
editmenu.add_cascade(label = 'Color Conversion', 
                     menu = colormenu, 
                     image = ico_new, 
                     compound = 'left', 
                     font = ('Helvetica', 10))
editmenu.add_separator()
editmenu.add_cascade(label = 'Noise Removal', 
                     menu = blurmenu, 
                     image = ico_new, 
                     compound = 'left', 
                     font = ('Helvetica', 10))

colormenu.add_command(label = ' Grayscale', 
                      command = grayscale, 
                      image = ico_edit, 
                      compound = 'left', 
                      font = ('Helvetica', 10))
colormenu.add_separator()
colormenu.add_command(label = ' Revert to Original', 
                      command = revert, 
                      image = ico_edit, 
                      compound = 'left', 
                      font = ('Helvetica', 10))
blurmenu.add_command(label = ' Gaussian Blur', 
                     command = gaussian_blur, 
                     image = ico_edit, 
                     compound = 'left', 
                     font = ('Helvetica', 10))
blurmenu.add_separator()
blurmenu.add_command(label = ' Median Blur', 
                     command = median_blur, 
                     image = ico_edit, 
                     compound = 'left', 
                     font = ('Helvetica', 10))

# Functions (Run menu)
menubar.add_cascade(label = 'Run', menu = runmenu)

runmenu.add_command(label = "Contour Detection", 
                    command = contour_detection, 
                    image = ico_new, 
                    compound = 'left', 
                    font = ('Helvetica', 10))
runmenu.add_separator()
runmenu.add_command(label = "Filter Contours", 
                    command = filter_contour, 
                    image = ico_new, 
                    compound = 'left', 
                    font = ('Helvetica', 10))


# Help
menubar.add_cascade(label =  'Help', menu = helpmenu)
helpmenu.add_command(label = ' Help', 
                     command = contact, 
                     image = ico_quit, 
                     compound = 'left', 
                     font = ('Helvetica', 10))


root['menu'] = menubar

center_window(root, 1200, 800)

root.mainloop()        
        