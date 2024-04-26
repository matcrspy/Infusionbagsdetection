from tkinter import *
from tkinter import messagebox, filedialog

import os
import cv2
import time
import numpy as np
import matplotlib as mpl

from copy import deepcopy
from PIL import Image, ImageTk
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.tools.infer.utility import draw_ocr_box_txt


from calendar import c
from concurrent.futures import process
from multiprocessing import dummy
import glob
import sys
import os

from matplotlib import pyplot as plt
from PIL import Image 
import PIL 
from pickle import FALSE, TRUE
import imutils

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
    global thresh, box_thresh, unclip_ratio, language
    # Default Values
    thresh = 0.3
    box_thresh = 0.5
    unclip_ratio = 2
    language = "en"
    
    # Open a file chooser dialog 
    path = filedialog.askopenfilename()
    # Ensure a file path was selected
    if len(path) > 0:
        # cv image bgr
        bgr = cv2.imdecode(np.fromfile(path, dtype = np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Open as a PIL image object
        pil_image = Image.fromarray(rgb)
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
    return
        
# ROI Manual Selector
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
    return

# ROI Range Selector
def select_roi_range():
    def get_param():
        global path
        global panelA
        global img, roi, out_img, temp
        
        try:
            x1 = int(v1.get().strip())
            x2 = int(v2.get().strip())
            y1 = int(v3.get().strip())
            y2 = int(v4.get().strip())
            
            if (x2 > roi.shape[1]) or (y2 > roi.shape[0]):
                messagebox.showwarning('Invalid Input', 
                                       'Input out of range')
            else:
                # Convert the ROI to PIL format 
                roi = roi[y1:y2, x1:x2]
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
        except:
            messagebox.showwarning('Invalid Input', 
                                   'Please enter a valid value')          

    if len(path) > 0:
        # Pop-up parameter setting window
        window = Toplevel(root)
        window.configure(bg = "white")
        window.title('Parameter Setting')
        center_window(window, 428, 160)
        # layout
        v1 = StringVar()
        v2 = StringVar()
        v3 = StringVar()
        v4 = StringVar()
        label1 = Label(window, 
                       text = 'ROI Selection', 
                       bg = '#00C1D3', 
                       fg = 'white', 
                       font = ('Helvetica', 18, 'bold'))
        label2 = Label(window, 
                       text = 'X range: ', 
                       bg = 'white', 
                       fg = 'black', 
                       font = ('Helvetica', 14),
                       anchor = W)
        label3 = Label(window, 
                       text = ' ~ ', 
                       bg = 'white', 
                       fg = 'black', 
                       font = ('Helvetica', 18, 'bold'))
        label4 = Label(window, 
                       text = 'Y range: ', 
                       bg = 'white', 
                       fg = 'black', 
                       font = ('Helvetica', 14),
                       anchor = W)
        label5 = Label(window, 
                       text = ' ~ ', 
                       bg = 'white', 
                       fg = 'black', 
                       font = ('Helvetica', 18, 'bold'))
        label6 = Label(window, 
                       bg = '#00C1D3', 
                       fg = 'white')
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
        entry3 = Entry(window, 
                       textvariable = v3,
                       bg = 'linen', 
                       width = 10, 
                       font = ('Helvetica', 14),
                       justify = CENTER)
        entry4 = Entry(window, 
                       textvariable = v4,
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
        
        label1.grid(row = 0, column = 0, columnspan = 6, sticky = W+E)
        label2.grid(row = 1, column = 0, sticky = W)
        entry1.grid(row = 1, column = 1, sticky = W)
        label3.grid(row = 1, column = 2)
        entry2.grid(row = 1, column = 3, sticky = W)
        label4.grid(row = 2, column = 0, sticky = W)
        entry3.grid(row = 2, column = 1, sticky = W)
        label5.grid(row = 2, column = 2)
        entry4.grid(row = 2, column = 3, sticky = W)
        btn1.grid(row = 3, column = 4, columnspan = 2, sticky = E)
        label6.grid(row = 4, column = 0, columnspan = 6, sticky = W+E)
    else:
        messagebox.showwarning('No Image', 
                               'Please first select an image')
    return

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
    return

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
    return

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

    
# Advanced parameter setting
def change_params():
    global thresh, box_thresh, unclip_ratio, language
    # Default Values
    thresh = 0.3
    box_thresh = 0.5
    unclip_ratio = 2
    language = "en"
    
    def get_param():
        global thresh, box_thresh, unclip_ratio, language
        try: 
            language = var.get()
            if language == "en" or language == "ch":
                language = var.get()
            else:
                language = "en"
        except:
            language = "en"
        try:
            thresh = float(v1.get().strip())
        except:
            thresh = 0.3
        try:
            box_thresh = float(v2.get().strip())
        except:
            box_thresh = 0.5
        try:
            unclip_ratio = float(v3.get().strip())
        except:
            unclip_ratio = 2
        
        txt.delete(1.0, 'end')
        txt.insert('end', 'Language: {}'.format(str(language)))
        txt.insert('end', '\nDB Threshold: {}'.format(thresh))
        txt.insert('end', '\nDB Box Threshold: {}'.format(box_thresh))
        txt.insert('end', '\nDB Unclip Ratio: {}'.format(unclip_ratio))
        
        window.destroy()
    
    # Parameter setting window
    window = Toplevel(root)
    window.configure(bg = "white")
    window.title('Advanced Parameter Setting')
    center_window(window, 450, 210)
    # layout
    v1 = StringVar()
    v2 = StringVar()
    v3 = StringVar()
    var = StringVar(value = " ")
    label1 = Label(window, 
                    text = 'Parameter', 
                    bg = '#00C1D3', 
                    fg = 'white', 
                    font = ('Helvetica', 18, 'bold'))
    label2 = Label(window, 
                    text = 'Language:', 
                    bg = 'white', 
                    fg = 'black', 
                    font = ('Helvetica', 14),
                    anchor = W)
    label3 = Label(window, 
                    text = 'det_db_thresh', 
                    bg = 'white', 
                    fg = 'black', 
                    font = ('Helvetica', 14))
    label4 = Label(window, 
                    text = 'det_db_box_thresh', 
                    bg = 'white', 
                    fg = 'black', 
                    font = ('Helvetica', 14),
                    anchor = W)
    label5 = Label(window, 
                    text = 'det_db_unclip_ratio', 
                    bg = 'white', 
                    fg = 'black', 
                    font = ('Helvetica', 14))
    label6 = Label(window, 
                    bg = '#00C1D3', 
                    fg = 'white')
    rb1 = Radiobutton(window,
                      text = "English",
                      variable = var,
                      value = "en",
                      font = ('Helvetica', 14))
    rb2 = Radiobutton(window,
                      text = "Chinese",
                      variable = var,
                      value = "ch",
                      font = ('Helvetica', 14))
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
    entry3 = Entry(window, 
                    textvariable = v3,
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
    
    label1.grid(row = 0, column = 0, columnspan = 5, sticky = W+E)
    
    label2.grid(row = 1, column = 0, columnspan = 2, sticky = W)
    rb1.grid(row = 1, column = 2, sticky = W)
    rb2.grid(row = 1, column = 3, sticky = E)
    
    label3.grid(row = 2, column = 0, sticky = W)
    entry1.grid(row = 2, column = 2, columnspan = 2, sticky = W+E)
    
    label4.grid(row = 3, column = 0, sticky = W)
    entry2.grid(row = 3, column = 2, columnspan = 2, sticky = W+E)
    
    label5.grid(row = 4, column = 0, sticky = W)
    entry3.grid(row = 4, column = 2, columnspan = 2, sticky = W+E)
    
    btn1.grid(row = 5, column = 4, sticky = E)
    label6.grid(row = 6, column = 0, columnspan = 5, sticky = W+E)

    return thresh, box_thresh, unclip_ratio, language

# Load ocr model
def load_model():
    global thresh, box_thresh, unclip_ratio, language
    
    # Load paddlrocr model
    ocr = PaddleOCR(use_gpu = False, 
                    use_angle_cls = True,
                    lang = language, 
                    det_db_thresh = thresh, 
                    det_db_box_thresh = box_thresh, 
                    det_db_unclip_ratio = unclip_ratio)

    return ocr

# Detection only
def text_detection():
    global panelA
    global img, roi, out_img, temp
    
    # Make sure an image is selcted
    if len(path) > 0:
        image = temp.copy()
        # Load the ocr model
        ocr = load_model()
        result = ocr.ocr(image, det = True, rec = False, cls = False)
        # Visualize the detection result
        im_show = draw_ocr(image, result, txts=None, scores=None)
        # Convert to pil
        visualized = Image.fromarray(im_show)
        w, h = visualized.size
        visualized = resize(visualized, w, h)
        # Update image panel with the resized result
        vis = ImageTk.PhotoImage(visualized)
        panelA.configure(image = vis)
        panelA.image = vis
        # Update output image and roi
        out_img = im_show #RGB
    else:
        messagebox.showwarning('No Image', 'Please first select an image')
    return

# Detection and Recognition
def text_recognition():
    global panelA
    global img, roi, out_img, temp
    
    # Make sure an image is selcted
    if len(path) > 0:
        image = temp.copy()
        # Load the ocr model
        ocr = load_model()
        result = ocr.ocr(image, det = True, rec = True, cls = False)
        # Obtain useful lists
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        # Visualize the recognition result
        # im_show = draw_ocr(image, boxes, txts, scores)
        
        img = Image.fromarray(image).convert('RGB')
        im_show = draw_ocr_box_txt(img, boxes, txts)
        
        # Convert to pil
        visualized = Image.fromarray(im_show)
        w, h = visualized.size
        # Only showing the visualized recognition
        visualized = visualized.crop((w/2, 0, w, h))
        visualized = resize(visualized, w, h)
        # Update image panel with the resized result
        vis = ImageTk.PhotoImage(visualized)
        panelA.configure(image = vis)
        panelA.image = vis
        # Update output image and roi
        out_img = im_show #RGB
        
        # Get text and score lists
        txt_list = [str(i) for i in txts]
        score_list = [str(i) for i in scores]
        combined = []
        combined = [i + j for i, j in zip(txt_list, score_list)]
        
        # Console message
        txt.delete(1.0, 'end')
        txt.tag_configure('title', font = ('Helvetica', 12, 'bold'))
        txt.insert('end', 'TEXTS AND SCORES', 'title')
        for i in range(len(txt_list)):
            txt.insert('end', '\n{}'.format(txt_list[i]), 'title')
            txt.insert('end', '   {}'.format(score_list[i]))
    else:
        messagebox.showwarning('No Image', 'Please first select an image')
    return

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
    return

# Help menu
def contact():
    messagebox.showinfo('Help', 'Welcome to the VNX-0-5 Toolkit Interface!' + 
                        '\nPlease contact support@visionx.org for any questions.' + 
                        '\nVisionX LLC. Copyright')

# Quit
def close():
    if messagebox.askyesno(title = 'Confirm to Quit', 
                           message = 'Are you sure you want to quit?'):
        root.destroy()


root = Tk()
root.configure(bg = 'white')
root.title('VNX-0-5')

# Initialize
panelA = None
panelB = None
path = ''
tpl_path = ''
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
# frame4 = Frame(root, bg='white')
frame4 = LabelFrame(root, bg = 'white', relief = GROOVE, bd = 3, 
                    text = 'Function Bar', font = ('Helvetica', 14))
# Console bar
frame5 = LabelFrame(root, bg = 'white', relief = GROOVE, bd = 3, 
                    text = 'Console', font = ('Helvetica', 14))

frame1.pack(side = 'top', fill = X)
frame2.pack(side = 'bottom', fill = X) 
frame3.pack(side = 'left', fill = BOTH, padx = 3, pady = 3, expand = 'Yes')
frame4.pack(side = 'top', anchor = N, padx = 3, pady = 3) 
frame5.pack(side = 'bottom', fill = BOTH, anchor = N, 
            padx = 3, pady = 3,expand = 'Yes') 

# Widgets 
label1 = Label(frame1, image = img1, bd = 0)
label2 = Label(frame2, image = img2, bd = 0)
label1.pack()
label2.pack()

# label3 = Label(frame4, 
#                 text = 'Function Bar', 
#                 bg = '#00C1D3', 
#                 fg = 'white', 
#                 width = 25, 
#                 font = ('Helvetica', 20, 'bold'))
# label3 = Label(frame4, 
#                text = 'Function Bar', 
#                bg = 'white', 
#                fg = '#00C1D3', 
#                width = 20, 
#                font = ('Helvetica', 18, 'bold'))
button1 = Button(frame4, 
                 text = 'Detection', 
                 command = text_detection,
                 bg = '#00C1D3', 
                 fg = 'white', 
                 width = 14, 
                 font = ('Helvetica', 14, 'bold'))
button2 = Button(frame4, 
                 text = 'Recognition', 
                 command = text_recognition,
                 bg = '#00C1D3', 
                 fg = 'white', 
                 width = 14, 
                 font = ('Helvetica', 14, 'bold'))
button3 = Button(frame4, 
                  text = 'New Image', 
                  command = select_image,
                  bg = '#00C1D3', 
                  fg = 'white', 
                  width = 14, 
                  font = ('Helvetica', 14, 'bold'))
button4 = Button(frame4, 
                  text = 'Save Image', 
                  command = save_image,
                  bg = '#00C1D3', 
                  fg = 'white', 
                  width = 14, 
                  font = ('Helvetica', 14, 'bold'))

# label3.grid(row = 0, column = 0, columnspan = 2)
button3.grid(row = 1, column = 0, padx = 3, pady = 3, sticky = N)
button4.grid(row = 2, column = 0, padx = 3, pady = 5, sticky = N)
button1.grid(row = 1, column = 1, padx = 3, sticky = E)
button2.grid(row = 2, column = 1, padx = 3, sticky = E)

# label3.pack(side = 'top', fill = X, expand = 'yes')
# button1.pack()
# button2.pack()

txt = Text(frame5, 
           width = 30,
           height = 20, 
           spacing1 = 5, 
           wrap = WORD, 
           font = ('Helvetica', 12))
scroll = Scrollbar(frame5)

scroll.pack(side = RIGHT, fill = Y)
txt.pack(side = LEFT, fill = Y)
scroll.config(command = txt.yview)
txt.config(yscrollcommand = scroll.set)

# Initial message
txt.insert(1.0, "VNX-0-5")

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
roimenu = Menu(editmenu, tearoff = False)
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

selectmenu.add_command(label = ' From file', 
                       command = select_image, 
                       image = ico_file, 
                       compound = 'left', 
                       font = ('Helvetica', 10))


# Pre-processing (Edit menu)
menubar.add_cascade(label = 'Edit', menu = editmenu)

editmenu.add_cascade(label = 'ROI Selection',
                     menu = roimenu,
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

roimenu.add_command(label = ' ROI Selector', 
                    command = select_roi, 
                    image = ico_edit, 
                    compound = 'left', 
                    font = ('Helvetica', 10))
roimenu.add_separator()
roimenu.add_command(label = ' ROI Location', 
                    command = select_roi_range, 
                    image = ico_edit, 
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

runmenu.add_command(label = "Text Detection (Only)", 
                    command = text_detection, 
                    image = ico_new, 
                    compound = 'left', 
                    font = ('Helvetica', 10))
runmenu.add_separator()
runmenu.add_command(label = "Text Detection and Recognition", 
                    command = text_recognition, 
                    image = ico_new, 
                    compound = 'left', 
                    font = ('Helvetica', 10))
runmenu.add_separator()
runmenu.add_command(label = "Advanced Parameter Setting", 
                    command = change_params, 
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
  
      