# coding: utf-8
import cv2
import numpy as np
import glob
import sys

filter_con_min = 0.25
filter_con_max = 0.6

#read images of .bmp format
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

#read images of .jpg format
def load_images_jpg(fold):
    pattern = fold + '\\*\\*.jpg'
    i_lst = []
    n_lst = []
    for path in glob.glob(pattern):
        print(path)
        n_lst.append(path)
        i_lst.append(cv2.imread(path))
    return i_lst, n_lst


def show_result(im, na, num, tag, max_num_show):
    if num < max_num_show:
        display_im = cv2.resize(im,(int(im.shape[1]/2), int(im.shape[0]/2)))
        cv2.imshow("{}: {}".format(str(num), tag), display_im)
        cv2.waitKey(0)


def blur(new_image, mode=1):
    if mode == 1:
        new_image = cv2.GaussianBlur(new_image, (13, 13), 0)
        # new_image = cv2.GaussianBlur(new_image, (0, 0), sigmaX = 2.5, sigmaY = 2.5)
    elif mode == 2:
        new_image = cv2.medianBlur(new_image, 5)
    elif mode == 3:
        new_image = cv2.bilateralFilter(new_image, 9, 75, 75)
    return new_image


def thresholding(corrected, mode=1):
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


def preprocess(src, show=True):
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
    im = blur(im, 2)
    if show:
        cv2.imshow("blur", im)
    th = thresholding(im, 2)  # binary
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return th, scaled_img


def filter_my_contours(image, contour_list):
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


def border_extraction(im, mode=2):
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
    filtered_con = filter_my_contours(th, contours_list)
    print("Filtered Contours num:{}".format(len(filtered_con)))
    result_im = np.copy(scaled_im)
    try:
        contour = filtered_con[0]
        # new_image = cv2.rectangle(new_image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
    except IndexError:
        print("Index Out of Range For Data: No contour Detected or contours are all filtered")
        filtered_con = None

    return result_im, filtered_con


def mark_convex(j, defects, im, cnt, draw=True):
    s, e, f, d = defects[j, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    if draw:
        cv2.line(im, start, end, [0, 255, 0], 2)
        cv2.circle(im, far, 5, [255, 0, 0], -1)
    return f


def locate_centroid(img, contour, text):
    M = cv2.moments(contour)  # calculating centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (155, 0, 0), 2, cv2.LINE_AA)


def locate_key_points(im, con):
    if con is not None:
        # this image has been located with a large bounding box
        # need to locate key points for this image
        bag_contour = con[0]
        hull = cv2.convexHull(bag_contour, returnPoints=False)
        defects = cv2.convexityDefects(bag_contour, hull)
        far1_d = far2_d = far1_df_idx = far2_df_idx = -1
        for j in range(defects.shape[0]):
            s, e, f, d = defects[j, 0]
            if d > far2_d:
                far1_d, far1_df_idx = far2_d, far2_df_idx
                far2_d, far2_df_idx = d, j
            elif d > far1_d:
                far1_d, far1_df_idx = d, j

        new_im = np.copy(im)
        far2_cnt_idx = mark_convex(far2_df_idx, defects, new_im, bag_contour)
        far1_cnt_idx = mark_convex(far1_df_idx, defects, new_im, bag_contour)
        start = min(far1_cnt_idx, far2_cnt_idx)
        end = max(far1_cnt_idx, far2_cnt_idx)
        minRect2 = cv2.minAreaRect(bag_contour[start:end + 1])  # find rotated minimum bounding box
        box = cv2.boxPoints(minRect2)
        box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(new_im, [box], 0, (255, 0, 0), 3)
        #cv2.drawContours(new_im, [bag_contour], 0, (0, 0, 255), 3)

        #draw bounding box
        
        minRect = cv2.minAreaRect(bag_contour)  # find rotated minimum bounding box
        box = cv2.boxPoints(minRect)
        box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(new_im, [box], 0, (0, 0, 255), 3)
        # x, y, w, h = cv2.boundingRect(contour)

        locate_centroid(new_im, bag_contour, "infusion bag")  # find centroid of bag
        locate_centroid(new_im, bag_contour[start:end + 1], "port")  # find centroid of port and label it
        result = new_im
    else:
        result = im

    return result


if __name__ == "__main__":
    show1 = 100
    show2 = True
    # folder = 'images'
    # images, names = load_images_jpg(folder)
    folder = 'img'
    images, names = load_images_bmp(folder)
    count = 0
    for (image, name) in zip(images, names):
        # show_result(images, names, show1)
        processed_img, scaled_img = preprocess(image, show2)
        #show_result(processed_img, name, count, "preprocessed", show1)

        border_images = border_extraction(processed_img, 1)
        #show_result(border_images, name, count, "edge image", show1)

        located, contour = locate_bags(border_images, scaled_img)
        # show_result(located, name, show1)

        if contour is None:
            print("Could not locate infusion bags in data: ".format(name))

        key_located = locate_key_points(scaled_img, contour)
        # show_result(scaled_imgs, name, count)
        show_result(key_located, name, count, "result", show1)
        count += 1
