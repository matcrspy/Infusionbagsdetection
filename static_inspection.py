import os
import cv2
import numpy as np

from matplotlib import pyplot as plt


def get_image(image_path):
    name_list = []
    img_list = []
    
    files = os.listdir(image_path)
    for file in files:
        img = cv2.imdecode(np.fromfile(image_path + file, dtype = np.uint8), cv2.IMREAD_COLOR)
        img_list.append(img)
        name = os.path.splitext(file)[0]
        name_list.append(name)
    return img_list, name_list

def capROI(image):
    # [y:y+h, x:x+w]
    roi = image[100:800, 2400:3400]
    return roi

def bagROI(image):
    # [y:y+h, x:x+w]
    roi = image[1000:3400, 1700:3900]
    return roi

def imgPreProcess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray

def liquidMask(image):
    image = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower = np.array([15, 10, 0], dtype = "uint8")
    upper = np.array([65, 43, 255], dtype = "uint8")
    
    mask = cv2.inRange(hsv, lower, upper)
    # plt.imshow(mask, 'gray')
    # plt.show()
    
    return mask

def filterContours(image, contours, lower, upper):
    filtered_contours = []
    imageArea = image.shape[0]*image.shape[1]
    
    for i in contours:
        area = cv2.contourArea(i)
        if (area>= lower*imageArea) and (area <= upper*imageArea):
            filtered_contours.append(i)
    
    return filtered_contours

# 内盖
def detect_cap(image):
    num_pixels = 0
    thres_value = 108
    result = ''
    
    image = capROI(image)
    gray = imgPreProcess(image)
    ret, thres = cv2.threshold(gray, thres_value, 255, cv2.THRESH_BINARY)
    # plt.imshow(thres, cmap='gray')
    # plt.axis('off')
    # plt.show()
    
    num_pixels = len(thres[thres == 255])
    # print(num_pixels)
    
    if num_pixels > 550000:
        result = '1(内盖异常)'
    else:
        result = '0(内盖正常)'
    return result

# 装量
def detect_liquid(image):
    result = ''
    
    image = bagROI(image)
    mask = liquidMask(image)
    
    num_pixels = len(mask[mask == 255])
    print(num_pixels)
    
    if num_pixels > 2900000 or num_pixels < 2650000:
        result = '1(装量异常)'
    else:
        result = '0(装量正常)'
    return result


# # 装量（轮廓）
# def detect_liquidContour(image):
#     result = ''
#     image = bagROI(image)
#     mask = liquidMask(image)
    
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_contours = filterContours(image, contours, 0.2, 0.99)
  
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.drawContours(image, filtered_contours, -1, (255, 0, 0), 5)
    
    
#     area = cv2.contourArea(filtered_contours[0])
   
#     if area < 670000:
#         result = '1(装量异常)'
#     else:
#         result = '0(装量正常)'
#     return result

# 废边
def detect_bagContour(image):
    thres_value = 12
    result = ''
    
    # img = bagROI(image)
    gray = imgPreProcess(image)
    ret, thres = cv2.threshold(gray, thres_value, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filterContours(image, contours, 0.5, 0.99)

    try:
        contour = filtered_contours[0]
    except IndexError:
        print("Index Out of Range: No contour Detected or contours are all filtered")
        return '0(废边正常)'
    
    perimeter = cv2.arcLength(filtered_contours[0], True)
    if perimeter < 5500:
        result = '1(废边异常)'
    else:
        result = '0(废边正常)'
    return result


# TopLeft, BottomRight, Center
def getPoints(image):

    # Parameters: Subject to change
    thres_value = 62

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create binary thresholded image
    ret, thres = cv2.threshold(gray, thres_value, 255, cv2.THRESH_BINARY)

    # findcontours(source image, contour retrieval mode, contour approximation method)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filterContoursForPoints(image, contours)

    # plt.imshow(thres)
    # plt.show()


    try:
        contour = filtered_contours[0]
    except IndexError:
        print("Index Out of Range: No contour Detected or contours are all filtered")
        return image, ((-1, -1), (-1, -1), (-1, -1))

    x, y, w, h = cv2.boundingRect(filtered_contours[0])

    TopLeft = (x, y)
    BottomRight = (x + w, y + h)
    Center = (x + w // 2, y + h // 2)

    # Uncomment to see plots
    image = cv2.drawContours(image, filtered_contours, -1, (255, 0, 0), 5)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=5)
    image = cv2.circle(image, BottomRight, radius=100, color=(0, 0, 255), thickness=5)
    image = cv2.circle(image, TopLeft, radius=100, color=(0, 0, 255), thickness=5)
    image = cv2.circle(image, Center, radius=100, color=(0, 0, 255), thickness=5)

    # plt.imshow(image)
    # plt.show()
    return image, (TopLeft, BottomRight, Center)

# Area proportion subject to change
def filterContoursForPoints(image, contours):
    filtered_contours = []
    imageArea = image.shape[0]*image.shape[1]
    for i in contours:
        area = cv2.contourArea(i)
        if (area >= 0.15 * imageArea) and (area <= 0.3 * imageArea):
            filtered_contours.append(i)
    return filtered_contours

def findPoints(contours):
    BottomRight = findBottomRight(contours)
    TopLeft = findTopLeft(contours)
    Center = ((BottomRight[0] + TopLeft[0]) // 2, (BottomRight[1] + TopLeft[1]) // 2)

    return BottomRight, TopLeft, Center

def findBottomRight(contours):
    BottomRight = (-1, -1)
    for point in contours:
        if point[0] + point[1] > BottomRight[0] + BottomRight[1]:
            BottomRight = (point[0], point[1])
    return BottomRight

def findTopLeft(contours):
    TopLeft = (float('inf'), float('inf'))
    for point in contours:
        if point[1] > 1000 and point[0] + point[1] < TopLeft[0] + TopLeft[1]:
            TopLeft = (point[0], point[1])
    return TopLeft

# 接口异常
def detectPort(image):
    image = portROI(image)
    black, yellow = detectBlack(image), detectYellow(image)
    if black: print("Detected Black")
    if yellow: print("Detected Yellow") 

    if black or yellow:
        return '1(接口异常)'

    return '0(接口正常)'

def detectBlack(image):
    #Thresh
    thres_value = 107
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    ret, thres = cv2.threshold(gray, thres_value, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filterCountersForPort(image, contours)

    result = False
    if makePortDecision(filtered_contours):
        result = True

    image = cv2.drawContours(image, filtered_contours, -1, (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()

    return result

def detectYellow(image):

    mid = np.array([57, 18, 49], dtype = "uint8")
    lower, upper = (15,0,0), (36, 255, 255)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    hsv_y = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
    hsv_y = cv2.cvtColor(hsv_y, cv2.COLOR_HSV2RGB)
    hsv_y = cv2.cvtColor(hsv_y, cv2.COLOR_RGB2BGR)
    hsv_y = cv2.cvtColor(hsv_y, cv2.COLOR_RGB2GRAY)


    contours, hierarchy = cv2.findContours(hsv_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filterCountersForPort(image, contours)

    result = False
    if makePortDecision(filtered_contours):
        result = True


    image = cv2.drawContours(image, filtered_contours, -1, (255, 0, 0), 1)


    plt.imshow(image)
    plt.show()


    result = False
    if makePortDecision(filtered_contours):
        result = True

    plt.imshow(image)
    plt.show()
    return result

def portROI(image):
    roi = image[900:1100, 2400:3200]
    return roi

def filterCountersForPort(image, contours):
    filtered_contours = []
    imageArea = image.shape[0]*image.shape[1]
    for i in contours:
        area = cv2.contourArea(i)
        if (area >= 0.001 * imageArea) and (area <= 0.80 * imageArea):
            filtered_contours.append(i)
    return filtered_contours if filtered_contours else []

def makePortDecision(contours):
    return True if len(contours) > 0 else False


if __name__ == "__main__":
    
    # path = 'D:/Qianxun/无内盖/'
    # path = 'D:/Qianxun/装量异常/'
    path = 'img/'
    out = "result/"


    images, names = get_image(path)
    
    
    # # Only the front side
    # i = 1
    # while(i<len(images)):
    #     print(names[i])
    #     print(detect_cap(images[i]))
    #     i = i+2
    # image = cv2.imread('img/10.bmp')
    # detectPort(image)


    i = 0
    for img in images:
        print(names[i])

        # getPoints
        image, result = getPoints(img)
        cv2.imwrite(out + names[i] + '.png', image)

        # Detect 接口异物
        # print(detectPort(img))
        i = i+1
    


