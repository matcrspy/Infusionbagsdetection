import cv2
from matplotlib import pyplot as plt
import argparse

image = cv2.imread("erode.jpg")
#generate grayscale image and display it
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
ghist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(ghist)
plt.show()
plt.waitKey(0)