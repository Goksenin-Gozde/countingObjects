import matplotlib.pyplot as plt
from skimage import io,data,segmentation,filters
import cv2
import numpy as np

#Resmi griye çeviriyoruz
img = io.imread("objets2.jpg")
gray_im = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#Contrast ayarları ve gama correction

gray_correct = np.array(255*(gray_im/255) ** 1.2 , dtype='uint8')
plt.subplot(221)
plt.title("Gama Correction y = 1.2 ")
plt.imshow(gray_correct , cmap='gray', vmin= 0 , vmax= 255)


#Contrast ayarları ve histogram eşitleme

gray_equ = cv2.equalizeHist(gray_im)
#plt.title('Histogram Equilization')
#plt.imshow(gray_correct, cmap='gray', vmin=0 ,vmax=255)

#Local adaptive threshold

thresh = cv2.adaptiveThreshold(gray_correct,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,255,19)
thresh = cv2.bitwise_not(thresh)
plt.subplot(222)
plt.title('Local Adaptive Threshold')
plt.imshow(thresh , cmap='gray', vmin = 0 , vmax = 255)

#Dilation and erosion

kernel = np.ones((15,15), np.uint8)
img_dilation = cv2.dilate(thresh,kernel,iterations=1)
img_erode = cv2.erode(img_dilation,kernel,iterations=1)

#Clean all noise after dilation and erosion

img_erode =  cv2.medianBlur(img_erode,7)

plt.subplot(223)
plt.title("Dilation + Erosion")
plt.imshow(img_erode, cmap = 'gray' , vmin = 0 , vmax = 255)


#Labeling

ret, labels = cv2.connectedComponents(img_erode)
label_hue = np.uint8(179 * labels /np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue,blank_ch,blank_ch])
labeled_img = cv2.cvtColor(labeled_img , cv2.COLOR_HSV2BGR)

plt.subplot(224)
plt.title('Objects Counted: '+ str(ret-2))
plt.imshow(labeled_img)

print("Objects number is: ", ret-2)
plt.show()