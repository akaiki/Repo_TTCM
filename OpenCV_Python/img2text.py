import cv2
import pytesseract as ptr
import numpy as np 
from matplotlib import pyplot as plt 

ptr.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
img = cv2.imread('test_hand_write.png')
img = cv2.resize(img, None, fx=2.5, fy=2.5)
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaptive_threshold = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91,11)

print("---Normal detect---\n" + ptr.image_to_string(img))
print("---Threshold detect---\n" + ptr.image_to_string(adaptive_threshold))
cv2.imshow('Nor', img)
cv2.imshow("adaptive th", adaptive_threshold)
cv2.waitKey(0)





# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()