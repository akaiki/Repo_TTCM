from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import pytesseract as ptr
import cv2
import numpy as np 
import os

ptr.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
img_path = 'TheSV.png'

# img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
# hImg, wImg, _ = img.shape
# conf = r'--oem 3 --psm 6 outputbase digits'
# boxes = ptr.image_to_data(img)
# for a,b in enumerate(boxes.splitlines()):
#         # print(b)
#         if a!=0:
#             b = b.split()
#             if len(b)==12:
#                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
#                 cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
#                 cv2.rectangle(img, (x,y), (x+w, y+h), (50, 50, 255), 2)

# cv2.imshow('boxed', img)
# cv2.waitKey(0)

#xám hóa ảnh đàu vào
img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)

identity = np.array((
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]))

edge = np.array((
    [-1,  -1,  -1],
    [-1, 8,  -1],
    [-1,  -1,  -1]))

boxblur = (1.0 / 9) * np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]])

gaussian = (1.0 / 16) * np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]])

emboss = np.array(
    [[-2, -1,  0],
     [-1,  1,  1],
     [ 0,  1,  2]])

square = np.array(
    [[ 0,  2,  0],
     [-2, -1,  2],
     [ 0, -2,  0]])

smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]))

laplacian = (1.0 / 16) * np.array(
    [[ 0,  0, -1,  0,  0],
     [ 0, -1, -2, -1,  0],
     [-1, -2, 16, -2, -1],
     [ 0, -1, -2, -1,  0],
     [ 0,  0, -1,  0,  0]])

sobelLeft = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]))

sobelRight = np.array((
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]))

sobelTop = np.array((
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]))

sobelBottom = np.array((
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]))


filters = [
    ("Identity", identity), #0
    ("Edge", edge), #1
    ("Box Blur", boxblur), #2
    ("Square", square), #3
    ("Gaussian", gaussian), #4
    ("Emboss", emboss), #5
    ("Small blur", smallBlur), #6
    ("Large blur", largeBlur), #7
    ("Sharpen", sharpen), #8
    ("Laplacian", laplacian), #9 
    ('Sobel Left', sobelLeft), #10
    ('Sobel Right', sobelRight), #11
    ('Sobel Top', sobelTop), #12
    ('Sobel Bottom', sobelBottom) #13
]

fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.1)

for i, filter in enumerate(filters):
    axes = fig.add_subplot(3, 5, i+1)
    axes.set(title=filter[0])
    axes.grid(False)
    axes.set_xticks([])
    axes.set_yticks([])
    img_out = cv2.filter2D(img, 0, filter[1])
    axes.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    print("------Filter: "+filter[0]+"------\n")
    print(ptr.image_to_string(img_out))
    print("\n---------------------------------")
plt.show()

# def show(i, filter, name):
#     img_out = cv2.filter2D(img, 0, filter[1])
#     name = filter[0]
#     print("------Filter: "+name+"------\n")
#     print(ptr.image_to_string(img_out))
#     print("\n---------------------------------")
    
#     cv2.imshow(name, img_out)
#     cv2.waitKey(0)


# name = sharpen[0]
# show(8,sharpen, name)

# for i, filter in enumerate(filters):
#     name = filter[0]
#     show(i,filter, name)

# img_out = cv2.filter2D(img, 0,emboss[1])
# img_out = cv2.filter2D(img_out, 0,gaussian[1])   
# img_out = cv2.filter2D(img_out, 0,sharpen[1])
# img_out = cv2.filter2D(img_out, 0,boxblur[1])
# img_out = cv2.filter2D(img_out, 0,edge[1])
# print(ptr.image_to_string(img_out))
# cv2.imshow("name", img_out)
# cv2.waitKey(0)
