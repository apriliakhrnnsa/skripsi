import numpy as np
from skimage.color import rgb2xyz, convert_colorspace
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import os

path = 'C:/Users/APRIL/PycharmProjects/pythonProject1/luka_campur/31.jpg'

image = imread(os.path.join(os.path.dirname(__file__), path), as_gray=False)
height, width, channel = image.shape

image_xyz = np.empty((height,width,channel))
#image_xyz = rgb2xyz(image)

for i in range(height):
    for j in range(width):
        for k in range(channel):
            R = image.item(i,j,0)
            G = image.item(i,j,1)
            B = image.item(i,j,2)

#            Rumus RGB ke XYZ dari buku
            X = (R*0.49000 + G*0.31000 + B*0.20000)/0.17697
            Y = (R*0.17697 + G*0.81240 + B*0.01063)/0.17697
            Z = (R*0.00000 + G*0.01000 + B*0.99000)/0.17697

            x = X/(X+Y+Z)
            y = Y/(X+Y+Z)
            z = Z/(X+Y+Z)

            image_xyz.itemset((i,j,0), float(x))
            image_xyz.itemset((i,j,1), float(y))
            image_xyz.itemset((i,j,2), float(z))

#            r = R/255
#            g = G/255
#            b = B/255

#           linearisasi nilai RGB, ** = pangkat
#            def sRGB(v):
#                if v <= 0.04045 :
#                    return v/12.92
#                else :
#                    return ((v + 0.055)/1.055)**2.4

#            r_ = sRGB(r)
#            g_ = sRGB(g)
#            b_ = sRGB(b)

#            RGB = np.array([r_, g_, b_])

#            M = np.array([[0.4124, 0.3576, 0,1805],
#                 [0.2126, 0.7152, 0.0722],
#                 [0.0193, 0.1192, 0.9506]])

#            XYZ = RGB * M

#            for x in range(0, len(RGB)):
#                baris = []
#                for y in range(0, len(RGB[0])):
#                    total = 0
#                    for z in range(0, len(RGB)):
#                        hasil = total + (RGB[x][z] * M[z][y])
#                    baris.append(hasil)
#                XYZ.append(baris)

#            for x in range(0, len(XYZ)):
#                for y in range(0, len(XYZ[0])):
#                    print(XYZ[x][y], end = '')
#                print()

#            Rumus sRGB ke XYZ
#            X = r_*0.412453 + g_*0.357580 + b_*0.180423
#            Y = r_*0.212671 + g_*0.715160 + b_*0.072169
#            Z = r_*0.019334 + g_*0.119193 + b_*0.950227

plt.subplot(121), imshow(image)
print(image.shape)
print(image)

plt.subplot(122), imshow(image_xyz)
print(image_xyz.shape)
print(image_xyz)

plt.show()