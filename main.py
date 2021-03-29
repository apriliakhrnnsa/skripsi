import numpy as np
from skimage.color import rgb2xyz, convert_colorspace, xyz2lab, rgb2lab
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import os

#path = 'C:/Users/APRIL/PycharmProjects/skripsi-april/luka_hitam/1.jpg'
#image = imread(os.path.join(os.path.dirname(__file__), path), as_gray=False)

image = imread('../skripsi-april/luka_hitam/1.jpg', as_gray=False)
height, width, channel = image.shape

image_xyz = np.empty((height,width,channel))
image_lab = np.empty((height,width,channel))

#image_xyz = rgb2xyz(image)
#image_lab = rgb2lab(image)

for i in range(height):
    for j in range(width):
        for k in range(channel):
            R = image.item(i,j,0)
            G = image.item(i,j,1)
            B = image.item(i,j,2)

#           Rumus RGB ke XYZ dari buku
            X = (R*0.49000 + G*0.31000 + B*0.20000)/0.17697
            Y = (R*0.17697 + G*0.81240 + B*0.01063)/0.17697
            Z = (R*0.00000 + G*0.01000 + B*0.99000)/0.17697

#           normalisasi nilai XYZ supaya berada dalam rentang 0 sampai 1
            x = X/(X+Y+Z)
            y = Y/(X+Y+Z)
            z = Z/(X+Y+Z)

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

#            Rumus sRGB ke XYZ
#            X = r_*0.412453 + g_*0.357580 + b_*0.180423
#            Y = r_*0.212671 + g_*0.715160 + b_*0.072169
#            Z = r_*0.019334 + g_*0.119193 + b_*0.950227

#           untuk mengeset nilai XYZ ke gambar
            image_xyz.itemset((i, j, 0), float(x))
            image_xyz.itemset((i, j, 1), float(y))
            image_xyz.itemset((i, j, 2), float(z))

#           ref XYZ dengan standar Illuminant D65 dan standar observer 2 derajat
            x_n = 95.0489
            y_n = 100
            z_n = 108.8840

            x_ = x/x_n
            y_ = y/y_n
            z_ = z/z_n

            def t(s):
                if s > 0.008856 :
                    return s**(1/3)
                else :
                    return (7.787037)*s + 16/116

            fx = t(x_)
            fy = t(y_)
            fz = t(z_)

            L = (116*fy) - 16
            a = 500*(fx - fy)
            b = 200*(fy - fz)

#           untuk mengeset nilai LAB ke gambar
            image_lab.itemset((i, j, 0), float(L))
            image_lab.itemset((i, j, 1), float(a))
            image_lab.itemset((i, j, 2), float(b))

mean1 = np.mean(image_lab1) #Mean citra LAB dengan rumus
mean2 = np.mean(image_lab2) #Mean citra LAB dari skimage

vars1 = np.var(image_lab1) #Varian citra LAB dengan rumus
vars2 = np.var(image_lab2) #Varian citra LAB dari skimage

print('Mean LAB Manual: ', mean1)
print('Mean LAB skimage: ', mean2)

print('Varian LAB Manual: ', vars1)
print('Varian LAB skimage: ', vars2)

#plt.title('Citra Asli', size=5)
plt.subplot(1,3,1), imshow(image)
#print(image.shape)
#print(image)

#plt.subplot(1,3,2), imshow(image_xyz)
#print(image_xyz.shape)
#print(image_xyz)

#plt.title('LAB Manual', size=5)
plt.subplot(1,3,2), imshow(image_lab1)
print(image_lab1.shape)
#print(image_lab1)

#plt.title('LAB skimage', size=5)
plt.subplot(1,3,3), imshow(image_lab2)
print(image_lab1.shape)
#print(image_lab2)

plt.show()
