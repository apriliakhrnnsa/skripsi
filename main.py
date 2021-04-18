import cv2
import numpy as np
from skimage.color import rgb2xyz, convert_colorspace, xyz2lab, rgb2lab
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import compare_images
from sklearn.cluster import KMeans

#path = 'C:/Users/APRIL/PycharmProjects/skripsi-april/luka_hitam/1.jpg'
#image = imread(os.path.join(os.path.dirname(__file__), path), as_gray=False)

image = imread('../skripsi-april/luka_hitam/1.jpg', as_gray=False)
height, width, channel = image.shape

image_xyz = np.empty((height,width,channel))
image_lab1 = np.empty((height,width,channel)) #Citra LAB dengan rumus dari buku

mask = imread('../skripsi-april/luka_hitam/1_region.png', as_gray=True)

#image_xyz = rgb2xyz(image)
#image_lab2 = rgb2lab(image, illuminant='D65', observer='2') #Citra LAB langsung dari skimage

#result = Image.blend(mask, image_lab1, alpha=0.5)
mask_resize = resize(mask, (height, width, 3))

hasil1 = compare_images(mask_resize, image, method='blend') #hasil overlay mask dengan LAB Manual
#hasil2 = compare_images(mask_resize, image_lab2, method='blend') #hasil overlay mask dengan LAB skimage

for i in range(height):
    for j in range(width):
        for k in range(channel):
            R = hasil1.item(i,j,0)
            G = hasil1.item(i,j,1)
            B = hasil1.item(i,j,2)

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
                if s > (6/29)**3 :
                    return s**(1/3)
                else :
                    return (1/3)*((29/6)**2)*s + 16/116

            fx = t(x_)
            fy = t(y_)
            fz = t(z_)

            L = (116*fy) - 16
            a = 500*(fx - fy)
            b = 200*(fy - fz)

#           untuk mengeset nilai LAB ke gambar
            image_lab1.itemset((i, j, 0), float(L))
            image_lab1.itemset((i, j, 1), float(a))
            image_lab1.itemset((i, j, 2), float(b))

#            RGB = np.array([r_, g_, b_])

#            M = np.array([[0.4124,  0.3576, 0,1805],
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

#k-means
img_n = image_lab1.reshape(image_lab1.shape[0]*image_lab1.shape[1], image_lab1.shape[2])
print('LAB reshape: ', img_n.shape)

kmeans = KMeans(n_clusters=3, random_state=0).fit(img_n)
img_km = kmeans.cluster_centers_[kmeans.labels_]

img_cluster = img_km.reshape(image_lab1.shape[0], image_lab1.shape[1], image_lab1.shape[2])

mean1 = np.mean(image_lab1) #Mean citra LAB dengan rumus
#mean2 = np.mean(image_lab2) #Mean citra LAB dari skimage

vars1 = np.var(image_lab1) #Varian citra LAB dengan rumus
#vars2 = np.var(image_lab2) #Varian citra LAB dari skimage

print('Mean LAB Manual: ', mean1)
#print('Mean LAB skimage: ', mean2)

print('Varian LAB Manual: ', vars1)
#print('Varian LAB skimage: ', vars2)

fig = plt.figure()
#ax1 = fig.add_subplot(131), imshow(image)
ax1 = fig.add_subplot(221), imshow(image_lab1)
ax2 = fig.add_subplot(222), imshow(img_cluster)
ax3 = fig.add_subplot(223), imshow(hasil1)
ax4 = fig.add_subplot(224), imshow(image)

print('Citra Asli: ',image.shape)
#print(image)

print('LAB Manual: ', image_lab1.shape)
#print(image_lab1)

print('Mask: ', mask_resize.shape)

plt.show()
