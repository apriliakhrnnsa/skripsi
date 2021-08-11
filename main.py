import cv2
import numpy as np
import glob
import warnings
from copy import deepcopy
from skimage.color import rgb2xyz, convert_colorspace, xyz2lab, rgb2lab, gray2rgb
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import compare_images
from sklearn.cluster import KMeans
from skimage.segmentation import clear_border, find_boundaries, mark_boundaries

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

hasil1 = compare_images(mask_resize, image, method='diff') #hasil overlay mask dengan LAB Manual
hasil2 = compare_images(hasil1, image, method='blend')

for i in range(height):
    for j in range(width):
        for k in range(channel):
            R = hasil2.item(i,j,0)
            G = hasil2.item(i,j,1)
            B = hasil2.item(i,j,2)

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

mean1 = np.mean(image_lab1) #Mean citra LAB dengan rumus
#mean2 = np.mean(image_lab2) #Mean citra LAB dari skimage

vars1 = np.var(image_lab1) #Varian citra LAB dengan rumus
#vars2 = np.var(image_lab2) #Varian citra LAB dari skimage

print('Mean LAB Manual: ', mean1)
#print('Mean LAB skimage: ', mean2)

print('Varian LAB Manual: ', vars1)
#print('Varian LAB skimage: ', vars2)

#k-means tanpa pakai sklearn
X = image_lab1
Y = deepcopy(X)

print(X)

#X = np.array([[[124, 147, 137], [117, 140, 130], [109, 132, 126]],
#			  [[106, 119, 128], [104, 117, 126], [103, 116, 125]],
#			  [[120, 143, 133], [112, 135, 127], [101, 114, 123]],
#			  [[155, 157, 144], [125, 136, 140], [121, 132, 136]],
#			  [[130, 153, 143], [122, 133, 137], [152, 154, 141]]])

print(X.shape)
#print(X.sum(axis=1))

class Kmeans:
	def __init__(self, K, max_iter=500):
		self.K = K
		self.max_iter = max_iter

#	untuk menghitung centroid baru (update centroid)
	def compute_centroids(self, X, labels):
		centroids = np.zeros((self.K, X.shape[2]))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			for k in range(self.K):
				centroids[k, :] = np.mean(X[labels == k, :], axis=0)
			#print(centroids)
			return centroids

#	untuk menghitung jarak dari tiap data ke tiap centroid
	def compute_distance(self, X, centroids):
		D = np.zeros((X.shape[0], X.shape[1], self.K))
		for k in range(self.K):
			dist = np.linalg.norm(X - centroids[k, :], axis=2)
			D[:, :, k] = np.square(dist)
			#print(D.shape)
			#print(D)
		return D

#	untuk mencari jarak terdeka
	def find_closest_cluster(self, D):
		a = np.argmin(D, axis=2)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				X[i,j] = self.centroids[a[i,j]]
		return a

	def compute_sse(self, X, labels, centroids):
		distance = np.zeros((X.shape[0], X.shape[1], self.K))
		for k in range(self.K):
			distance[labels == k, :] = np.linalg.norm(X[labels == k, :] - centroids[k, :], axis=0)
			return np.sum(np.square(distance))

	def fit(self, X):
		#self.centroids = np.array([[255, 0, 0], [255, 255, 0], [0, 0, 0]])
		self.centroids = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
		#self.centroids = np.array([[0.4125, 0.2127, 0.0193], [0.7700, 0.9278, 0.1385], [0, 0, 0]])
		#print(self.centroids.shape)
		for i in range(self.max_iter):
			old_centroids = self.centroids
			distance = self.compute_distance(X, old_centroids)
			self.labels = self.find_closest_cluster(distance)
			self.centroids = self.compute_centroids(X, self.labels)

			if np.all(old_centroids == self.centroids):
				break

		self.error = self.compute_sse(X, self.labels, self.centroids)


	def predict(self, X):
		old_centroids = self.centroids
		distance = self.compute_distance(X, old_centroids)
		return self.find_closest_cluster(distance)

X_km = np.empty((X.shape[0], X.shape[1], X.shape[2]))
kmeans = Kmeans(K=3, max_iter=500)
kmeans.fit(X_km)
centroids = kmeans.centroids

def labtorgb(X):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			for k in range(X.shape[2]):
				X.itemset((i, j, 0), float(L))
				X.itemset((i, j, 1), float(a))
				X.itemset((i, j, 2), float(b))

				x_n = 95.0489
				y_n = 100
				z_n = 108.8840

				def t(s):
					if s > 6/29:
						return s**3
					else:
						return s*((6/29)**2)*(s - (16/116))

				x_1 = ((L + 16)/116) + (a/500)
				y_1 = (L + 16)/116
				z_1 = ((L + 16)/116) + (b/200)

				fx1 = t(x_1)
				fy1 = t(y_1)
				fz1 = t(z_1)

				x1 = x_n*fx1
				y1 = y_n*fy1
				z1 = z_n*fz1

				r1 = (x1*3.2404542) + (y1*(-1.5371385)) + (z1*(-0.4985314))
				g1 = (x1*(-0.9692660)) + (y1*1.8760108) + (z1*0.0415560)
				b1 = (x1*0.0556434) + (y1*(-0.2040259)) + (z1*1.0572252)

				X_km.itemset((i, j, 0), float(r1))
				X_km.itemset((i, j, 1), float(g1))
				X_km.itemset((i, j, 2), float(b1))
	return X_km

ig_km = labtorgb(X_km)

def gaussian_filter_2d(w, sigma):
	# we must be odd, or we enforce it
	w = w + (w % 2 == 0)

	# create an array of size w x w
	F = np.zeros([w, w])
	# window w must be odd if it isn't add 1 to it
	w = w + ((w % 2) == 0)
	mid = w // 2
	X = np.arange(w) - mid
	denom = 1 / (2 * np.pi * sigma ** 2)
	denom_s = 1 / (2 * sigma ** 2)
	for i in X:
		for j in X:
			power_val = (i ** 2 + j ** 2) * denom_s
			F[i + mid, j + mid] = np.exp(-power_val) * denom
	return F


def convolve_2d_global(X, K):
	# pad X
	(m, n) = X.shape
	wl = K.shape[0]
	w = wl // 2
	Xp = np.pad(X, w, 'symmetric')
	Z = np.zeros(X.shape)
	for i in range(m):
		for j in range(n):
			W = Xp[i:i + wl, j:j + wl]
			Z[i, j] = np.sum(np.dot(W, K))
	return Z


def gaussian_kernel(X, F):
	Z = np.sum(np.dot(X, F))
	return Z


def get_neighbors(X, w, i, j):
	X = np.pad(X, w, 'symmetric')
	return X[i:i + w:, j:j + w]


def meanshift(X, w, sigma):
	# store original data
	Xc = np.array(X)
	F = gaussian_filter_2d(w, sigma)

	# get all neighbors
	(m, n) = X.shape
	for i in range(m):
		for j in range(n):
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				Xn = get_neighbors(X, w, i, j)
				upper_par = (Xn - X[i, j]) * Xn
				lower_par = (Xn - X[i, j])
				new_x = gaussian_kernel(upper_par, F) / gaussian_kernel(lower_par, F)
				Xc[i, j] = new_x
	X = Xc
	return X

#RGB to grayscale follows
#imgrey = 0.3*R + 0.59*G + 0.11*B
#imgrey = 0.3*hasil2[:,:,0] + 0.59*hasil2[:,:,1] + 0.11*hasil2[:,:,2]
imgrey = 0.3*image_lab1[:,:,0] + 0.59*image_lab1[:,:,1] + 0.11*image_lab1[:,:,2]
#cast to int
#imgrey = imgrey.astype(int)

#plt.figure(0)
#plt.imshow(imgrey)

w = 3
sigma = 35
F = gaussian_filter_2d(w,sigma)
#Z= convolve_2d_global(imgrey, F)
Xk = meanshift(imgrey, w, sigma)
Xms = gray2rgb(Xk)

fig = plt.figure()
ax1 = fig.add_subplot(131), imshow(Y)
ax2 = fig.add_subplot(132), imshow(X)
ax3 = fig.add_subplot(133), imshow(X_km)

plt.show()
