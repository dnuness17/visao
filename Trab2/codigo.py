#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import functions as f
#import imutils
#from functions import homography

t = 10                # true values 
s = 4                 # Samples
e = 0.8               # Outlier proportion
p = 0.99              # Probability of outlier-free samples

MIN_MATCH_COUNT = 10
img1 = cv.imread('comicsStarWars01.jpg',0) # queryImage
img2 = cv.imread('comicsStarWars02.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# KNN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# launch exception if len(good) is too low 
try:
    assert(len(good) > MIN_MATCH_COUNT)
except AssertionError:
    raise AssertionError("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

src_points = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
dest_points = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

#%%
# Normalization
scaler_1 = f.Normalization()
scaler_1.fit(src_points)
scaler_2 = f.Normalization()
scaler_2.fit(dest_points)

H = f.homography(src_points,dest_points,scaler_1,scaler_2,s,t,p,e)

M, mask = cv.findHomography(src_points, dest_points, cv.RANSAC,5.0)

#%%
# Achando a matriz homogênea por otimizações

# Symmetric_transfer_error
O_ste = f.optimize_ste(H, src_points, dest_points)

#%%

# Reprojection_error

m1, n1 = src_points.shape
C = np.ones((m1,1))
src_aux = np.concatenate((src_points,C),1)

P = np.vstack((H,(src_aux)))   # Shape of P: (3+N) x 3.  P Contains the homography from Ransac and points of the first image
R, src_hat = f.optimize_min(P, src_points, dest_points)

#%%

# OpenCV
img3 = cv.warpPerspective(img2, np.linalg.inv(M), (img2.shape[1],img2.shape[0]))
img4 = cv.warpPerspective(img1, M, (img1.shape[1],img1.shape[0]))

# RANSAC
img5 = cv.warpPerspective(img2, np.linalg.inv(H), (img2.shape[1],img2.shape[0]))
img6 = cv.warpPerspective(img1, H, (img1.shape[1],img1.shape[0]))

#%%

# Otimização/Symmetric_transfer_error 
img7 = cv.warpPerspective(img2, np.linalg.inv(O_ste), (img2.shape[1],img2.shape[0]))
img8 = cv.warpPerspective(img1, O_ste, (img1.shape[1],img1.shape[0]))

#%%

# Otimização/Reprojection_error
img9 = cv.warpPerspective(img2, np.linalg.inv(R), (img2.shape[1],img2.shape[0]))
img10 = cv.warpPerspective(img1, R, (img1.shape[1],img1.shape[0]))

# Plot das imagens geradas
#%%

fig = plt.figure(figsize=(30,10))
ax1 = fig.add_subplot(5,2,1)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
ax1 = fig.add_subplot(5,2,2)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
ax1 = fig.add_subplot(5,2,3)
plt.title('Segunda imagem após transformação de referência')
plt.imshow(img3, 'gray')
ax1 = fig.add_subplot(5,2,4)
plt.title('Primeira imagem após transformação de referência')
plt.imshow(img4, 'gray')
ax1 = fig.add_subplot(5,2,5)
plt.title('Segunda imagem após transformação usando RANSAC')
plt.imshow(img5, 'gray')
ax1 = fig.add_subplot(5,2,6)
plt.title('Primeira imagem após transformação usando RANSAC')
plt.imshow(img6, 'gray')
ax1 = fig.add_subplot(5,2,7)
plt.title('Segunda imagem após transformação usando Otimização 1')
plt.imshow(img7, 'gray')
ax1 = fig.add_subplot(5,2,8)
plt.title('Primeira imagem após transformação usando Otimização 1')
plt.imshow(img8, 'gray')

#%%

ax1 = fig.add_subplot(5,2,7)
plt.title('Segunda imagem após transformação usando Otimização 1')
plt.imshow(img7, 'gray')
ax1 = fig.add_subplot(5,2,8)
plt.title('Primeira imagem após transformação usando Otimização 1')
plt.imshow(img8, 'gray')

#%%

ax1 = fig.add_subplot(5,2,9)
plt.title('Segunda imagem após transformação usando Otimização 2')
plt.imshow(img9, 'gray')
ax1 = fig.add_subplot(5,2,10)
plt.title('Primeira imagem após transformação usando Otimização 2')
plt.imshow(img10, 'gray')



#%%
plt.show()


# %%

#%%

scaler = Normalization()
aux = scaler.fit_transform(src_points)

plt.plot(src_points[:,0],src_points[:,1],'.')
plt.show()
plt.figure()
plt.plot(aux[:,0],aux[:,1],'.')
plt.show()
# %%
