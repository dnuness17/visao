#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import functions as f
#import imutils
#from functions import homography


MIN_MATCH_COUNT = 10
img1 = cv.imread('comicsStarWars01.jpg',0) # queryImage
img2 = cv.imread('comicsStarWars02.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


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


t = 10                # true values 
s = 4                 # Samples
e = 0.80              # Outlier proportion
p = 0.99              # Probability of outlier-free samples

H = f.homography(src_points,dest_points,s,t,p,e)

M, mask = cv.findHomography(src_points, dest_points, cv.RANSAC,5.0)

#%%

#homography(kp1,kp2)

#M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
M,mask = homography(...)
matchesMask = mask.ravel().tolist()

img4 = cv.warpPerspective(img1, M, (img1.shape[1],img1.shape[0])) #, None) #, flags[, borderMode[, borderValue]]]]	)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


fig = plt.figure()
fig, axs = plt.subplots(2,2,figsize=(30,15))
ax1 = fig.add_subplot(2,2,1)
plt.imshow(img3, 'gray')
ax1 = fig.add_subplot(2,2,2)
plt.title('First image')
plt.imshow(img1,'gray')
ax1 = fig.add_subplot(2,2,3)
plt.title('Second image')
plt.imshow(img2,'gray')
ax1 = fig.add_subplot(2,2,4)
plt.title('First image after transformation')
plt.imshow(img4,'gray')
plt.show()