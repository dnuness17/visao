import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils

def homography(src_points,dest_points,s,t,p,e):

    # RANSAC starting parameters
    N = np.ceil(np.log10(1-p) / np.log10(1 - (1-e)**s),dtype='int32') # Number of iter.
    T = (1-e)*N # desired percentage of inliers
    best_S = None # best group of inliers
    best_perc_inliers = 0 # best percentage of inliers
    sample_count = 0

    while N > sample_count:

        # choose random points
        chosen_index = np.random.randint(
            low=0,
            high=src_points.shape[0],
            size=s
        )

        # pick random points
        rand_src_points = src_points[chosen_index,:]
        rand_dest_points = dest_points[chosen_index,:]
        
        H = calculate_dlt(rand_src_points,rand_dest_points)

        proj_points = H @ src_points.T
        proj_points = proj_points.T[:,:-1] # ignore last coordinate

        # matrix with projection errors
        diff = abs(proj_points - dest_points)

        # only samples where both coordinates are less than t from true values
        is_inlier = np.all(diff[diff < t],axis=1) 

        best_perc = len(is_inlier)/src_points.shape[0]

        if best_perc > best_perc_inliers:
            best_perc_inliers = best_perc
            best_S = (src_points[is_inlier,:],dest_points[is_inlier,:])

        if 1-best_perc < e:
            e = 1-best_perc
            N = np.ceil(np.log10(1-p) / np.log10(1 - (1-e)**s),dtype='int32')
        
        sample_count += 1

    # if conditions were found
    H = calculate_dlt(src_points=best_S[0],dest_points=best_S[1])
    return H
         
def calculate_dlt(src_points,dest_points):
    A = []

    # normalize
    scaler_1 = Normalization()
    scaler_2 = Normalization()
    src_points = scaler_1.fit_transform(src_points)
    dest_points = scaler_2.fit_transform(dest_points)
    T_src = scaler_1.get_T()
    T_dest = scaler_2.get_T()

    for i in range(src_points.shape[0]):
        x = src_points[i,0]
        y = src_points[i,1]
        xl = dest_points[i,0]
        yl = dest_points[i,1]
        A.append(np.array([0,0,0,-xl,-yl,-1,yl*x,yl*y,yl]))
        A.append(np.array([x,y,1,0,0,0,-xl*x,-xl*y,-xl]))

    A = np.array(A)
    _,_,V = np.linalg.svd(A)
    last_line = V[-1,:]/V[-1,-1] # normalize by the last value

    H = last_line.reshape(3,3)
    H = np.linalg.inv(T_dest) @ H @ T_src
    H = H/H[-1,-1]
    return H

class Normalization():

    def __init__(self):
        self.x_centroid = None
        self.y_centroid = None
        self.x_scale = None
        self.y_scale = None

    def fit_transform(self,points):

        # define centroid
        self.x_centroid = points[:,0].mean()
        self.y_centroid = points[:,1].mean()

        # center
        centered_points = points - np.array([[self.x_centroid,self.y_centroid]])

        # find scales
        self.x_scale = np.mean(abs(centered_points[:,0]))
        self.y_scale = np.mean(abs(centered_points[:,1]))

        normalized_points = centered_points / np.array([[self.x_scale,self.y_scale]])
        ones = np.ones((points.shape[0],1),dtype='float64')

        # return homogeneous coordinate points
        return np.hstack((normalized_points,ones))

    def inverse_transform(self,points):
        # Function made for 2D points

        # check if fit was executed at least once
        assert(x_centroid is not None)

        T = self.get_T()
        inverse_points = np.linalg.inv(T) @ points.T
        estimated_points = inverse_points.T 
        estimated_points = estimated_points / estimated_points[:,-1].reshape(-1,1)

        return estimated_points

    def get_T(self):
        # transformation to normalize the points

        # check if fit was already executed
        assert(x_centroid is not None)

        # translation
        T1 = np.array(
            [
                [1,0,-self.x_centroid],
                [0,1,-self.y_centroid],
                [0,0,1]
            ])

        # scale
        T2 = np.array(
            [
                [1/self.x_scale,0,0],
                [0,1/self.y_scale,0],
                [0,0,1]
            ])

        return T2 @ T1