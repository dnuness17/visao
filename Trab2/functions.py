import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils

def homography(src_points,dest_points):
    # normalize
    scaler_1 = Normalization()
    scaler_2 = Normalization()
    norm_kp1 = scaler_1.fit_transform(src_points)
    norm_kp2 = scaler_1.fit_transform(dest_points)

    



## def dlt

## def ransac

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