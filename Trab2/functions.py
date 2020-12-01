import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
from scipy.optimize import minimize, root, least_squares
#import imutils

def homography(src_points,dest_points,scaler_1,scaler_2,s,t,p,e):

    # RANSAC starting parameters
    N = np.ceil(np.log10(1-p) / np.log10(1 - (1-e)**s)) # Number of iter.
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
        
        H = calculate_dlt(
            src_points=rand_src_points,
            dest_points=rand_dest_points,
            scaler_1=scaler_1,
            scaler_2=scaler_1
        )

        m, n = src_points.shape
        C = np.ones((m,1))                              
        src_points_new = np.concatenate((src_points,C),1)  
        proj_points = np.dot(H,np.transpose(src_points_new)).T
        proj_points = proj_points / proj_points[:,-1].reshape(-1,1)
        proj_points = proj_points[:,:-1] # ignore last coordinate

        # matrix with projection errors
        diff = abs(proj_points - dest_points)

        # only samples where both coordinates are less than t from true values
        #is_inlier = np.all(diff[diff < t]) 
        is_inlier = np.all(diff < t,axis = 1) 

        best_perc = sum(is_inlier)/src_points.shape[0]

        if best_perc > best_perc_inliers:
            best_perc_inliers = best_perc
            best_S = (src_points[is_inlier,:],dest_points[is_inlier,:])

        # e = 1-best_perc
        # N = np.ceil(np.log10(1-p) / np.log10(1 - (1-e)**s))

        if best_perc > 1-e:
            break
        
        sample_count += 1

    # if conditions were found
    H = calculate_dlt(
        src_points=best_S[0],
        dest_points=best_S[1],
        scaler_1=scaler_1,
        scaler_2=scaler_1
    )
    return H
         
def calculate_dlt(src_points,dest_points,scaler_1,scaler_2):
    A = []

    # normalize
    src_points = scaler_1.transform(src_points)
    dest_points = scaler_2.transform(dest_points)
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

    def fit(self,points):

        # define centroid
        self.x_centroid = points[:,0].mean()
        self.y_centroid = points[:,1].mean()

        # center
        centered_points = points - np.array([[self.x_centroid,self.y_centroid]])

        # find scales
        self.x_scale = np.mean(abs(centered_points[:,0]))
        self.y_scale = np.mean(abs(centered_points[:,1]))

    def transform(self,points):
        ones = np.ones((points.shape[0],1),dtype='float64')
        homogeneous_points = np.hstack((points,ones))
        
        aux = np.transpose(self.get_T() @ homogeneous_points.T)
        return aux / aux[:,-1].reshape(-1,1)
        
    def fit_transform(self,points):
        self.fit(points)
        return self.transform(points)

    def inverse_transform(self,points):
        # Function made for 2D points

        # check if fit was executed at least once
        assert(self.x_centroid is not None)

        T = self.get_T()
        inverse_points = np.linalg.inv(T) @ points.T
        estimated_points = inverse_points.T 
        estimated_points = estimated_points / estimated_points[:,-1].reshape(-1,1)

        return estimated_points

    def get_T(self):
        # transformation to normalize the points

        # check if fit was already executed
        assert(self.x_centroid is not None)

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

def symmetric_transfer_error(H, src, dst):
    H = H.reshape((3, 3))
    H_inv = np.linalg.inv(H)
  
    dst_pred = np.dot(H, src)
    src_pred = np.dot(H_inv, dst)

    src_loss = np.linalg.norm((src_pred/src_pred[-1,:])-src, axis=0)
    dst_loss = np.linalg.norm((dst_pred/dst_pred[-1,:])-dst, axis=0)
    cost = src_loss**2 + dst_loss**2
    return sum(cost) 

def optimize_ste(H_ini, src, dst):
  
    m1, n1 = src.shape
    C = np.ones((m1,1))                              
    src_new = np.transpose(np.concatenate((src,C),1)) 

    m2, n2 = dst.shape
    D = np.ones((m2,1))                              
    dst_new = np.transpose(np.concatenate((dst,D),1))

    res = minimize(
        symmetric_transfer_error, 
        H_ini.flatten(), 
        args=(src_new, dst_new), 
        method='Powell', 
        options={'maxiter': 2000})
    H = res.x.reshape((3, 3))
  
    return H/H[-1,-1]       

def reprojection_error (param, src, dst):

    aux = param[0:9]
    H = aux.reshape((3, 3))
  
    aux = param[9:param.shape[0]]
    src_hat = aux.reshape((src.shape[1], 3)).T  # vector 3xN (N - number of points)
  
    dst_hat = np.dot(H,src_hat)

    src_loss = np.linalg.norm((src_hat/src_hat[-1,:])-src, axis=0)
    dst_loss = np.linalg.norm((dst_hat/dst_hat[-1,:])-dst, axis=0)
    cost = src_loss + dst_loss
  
    return np.sum(cost)

def optimize_min(P, src0, dst0):
  
    m1, n1 = src0.shape
    C = np.ones((m1,1))
    src = np.transpose(np.concatenate((src0,C),1)) 

    m2, n2 = dst0.shape
    D = np.ones((m2,1))                              
    dst = np.transpose(np.concatenate((dst0,D),1))

    res = minimize(
        reprojection_error, 
        P.ravel(), 
        method='Powell', 
        args=(src, dst), 
        options={'maxiter': 2000})  
    #'Powell' 'CG' 'BFGS' 'L-BFGS-B' 'TNC' 'COBYLA' 'SLSQP'
  
    H = res.x[0:9].reshape((3, 3))
    H = H/H[-1,-1]

    aux = res.x[9:res.x.shape[0]]
    src_hat = aux.reshape((src.shape[1], 3)).T
    src_hat = src_hat/src_hat[-1,:]

  
    return H, src_hat