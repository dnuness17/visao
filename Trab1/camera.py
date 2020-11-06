import numpy as np
import transformations as t

class Camera:
    #O iii
    def __init__(self,Ox,Oy,Sx,Sy,Steta,f,Xw,Yw,Zw,Mwc):
        self.Ox = Ox
        self.Oy = Oy
        self.Sx = Sx
        self.Sy = Sy
        self.Steta = Steta
        self.f = f
        self.Xw = Xw
        self.Yw = Yw
        self.Zw = Zw
        self.Mwc = Mwc

    def get_intrinsic_matrix(self):
        
        # intrinsic parameters matrix
        K = np.array([
            [self.f*self.Sx, self.f*self.Steta,self.Ox],
            [0,self.f*self.Sy,self.Oy],
            [0,0,1]
        ])
        
        # standard projection matrix
        pi0 = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0]
        ])

        return K @ pi0

    def set_intrinsic_parameters(
            self,Ox=None,Oy=None,Sx=None,Sy=None,
            Steta=None,f=None):

        self.Ox = Ox if Ox is not None else self.Ox
        self.Oy = Oy if Oy is not None else self.Oy
        self.Sx = Sx if Sx is not None else self.Sx
        self.Sy = Sy if Sy is not None else self.Sy
        self.Steta = Steta if Steta is not None else self.Steta
        self.f = f if f is not None else self.f

    def get_position(self):
        return np.array([[self.Xw,self.Yw,self.Zw,1]]).T

    def set_position(self,P):
        assert(len(P) == 4)
        P = P.flatten()
        self.Xw = P[0]/P[3]
        self.Yw = P[1]/P[3]
        self.Zw = P[2]/P[3]

    def rotate_x(self,teta,degree=True):
        R = t.rotate_x(teta,degree)
        new_position = R @ self.get_position()
        self.set_position(new_position)
        self.Mwc = self.Mwc @ np.linalg.inv(R)

    def rotate_y(self,teta,degree=True):
        R = t.rotate_y(teta,degree)
        new_position = R @ self.get_position()
        self.set_position(new_position)
        self.Mwc = self.Mwc @ np.linalg.inv(R)

    def rotate_z(self,teta,degree=True):
        R = t.rotate_z(teta,degree)
        new_position = R @ self.get_position()
        self.set_position(new_position)
        self.Mwc = self.Mwc @ np.linalg.inv(R)

    def rotate_x_own(self,teta,degree=True):
        R = t.rotate_x(teta,degree)
        self.Mwc = R @ self.Mwc

    def rotate_y_own(self,teta,degree=True):
        R = t.rotate_y(teta,degree)
        self.Mwc = R @ self.Mwc

    def rotate_z_own(self,teta,degree=True):
        R = t.rotate_z(teta,degree)
        self.Mwc = R @ self.Mwc

    def translate(self,dx,dy,dz):
        T = t.translation(dx,dy,dz)
        new_position = T @ self.get_position()
        print(new_position)
        self.set_position(new_position)
        self.Mwc = np.linalg.inv(T) @ self.Mwc 
        print(self.Mwc)

    def translate_own(self,dx,dy,dz):
        T = t.translation(dx,dy,dz)
        position_in_camera_frame = np.linalg.inv(self.Mwc) @ self.get_position()
        new_position_in_camera_frame = T @ position_in_camera_frame
        new_position = self.Mwc @ new_position_in_camera_frame
        self.set_position(new_position)
        self.Mwc = self.Mwc @ np.linalg.inv(T)

