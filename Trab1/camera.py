import numpy as np
from transformations import rotate_x,rotate_y,rotate_z,translation

class Camera:
    
    def __init__(self,Ox,Oy,Sx,Sy,Steta,f,Xw,Yw,Zw,Mcw):
        self.Ox = Ox
        self.Oy = Oy
        self.Sx = Sx
        self.Sy = Sy
        self.Steta = Steta
        self.f = f
        self.Xw = Xw
        self.Yw = Yw
        self.Zw = Zw
        self.Mcw = Mcw

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
        self.Xw = P[0]
        self.Yw = P[1]
        self.Zw = P[2]

    def get_Mcw(self):
        return self.Mcw

    

        

