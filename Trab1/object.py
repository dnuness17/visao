import numpy as np
import transformations as t


class Object:

    def __init__(self,Xw,Yw,Zw,Mwo,points):
        self.Xw = Xw
        self.Yw = Yw
        self.Zw = Zw
        self.Mwo = Mwo
        self.points = points

    def get_position(self):
        return np.array([[self.Xw,self.Yw,self.Zw,1]]).T

    def set_position(self,P):
        assert(len(P) == 4)
        self.Xw = P[0]
        self.Yw = P[1]
        self.Zw = P[2]

    def rotate_x(self,teta,degree=True):
        R = t.rotate_x(teta,degree)
        new_position = R @ self.get_position()
        self.set_position(new_position)
        self.Mwo = self.Mwo @ np.inv(R)

    def rotate_y(self,teta,degree=True):
        R = t.rotate_y(teta,degree)
        new_position = R @ self.get_position()
        self.set_position(new_position)
        self.Mwo = self.Mwo @ np.inv(R)

    def rotate_z(self,teta,degree=True):
        R = t.rotate_z(teta,degree)
        new_position = R @ self.get_position()
        self.set_position(new_position)
        self.Mwo = self.Mwo @ np.inv(R)

    def rotate_x_own(self,teta,degree=True):
        R = t.rotate_x(teta,degree)
        self.Mwo = R @ self.Mwo

    def rotate_y_own(self,teta,degree=True):
        R = t.rotate_y(teta,degree)
        self.Mwo = R @ self.Mwo

    def rotate_z_own(self,teta,degree=True):
        R = t.rotate_z(teta,degree)
        self.Mwo = R @ self.Mwo

    def translate(self,dx,dy,dz):
        T = t.translation(dx,dy,dz)
        new_position = T @ self.get_position()
        self.set_position(new_position)
        self.Mwo = np.inv(T) @ self.Mwo 

    def translate_own(self,dx,dy,dz):
        T = t.translation(dx,dy,dz)
        position_in_camera_frame = np.inv(self.Mwo) @ self.get_position()
        new_position_in_camera_frame = T @ position_in_camera_frame
        new_position = self.Mwo @ new_position_in_camera_frame
        self.set_position(new_position)
        self.Mwo = self.Mwo @ np.inv(T)
    
    