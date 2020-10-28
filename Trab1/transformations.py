import numpy as np
from math import pi,sin,cos

def translation(dx,dy,dz):
    """ Function to create Translation matrix

    Args:
        dx (float): Rotation in X axis
        dy (float): Rotation in Y axis
        dz (float): Rotation in Z axis

    Returns:
        np.array: Translation Matrix
    """    
    return np.array([
    [1,0,0,dx],
    [0,1,0,dy],
    [0,0,1,dz],
    [0,0,0,1]
    ])

def rotate_x(teta,degree=True):
    """ Function to create Rotation matrix over X axis
    Args:
        teta (float): Angle to rotate
        degree (bool, optional): If teta is in degrees. 
        Defaults to True.

    Returns:
        np.array: Rotation matrix
    """    

    angle = teta * pi/180 if degree else teta
    return np.array([
        [1,0,0,0],
        [0,cos(angle),-sin(angle),0],
        [0,sin(angle),cos(angle),0],
        [0,0,0,1]
    ])

def rotate_y(teta,degree=True):
    """ Function to create Rotation matrix over Y axis
    Args:
        teta (float): Angle to rotate
        degree (bool, optional): If teta is in degrees. 
        Defaults to True.

    Returns:
        np.array: Rotation matrix
    """   

    angle = teta * pi/180 if degree else teta
    return np.array([
        [cos(angle),0,sin(angle),0],
        [0,1,0,0],
        [-sin(angle),0,cos(angle),0],
        [0,0,0,1]
    ])

def rotate_z(teta,degree=True):
    """ Function to create Rotation matrix over Z axis
    Args:
        teta (float): Angle to rotate
        degree (bool, optional): If teta is in degrees. 
        Defaults to True.

    Returns:
        np.array: Rotation matrix
    """   

    angle = teta * pi/180 if degree else teta
    return np.array([
        [cos(angle),-sin(angle),0,0],
        [sin(angle),cos(angle),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])

