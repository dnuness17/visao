import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import sys

# Nome dos arquivos
file_name_0 = "Trab3/camera-00.mp4" 
file_name_1 = "Trab3/camera-01.mp4" 
file_name_2 = "Trab3/camera-02.mp4" 
file_name_3 = "Trab3/camera-03.mp4" 

# Pegando o aruco para comparação
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()

# Loading dos videos
vid_0 = cv2.VideoCapture(file_name_0)
vid_1 = cv2.VideoCapture(file_name_1)
vid_2 = cv2.VideoCapture(file_name_2)
vid_3 = cv2.VideoCapture(file_name_3)

# apenas para testes !!!!
aux_0 = 0
aux_1 = 0
aux_2 = 0
aux_3 = 0

while True:

    # Imagens do momento do vídeo
    _, img_0 = vid_0.read()
    if img_0 is None:
        print("Empty Frame")
        break

    _, img_1 = vid_1.read()
    if img_1 is None:
        print("Empty Frame")
        break

    _, img_2 = vid_2.read()
    if img_2 is None:
        print("Empty Frame")
        break

    _, img_3 = vid_3.read()
    if img_3 is None:
        print("Empty Frame")
        break

    # Pegando quinas do aruco em cada imagem    

    gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    corners_0, ids, rejectedImgPoints = aruco.detectMarkers(gray_0, aruco_dict, parameters=parameters)
    #frame_markers = aruco.drawDetectedMarkers(img_0.copy(), corners, ids)
    #cv2.imshow('output', frame_markers)
    
    if aux_0 == 0:
        print(corners_0)
        aux_0 = 1

    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    corners_1, ids, rejectedImgPoints = aruco.detectMarkers(gray_1, aruco_dict, parameters=parameters)
    #frame_markers = aruco.drawDetectedMarkers(img_0.copy(), corners, ids)
    #cv2.imshow('output', frame_markers)
    
    if aux_1 == 0:
        print(corners_1)
        aux_1 = 1

    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    corners_2, ids, rejectedImgPoints = aruco.detectMarkers(gray_2, aruco_dict, parameters=parameters)
    #frame_markers = aruco.drawDetectedMarkers(img_0.copy(), corners, ids)
    #cv2.imshow('output', frame_markers)
    
    if aux_2 == 0:
        print(corners_2)
        aux_2 = 1    
    
    gray_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
    corners_3, ids, rejectedImgPoints = aruco.detectMarkers(gray_3, aruco_dict, parameters=parameters)
    #frame_markers = aruco.drawDetectedMarkers(img_0.copy(), corners, ids)
    #cv2.imshow('output', frame_markers)
    
    if aux_3 == 0:
        print(corners_3)
        aux_3 = 1

    
    # Interrompe o ciclo quando o vídeo acaba
    if cv2.waitKey(1) == ord('q'):
        break