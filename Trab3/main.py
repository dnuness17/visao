#%% Inicialização

import cv2
import numpy as np
from numpy.linalg import inv
from cv2 import aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from camera import Camera

# Informação da calibração de cada câmera
cam0 = Camera("0.json")
cam1 = Camera("1.json")
cam2 = Camera("2.json")
cam3 = Camera("3.json")

# Nome dos arquivos
file_name_0 = "camera-00.mp4" 
file_name_1 = "camera-01.mp4" 
file_name_2 = "camera-02.mp4" 
file_name_3 = "camera-03.mp4" 

# Pegando o aruco para comparação
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters =  aruco.DetectorParameters_create()

# Loading dos videos
vid_0 = cv2.VideoCapture(file_name_0)
vid_1 = cv2.VideoCapture(file_name_1)
vid_2 = cv2.VideoCapture(file_name_2)
vid_3 = cv2.VideoCapture(file_name_3)

#%% Aruco

# array para armazenar os arucos que deram certo no frame (ex: [0,1,3])
frame_atual = np.array([],dtype=int)

# lista para armazenar o conjunto dos arucos que deram certo (ex: [[0,1] , [0,1,3], ...])
aruco_ok = []

# ponto central a cada frame
centro_0 = []
centro_1 = []
centro_2 = []
centro_3 = []

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

    # Pegando quinas do aruco  
    gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    corners_0, ids, rejectedImgPoints = aruco.detectMarkers(gray_0, aruco_dict, parameters=parameters)

    # se tiver detecção com as 4 quinas, retira a distorção
    if corners_0 and corners_0[0].shape[1] == 4:
        frame_atual = np.append(frame_atual,0)

        corners_0 = corners_0[0][0]
        quinas = np.array(cv2.undistortPoints(
            src=corners_0,
            cameraMatrix=cam0.I,
            distCoeffs=cam0.cd,
            P=cam0.I)).reshape(4,2)  # Salvar esses pontos num array (o mesmo que salva os -1)
        x = np.mean(quinas[:,0])
        y = np.mean(quinas[:,1])

        centro_0.append(np.array([x,y,1]).reshape(3,1))
    else:
        centro_0.append(None)

    # Pegando quinas do aruco   
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    corners_1, ids, rejectedImgPoints = aruco.detectMarkers(gray_1, aruco_dict, parameters=parameters)

    # se tiver detecção com as 4 quinas, retira a distorção
    if corners_1 and corners_1[0].shape[1] == 4:
        frame_atual = np.append(frame_atual,1)

        corners_1 = corners_1[0][0]
        quinas = np.array(cv2.undistortPoints(
            src=corners_1,
            cameraMatrix=cam1.I,
            distCoeffs=cam1.cd,
            P=cam1.I)).reshape(4,2)  # Salvar esses pontos num array (o mesmo que salva os -1)
        x = np.mean(quinas[:,0])
        y = np.mean(quinas[:,1])

        centro_1.append(np.array([x,y,1]).reshape(3,1))
    else:
        centro_1.append(None)

    # Pegando quinas do aruco  
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    corners_2, ids, rejectedImgPoints = aruco.detectMarkers(gray_2, aruco_dict, parameters=parameters)

    # se tiver detecção com as 4 quinas, retira a distorção
    if corners_2 and corners_2[0].shape[1] == 4:
        frame_atual = np.append(frame_atual,2)

        corners_2 = corners_2[0][0]
        quinas = np.array(cv2.undistortPoints(
            src=corners_2,
            cameraMatrix=cam2.I,
            distCoeffs=cam2.cd,
            P=cam2.I)).reshape(4,2)  # Salvar esses pontos num array (o mesmo que salva os -1)
        x = np.mean(quinas[:,0])
        y = np.mean(quinas[:,1])

        centro_2.append(np.array([x,y,1]).reshape(3,1))
    else:
        centro_2.append(None)
    
    # Pegando quinas do aruco   
    gray_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
    corners_3, ids, rejectedImgPoints = aruco.detectMarkers(gray_3, aruco_dict, parameters=parameters)

    # se tiver detecção com as 4 quinas, retira a distorção
    if corners_3 and corners_3[0].shape[1] == 4:
        frame_atual = np.append(frame_atual,3)

        corners_3 = corners_3[0][0]
        quinas = np.array(cv2.undistortPoints(
            src=corners_3,
            cameraMatrix=cam3.I,
            distCoeffs=cam3.cd,
            P=cam3.I)).reshape(4,2)  # Salvar esses pontos num array (o mesmo que salva os -1)
        x = np.mean(quinas[:,0])
        y = np.mean(quinas[:,1])

        centro_3.append(np.array([x,y,1]).reshape(3,1))
    else:
        centro_3.append(None)
    
    # adiciona a lista dos frames ok no aruco_ok e zera os indices frame_atual
    aruco_ok.append(frame_atual)
    frame_atual = np.array([],dtype=int)

    # Interrompe o ciclo quando o vídeo acaba
    if cv2.waitKey(1) == ord('q'):
        break      

#%% Triangulação

# listas para facilitar o loop
camera_list = [cam0,cam1,cam2,cam3]
centro_list = [centro_0,centro_1,centro_2,centro_3]

# lista para abrigar resultados
x_final = []
y_final = []
z_final = []

# itera frame por frame pegando lista das cameras que estavam ok
for index,frame in enumerate(aruco_ok):

    # lista de vetores para a conta
    W_list = []
    B_list = []

    # pega cameras que estavam ok
    for cam_id in frame:
        W = inv(camera_list[cam_id].I @ camera_list[cam_id].R) @ centro_list[cam_id][index]
        RT = inv(camera_list[cam_id].R) @ camera_list[cam_id].T

        W_list.append(W)
        B_list.append(RT)
    
    # confere o caso para montar as matrizes certas
    if len(W_list) == 2:
        A0 = np.hstack((-np.eye(3),W_list[0],np.zeros((3,1))))
        A1 = np.hstack((-np.eye(3),np.zeros((3,1)),W_list[1]))
        A = np.vstack((A0,A1))
    elif len(W_list) == 3:
        A0 = np.hstack((-np.eye(3),W_list[0],np.zeros((3,2))))
        A1 = np.hstack((-np.eye(3),np.zeros((3,1)),W_list[1],np.zeros((3,1))))
        A2 = np.hstack((-np.eye(3),np.zeros((3,2)),W_list[2]))
        A = np.vstack((A0,A1,A2))
    elif len(W_list) == 4:
        A0 = np.hstack((-np.eye(3),W_list[0],np.zeros((3,3))))
        A1 = np.hstack((-np.eye(3),np.zeros((3,1)),W_list[1],np.zeros((3,2))))
        A2 = np.hstack((-np.eye(3),np.zeros((3,2)),W_list[2],np.zeros((3,1))))
        A3 = np.hstack((-np.eye(3),np.zeros((3,3)),W_list[3]))
        A = np.vstack((A0,A1,A2,A3))
    
    if len(W_list) > 1:
        B = np.vstack(tuple(B_list))

        aux,_,_,_ = np.linalg.lstsq(A, B, rcond=None)
        x_final.append(aux[0,0])
        y_final.append(aux[1,0])
        z_final.append(aux[2,0])    

#%% Plot

def set_axes_equal(ax):

  x_limits = ax.get_xlim3d()
  y_limits = ax.get_ylim3d()
  z_limits = ax.get_zlim3d()

  x_range = abs(x_limits[1] - x_limits[0])
  x_middle = np.mean(x_limits)
  y_range = abs(y_limits[1] - y_limits[0])
  y_middle = np.mean(y_limits)
  z_range = abs(z_limits[1] - z_limits[0])
  z_middle = np.mean(z_limits)

  plot_radius = 0.5*max([x_range, y_range, z_range])

  ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
  ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
  ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_final, y_final, z_final)
set_axes_equal(ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# %%
