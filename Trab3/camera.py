import numpy as np
import json 

class Camera:

    def __init__(self,arq_name):

        # Opening JSON file 
        f = open(arq_name,) 
  
        # returns JSON object as a dictionary 
        data = json.load(f)

        # Coeficientes de distorção
        aux_cd = [data['distortion']['doubles'][0],data['distortion']['doubles'][1],data['distortion']['doubles'][2],data['distortion']['doubles'][3],data['distortion']['doubles'][4]]
        self.cd = np.array([aux_cd[0],aux_cd[1],aux_cd[2],aux_cd[3],aux_cd[4]])

        # Matriz dos parâmetros intrínsecos
        auxI = data['intrinsic']['doubles']
        self.I = np.array([
            [auxI[0],auxI[1],auxI[2]],
            [auxI[3],auxI[4],auxI[5]],
            [auxI[6],auxI[7],auxI[8]]
        ])

        # Matriz dos parâmetros extrínsecos
        auxE = data['extrinsic']['tf']['doubles']
        E = np.array([
            [auxE[0],auxE[1],auxE[2],auxE[3]],
            [auxE[4],auxE[5],auxE[6],auxE[7]],
            [auxE[8],auxE[9],auxE[10],auxE[11]],
            [auxE[12],auxE[13],auxE[14],auxE[15]],
        ])

        E = np.linalg.inv(E)
        self.R = E[:3,:3]
        self.T = E[:3,3].reshape(3,1)

        # Closing file 
        f.close() 