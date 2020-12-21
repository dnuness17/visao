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
        self.E = np.array([
            [auxE[0],auxE[1],auxE[2]],
            [auxE[3],auxE[4],auxE[5]],
            [auxE[6],auxE[7],auxE[8]]
        ])

        # Closing file 
        f.close() 

    def distorcer(self,pontos):

        # Passando ponto a ponto para aplicação da distorção
        [a,b] = np.shape(pontos)    # tamanho da matriz de pontos
        Md = np.zeros_like(pontos)  # matriz de zeros que serão adicionados os pontos já distorcidos posteriormente
        for i in np.arange(0,a):
            for j in np.arange(0,b):
                mod = np.abs(pontos[i,j]) # módulo do ponto
                # função de distorção
                f = 1 + self.cd[0]*mod + self.cd[1]*mod**2 + self.cd[2]*mod**3 + self.cd[3]*mod**4 + self.cd[4]*mod**5  
                Md[i,j] = f*pontos[i,j]
        return Md
