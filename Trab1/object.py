import numpy as np


class Object:

    def __init__(self,Xw,Yw,Zw,Mow,points):
        self.Xw = Xw
        self.Yw = Yw
        self.Zw = Zw
        self.Mow = Mow
        self.points = points

    ## CRIAR MATRIZES DE ROTAÇÃO e TRANSLACAO
    
    