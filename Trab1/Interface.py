# Import das bibliotecas 
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D 
import transformations as t
from camera import Camera
from object import Object

# Settings da janela de trabalho
menu_principal = Tk()
menu_principal.title("Trabalho 1")
largura_screen = menu_principal.winfo_screenwidth()
altura_screen = menu_principal.winfo_screenheight()
largura = 0.78*largura_screen
altura = 0.8*altura_screen
posx = largura_screen/2 - largura/2
posy = altura_screen/2 - altura/2 - 0.05*altura_screen
menu_principal.geometry("%dx%d+%d+%d" % (largura,altura, posx, posy))

# Criando objeto (casa)

house = np.array([[0,         0,         0],
         [6,  -10.0000,         0],
         [6, -10.0000,   12.0000],
         [6,  -10.4000,   11.5000],
         [6,   -5.0000,   16.0000],
         [6,         0,   12.0000],
         [6,    0.5000,   11.4000],
         [6,         0,   12.0000],
         [6,         0,         0],
  [-6.0000,         0,         0],
  [-6.0000,   -5.0000,         0],
  [-6.0000,  -10.0000,         0],
         [6,  -10.0000,         0],
         [6,  -10.0000,   12.0000],
[-6.0000,  -10.0000,   12.0000],
  [-6.0000,         0,   12.0000],
         [6,         0,   12.0000],
         [6,  -10.0000,   12.0000],
         [6,  -10.5000,   11.4000],
  [-6.0000,  -10.5000,   11.4000],
  [-6.0000,  -10.0000,   12.0000],
  [-6.0000,   -5.0000,   16.0000],
         [6,   -5.0000,   16.0000],
         [6,    0.5000,   11.4000],
  [-6.0000,    0.5000,   11.4000],
  [-6.0000,         0,   12.0000],
  [-6.0000,   -5.0000,   16.0000],
  [-6.0000,  -10.0000,   12.0000],
  [-6.0000,  -10.0000,         0],
  [-6.0000,   -5.0000,         0],
  [-6.0000,         0,         0],
  [-6.0000,         0,   12.0000],
  [-6.0000,         0,         0]])

house = np.transpose(house/12)

#Representing the object in homogeneous coordinates
#create row of ones
num_columns = np.size(house,1)
ones_line = np.ones(num_columns)


#add to the house matrix to represent the house in homogeneous coordinates
house = np.vstack([house, ones_line])

# Transladar a casa para cima

aux_house = t.translation(0,-0.4375,4)
house = Object(0,-0.4375,4,np.linalg.inv(aux_house),house)

# Criando camera

global camera
camera = Camera(0,0,1,1,0,14,0,0,0,np.eye(4))

# Labels
label_ref = Label(menu_principal, text = "Referêncial:")
label_ref.place(x=0.02*largura, y=0.02*altura)

label_eixo = Label(menu_principal, text = "Eixo:")
label_eixo.place(x=0.02*largura, y=0.13*altura)

label_valores = Label(menu_principal, text = "Valores:")
label_valores.place(x=0.3*largura, y=0.02*altura)

label_ox = Label(menu_principal, text = "Ox")
label_ox.place(x=0.52*largura, y=0.02*altura)

label_oy = Label(menu_principal, text = "Oy")
label_oy.place(x=0.59*largura, y=0.02*altura)

label_fc = Label(menu_principal, text = "Distância focal")
label_fc.place(x=0.64*largura, y=0.02*altura)

label_sx = Label(menu_principal, text = "Sx")
label_sx.place(x=0.733*largura, y=0.02*altura)

label_sy = Label(menu_principal, text = "Sy")
label_sy.place(x=0.803*largura, y=0.02*altura)

label_st = Label(menu_principal, text = "Steta")
label_st.place(x=0.866*largura, y=0.02*altura)


# Buttons

def transladar_action():
    if check_x_var.get() == 1:
        vetor = np.array([valor_T.get(),0,0])
    if check_y_var.get() == 1:
        vetor = np.array([0,valor_T.get(),0])
    if check_z_var.get() == 1:
        vetor = np.array([0,0,valor_T.get()])        
    if check_camera_var.get() == 1:
        if check_ambiente_var.get() == 1:
            camera.translate(vetor[0],vetor[1],vetor[2])
            plotar_3d()
            plotar_2d()
        if check_proprio_var.get() == 1:
            camera.translate_own(vetor[0],vetor[1],vetor[2]) 
            plotar_3d()  
            plotar_2d() 
    if check_objeto_var.get() == 1:
        if check_objeto_var.get() == 1:
            house.translate(vetor[0],vetor[1],vetor[2])
            plotar_3d()
            plotar_2d()
        if check_proprio_var.get() == 1:
            house.translate_own(vetor[0],vetor[1],vetor[2]) 
            plotar_3d()  
            plotar_2d()     

transladar = Button(menu_principal, text = "Transladar  :", relief = "raised", command = transladar_action)
transladar.place(x=0.3*largura, y=0.07*altura)

def rotacionar_action():      
    if check_camera_var.get() == 1:
        if check_ambiente_var.get() == 1:
            if check_x_var.get() == 1:
                camera.rotate_x(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_y_var.get() == 1:
                camera.rotate_y(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_z_var.get() == 1:
                camera.rotate_z(valor_R.get())
                plotar_3d()
                plotar_2d()    
        if check_proprio_var.get() == 1:
            if check_x_var.get() == 1:
                camera.rotate_x_own(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_y_var.get() == 1:
                camera.rotate_y_own(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_z_var.get() == 1:
                camera.rotate_z_own(valor_R.get())  
                plotar_3d() 
                plotar_2d()
    if check_objeto_var.get() == 1:
        if check_ambiente_var.get() == 1:
            if check_x_var.get() == 1:
                house.rotate_x(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_y_var.get() == 1:
                house.rotate_y(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_z_var.get() == 1:
                house.rotate_z(valor_R.get()) 
                plotar_3d() 
                plotar_2d()  
        if check_proprio_var.get() == 1:
            if check_x_var.get() == 1:
                house.rotate_x_own(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_y_var.get() == 1:
                house.rotate_y_own(valor_R.get())
                plotar_3d()
                plotar_2d()
            if check_z_var.get() == 1:
                house.rotate_z_own(valor_R.get())  
                plotar_3d() 
                plotar_2d() 

rotacionar = Button(menu_principal, text = "Rotacionar :", relief = "raised", command = rotacionar_action)
rotacionar.place(x=0.3*largura, y=0.17*altura)

# Check boxs

def check_camera_command():
    check_objeto.deselect()
check_camera_var = IntVar(value=1)
check_camera = Checkbutton(menu_principal, text="Câmera", variable = check_camera_var, command = check_camera_command)
check_camera.place(x=0.02*largura, y=0.05*altura)

def check_objeto_command():
    check_camera.deselect()
check_objeto_var = IntVar(value=0)
check_objeto = Checkbutton(menu_principal, text="Objeto", variable = check_objeto_var, command = check_objeto_command)
check_objeto.place(x=0.02*largura, y=0.08*altura)

def check_ambiente_command():
    check_proprio.deselect()
check_ambiente_var = IntVar(value=1)
check_ambiente = Checkbutton(menu_principal, text="Ambiente", variable = check_ambiente_var, command = check_ambiente_command)
check_ambiente.place(x=0.15*largura, y=0.05*altura)

def check_proprio_command():
    check_ambiente.deselect()
check_proprio_var = IntVar(value=0)
check_proprio = Checkbutton(menu_principal, text="Próprio", variable = check_proprio_var, command = check_proprio_command)
check_proprio.place(x=0.15*largura, y=0.08*altura)

def check_x_command():
    check_y.deselect()
    check_z.deselect()

check_x_var = IntVar(value=1)
check_x = Checkbutton(menu_principal, text="x", variable = check_x_var, command = check_x_command)
check_x.place(x=0.02*largura, y=0.16*altura)

def check_y_command():
    check_x.deselect()
    check_z.deselect()
check_y_var = IntVar(value=0)
check_y = Checkbutton(menu_principal, text="y", variable = check_y_var, command = check_y_command)
check_y.place(x=0.02*largura, y=0.19*altura)

def check_z_command():
    check_x.deselect()
    check_y.deselect()
check_z_var = IntVar(value=0)
check_z = Checkbutton(menu_principal, text="z", variable = check_z_var, command = check_z_command)
check_z.place(x=0.02*largura, y=0.22*altura)

# Entrada de texto

valor_T = DoubleVar(value=0)
valor_transladar = Entry(menu_principal, width = round(0.01*largura), textvariable = valor_T)
valor_transladar.place(x=0.3*largura, y=0.12*altura)

valor_R = DoubleVar(value=0)
valor_rotacionar = Entry(menu_principal, width = round(0.01*largura), textvariable = valor_R)
valor_rotacionar.place(x=0.3*largura, y=0.22*altura)

# Escorrega
def ativar_Ox(Ox):
    global camera
    camera.Ox = float(Ox)
    plotar_2d()
Ox = DoubleVar(value=camera.Ox)
slide_ox = Scale(menu_principal, from_=1, to=-1, orient = VERTICAL, resolution = 0.1, variable = Ox, command = ativar_Ox)
slide_ox.place(x=0.50*largura, y=0.1*altura)

def ativar_Oy(Oy):
    global camera
    camera.Oy = float(Oy)
    plotar_2d()
Oy = DoubleVar(value=camera.Oy)
slide_oy = Scale(menu_principal, from_=1, to=-1, orient = VERTICAL, resolution = 0.1, variable = Oy, command = ativar_Oy)
slide_oy.place(x=0.57*largura, y=0.1*altura)

def ativar_Fc(Fc):
    global camera
    camera.f = float(Fc)
    plotar_2d()
Fc = DoubleVar(value=camera.f)
slide_fc = Scale(menu_principal, from_=100, to=1, orient = VERTICAL, resolution = 0.1, variable = Fc, command = ativar_Fc)
slide_fc.place(x=0.64*largura, y=0.1*altura)

def ativar_Sx(Sx):
    global camera
    camera.Sx = float(Sx)
    plotar_2d()
Sx = DoubleVar(value=1)
slide_sx = Scale(menu_principal, from_=10, to=0, orient = VERTICAL, resolution = 0.1, variable = Sx, command = ativar_Sx)
slide_sx.place(x=0.71*largura, y=0.1*altura)

def ativar_Sy(Sy):
    global camera
    camera.Sy = float(Sy)
    plotar_2d()
Sy = DoubleVar(value=1)
slide_sy = Scale(menu_principal, from_=10, to=0, orient = VERTICAL, resolution = 0.1, variable = Sy, command = ativar_Sy)
slide_sy.place(x=0.78*largura, y=0.1*altura)

def ativar_Steta(Steta):
    global camera
    camera.Steta = float(Steta)
    plotar_2d()
Steta = DoubleVar(value=1)
slide_st = Scale(menu_principal, from_=10, to=0, orient = VERTICAL, resolution = 0.1, variable = Steta, command = ativar_Steta)
slide_st.place(x=0.85*largura, y=0.1*altura)

def plotar_3d():
    plt.close('all')
    fig0 = plt.figure()
    ax0 = plt.axes(projection='3d')
    ax0.set_xlim([-5,5])
    ax0.set_ylim([-5,5])
    ax0.set_zlim([0,10])
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    pontos_aux = np.linalg.inv(house.Mwo)@house.points
    ax0.plot3D(pontos_aux[0,:], pontos_aux[1,:], pontos_aux[2,:], 'black')

    I = np.eye(4)[:,0:3]
    ec = camera.Mwc@I
    eo = house.Mwo@I

    e1c = ec[0,:]  # eixos do referencial camera
    e2c = ec[1,:]
    e3c = ec[2,:]
    e1o = eo[0,:]  # eixos do referencial objeto
    e2o = eo[1,:]
    e3o = eo[2,:]
    Xo = house.Xw
    Yo = house.Yw
    Zo = house.Zw
    Xc = camera.Xw
    Yc = camera.Yw
    Zc = camera.Zw
    ax0.quiver(Xc,Yc,Zc,e1c[0],e1c[1],e1c[2],color='red',pivot='tail',length=1.5)
    ax0.quiver(Xc,Yc,Zc,e2c[0],e2c[1],e2c[2],color='green',pivot='tail',length=1.5)
    ax0.quiver(Xc,Yc,Zc,e3c[0],e3c[1],e3c[2],color='blue',pivot='tail',length=1.5)
    ax0.quiver(Xo,Yo,Zo,e1o[0],e1o[1],e1o[2],color='red',pivot='tail',length=1.5)
    ax0.quiver(Xo,Yo,Zo,e2o[0],e2o[1],e2o[2],color='green',pivot='tail',length=1.5)
    ax0.quiver(Xo,Yo,Zo,e3o[0],e3o[1],e3o[2],color='blue',pivot='tail',length=1.5)

    chart1_type = FigureCanvasTkAgg(fig0, menu_principal)
    chart1_type.get_tk_widget().place(x=-0.05*largura, y=0.3*altura)

def plotar_2d():
    plt.close('all')
    pontos_2d = camera.get_intrinsic_matrix()@np.linalg.inv(camera.Mwc)@house.Mwo@house.points
    pontos_2d = pontos_2d/pontos_2d[-1,:]

    # Creating a separate 2D figure
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.grid()
    ax1.set_aspect('equal')
    ax1.set_xlim([-5,5])
    ax1.set_ylim([-5,5])
    ax1.plot(pontos_2d[0,:], pontos_2d[1,:])
    #ax1.quiver(0,0,1,0,color='red',pivot='tail')
    #ax1.quiver(0,0,0,1,color='green',pivot='tail')
    chart2_type = FigureCanvasTkAgg(fig1, menu_principal)
    chart2_type.get_tk_widget().place(x=0.5*largura, y=0.3*altura)

# Plot

plotar_3d()
plotar_2d()

menu_principal.mainloop()

#plt.show()