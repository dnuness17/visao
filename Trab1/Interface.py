# Import das bibliotecas 
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D 

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
label_sx.place(x=0.74*largura-1, y=0.02*altura)

label_sy = Label(menu_principal, text = "Sy")
label_sy.place(x=0.81*largura-1, y=0.02*altura)

# Buttons

transladar = Button(menu_principal, text = "Transladar  :", relief = "raised")
transladar.place(x=0.3*largura, y=0.07*altura)

rotacionar = Button(menu_principal, text = "Rotacionar :", relief = "raised")
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
Ox = DoubleVar(value=0)
slide_ox = Scale(menu_principal, from_=1, to=-1, orient = VERTICAL, resolution = 0.1, variable = Ox)
slide_ox.place(x=0.50*largura, y=0.1*altura)
Oy = DoubleVar(value=0)
slide_oy = Scale(menu_principal, from_=1, to=-1, orient = VERTICAL, resolution = 0.1, variable = Oy)
slide_oy.place(x=0.57*largura, y=0.1*altura)
Fc = DoubleVar(value=1)
slide_fc = Scale(menu_principal, from_=100, to=1, orient = VERTICAL, resolution = 0.1, variable = Fc)
slide_fc.place(x=0.64*largura, y=0.1*altura)
Sx = DoubleVar(value=1)
slide_sx = Scale(menu_principal, from_=10, to=0, orient = VERTICAL, resolution = 0.1, variable = Sx)
slide_sx.place(x=0.71*largura, y=0.1*altura)
Sy = DoubleVar(value=1)
slide_sy = Scale(menu_principal, from_=10, to=0, orient = VERTICAL, resolution = 0.1, variable = Sy)
slide_sy.place(x=0.78*largura, y=0.1*altura)

# Creating a separate 3D figure
fig0 = plt.figure()
ax0 = plt.axes(projection='3d')

# Creating a separate 2D figure
fig1 = plt.figure()
ax1 = plt.axes()
ax1.grid()
ax1.set_aspect('equal')

# Plot

chart1_type = FigureCanvasTkAgg(fig0, menu_principal)
chart1_type.get_tk_widget().place(x=-0.05*largura, y=0.3*altura)

chart2_type = FigureCanvasTkAgg(fig1, menu_principal)
chart2_type.get_tk_widget().place(x=0.5*largura, y=0.3*altura)

menu_principal.mainloop()

