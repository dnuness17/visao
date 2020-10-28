import numpy as np
import matplolib.pyplot as plt

def create_figure_with_axes(
    e1=np.array([1,0,0]),
    e2=np.array([0,1,0]),
    e3=np.array([0,0,1]),
    point=np.array([0,0,0,1]),
    fig_size=(10,5),
    xlim=[-2,2],
    ylim=[-2,2],
    zlim=[-2,2]):
    

  # create axes
  axes = plt.axes(projection='3d')
  axes.set_xlim(xlim)
  axes.set_xlabel("x axis")
  axes.set_ylim(ylim)
  axes.set_ylabel("y axis")
  axes.set_zlim(zlim)
  axes.set_zlabel("z axis")

  # plot arrows
  axes.quiver(point[0],point[1],point[2],e1,0,0,color='red',pivot='tail',length=1.5)
  axes.quiver(point[0],point[1],point[2],0,e2,0,color='green',pivot='tail',length=1.5)
  axes.quiver(point[0],point[1],point[2],0,0,e3,color='blue',pivot='tail',length=1.5)

  return axes

def set_axes_equal(ax):
  """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

  Args:
      ax (matplotlib.axes): Axes to be set equal
  """  
  
  x_limits = ax.get_xlim3d()
  y_limits = ax.get_ylim3d()
  z_limits = ax.get_zlim3d()

  x_range = abs(x_limits[1] - x_limits[0])
  x_middle = np.mean(x_limits)
  y_range = abs(y_limits[1] - y_limits[0])
  y_middle = np.mean(y_limits)
  z_range = abs(z_limits[1] - z_limits[0])
  z_middle = np.mean(z_limits)

  # The plot bounding box is a sphere in the sense of the infinity
  # norm, hence I call half the max range the plot radius.
  plot_radius = 0.5*max([x_range, y_range, z_range])

  ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
  ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
  ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])