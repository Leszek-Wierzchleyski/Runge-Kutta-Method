import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def RungeKutta(f,y0,h,t0,tf):

    """"Implements the Runge-Kutta method for a first order ordinary differential equation depending on a single
    variable. Obtains and graphs approximate numerical solutions for a given time period and with a given initial
    value."""

    N = int((tf - t0) / h)

    t = np.linspace(t0, tf, N + 1 )

    y = y0

    yn = [y0]

    for i in range(N):
        k1 = f(y)
        k2 = f(y + h * 0.5 * k1)
        k3 = f(y + h * 0.5 * k2)
        k4 = f(y + h * k3)
        y = y + (1 / 6) * h *(k1 + 2 * k2 + 2 * k3 + k4)
        yn.append(y)
    plt.plot(t, yn)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('yn')
    plt.show()
    return t, yn
    
    
    def RungKutta2var(f,y0,h,x0,xf):

    """"Implements the Runge-Kutta method for a first order ordinary differential equation depending on two variables.
    Obtains and graphs approximate numerical solutions for a given time period and with a given initial value."""

    N = int((xf - x0) / h)

    x = np.linspace(x0, xf, N + 1)

    y = y0

    yn = [y0]

    for i in range(N):
        k1 = f(x[i],y)
        k2 = f((x[i] + 0.5 * h), (y + h * 0.5 * k1))
        k3 = f((x[i] + 0.5 * h), (y + h * 0.5 * k2))
        k4 = f((x[i] + h), (y + h * k3))
        y = y + (1 / 6) * h * (k1 + 2 * k2 + 2 * k3 + k4)
        yn.append(y)
    plt.plot(x, yn)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('yn')
    plt.show()
    return x, yn


def RungeKutta2ndOrder(f, y0, z0, h, x0, xf):
    """"Implements the Runge-Kutta method for a second order ordinary differential equation depending on a two
        variables. An first order diferential terms should be treated as z. Obtains and graphs approximate numerical
        solutions for a given time period and with a given initial
        value."""
    N = int((xf - x0) / h)

    x = np.linspace(x0, xf, N + 1)

    y = y0

    yn = [y0]

    z = z0

    zn = [z0]

    for i in range(N):
        F1 = z
        k1 = f(x[i], y, z)
        k2 = f((x[i] + 0.5 * h), (y + F1 * 0.5 * h), (z + k1 * 0.5 * h))
        F2 = z + 0.5 * k1 * h
        k3 = f((x[i] + 0.5 * h), (y + h * 0.5 * F2), (z + h * 0.5 * k2))
        F3 = z + 0.5 * k2 * h
        k4 = f((x[i] + h), (y + h * F3), (z + h * k3))
        F4 = z + k3 * h
        z = z + (1 / 6) * h * (k1 + 2 * k2 + 2 * k3 + k4)
        zn.append(z)
        y = y + (1 / 6) * h * (F1 + 2 * F2 + 2 * F3 + F4)
        yn.append(y)
    fig = plt.figure(1)
    ax = fig.gca(projection = '3d')
    ax.plot_trisurf(x, yn, zn)
    plt.grid() 
    plt.xlabel('x')
    plt.ylabel('yn')
    ax.set_zlabel('dy/dx')
    plt.show()
    plt.figure(2)
    plt.plot(x, yn)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('yn')
    plt.show()
    return x, yn, zn


def RungeKutta3D(f, x0, y0, z0, h, t0, tf):
    """Implements the Runge-Kutta method to calculate and plot a 3 dimensional trajectory, based on a set of three first
     order ordinary differential equations."""

    N = int((tf - t0) / h)

    t = np.linspace(t0, tf, N + 1)
    x = x0
    y = y0
    z = z0

    xn = [x0]
    yn = [y0]
    zn = [z0]

    for i in range(N):
        k1 = f(x, y, z)
        k2 = f((x + 0.5 * h * k1[0]), (y + h * 0.5 * k1[1]), (z + h * 0.5 * k1[2]))
        k3 = f((x + 0.5 * h * k2[0]), (y + h * 0.5 * k2[1]), (z + h * 0.5 * k2[2]))
        k4 = f((x + h * k3[0]), (y + h * k3[1]), (z + h * k3[2]))
        x = x + (1 / 6) * h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + (1 / 6) * h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z = z + (1 / 6) * h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        xn.append(x)
        yn.append(y)
        zn.append(z)
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot(xn, yn, zn)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return xn, yn, zn
