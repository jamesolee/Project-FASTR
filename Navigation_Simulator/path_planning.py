import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Define control points for a closed loop
# theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
# x = np.cos(theta) + 0.1 * np.random.randn(8)
# y = np.sin(theta) + 0.1 * np.random.randn(8)
# z = np.sin(2*theta) + 0.1 * np.random.randn(8)

x = [0, 1, 0, 1, 0]
y = [0, 0, 1, 1, 0]
z = [0.1, 0.1, 0.1, 0.1, 0]


def hermite_path(x, y, z, dx, dy, dz):
    i = 0
    while i < len(dx):
        dx[i] = 7 * dx[i]
        dy[i] = 7 * dy[i]
        dz[i] = 7 * dz[i]
        i += 1

    p_x = []
    p_y = []
    p_z = []
    v_x = []
    v_y = []
    v_z = []

    i = 0
    t = []
    
    while i < len(x) - 1:
        t.append(i)
        i += 0.01

    h1 = []
    h2 = []
    h3 = []
    h4 = []
    dh1 = []
    dh2 = []
    dh3 = []
    dh4 = []

    for i in t:
        s = i % 1
        h1.append(2 * s**3 - 3 * s**2 + 1)
        h2.append(-2 * s**3 + 3 * s**2)
        h3.append(s**3 -2 * s**2 + s)
        h4.append(s**3 - s**2)
        dh1.append(6 * s**2 - 6 * s)
        dh2.append(-6 * s**2 + 6 * s)
        dh3.append(3 * s**2 - 4 * s + 1)
        dh4.append(3 * s**2 - 2 * s)

    
    dt = 0
    while dt < len(t) - 1:
        i = int(dt/100)
        p_x.append(float(h1[dt]) * float(x[i]) + float(h2[dt]) * float(x[i + 1]) + float(h3[dt]) * dx[i] + float(h4[dt]) * float(dx[i + 1]))
        p_y.append(float(h1[dt]) * float(y[i]) + float(h2[dt]) * float(y[i + 1]) + float(h3[dt]) * dy[i] + float(h4[dt]) * float(dy[i + 1]))
        p_z.append(float(h1[dt]) * float(z[i]) + float(h2[dt]) * float(z[i + 1]) + float(h3[dt]) * dz[i] + float(h4[dt]) * float(dz[i + 1]))
        v_x.append(float(dh1[dt]) * float(x[i]) + float(dh2[dt]) * float(x[i + 1]) + float(dh3[dt]) * dx[i] + float(dh4[dt]) * float(dx[i + 1]))
        v_y.append(float(dh1[dt]) * float(y[i]) + float(dh2[dt]) * float(y[i + 1]) + float(dh3[dt]) * dy[i] + float(dh4[dt]) * float(dy[i + 1]))
        v_z.append(float(dh1[dt]) * float(z[i]) + float(dh2[dt]) * float(z[i + 1]) + float(dh3[dt]) * dz[i] + float(dh4[dt]) * float(dz[i + 1]))
        dt += 1

    return p_x, p_y, p_z, v_x, v_y, v_z


def catmull_rom_path(x, y, z):
    x = [2 * x[0] - x[1]] + x + [2 * x[len(x) - 1] - x[len(x) - 2]]
    y = [2 * y[0] - y[1]] + y + [2 * y[len(y) - 1] - y[len(y) - 2]]
    z = [2 * z[0] - z[1]] + z + [2 * z[len(z) - 1] - z[len(z) - 2]]

    p_x = []
    p_y = []
    p_z = []
    v_x = []
    v_y = []
    v_z = []

    i = 0
    t = []
    
    while i < len(x) - 3:
        k = i % 1
        t.append(float(k))
        i += 0.01


    dt = 0
    while dt < len(t) - 1:
        i = int(dt/100) 
        p_x.append(0.5 * (2 * float(x[i+1]) + (-float(x[i]) + float(x[i + 2]))*t[dt] + (2 * float(x[i]) - 5 * float(x[i+1]) + 4 * float(x[i+2]) -float(x[i+3]))*t[dt] ** 2 + (-float(x[i]) + 3 * float(x[i+1]) -3 * float(x[i+2]) + float(x[i+3]))*t[dt] ** 3))
        p_y.append(0.5 * (2 * float(y[i+1]) + (-float(y[i]) + float(y[i + 2]))*t[dt] + (2 * float(y[i]) - 5 * float(y[i+1]) + 4 * float(y[i+2]) -float(y[i+3]))*t[dt] ** 2 + (-float(y[i]) + 3 * float(y[i+1]) -3 * float(y[i+2]) + float(y[i+3]))*t[dt] ** 3))
        p_z.append(0.5 * (2 * float(z[i+1]) + (-float(z[i]) + float(z[i + 2]))*t[dt] + (2 * float(z[i]) - 5 * float(z[i+1]) + 4 * float(z[i+2]) -float(z[i+3]))*t[dt] ** 2 + (-float(z[i]) + 3 * float(z[i+1]) -3 * float(z[i+2]) + float(z[i+3]))*t[dt] ** 3))
        v_x.append(0.5 * ((-float(x[i]) + float(x[i + 2])) + 2 *(2 * float(x[i]) - 5 * float(x[i+1]) + 4 * float(x[i+2]) -float(x[i+3]))*t[dt] + 3 *(-float(x[i]) + 3 * float(x[i+1]) -3 * float(x[i+2]) + float(x[i+3]))*t[dt] ** 2))
        v_y.append(0.5 * ((-float(y[i]) + float(y[i + 2])) + 2 *(2 * float(y[i]) - 5 * float(y[i+1]) + 4 * float(y[i+2]) -float(y[i+3]))*t[dt] + 3 *(-float(y[i]) + 3 * float(y[i+1]) -3 * float(y[i+2]) + float(y[i+3]))*t[dt] ** 2))
        v_z.append(0.5 * ((-float(z[i]) + float(z[i + 2])) + 2 *(2 * float(z[i]) - 5 * float(z[i+1]) + 4 * float(z[i+2]) -float(z[i+3]))*t[dt] + 3 *(-float(z[i]) + 3 * float(z[i+1]) -3 * float(z[i+2]) + float(z[i+3]))*t[dt] ** 2))
        dt += 1
    
    return p_x, p_y, p_z, v_x, v_y, v_z


def natural_cubic_path(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    t= np.arange(len(x))
    p_x = CubicSpline(t, x, bc_type='natural')
    p_y = CubicSpline(t, y, bc_type='natural')
    p_z = CubicSpline(t, z, bc_type='natural')

    t_smooth = np.linspace(t[0], t[-1], 600)
    x_smooth = list(p_x(t_smooth))
    y_smooth = list(p_y(t_smooth))
    z_smooth = list(p_z(t_smooth))

    v_x = list(p_x(t_smooth, 1))
    v_y = list(p_y(t_smooth, 1))
    v_z = list(p_z(t_smooth, 1))

    return x_smooth, y_smooth, z_smooth, v_x, v_y, v_z


def TCB_path(x, y, z, samples=100):
    T = 0.01
    C = -0.3
    B = 0.4
    x = [2*x[0]-x[1]] + list(x) + [2*x[-1]-x[-2]]
    y = [2*y[0]-y[1]] + list(y) + [2*y[-1]-y[-2]]
    z = [2*z[0]-z[1]] + list(z) + [2*z[-1]-z[-2]]

    p_x, p_y, p_z, v_x, v_y, v_z = [], [], [], [], [], []

    # compute tangents
    dx, dy, dz = [], [], []
    for i in range(1, len(x)-1):
        dx.append(2 * ((1 - T) * ((1 + C) * (1 + B) * (x[i] - x[i-1]) / 2 + (1 - C) * (1 - B) * (x[i+1] - x[i]) / 2)))
        dy.append(2 * ((1 - T) * ((1 + C) * (1 + B) * (y[i] - y[i-1]) / 2 + (1 - C) * (1 - B) * (y[i+1] - y[i]) / 2)))
        dz.append(2 * ((1 - T) * ((1 + C) * (1 + B) * (z[i] - z[i-1]) / 2 + (1 - C) * (1 - B) * (z[i+1] - z[i]) / 2)))

    # interpolate segments
    for i in range(len(x)-3):
        for s in np.linspace(0, 1, samples):
            h1 = 2*s**3 - 3*s**2 + 1
            h2 = -2*s**3 + 3*s**2
            h3 = s**3 - 2*s**2 + s
            h4 = s**3 - s**2

            dh1 = 6*s**2 - 6*s
            dh2 = -6*s**2 + 6*s
            dh3 = 3*s**2 - 4*s + 1
            dh4 = 3*s**2 - 2*s

            px = h1*x[i+1] + h2*x[i+2] + h3*dx[i] + h4*dx[i+1]
            py = h1*y[i+1] + h2*y[i+2] + h3*dy[i] + h4*dy[i+1]
            pz = h1*z[i+1] + h2*z[i+2] + h3*dz[i] + h4*dz[i+1]

            vx = dh1*x[i+1] + dh2*x[i+2] + dh3*dx[i] + dh4*dx[i+1]
            vy = dh1*y[i+1] + dh2*y[i+2] + dh3*dy[i] + dh4*dy[i+1]
            vz = dh1*z[i+1] + dh2*z[i+2] + dh3*dz[i] + dh4*dz[i+1]

            p_x.append(px); p_y.append(py); p_z.append(pz)
            v_x.append(vx); v_y.append(vy); v_z.append(vz)

    return p_x, p_y, p_z, v_x, v_y, v_z



def spline_path(x,y,z):
    # Create a periodic spline
    print([x, y, z])
    tck, u = interpolate.splprep([x, y, z], s=0, per=True)

    # Generate points on the spline
    u_new = np.linspace(0, 1, 100)
    x_new, y_new, z_new = interpolate.splev(u_new, tck)
    vx, vy, vz = interpolate.splev(u_new, tck, der=1)

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    v_max = np.max(v)

    return x_new, y_new, z_new, vx/v_max, vy/v_max, vz/v_max


def main():
    x_new, y_new, z_new, vx, vy, vz = spline_path(x,y,z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_new, y_new, z_new, label='Closed spline')
    ax.plot(x_new[0:len(x_new)//2], y_new[0:len(x_new)//2], z_new[0:len(x_new)//2],'g.', label='Closed spline')
    ax.scatter(x, y, z, c='red', label='Control points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    plt.figure()
    plt.plot(vx, label='V_x')
    plt.plot(vy, label='V_y')
    plt.plot(vz, label='V_z')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()