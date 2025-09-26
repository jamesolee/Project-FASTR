import numpy as np
from scipy import interpolate
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
        i += 0.1

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
        i = int(dt/10)
        p_x.append(float(h1[dt]) * float(x[i]) + float(h2[dt]) * float(x[i + 1]) + float(h3[dt]) * dx[i] + float(h4[dt]) * float(dx[i + 1]))
        p_y.append(float(h1[dt]) * float(y[i]) + float(h2[dt]) * float(y[i + 1]) + float(h3[dt]) * dy[i] + float(h4[dt]) * float(dy[i + 1]))
        p_z.append(float(h1[dt]) * float(z[i]) + float(h2[dt]) * float(z[i + 1]) + float(h3[dt]) * dz[i] + float(h4[dt]) * float(dz[i + 1]))
        v_x.append(float(dh1[dt]) * float(x[i]) + float(dh2[dt]) * float(x[i + 1]) + float(dh3[dt]) * dx[i] + float(dh4[dt]) * float(dx[i + 1]))
        v_y.append(float(dh1[dt]) * float(y[i]) + float(dh2[dt]) * float(y[i + 1]) + float(dh3[dt]) * dy[i] + float(dh4[dt]) * float(dy[i + 1]))
        v_z.append(float(dh1[dt]) * float(z[i]) + float(dh2[dt]) * float(z[i + 1]) + float(dh3[dt]) * dz[i] + float(dh4[dt]) * float(dz[i + 1]))
        dt += 1

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