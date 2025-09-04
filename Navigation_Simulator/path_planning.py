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