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


def hermite_path(x, y, z, dx, dy, dz, samples_per_segment=100):
    dx = [9*d for d in dx]
    dy = [9*d for d in dy]
    dz = [9*d for d in dz]

    p_x, p_y, p_z = [], [], []
    v_x, v_y, v_z = [], [], []

    num_segments = len(x) - 1
    total_samples = samples_per_segment * num_segments

    for dt in range(total_samples):
        # Determine which segment
        segment = dt // samples_per_segment
        s = (dt % samples_per_segment) / samples_per_segment

        h1 = 2*s**3 - 3*s**2 + 1
        h2 = -2*s**3 + 3*s**2
        h3 = s**3 - 2*s**2 + s
        h4 = s**3 - s**2

        dh1 = 6*s**2 - 6*s
        dh2 = -6*s**2 + 6*s
        dh3 = 3*s**2 - 4*s + 1
        dh4 = 3*s**2 - 2*s

        p_x.append(h1*x[segment] + h2*x[segment+1] + h3*dx[segment] + h4*dx[segment+1])
        p_y.append(h1*y[segment] + h2*y[segment+1] + h3*dy[segment] + h4*dy[segment+1])
        p_z.append(h1*z[segment] + h2*z[segment+1] + h3*dz[segment] + h4*dz[segment+1])

        v_x.append(dh1*x[segment] + dh2*x[segment+1] + dh3*dx[segment] + dh4*dx[segment+1])
        v_y.append(dh1*y[segment] + dh2*y[segment+1] + dh3*dy[segment] + dh4*dy[segment+1])
        v_z.append(dh1*z[segment] + dh2*z[segment+1] + dh3*dz[segment] + dh4*dz[segment+1])

    return p_x, p_y, p_z, v_x, v_y, v_z

def catmull_rom_path(x, y, z, samples_per_segment=100):
    # Extend endpoints to allow smooth spline
    x = [2*x[0] - x[1]] + x + [2*x[-1] - x[-2]]
    y = [2*y[0] - y[1]] + y + [2*y[-1] - y[-2]]
    z = [2*z[0] - z[1]] + z + [2*z[-1] - z[-2]]

    p_x, p_y, p_z = [], [], []
    v_x, v_y, v_z = [], [], []

    num_segments = len(x) - 3
    total_samples = samples_per_segment * num_segments

    for dt in range(total_samples):
        # Determine segment and local parameter s
        segment = dt // samples_per_segment
        s = (dt % samples_per_segment) / samples_per_segment

        # Shortcut for points
        p0, p1, p2, p3 = x[segment], x[segment+1], x[segment+2], x[segment+3]
        q0, q1, q2, q3 = y[segment], y[segment+1], y[segment+2], y[segment+3]
        r0, r1, r2, r3 = z[segment], z[segment+1], z[segment+2], z[segment+3]

        # Position
        p_x.append(0.5 * (2*p1 + (-p0 + p2)*s + (2*p0 -5*p1 +4*p2 -p3)*s**2 + (-p0 +3*p1 -3*p2 +p3)*s**3))
        p_y.append(0.5 * (2*q1 + (-q0 + q2)*s + (2*q0 -5*q1 +4*q2 -q3)*s**2 + (-q0 +3*q1 -3*q2 +q3)*s**3))
        p_z.append(0.5 * (2*r1 + (-r0 + r2)*s + (2*r0 -5*r1 +4*r2 -r3)*s**2 + (-r0 +3*r1 -3*r2 +r3)*s**3))

        # Velocity
        v_x.append(0.5 * ((-p0 + p2) + 2*(2*p0 -5*p1 +4*p2 -p3)*s + 3*(-p0 +3*p1 -3*p2 +p3)*s**2))
        v_y.append(0.5 * ((-q0 + q2) + 2*(2*q0 -5*q1 +4*q2 -q3)*s + 3*(-q0 +3*q1 -3*q2 +q3)*s**2))
        v_z.append(0.5 * ((-r0 + r2) + 2*(2*r0 -5*r1 +4*r2 -r3)*s + 3*(-r0 +3*r1 -3*r2 +r3)*s**2))

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

import numpy as np

def TCB_path(x, y, z, samples=100):
    T = 0.01
    C = -0.3
    B = 0.4

    # Extend endpoints
    x = [2*x[0]-x[1]] + list(x) + [2*x[-1]-x[-2]]
    y = [2*y[0]-y[1]] + list(y) + [2*y[-1]-y[-2]]
    z = [2*z[0]-z[1]] + list(z) + [2*z[-1]-z[-2]]

    p_x, p_y, p_z = [], [], []
    v_x, v_y, v_z = [], [], []

    # compute tangents
    dx, dy, dz = [], [], []
    for i in range(1, len(x)-1):
        dx.append(2*(1-T)*((1+C)*(1+B)*(x[i]-x[i-1])/2 + (1-C)*(1-B)*(x[i+1]-x[i])/2))
        dy.append(2*(1-T)*((1+C)*(1+B)*(y[i]-y[i-1])/2 + (1-C)*(1-B)*(y[i+1]-y[i])/2))
        dz.append(2*(1-T)*((1+C)*(1+B)*(z[i]-z[i-1])/2 + (1-C)*(1-B)*(z[i+1]-z[i])/2))

    # interpolate segments
    for i in range(len(x)-3):
        # dx/dy/dz indexing: adjust because dx starts from index 1 of x
        ti = i  # corresponds to x[i+1] in position formula
        for s in np.linspace(0,1,samples):
            h1 = 2*s**3 - 3*s**2 + 1
            h2 = -2*s**3 + 3*s**2
            h3 = s**3 - 2*s**2 + s
            h4 = s**3 - s**2

            dh1 = 6*s**2 - 6*s
            dh2 = -6*s**2 + 6*s
            dh3 = 3*s**2 - 4*s + 1
            dh4 = 3*s**2 - 2*s

            px = h1*x[i+1] + h2*x[i+2] + h3*dx[ti] + h4*dx[ti+1]
            py = h1*y[i+1] + h2*y[i+2] + h3*dy[ti] + h4*dy[ti+1]
            pz = h1*z[i+1] + h2*z[i+2] + h3*dz[ti] + h4*dz[ti+1]

            vx = dh1*x[i+1] + dh2*x[i+2] + dh3*dx[ti] + dh4*dx[ti+1]
            vy = dh1*y[i+1] + dh2*y[i+2] + dh3*dy[ti] + dh4*dy[ti+1]
            vz = dh1*z[i+1] + dh2*z[i+2] + dh3*dz[ti] + dh4*dz[ti+1]

            p_x.append(px)
            p_y.append(py)
            p_z.append(pz)
            v_x.append(vx)
            v_y.append(vy)
            v_z.append(vz)

    return p_x, p_y, p_z, v_x, v_y, v_z

# def TCB_path(x, y, z, samples=100):
#     T = 0.01
#     C = -0.3
#     B = 0.4
#     x = [2*x[0]-x[1]] + list(x) + [2*x[-1]-x[-2]]
#     y = [2*y[0]-y[1]] + list(y) + [2*y[-1]-y[-2]]
#     z = [2*z[0]-z[1]] + list(z) + [2*z[-1]-z[-2]]

#     p_x, p_y, p_z, v_x, v_y, v_z = [], [], [], [], [], []

#     # compute tangents
#     dx, dy, dz = [], [], []
#     for i in range(1, len(x)-1):
#         dx.append(2 * ((1 - T) * ((1 + C) * (1 + B) * (x[i] - x[i-1]) / 2 + (1 - C) * (1 - B) * (x[i+1] - x[i]) / 2)))
#         dy.append(2 * ((1 - T) * ((1 + C) * (1 + B) * (y[i] - y[i-1]) / 2 + (1 - C) * (1 - B) * (y[i+1] - y[i]) / 2)))
#         dz.append(2 * ((1 - T) * ((1 + C) * (1 + B) * (z[i] - z[i-1]) / 2 + (1 - C) * (1 - B) * (z[i+1] - z[i]) / 2)))

#     # interpolate segments
#     for i in range(len(x)-3):
#         for s in np.linspace(0, 1, samples):
#             h1 = 2*s**3 - 3*s**2 + 1
#             h2 = -2*s**3 + 3*s**2
#             h3 = s**3 - 2*s**2 + s
#             h4 = s**3 - s**2

#             dh1 = 6*s**2 - 6*s
#             dh2 = -6*s**2 + 6*s
#             dh3 = 3*s**2 - 4*s + 1
#             dh4 = 3*s**2 - 2*s

#             px = h1*x[i+1] + h2*x[i+2] + h3*dx[i] + h4*dx[i+1]
#             py = h1*y[i+1] + h2*y[i+2] + h3*dy[i] + h4*dy[i+1]
#             pz = h1*z[i+1] + h2*z[i+2] + h3*dz[i] + h4*dz[i+1]

#             vx = dh1*x[i+1] + dh2*x[i+2] + dh3*dx[i] + dh4*dx[i+1]
#             vy = dh1*y[i+1] + dh2*y[i+2] + dh3*dy[i] + dh4*dy[i+1]
#             vz = dh1*z[i+1] + dh2*z[i+2] + dh3*dz[i] + dh4*dz[i+1]

#             p_x.append(px); p_y.append(py); p_z.append(pz)
#             v_x.append(vx); v_y.append(vy); v_z.append(vz)

#     return p_x, p_y, p_z, v_x, v_y, v_z

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