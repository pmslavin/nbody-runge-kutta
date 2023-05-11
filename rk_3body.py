"""
  This file contains a reference implementation of the 'classical' form
  of the fourth-order Runge-Kutta solver. This is applied to solve the
  three-body problem, represented as a system of first-order differential
  equations describing gravitational attraction.

  Note that the implementation is structured for clarity rather than
  performance; vectors and loops are avoided in order to emphasise the
  component steps of Runge-Kutta within each iteration.

  - Paul Slavin
"""
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

points = []

class Body():
    def __init__(self, M, x, y, vx=0.0, vy=0.0):
        self.M = M
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)

    def distance(self, rhs):
        return math.sqrt((rhs.x - self.x)**2 + (rhs.y - self.y)**2)

    def __str__(self):
        return f"M: {self.M:8g} ({self.x:12.9f}, {self.y:12.9f}) v_x: {self.vx:10.8f}, v_y: {self.vy:10.8f}"


def f0(b0, b1, b2):
    dv_x = 0.0
    dv_y = 0.0

    GM_1 = -G*b1.M

    rdiff_x_1 = b0.x - b1.x
    rdiff_y_1 = b0.y - b1.y

    rcubed_1 = b0.distance(b1)**3

    dv_x += GM_1*rdiff_x_1/rcubed_1
    dv_y += GM_1*rdiff_y_1/rcubed_1

    GM_2 = -G*b2.M

    rdiff_x_2 = b0.x - b2.x
    rdiff_y_2 = b0.y - b2.y

    rcubed_2 = b0.distance(b2)**3

    dv_x += GM_2*rdiff_x_2/rcubed_2
    dv_y += GM_2*rdiff_y_2/rcubed_2

    return dv_x, dv_y, b0.vx, b0.vy


def plot_animated(points, footnote=None):
    def update_plot(i, data, sct):
        try:
            sct.set_offsets(data[i])
        except IndexError:
            exit(0)
        return sct

    fig, ax = plt.subplots()
    colours = ("brown", "green", "blue")
    ax.set(xlim=(-2.0, 2.0), ylim=(-1.5, 1.5))  # C&M periodic
    #ax.set(xlim=(-400000, 400000), ylim=(-400000, 400000))  # Earth-moon

    if footnote:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])
        plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=6, bbox={"facecolor":"white", "alpha":0.5, "pad":4})

    init_x = [p[0] for p in points[0]]
    init_y = [p[1] for p in points[0]]
    sct = plt.scatter(init_x, init_y, c=colours)
    ani = animation.FuncAnimation(fig, update_plot, interval=5, fargs=(points, sct))
    #writer = animation.FFMpegWriter(fps=60, bitrate=-1)
    #ani.save("out.mp4", writer=writer)
    #writer = animation.PillowWriter(fps=60, bitrate=-1)
    #ani.save('out.gif', writer=writer)

    plt.show()

def make_footnote_text(*bodies):
    lines = []
    for b in bodies:
        lines.append(str(b))

    return '\n'.join(lines)

"""
  Earth-Moon system
M_e = 5.972e24
M_m = 7.34767309e22
dist_em = 384400
b1 = Body(M_e, 0, 0)
b2 = Body(M_m, 0, dist_em, -31400, 0)
b3 = Body(0, 0, 0.0001)

G = 6.67408313131313e-11
h = 0.0002
t = 0
t_f = 100
"""
"""
  Chencimer & Montgomery (2000) periodic initial conditions
  https://arxiv.org/pdf/math/0011268.pdf

  These create a system with a period of 6.32591398s.
  Note that G == 1 is required for these values.
"""
b1 = Body(1.0, -0.97000436, 0.24208753, 0.4662036850, 0.4323657300)
b2 = Body(1.0, 0, 0, -0.933240737, -0.86473146)
b3 = Body(1.0, 0.97000436, -0.24208753, 0.4662036850, 0.4323657300)

G = 1
h = 0.0002
t = 0
t_f = 20

print("b1: ", b1)
print("b2: ", b2)
print("b3: ", b3)
print("")

step = 0
while t < t_f:
    # Calculate slope k1 at starting point
    k1_1_x, k1_1_y, k1_1_vx, k1_1_vy = f0(b1, b2, b3)
    k1_2_x, k1_2_y, k1_2_vx, k1_2_vy = f0(b2, b1, b3)
    k1_3_x, k1_3_y, k1_3_vx, k1_3_vy = f0(b3, b1, b2)

    # Extend slope k1 to midpoint for position: r + dv/dt*t/2
    b1_k1_x = b1.x + k1_1_vx*h/2
    b2_k1_x = b2.x + k1_2_vx*h/2
    b3_k1_x = b3.x + k1_3_vx*h/2

    b1_k1_y = b1.y + k1_1_vy*h/2
    b2_k1_y = b2.y + k1_2_vy*h/2
    b3_k1_y = b3.y + k1_3_vy*h/2

    # Extend slope k1 to midpoint for velocity: dv/dt + d2v/dt2*h/2
    b1_k1_vx = b1.vx + k1_1_x*h/2
    b2_k1_vx = b2.vx + k1_2_x*h/2
    b3_k1_vx = b3.vx + k1_3_x*h/2

    b1_k1_vy = b1.vy + k1_1_y*h/2
    b2_k1_vy = b2.vy + k1_2_y*h/2
    b3_k1_vy = b3.vy + k1_3_y*h/2

    # Create midpoint func args from slope k1
    b1_t = Body(b1.M, b1_k1_x, b1_k1_y, b1_k1_vx, b1_k1_vy)
    b2_t = Body(b2.M, b2_k1_x, b2_k1_y, b2_k1_vx, b2_k1_vy)
    b3_t = Body(b3.M, b3_k1_x, b3_k1_y, b3_k1_vx, b3_k1_vy)

    # Calculate slope k2 at midpoint
    k2_1_x, k2_1_y, k2_1_vx, k2_1_vy = f0(b1_t, b2_t, b3_t)
    k2_2_x, k2_2_y, k2_2_vx, k2_2_vy = f0(b2_t, b1_t, b3_t)
    k2_3_x, k2_3_y, k2_3_vx, k2_3_vy = f0(b3_t, b1_t, b2_t)

    # Extend slope k2 to midpoint for position: r + dv/dt*t/2
    b1_k2_x = b1.x + k2_1_vx*h/2
    b2_k2_x = b2.x + k2_2_vx*h/2
    b3_k2_x = b3.x + k2_3_vx*h/2

    b1_k2_y = b1.y + k2_1_vy*h/2
    b2_k2_y = b2.y + k2_2_vy*h/2
    b3_k2_y = b3.y + k2_3_vy*h/2

    # Extend slope k2 to midpoint for velocity: dv/dt + d2v/dt2*h/2
    b1_k2_vx = b1.vx + k2_1_x*h/2
    b2_k2_vx = b2.vx + k2_2_x*h/2
    b3_k2_vx = b3.vx + k2_3_x*h/2

    b1_k2_vy = b1.vy + k2_1_y*h/2
    b2_k2_vy = b2.vy + k2_2_y*h/2
    b3_k2_vy = b3.vy + k2_3_y*h/2

    # Create midpoint func args from slope k2
    b1_t = Body(b1.M, b1_k2_x, b1_k2_y, b1_k2_vx, b1_k2_vy)
    b2_t = Body(b2.M, b2_k2_x, b2_k2_y, b2_k2_vx, b2_k2_vy)
    b3_t = Body(b3.M, b3_k2_x, b3_k2_y, b3_k2_vx, b3_k2_vy)

    # Calculate slope k3 at midpoint
    k3_1_x, k3_1_y, k3_1_vx, k3_1_vy = f0(b1_t, b2_t, b3_t)
    k3_2_x, k3_2_y, k3_2_vx, k3_2_vy = f0(b2_t, b1_t, b3_t)
    k3_3_x, k3_3_y, k3_3_vx, k3_3_vy = f0(b3_t, b1_t, b2_t)

    # Extend slope k3 to endpoint for position: r + dv/dt*t/2
    b1_k3_x = b1.x + k3_1_vx*h
    b2_k3_x = b2.x + k3_2_vx*h
    b3_k3_x = b3.x + k3_3_vx*h

    b1_k3_y = b1.y + k3_1_vy*h
    b2_k3_y = b2.y + k3_2_vy*h
    b3_k3_y = b3.y + k3_3_vy*h

    # Extend slope k3 to endpoint for velocity: dv/dt + d2v/dt2*h/2
    b1_k3_vx = b1.vx + k3_1_x*h
    b2_k3_vx = b2.vx + k3_2_x*h
    b3_k3_vx = b3.vx + k3_3_x*h

    b1_k3_vy = b1.vy + k3_1_y*h
    b2_k3_vy = b2.vy + k3_2_y*h
    b3_k3_vy = b3.vy + k3_3_y*h

    # Create endpoint func args from slope k3
    b1_t = Body(b1.M, b1_k3_x, b1_k3_y, b1_k3_vx, b1_k3_vy)
    b2_t = Body(b2.M, b2_k3_x, b2_k3_y, b2_k3_vx, b2_k3_vy)
    b3_t = Body(b3.M, b3_k3_x, b3_k3_y, b3_k3_vx, b3_k3_vy)

    # Calculate slope k4 at endpoint
    k4_1_x, k4_1_y, k4_1_vx, k4_1_vy = f0(b1_t, b2_t, b3_t)
    k4_2_x, k4_2_y, k4_2_vx, k4_2_vy = f0(b2_t, b1_t, b3_t)
    k4_3_x, k4_3_y, k4_3_vx, k4_3_vy = f0(b3_t, b1_t, b2_t)

    # Extend weighted slopes to endpoint for position
    b1_h_x = b1.x + 1/6*(k1_1_vx + 2*(k2_1_vx + k3_1_vx) + k4_1_vx)*h
    b2_h_x = b2.x + 1/6*(k1_2_vx + 2*(k2_2_vx + k3_2_vx) + k4_2_vx)*h
    b3_h_x = b3.x + 1/6*(k1_3_vx + 2*(k2_3_vx + k3_3_vx) + k4_3_vx)*h

    b1_h_y = b1.y + 1/6*(k1_1_vy + 2*(k2_1_vy + k3_1_vy) + k4_1_vy)*h
    b2_h_y = b2.y + 1/6*(k1_2_vy + 2*(k2_2_vy + k3_2_vy) + k4_2_vy)*h
    b3_h_y = b3.y + 1/6*(k1_3_vy + 2*(k2_3_vy + k3_3_vy) + k4_3_vy)*h

    # Extend weighted slopes to endpoint for velocity
    b1_h_vx = b1.vx + 1/6*(k1_1_x + 2*(k2_1_x + k3_1_x) + k4_1_x)*h
    b2_h_vx = b2.vx + 1/6*(k1_2_x + 2*(k2_2_x + k3_2_x) + k4_2_x)*h
    b3_h_vx = b3.vx + 1/6*(k1_3_x + 2*(k2_3_x + k3_3_x) + k4_3_x)*h

    b1_h_vy = b1.vy + 1/6*(k1_1_y + 2*(k2_1_y + k3_1_y) + k4_1_y)*h
    b2_h_vy = b2.vy + 1/6*(k1_2_y + 2*(k2_2_y + k3_2_y) + k4_2_y)*h
    b3_h_vy = b3.vy + 1/6*(k1_3_y + 2*(k2_3_y + k3_3_y) + k4_3_y)*h

    b1.x = b1_h_x
    b1.y = b1_h_y
    b1.vx = b1_h_vx
    b1.vy = b1_h_vy

    b2.x = b2_h_x
    b2.y = b2_h_y
    b2.vx = b2_h_vx
    b2.vy = b2_h_vy

    b3.x = b3_h_x
    b3.y = b3_h_y
    b3.vx = b3_h_vx
    b3.vy = b3_h_vy

    t += h
    step += 1

    #print("b1: ", b1)
    #print("b2: ", b2)
    #print("b3: ", b3)
    #print("")
    if step % 100 == 0:
        points.append([(b1.x, b1.y), (b2.x, b2.y), (b3.x, b3.y)])

footnote = make_footnote_text(b1, b2, b3)
plot_animated(points, footnote)
