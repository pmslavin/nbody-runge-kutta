"""
  N-body simulator using a fourth-order Runge-Kutta scheme
"""
import atexit
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rkfuncs

bodies = []
points = []


class Body():
    def __init__(self, M, x, y, vx=0.0, vy=0.0):
        self.M = float(M)
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)

    def __str__(self):
        return f"M: {self.M:8g} ({self.x:12.9f}, {self.y:12.9f}) v_x: {self.vx:10.8f}, v_y: {self.vy:10.8f}"


@atexit.register
def show_cursor():
    # meh
    print('\033[?25h', end="")


def f1(bodies):
    dv_x, dv_y, vx, vy = [], [], [], []
    for i0, lhs in enumerate(bodies):
        dv_x.append(0.0)
        dv_y.append(0.0)
        vx.append(lhs.vx)
        vy.append(lhs.vy)
        for rhs in [b for i1, b in enumerate(bodies) if i0 != i1]:
            GM = -G*rhs.M
            rdiff_x = lhs.x - rhs.x
            rdiff_y = lhs.y - rhs.y

            rcubed = ((rhs.x - lhs.x)**2 + (rhs.y - lhs.y)**2)**(3/2)

            dv_x[i0] += GM*rdiff_x/rcubed
            dv_y[i0] += GM*rdiff_y/rcubed

    return dv_x, dv_y, vx, vy

f1 = rkfuncs.gravity_first_order


def make_increment_args(bodies, dv_x, dv_y, vx, vy, h):
    args = []
    for idx, b in enumerate(bodies):
        # Extend slope for position: r + dv/dt*h
        b1_x = b.x + vx[idx]*h
        b1_y = b.y + vy[idx]*h

        # Extend slope for velocity: dv/dt + d2v/dt2*h
        b1_vx = b.vx + dv_x[idx]*h
        b1_vy = b.vy + dv_y[idx]*h

        # Create func args at increment
        args.append(Body(b.M, b1_x, b1_y, b1_vx, b1_vy))

    return args


def plot_animated(points, ax_scale, footnote, write_mp4=False):
    import itertools
    import matplotlib.colors as mcolors
    def update_plot(i, data, sct):
        try:
            sct.set_offsets(data[i])
        except IndexError:
            exit(0)
        return sct

    def make_colours(count):
        colours = mcolors.XKCD_COLORS
        cyc = []
        idx = 0
        for c in itertools.cycle(colours):
            cyc.append(c)
            idx += 1
            if idx == count:
                break
        return cyc

    fig, ax = plt.subplots()
    ax.set(xlim=ax_scale[0], ylim=ax_scale[1])

    if footnote:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])
        plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=6, bbox={"facecolor": "white", "alpha": 0.5, "pad": 4})

    init_x = [p[0] for p in points[0]]
    init_y = [p[1] for p in points[0]]
    colours = make_colours(len(points[0]))
    sct = plt.scatter(init_x, init_y, c=colours)

    fps = 60
    frame_args = {"cache_frame_data": False}
    if write_mp4:
        frame_args["frames"] = len(points)

    ani = animation.FuncAnimation(fig, update_plot, interval=1000/fps, fargs=(points, sct), **frame_args)

    if write_mp4:
        writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
        ani.save("out.mp4", writer=writer)

    plt.show()


def make_footnote_text(bodies):
    lines = []
    for b in bodies:
        lines.append(str(b))

    return '\n'.join(lines)

"""
  Earth-Moon system
M_e = 5.972e24        # Earth Kg
M_m = 7.34767309e22   # Moon Kg
dist_em = 384400      # Mean Earth-Moon Km
bodies.append(Body(M_e, 0, 0))
bodies.append(Body(M_m, 0, dist_em, -31410, 0))

# Asteroid chaos...
for i in range(50):
    bodies.append(Body(random.random()*1e8, random.gauss(-2.5e5, 2.5e5), random.gauss(-2.5e5, 2.5e5),
                       random.uniform(-1,1)*1e5, random.uniform(-1,1)*1e5))

G = 6.67408313131313e-11   # N.m^2/Kg^2
h = 0.002
t = 0
t_f = 60
ax_scale = ((-4e5, 4e5), (-4e5, 4e5))
"""
"""
  Chencimer & Montgomery (2000) periodic initial conditions
  https://arxiv.org/pdf/math/0011268.pdf

  These create a system with a period of 6.32591398s.
  Note that G == 1 is required for these values.
"""
bodies.append(Body(1.0, -0.97000436, 0.24208753, 0.4662036850, 0.4323657300))
bodies.append(Body(1.0, 0, 0, -0.933240737, -0.86473146))
bodies.append(Body(1.0, 0.97000436, -0.24208753, 0.4662036850, 0.4323657300))

G = 1
h = 0.002
t = 0
t_f = 6.32591398*6
ax_scale = ((-2.0, 2.0), (-1.5, 1.5))

print('\033[?25l', end="")
step = 0
t0 = time.time()
while t < t_f:
    k1_x, k1_y, k1_vx, k1_vy = f1(bodies)
    increments = make_increment_args(bodies, k1_x, k1_y, k1_vx, k1_vy, h/2)

    k2_x, k2_y, k2_vx, k2_vy = f1(increments)
    increments = make_increment_args(bodies, k2_x, k2_y, k2_vx, k2_vy, h/2)

    k3_x, k3_y, k3_vx, k3_vy = f1(increments)
    increments = make_increment_args(bodies, k3_x, k3_y, k3_vx, k3_vy, h)

    k4_x, k4_y, k4_vx, k4_vy = f1(increments)

    for idx, b in enumerate(bodies):
        b_h_x = b.x + 1/6*(k1_vx[idx] + 2*(k2_vx[idx] + k3_vx[idx]) + k4_vx[idx])*h
        b_h_y = b.y + 1/6*(k1_vy[idx] + 2*(k2_vy[idx] + k3_vy[idx]) + k4_vy[idx])*h
        b_h_vx = b.vx + 1/6*(k1_x[idx] + 2*(k2_x[idx] + k3_x[idx]) + k4_x[idx])*h
        b_h_vy = b.vy + 1/6*(k1_y[idx] + 2*(k2_y[idx] + k3_y[idx]) + k4_y[idx])*h

        b.x = b_h_x
        b.y = b_h_y
        b.vx = b_h_vx
        b.vy = b_h_vy

    t += h
    step += 1

    if step % 10 == 0:
        t1 = time.time()
        t_h, t_h_s = divmod(t1-t0, 3600)
        t_m, t_s = divmod(t_h_s, 60)
        steps = int(t_f/h)
        if step > steps-10:
            # Round up last print
            step = steps
        sw = int(math.log10(steps)+1)
        print(f"\rstep: {step:>{sw}}/{steps:>{sw}}  {100*step/steps:4.2f}%  "
                f"{t_h:02.0f}:{t_m:02.0f}:{t_s:02.0f}  ", end='')
        points.append([(b.x, b.y) for b in bodies])
print(f"{steps/(time.time()-t0):.2f} steps/s")
footnote = make_footnote_text(bodies) if len(bodies)<=4 else None
plot_animated(points, ax_scale, footnote, write_mp4=False)
