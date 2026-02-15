import random

def catmull_rom_spline(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0]) +
        (-p0[0] + p2[0]) * t +
        (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
        (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1]) +
        (-p0[1] + p2[1]) * t +
        (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
        (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
    )
    return x, y

def ease_in_out_quad_random(t):
    accel = 1 + random.uniform(-0.2, 0.2)
    decel = 1 + random.uniform(-0.2, 0.2)
    if t < 0.5:
        return 2 * accel * t * t
    return -1 + (4 - 2 * decel * t) * t

def linear(t): return t
def ease_out_cubic(t): return 1 - pow(1-t, 3)
def ease_in_out_s_curve(t): return t*t*(3 - 2*t)