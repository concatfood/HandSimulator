import glm
import math
import random


x = None
y = None
z = None


def reset_light():
    global x
    global y
    global z

    x = None
    y = None
    z = None


def sample_from_unit_sphere(radius):
    global x
    global y
    global z

    u = random.random()
    v = random.random()

    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)

    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    return x, y, z


def get_light_inv_dir(scene, radius):
    if scene == 'random':
        global x
        global y
        global z

        if x is None or y is None or z is None:
            x, y, z = sample_from_unit_sphere(radius)

        return glm.vec3(x, y, z)


def get_light_pos(scene, pos, radius):
    if scene == 'random':
        global x
        global y
        global z

        if x is None or y is None or z is None:
            x, y, z = sample_from_unit_sphere(radius)

        return pos + glm.vec3(x, y, z)
