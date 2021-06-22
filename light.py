import glm
import math
from random import random


x = None
y = None
z = None


def sample_from_unit_sphere():
    global x
    global y
    global z

    u = random()
    v = random()

    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)

    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    return x, y, z


def get_light_inv_dir(scene):
    if scene == 'forest':
        return glm.vec3(0.5, 0.1, 0.0)
        # return glm.vec3(0.5, 0.0, 0.0)
    elif scene == 'random':
        global x
        global y
        global z

        if x is None or y is None or z is None:
            x, y, z = sample_from_unit_sphere()

        return glm.vec3(x, y, z)


def get_light_pos(scene, pos):
    if scene == 'forest':
        return pos + glm.vec3(0.0, 1.0, 0.0)
    elif scene == 'random':
        global x
        global y
        global z

        if x is None or y is None or z is None:
            x, y, z = sample_from_unit_sphere()

        return pos + glm.vec3(x, y, z)
