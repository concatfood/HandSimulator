import glm
import math
from settings import *


# used for performing predefined articulations (currently not used)
def test(i_hand, f):
    projection_matrix = glm.perspective(glm.radians(fov), res[0] / res[1], near, far)
    position = glm.vec3(0.0, 0.0, 0.5)
    # position = glm.vec3(0.5, 0.25, 0.0)
    direction = glm.vec3(0.0, 0.0, -1.0)
    # direction = glm.vec3(-1.0, -0.25, 0.0)
    up = glm.vec3(0.0, 1.0, 0.0)
    view_matrix = glm.lookAt(position, position + direction, up)
    model_matrix = glm.mat4()
    # # left: glm.vec3(-0.2, 0.0, 0.0)    right: glm.vec3(0.1, 0.0, 0.0)
    # # up: glm.vec3(0.0, 0.13, 0.0)      down: glm.vec3(0.0, -0.1, 0.0)

    if i_hand == 0:
        # rot = glm.rotate(glm.mat4(), math.pi * (current_time - start_time), glm.vec3(1.0, 0.0, 0.0))
        rot = glm.rotate(glm.mat4(), math.pi * f / FPS_ANIMATION, glm.vec3(1.0, 0.0, 0.0))
        trans = glm.translate(glm.mat4(), glm.vec3(-0.08, -0.08 * math.sin(math.sqrt(2) / 2 * 2 * math.pi * f /
                                                                           FPS_ANIMATION), 0.0))
        model_matrix = trans * rot
    elif i_hand == 1:
        # rot = glm.rotate(glm.mat4(), -math.pi * (current_time - start_time), glm.vec3(1.0, 0.0, 0.0))
        rot = glm.rotate(glm.mat4(), -math.pi * f / FPS_ANIMATION, glm.vec3(1.0, 0.0, 0.0))
        trans = glm.translate(glm.mat4(), glm.vec3(0.01, 0.08 * math.sin(math.sqrt(2) / 2 * 2 * math.pi * f /
                                                                         FPS_ANIMATION), 0.0))
        model_matrix = trans * rot

    return projection_matrix, view_matrix, model_matrix
