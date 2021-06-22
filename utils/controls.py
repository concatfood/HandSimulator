import glfw
import glm
from glm import *

horizontal_angle = pi()
initial_fov = 45.0
fov_current = initial_fov
last_time = None
mouse_speed = 0.0005
position = glm.vec3(1.07374847, -0.24665557,  1.46311718)   # position = glm.vec3(0.0, 0.0, 0.5)
speed = 1.0
vertical_angle = 0.0


# compute matrices needed for the MVP matrix
def compute_matrices_from_inputs(window, res, fov, near, far):
    global fov_current
    global horizontal_angle
    global last_time
    global position
    global vertical_angle

    if last_time is None:
        last_time = glfw.get_time()
        fov_current = fov

    current_time = glfw.get_time()
    delta_time = float(current_time - last_time)

    # mouse input
    x_pos, y_pos = glfw.get_cursor_pos(window)

    glfw.set_cursor_pos(window, res[0] / 2, res[1] / 2)

    horizontal_angle += mouse_speed * float(res[0] / 2 - x_pos)
    vertical_angle += mouse_speed * float(res[1] / 2 - y_pos)

    direction = glm.vec3(cos(vertical_angle) * sin(horizontal_angle),
                         sin(vertical_angle),
                         cos(vertical_angle) * cos(horizontal_angle))
    right = glm.vec3(sin(horizontal_angle - pi() / 2.0), 0, cos(horizontal_angle - pi() / 2.0))
    up = glm.vec3(glm.cross(right, direction))

    # keyboard input
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        position += direction * delta_time * speed

    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        position -= direction * delta_time * speed

    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        position += right * delta_time * speed

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        position -= right * delta_time * speed

    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
        position -= up * delta_time * speed

    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        position += up * delta_time * speed

    if glfw.get_key(window, glfw.KEY_PAGE_UP) == glfw.PRESS:
        fov_current -= 5 * delta_time

    if glfw.get_key(window, glfw.KEY_PAGE_DOWN) == glfw.PRESS:
        fov_current += 5 * delta_time

    # compute matrices
    projection_matrix = glm.perspective(glm.radians(fov_current), res[0] / res[1], near, far)
    view_matrix = glm.lookAt(position, position + direction, up)
    model_matrix = glm.mat4()

    last_time = current_time

    return projection_matrix, view_matrix, model_matrix
