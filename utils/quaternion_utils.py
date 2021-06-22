import glm
from glm import *


def rotation_between_vectors(start, dest):
    start = normalize(start)
    dest = normalize(dest)

    cos_theta = dot(start, dest)

    if cos_theta < -1 + 0.001:
        rotation_axis = cross(vec3(0.0, 0.0, 1.0), start)

        if length2(rotation_axis) < 0.01:
            rotation_axis = cross(vec3(1.0, 0.0, 0.0), start)

        rotation_axis = normalize(rotation_axis)
        return angleAxis(glm.radians(180.0), rotation_axis)

    rotation_axis = cross(start, dest)

    s = sqrt((1 + cos_theta) * 2)
    invs = 1 / s

    return quat(s * 0.5, rotation_axis.x * invs, rotation_axis.y * invs, rotation_axis.z * invs)


def look_at(direction, desired_up):
    if length2(direction) < 0.0001:
        return quat()
    
    right = cross(direction, desired_up)
    desired_up = cross(right, direction)
    
    rot1 = rotation_between_vectors(vec3(0.0, 0.0, 1.0), direction)
    new_up = rot1 * vec3(0.0, 1.0, 0.0)
    rot2 = rotation_between_vectors(new_up, desired_up)
    
    return rot2 * rot1


def rotate_towards(q1, q2, max_angle):
    if max_angle < 0.001:
        return q1
    
    cos_theta = dot(q1, q2)
    
    if cos_theta > 0.9999:
        return q2
    
    if cos_theta < 0:
        q1 = q1 * -1.0
        cos_theta *= -1.0
    
    angle = acos(cos_theta)
    
    if angle < max_angle:
        return q2
    
    t = max_angle / angle
    angle = max_angle
    
    res = (sin((1.0 - t) * angle) * q1 + sin(t * angle) * q2) / sin(angle)
    res = normalize(res)
    return res
