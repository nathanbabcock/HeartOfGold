from math import pi, sqrt, inf, cos, sin, tan, atan2
from rlbot.utils.game_state_util import Vector3, Rotator
from util.vec import Vec3

from rlutilities.simulation import Ball, Car, Field, Game, Input
from rlutilities.mechanics import Aerial, Dodge
from rlutilities.linear_algebra import *

# A bunch of helper functions for working with the rlutilities package

def veclen(vec: vec3) -> float:
    return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def setveclen(vec: vec3, new_len: float):
    return vec * new_len / veclen(vec)

def closest_point_on_obb(obb, vec):
    # get the local coordinates of b
    vec_local = dot(vec - obb.center, obb.orientation)

    # clip those coordinates to find the closest point (in local coordinates)
    closest_local = vec3(min(max(vec_local[0], -obb.half_width[0]), obb.half_width[0]), min(max(vec_local[1], -obb.half_width[1]), obb.half_width[1]), min(max(vec_local[2], -obb.half_width[2]), obb.half_width[2]))

    # transform back to world coordinates
    return dot(obb.orientation, closest_local) + obb.center
  
def angle_to_mat3(theta):
    return mat3(cos(theta), -sin(theta), 0,  sin(theta), cos(theta), 0, 0, 0, 0)

def to_vec3(_Vec3: Vector3) -> vec3:
    return vec3(_Vec3.x, _Vec3.y, _Vec3.z)

def to_Vector3(_vec3: vec3):
    return Vector3(_vec3[0], _vec3[1], _vec3[2])

def to_Vec3(_vec3: vec3):
    return Vec3(_vec3[0], _vec3[1], _vec3[2])

def to_Vec3(_Vector3: Vector3):
    return Vec3(_Vector3.x, _Vector3.y, _Vector3.z)

def rotation_to_euler(theta: mat3) -> Rotator:
  return Rotator(
    atan2(theta[2, 0], norm(vec2(theta[0, 0], theta[1, 0]))),
    atan2(theta[1, 0], theta[0, 0]),
    atan2(-theta[2, 1], theta[2, 2])
  )

def project(u: vec3, v: vec3) -> vec3:
    return (dot(u, v) / norm(v)**2) * v

def get_closest_point_on_trajectory(b: Ball, target: vec3):
    closest = vec3(target)
    while True:
        b.step(1.0 / 120.0)
        if abs(veclen(target - b.location)) > abs(veclen(target - closest)):
            return closest
        closest = vec3(b.location)

def copy_aerial(aerial: Aerial, car: Car):
    aerial_copy = Aerial(car)
    aerial_copy.target = vec3(aerial.target)
    aerial_copy.arrival_time = aerial.arrival_time
    aerial_copy.target_orientation = aerial.target_orientation
    aerial_copy.up = aerial.up
    aerial_copy.angle_threshold = aerial.angle_threshold
    aerial_copy.reorient_distance = aerial.reorient_distance
    aerial_copy.throttle_distance = aerial.throttle_distance
    return aerial_copy

def aerial_step(aerial: Aerial, car: Car, rotation_input = None, dt = 1.0 / 60.0):
    return_value = aerial.step(dt)
    if rotation_input is not None and aerial.arrival_time - car.time <= 0.5:
        aerial.controls.pitch = rotation_input.pitch
        aerial.controls.yaw = rotation_input.yaw
        aerial.controls.roll = rotation_input.roll
        # aerial.controls.boost = rotation_input.boost
    return return_value