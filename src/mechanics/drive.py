from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import CarState

from math import pi

from util.vec import Vec3
from util.drive import steer_toward_target
from util.rlutilities import *

from rlutilities.simulation import Car
from rlutilities.linear_algebra import vec3, angle_between

def drive_at(self, car: CarState, target: vec3):
    controls = SimpleControllerState()
    # target_Vec3 = Vec3(target[0], target[1], target[2])

    # # warning: lazy conversions and variable scope
    # if angle_between(target - to_vec3(car.physics.location), self.game.my_car.forward()) > pi / 2:
    #     controls.boost = False
    #     controls.handbrake = True
    # else:
    #     controls.boost = True
    #     controls.handbrake = False

    # # Be smart about not using boost at max speed
    # # if Vec3(car.physics.velocity).length() > self.boost_analysis.frames[-1].speed - 10:
    # #     controls.boost = False

    # controls.steer = steer_toward_target(car, target_Vec3)
    # controls.throttle = 1
    print('WARNING, drive_at is deprecated in favor of Intercept.get_controls()')
    assert False
    return controls