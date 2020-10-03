from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import CarState

from math import pi

from util.vec import Vec3
from util.drive import steer_toward_target
from util.rlutilities import *

from rlutilities.simulation import Car
from rlutilities.linear_algebra import vec3, angle_between

def drive_at(self, car: CarState, target: Vec3):
    controls = SimpleControllerState()

    # warning: lazy conversions and variable scope
    if angle_between(to_vec3(target) - to_vec3(car.physics.location), self.game.my_car.forward()) > pi / 2:
        controls.boost = False
        controls.handbrake = True
    else:
        controls.boost = True
        controls.handbrake = False

    # Be smart about not using boost at max speed
    # if Vec3(car.physics.velocity).length() > self.boost_analysis.frames[-1].speed - 10:
    #     controls.boost = False

    controls.steer = steer_toward_target(car, target)
    controls.throttle = 1
    return controls