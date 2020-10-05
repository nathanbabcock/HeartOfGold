from rlutilities.simulation import Car, Ball
from rlutilities.linear_algebra import *
from analysis.throttle import ThrottleAnalysis
from analysis.boost import BoostAnalysis
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import CarState
from util.drive import steer_toward_target
from util.vec import Vec3
from util.rlutilities import to_vec3
from math import pi

class Intercept():
    def __init__(self, location: vec3, boost = True):
        self.location = location
        self.boost = boost
        self.time = None

    def simulate(self, car: Car, ball: Ball = None) -> vec3:
        return None

    # warning: lazy conversions and variable scope
    def get_controls(self, car_state: CarState, car: Car):
        controls = SimpleControllerState()
        target_Vec3 = Vec3(self.location[0], self.location[1], self.location[2])

        if angle_between(self.location - to_vec3(car_state.physics.location), car.forward()) > pi / 2:
            controls.boost = False
            controls.handbrake = True
        else:
            controls.boost = True
            controls.handbrake = False

        # Be smart about not using boost at max speed
        # if Vec3(car.physics.velocity).length() > self.boost_analysis.frames[-1].speed - 10:
        #     controls.boost = False

        controls.steer = steer_toward_target(car_state, target_Vec3)
        controls.throttle = 1
        return controls

    @staticmethod
    def calculate(car: Car, ball: Ball, ball_predictions = None):
        # Init vars
        c = Car(car)
        b = Ball(ball)

        # Generate predictions of ball path
        if ball_predictions is None:
            ball_predictions = [vec3(b.location)]
            for i in range(60*5):
                b.step(1.0 / 60.0)
                ball_predictions.append(vec3(b.location))

        # Gradually converge on ball location by aiming at a location, checking time to that location,
        # and then aiming at the ball's NEW position. Guaranteed to converge (typically in <10 iterations)
        # unless the ball is moving away from the car faster than the car's max boost speed
        intercept = Intercept(b.location)
        intercept.boost = True
        i = 0
        max_tries = 25
        while i < max_tries:
            analyzer = BoostAnalysis() if intercept.boost else ThrottleAnalysis()
            analysis = analyzer.travel_distance(norm(intercept.location - c.location), norm(c.velocity))
            ball_index = int(round(analysis.time * 60))
            if ball_index >= len(ball_predictions):
                intercept.location = ball_predictions[-1]
                intercept.time = len(ball_predictions) * 60
                break
            ball_location = ball_predictions[ball_index]
            if norm(ball_location - intercept.location) <= b.collision_radius + c.hitbox().half_width[0]:
                if i != 1: print(f'Intercept convergence in {i} iterations')
                break
            intercept.location = ball_location
            intercept.time = ball_index / 60.0
            i += 1

        if i >= max_tries:
            print(f'Warning: max tries ({max_tries}) exceeded for calculating intercept')

        return intercept
