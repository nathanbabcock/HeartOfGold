from util.rlutilities import *
from rlutilities.simulation import Navigator, Curve
from rlutilities.mechanics import FollowPath
from random import randint
import time, random

def init_path(self):
    print('init path')

    # Init variables
    b = Ball(self.game.ball)
    c = Car(self.game.my_car)
    t = to_vec3(self.training_target_location)

    # Init the action

    # Get ball predictions
    self.ball_predictions = [vec3(b.location)]
    for i in range(60*5):
        b.step(1.0 / 60.0)
        self.ball_predictions.append(vec3(b.location))

    # Aim at a random point on the ball's trajectory
    slice_index = randint(0, len(self.ball_predictions) - 1)
    # self.navigator = Navigator(c)
    # self.navigator.analyze_surroundings(3.0)
    # self.action.path = self.navigator.path_to(self.ball_predictions[slice_index], c.forward(), self.action.arrival_speed)
    # self.action.arrival_time = self.game.my_car.time + 3.0 # slice_index * 1.0 / 60.0
    self.curve = Curve([vec3(c.location), self.ball_predictions[slice_index]])

    print(self.curve.point_at(0))
    print(self.curve.point_at(0.3))
    print(self.curve.point_at(0.6))
    print(self.curve.point_at(1))
    print(self.curve.point_at(100))
    print(self.curve.point_at(1000))