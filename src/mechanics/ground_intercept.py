from rlutilities.simulation import Car, Ball
from rlutilities.linear_algebra import *

def time_to_travel(delta, v_i):

def ground_intercept(self):
    # Init vars
    c = Car(self.game.my_car)
    b = Ball(self.game.ball)

    # Generate predictions of ball path
    self.ball_predictions = [vec3(b.location)]
    for i in range(60*5):
        b.step(1.0 / 60.0)
        self.ball_predictions.append(vec3(b.location))

    # Start with a rough guess
    time = time_to_travel(norm(b - c), norm(c.velocity)
