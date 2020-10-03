from rlutilities.simulation import Car, Ball
from rlutilities.linear_algebra import *
from analysis.throttle import *
from analysis.boost import *

def ground_intercept(self) -> vec3:
    # Init vars
    c = Car(self.game.my_car)
    b = Ball(self.game.ball)

    # Generate predictions of ball path
    self.ball_predictions = [vec3(b.location)]
    for i in range(60*5):
        b.step(1.0 / 60.0)
        self.ball_predictions.append(vec3(b.location))

    # Gradually converge on ball location by 
    intercept = b.location
    i = 0
    max_tries = 25
    while i < max_tries:
        analysis = self.boost_analysis.travel_distance(norm(intercept - c.location), norm(c.velocity))
        ball_index = int(round(analysis.time * 60))
        if ball_index > len(self.ball_predictions):
            return self.ball_predictions[-1]
        ball_location = self.ball_predictions[ball_index]
        if norm(ball_location - intercept) <= 100:
            if i != 1: print(f'Intercept convergence in {i} iterations')
            return ball_location
        intercept = ball_location
        i += 1

    print(f'Warning: max tries ({max_tries}) exceeded for calculating intercept!')
    return intercept

