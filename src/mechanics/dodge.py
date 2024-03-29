from rlbot.agents.base_agent import SimpleControllerState
from rlutilities.simulation import Car, Ball
from rlutilities.mechanics import Dodge
from rlutilities.linear_algebra import *
from util.rlutilities import *
from util.drive import steer_toward_target
from mechanics.drive import *

from analysis.boost import *
from analysis.throttle import *

from random import randint, uniform, choice, gauss
from time import time

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def random_dodge(car: Car, seed: Dodge = None, epsilon: float = 1.0) -> Dodge:
    directions = [
        vec2(0.0, 0.0), # double jump (up)
        vec2(car.forward()),
        vec2(car.left()),
        vec2(-1.0 * car.left()), # right
        vec2(-1.0 * car.forward()), # back
        vec2(car.forward() + car.left()), # forward left
        vec2(car.forward() - car.left()), # forward right
        vec2(-1.0 * car.forward() + car.left()), # back left
        vec2(-1.0 * car.forward() - car.left()), # back right
    ]

    trigger_distance_absolute_min = trigger_distance_min = 50
    trigger_distance_absolute_max = trigger_distance_max = 2500
    trigger_distance_absolute_range = trigger_distance_absolute_max - trigger_distance_absolute_min
    if seed is not None:
        trigger_distance_min = int(round(max(seed.trigger_distance - epsilon * trigger_distance_absolute_range / 2, trigger_distance_absolute_min)))
        trigger_distance_max = int(round(min(seed.trigger_distance + epsilon * trigger_distance_absolute_range / 2, trigger_distance_absolute_max)))

    if trigger_distance_max < trigger_distance_min:
        trigger_distance_max = trigger_distance_min

    delay_absolute_min = delay_min = 0.3
    delay_absolute_max = delay_max = 0.6
    delay_absolute_range = delay_absolute_max - delay_absolute_min
    if seed is not None:
        delay_min = max(seed.delay - epsilon * delay_absolute_range / 2, delay_absolute_min)
        delay_max = min(seed.delay + epsilon * delay_absolute_range / 2, delay_absolute_max)

    if seed is not None and epsilon < 0.15:
        direction = seed.direction
    else:
        direction = choice(directions)

    dodge = Dodge(car)
    dodge.duration = 0.15
    # dodge.delay = uniform(delay_min, delay_max)
    dodge.delay = 0.3
    dodge.trigger_distance = randint(trigger_distance_min, trigger_distance_max)
    dodge.direction = direction
    dodge.preorientation = axis_to_rotation(vec3(gauss(0.0, 0.4), gauss(0.0, 0.4), gauss(0.0, 0.4)) + car.forward())
    # dodge.preorientation = dot(axis_to_rotation(vec3(0, 0, 3)), car.rotation)
    # dodge.direction = vec2(uniform(-1.0, 1.0), uniform(-1.0, 1.0))
    return dodge

def get_dodge(self, car: Car, ball: Ball, target: vec3):
    # Record the best result (min dist from target)
    best_dodge = None
    min_error = None
    num_sims = 0
    tick_deadline = self.tick_start + (1.0 / 120.0)
    while time() < tick_deadline:
        if self.best_dodge is not None:
            total_dist = self.best_dodge_start_dist - self.best_dodge.trigger_distance
            cur_dist = norm(self.intercept.location - car.location) - self.best_dodge.trigger_distance
            epsilon = cur_dist / total_dist
            dodge = random_dodge(car, self.best_dodge, epsilon)
        else:
            dodge = random_dodge(car)
        # error = dodge.simulate_hit(car, ball, target)
        error = simulate_dodge(dodge, car, ball, target, self.intercept.location)
        num_sims += 1

        # Check if ball not hit
        if norm(error) > 999999:
            continue

        # Set new best dodge
        if min_error is None or norm(error) < norm(min_error):
            min_error = error
            best_dodge = dodge
            best_dodge.error = error

    # print(f'Tried {num_sims} dodges this tick')
    self.best_dodge_sims += num_sims
    return best_dodge

def simulate_dodge(dodge: Dodge, car: Car, ball: Ball, target: vec3, intercept: vec3):
    # First, advance the car to the trigger distance
    car = Car(car)
    ball = Ball(ball)
    dt = 1.0 / 60.0
    direction = normalize(intercept - car.location)
    translation_dist = norm(intercept - car.location) - dodge.trigger_distance
    translation = direction * translation_dist
    sim_start_state: ThrottleFrame = BoostAnalysis().travel_distance(translation_dist, norm(car.velocity))
    car.velocity = direction * sim_start_state.speed
    car.location += translation
    car.time += sim_start_state.time

    # Advance the ball the same amount
    while ball.time < car.time:
        ball.step(dt)

    # Then run the simulation in C++
    return dodge.simulate_hit(car, ball, target)

##################################################################################################################
############################################## BUILD THE WALL ####################################################
##################################################################################################################

def copy_dodge(dodge: Dodge, car: Car) -> Dodge:
    dodge_copy = Dodge(car)
    dodge_copy.delay = dodge.delay
    dodge_copy.trigger_distance = dodge.trigger_distance
    dodge_copy.duration = dodge.duration
    dodge_copy.direction = dodge.direction
    dodge_copy.preorientation = dodge.preorientation
    return dodge_copy

def simulate_dodge_old(self, dodge: Dodge, target: vec3 = None):
    if target is None:
        target = self.target

    # Sanity checks
    assert target is not None

    # Init vars
    c = Car(self.game.my_car)
    b = Ball(self.game.ball)
    t = vec3(target)
    dt = 1.0 / 60.0
    dodge_copy = copy_dodge(dodge, c)
    hit = False
    min_error = None
    self.ball_predictions = [vec3(b.location)]

    # Simulate dodge controls, car, car+ball collision
    for i in range(60*3):
        dodge_copy.step(dt)
        c.step(dodge_copy.controls, dt)
        b.step(dt, c)

        # Check if we hit the ball yet
        if norm(b.location - c.location) < c.hitbox().half_width[0] + b.collision_radius:
            hit = True
            print("DODGE HIT")

        # Measure dist from target
        error = t - b.location
        if hit and (min_error == None or norm(error) < norm(min_error)):
            min_error = error

        # Record trajectory
        self.ball_predictions.append(vec3(b.location))

    if not hit: return None
    return min_error

def try_init_dodge(self):
    if self.dodge is not None and self.dodge.finished:
        self.dodge = None
    if self.dodge is not None and norm(to_vec3(self.ground_target) - self.game.my_car.location) < 100:
        self.dodge = None
    sim_dodge = None
    dodge_error = None
    if self.dodge is None:
        sim_dodge = random_dodge(self.game.my_car)
        sim_dodge.direction = vec2(self.game.my_car.location - self.game.ball.location)
        dodge_error = simulate_dodge(self, sim_dodge)
    if dodge_error is not None:
        self.dodge = sim_dodge
        self.dodge_started = False
        self.dodge_error = dodge_error

def dodge_controls(self, carState):
    # print('dodge_controls')
    dist = norm(to_vec3(self.ground_target) - self.game.my_car.location)

    # Not close enough to trigger the dodge
    if not self.dodge_started and dist > self.dodge.trigger_distance:
        return drive_at(self, carState, self.ground_target)

    # Close enough
    self.dodge_started = True
    self.dodge.step(self.game.time_delta)
    return self.dodge.controls

def init_dodge_old(self):
    # Get values
    t = to_vec3(self.training_target_location)
    b = Ball(self.game.ball)
    c = Car(self.game.cars[self.index])
    dt = 1.0 / 60.0

    # Try a random dodge
    dodge = random_dodge(c)

    # Simulate and check for contact
    hit = False
    for i in range(60*3):
        dodge.step(dt)
        c.step(dodge.controls, dt)
        b.step(dt)

        # Noclip simulation, but treat both entities as spheres and check for collision
        if norm(b.location - c.location) < c.hitbox().half_width[0] + b.collision_radius:
            hit = True
            break

        # Please hit the ball during the flip, not after
        if dodge.finished and not hit:
            break

    if hit:
        self.dodge = Dodge(self.game.my_car)
        self.dodge.delay = dodge.delay
        self.dodge.duration = dodge.duration
        self.dodge.direction = vec2(dodge.direction)
        self.dodge.preorientation = dodge.preorientation
        print('DODGE HIT!')

def simulate_dodge_old(self):
    # Re-simulate the aerial every frame
    self.closest_dist = None
    self.closest_point = None
    self.closest_vec = None
    t = to_vec3(self.training_target_location)
    b = Ball(self.game.ball)
    c = Car(self.game.cars[self.index])
    dt = 1.0 / 60.0
    hit = False

    # Simulate current aerial
    self.ball_predictions = [vec3(b.location)]
    dodge_copy = copy_dodge(self.dodge, c)
    for i in range(60*5):
        # Simulate
        dodge_copy.step(dt)
        c.step(dodge_copy.controls, dt)
        b.step(dt, c)

        # Check if we hit the ball yet
        if norm(b.location - c.location) < (c.hitbox().half_width[0] + b.collision_radius) * 1.05:
            hit = True

        # Measure dist from target
        dist = norm(t - b.location)
        if hit and (self.closest_dist == None or dist < self.closest_dist):
            self.closest_dist = dist
            self.closest_point = vec3(b.location)
            self.closest_vec = b.location - t

        # Record trajectory
        self.ball_predictions.append(vec3(b.location))

    if not hit:
        self.closest_point = vec3(999999, 999999, 999999)
        self.closest_vec = b.location - self.closest_point
        self.closest_dist = norm(self.closest_vec)

    if self.avg_error == None:
        self.avg_error = self.closest_vec
        self.cur_sims = 1
    else:
        self.avg_error = self.avg_error + (self.closest_vec - self.avg_error) / (self.cur_sims + 1)
        self.cur_sims += 1


def simulate_alternate_dodges_old(self):
    dt = 1.0 / 60.0
    avg_sim_time = 0
    num_sims = 0
    tick_deadline = self.tick_start + (1.0 / 120.0)
    while time() + avg_sim_time < tick_deadline:
        # Start timer
        sim_start = time()

        # Simulate ALTERNATE dodge
        t = to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        alt_closest_dist = None
        alt_closest_point = None
        alt_closest_vec = None
        alt_ball_predictions = [vec3(b.location)]
        alt_dodge_hit = False
        dodge_copy = copy_dodge(self.dodge, c)

        # Adjust params
        # dodge_copy.delay = 0.5
        dodge_copy.preorientation = dot(axis_to_rotation(vec3(uniform(-1.0, 1.0), uniform(-1.0, 1.0), uniform(-1.0, 1.0))), c.rotation)
        # dodge_copy.preorientation = dot(axis_to_rotation(vec3(0, 0, 3)), c.rotation)
        # dodge_copy.direction = vec2(uniform(-1.0, 1.0), uniform(-1.0, 1.0))
        dodge_copy.delay = uniform(0.5, 0.9)
        
        for i in range(60*5):
            # Simulate
            dodge_copy.step(dt)
            c.step(dodge_copy.controls, dt)
            b.step(dt, c)

            # Check if we hit the ball yet
            if norm(b.location - c.location) < (c.hitbox().half_width[0] + b.collision_radius) * 1.05:
                alt_dodge_hit = True

            # Please hit the ball during the flip, not after
            if dodge_copy.finished and not alt_dodge_hit:
                break

            # Measure dist from target
            dist = norm(t - b.location)
            if alt_dodge_hit and (alt_closest_dist == None or dist < alt_closest_dist):
                alt_closest_point = vec3(b.location)
                alt_closest_vec = b.location - t
                alt_closest_dist = norm((self.avg_error + alt_closest_vec) / 2)

            # Record trajectory
            alt_ball_predictions.append(vec3(b.location))

        # We found a better dodge!
        if alt_dodge_hit and alt_closest_dist < self.closest_dist:
            self.ball_predictions = alt_ball_predictions
            self.dodge.direction = dodge_copy.direction
            self.dodge.delay = dodge_copy.delay
            self.dodge.preorientation = dodge_copy.preorientation
            self.dodge.duration = dodge_copy.duration
            self.closest_dist = alt_closest_dist

        # Update tick time estimation
        time_this_sim = time() - sim_start
        avg_sim_time = avg_sim_time + (time_this_sim - avg_sim_time) / (num_sims + 1)
        num_sims += 1

    print('tried', num_sims)

def simulate_alternate_dodges(self):
    assert self.ground_target is not None

    tick_deadline = self.tick_start + (1.0 / 120.0)
    dodges_tried = 0
    while time() < tick_deadline:
        c = Car(self.game.my_car)
        t = to_vec3(self.ground_target)
        dodge = random_dodge(c)
        perturbator = vec3(randint(-200, 200), randint(-200, 200), randint(-200, 200))
        ground_target = t + perturbator
        ground_target = Vec3(ground_target[0], ground_target[1], ground_target[2])

        error = simulate_dodge(self, dodge, ground_target)
        dodges_tried += 1

        if error is not None and (self.dodge_error is None or norm(error) < norm(self.dodge_error)):
            self.dodge = dodge
            self.ground_target = Vec3(ground_target[0], ground_target[1], ground_target[2])
            print('Found a better dodge!')

    print(f'Tried {dodges_tried} dodges')