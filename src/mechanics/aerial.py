from util.rlutilities import *
from random import randint
import time, random

def init_aerial(self):
    print('> init_aerial()')

    # Get values
    t = to_vec3(self.training_target_location)
    b = Ball(self.game.ball)
    c = Car(self.game.cars[self.index])

    # Initialize aerial
    self.aerial = Aerial(self.game.cars[self.index])
    for i in range(60*5):
        b.step(1.0 / 60.0)
        if b.location[2] <= 500:
            continue
        self.aerial.target = b.location
        self.aerial.arrival_time = b.time

        # Simulate an aerial
        dt = 1.0 / 60.0
        car_copy = Car(self.game.cars[self.index])
        ball_copy = Ball(self.game.ball)
        aerial_copy = copy_aerial(self.aerial, car_copy)

        # same as aerial.simulate()
        for i in range(60*5):
            aerial_copy.step(dt)
            car_copy.step(aerial_copy.controls, dt)
            ball_copy.step(dt, car_copy)

            if aerial_copy.finished:
                break

        # Now we've simulated the aerial until completion
        if norm(car_copy.location - self.aerial.target) < 100:
            # self.ball_predictions = [vec3(ball_copy.location)]
            # for i in range(60*5):
            #     car_copy.step(aerial_copy.controls, dt)
            #     ball_copy.step(dt, car_copy)
            #     self.ball_predictions.append(vec3(ball_copy.location))
            break

    self.original_aerial_target = vec3(self.aerial.target)
    self.avg_aerial_error = None
    print('aerial target', self.aerial.target)
    print('aerial TIME', self.aerial.arrival_time)

def simulate_aerial(self):
    # Re-simulate the aerial every frame
    self.closest_dist = None
    self.closest_point = None
    self.closest_vec = None
    t = to_vec3(self.training_target_location)
    b = Ball(self.game.ball)
    c = Car(self.game.cars[self.index])
    dt = 1.0 / 60.0

    # Simulate current aerial
    self.ball_predictions = [vec3(b.location)]
    aerial_copy = copy_aerial(self.aerial, c)
    for i in range(60*5):
        # Simulate
        aerial_step(aerial_copy, c, self.rotation_input, dt)
        c.step(aerial_copy.controls, dt)
        b.step(dt, c)

        # Measure dist from target
        dist = norm(t - b.location)
        if self.closest_dist == None or dist < self.closest_dist:
            self.closest_dist = dist
            self.closest_point = vec3(b.location)
            self.closest_vec = b.location - t

        # Record trajectory
        self.ball_predictions.append(vec3(b.location))

    if self.avg_aerial_error == None:
        self.avg_aerial_error = self.closest_vec
        self.cur_aerial_sims = 1
    else:
        self.avg_aerial_error = self.avg_aerial_error + (self.closest_vec - self.avg_aerial_error) / (self.cur_aerial_sims + 1)
        self.cur_aerial_sims += 1

def simulate_alternate_aerials(self):
    dt = 1.0 / 60.0
    avg_sim_time = 0
    num_sims = 0
    tick_deadline = self.tick_start + (1.0 / 120.0)
    while time.time() + avg_sim_time < tick_deadline:
        # Start timer
        sim_start = time.time()

        # Simulate ALTERNATE aerial
        t = to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        alt_closest_dist = None
        alt_closest_point = None
        alt_closest_vec = None
        alt_ball_predictions = [vec3(b.location)]
        alt_aerial_hit = False
        aerial_copy = copy_aerial(self.aerial, c)

        rotation_input = None
        if aerial_copy.arrival_time - c.time < 0.5:
            rotatobator = Input()
            rotatobator.pitch = random.uniform(-1.0, 1.0)
            rotatobator.yaw = random.uniform(-1.0, 1.0)
            rotatobator.roll = random.uniform(-1.0, 1.0)
            # rotatobator.boost = True
            rotation_input = rotatobator
        # else:
        perturbator = vec3(randint(-int(2 * b.collision_radius), int(2 * b.collision_radius)), randint(-int(2 * b.collision_radius), int(2 * b.collision_radius)), randint(-int(2 * b.collision_radius), int(2 * b.collision_radius)))
        aerial_copy.target = self.original_aerial_target + perturbator
        
        for i in range(60*5):
            # Simulate
            aerial_step(aerial_copy, c, rotation_input, dt)
            c.step(aerial_copy.controls, dt)
            b.step(dt, c)

            # Bail on hitting wall or ground
            if abs(b.location[0]) > 4096 - b.collision_radius:
                break
            if abs(b.location[1]) > 5120 - b.collision_radius:
                break
            if abs(b.location[2]) < b.collision_radius * 1.05:
                break

            # Check if we hit the ball yet
            if norm(b.location - c.location) < 200:
                alt_aerial_hit = True

            # Measure dist from target
            dist = norm(t - b.location)
            if alt_closest_dist == None or dist < alt_closest_dist:
                # alt_closest_dist = dist
                alt_closest_point = vec3(b.location)
                # alt_closest_dist = norm((self.closest_point + alt_closest_point) / 2)
                alt_closest_vec = b.location - t
                # average this delta with the previous average
                # alt_closest_dist = norm(self.avg_aerial_error + (alt_closest_vec - self.avg_aerial_error) / (self.cur_aerial_sims + 1))
                alt_closest_dist = norm((self.avg_aerial_error + alt_closest_vec) / 2)

            # Record trajectory
            alt_ball_predictions.append(vec3(b.location))

        # We found a better aerial!
        if alt_closest_dist < self.closest_dist:
            self.ball_predictions = alt_ball_predictions
            self.aerial.target = aerial_copy.target
            self.rotation_input = rotation_input
            # self.closest_point = alt_closest_point
            self.closest_dist = alt_closest_dist

        # Update tick time estimation
        time_this_sim = time.time() - sim_start
        avg_sim_time = avg_sim_time + (time_this_sim - avg_sim_time) / (num_sims + 1)
        num_sims += 1

    # print('Alternate realities visited during this hundreth of a second:', num_sims)
    # print('avg sim time:', avg_sim_time)
    # print('tick time:', 1.0 / 120.0)
