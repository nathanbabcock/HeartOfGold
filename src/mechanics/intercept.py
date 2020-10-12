from rlutilities.simulation import Car, Ball
from rlutilities.linear_algebra import *
from analysis.throttle import ThrottleAnalysis
from analysis.boost import BoostAnalysis
from analysis.jump import JumpAnalysis
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import CarState
from util.drive import steer_toward_target
from util.vec import Vec3
from util.rlutilities import to_vec3, rotation_to_euler
from math import pi

class Intercept():
    def __init__(self, location: vec3, boost = True):
        self.location = location
        self.boost = boost
        self.time = None
        self.purpose = None # rip
        self.dodge = False

    def simulate(self, bot) -> vec3:
        # print('simulate intercept')

        # Init vars
        c = Car(bot.game.my_car)
        b = Ball(bot.game.ball)
        t = vec3(bot.target)
        intercept = self.location
        dt = 1.0 / 60.0
        hit = False
        min_error = None

        # Drive towards intercept (moving in direction of c.forward())
        c.rotation = look_at(intercept, c.up())
        direction = normalize(intercept - c.location)#c.forward()
        advance_distance = norm(intercept - c.location) - c.hitbox().half_width[0] - b.collision_radius
        translation = direction * advance_distance
        sim_start_state: ThrottleFrame = BoostAnalysis().travel_distance(advance_distance, norm(c.velocity))
        c.velocity = direction * sim_start_state.speed
        c.location += translation
        c.time += sim_start_state.time
        bot.ball_predictions = [vec3(b.location)]

        while b.time < c.time:
            b.step(dt)
            bot.ball_predictions.append(vec3(b.location))

        # print(c.time, b.time)
        # print(c.location, b.location)

        # Simulate the collision and resulting
        for i in range(60*3):
            c.location += c.velocity * dt
            b.step(dt, c)

            # Check if we hit the ball yet
            if norm(b.location - c.location) < (c.hitbox().half_width[0] + b.collision_radius) * 1.05:
                hit = True
                # print('hit')

            # Measure dist from target
            error = t - b.location
            if hit and (min_error == None or norm(error) < norm(min_error)):
                min_error = error

            # Record trajectory
            bot.ball_predictions.append(vec3(b.location))

        if not hit: return None
        return min_error

    # warning: lazy conversions and variable scope
    def get_controls(self, car_state: CarState, car: Car):
        controls = SimpleControllerState()
        target_Vec3 = Vec3(self.location[0], self.location[1], self.location[2])

        if angle_between(self.location - to_vec3(car_state.physics.location), car.forward()) > pi / 2:
            controls.boost = False
            controls.handbrake = True
        elif angle_between(self.location - to_vec3(car_state.physics.location), car.forward()) > pi / 4:
            controls.boost = False
            controls.handbrake = False
        else:
            controls.boost = self.boost
            controls.handbrake = False

        # Be smart about not using boost at max speed
        # if Vec3(car.physics.velocity).length() > self.boost_analysis.frames[-1].speed - 10:
        #     controls.boost = False

        controls.steer = steer_toward_target(car_state, target_Vec3)
        controls.throttle = 1
        return controls

    @staticmethod
    def calculate(car: Car, ball: Ball, target: vec3, ball_predictions = None):
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
        intercept.purpose = 'ball'
        intercept.boost = True
        intercept_ball_position = vec3(b.location)
        i = 0
        max_tries = 100
        analyzer = BoostAnalysis() if intercept.boost else ThrottleAnalysis()
        while i < max_tries:
            # Find optimal spot to hit the ball
            optimal_hit_vector = normalize(target - intercept_ball_position) * b.collision_radius
            optimal_hit_location = intercept_ball_position - optimal_hit_vector

            # Find ideal rotation, unless it intersects with ground
            optimal_rotation = look_at(optimal_hit_vector, vec3(0, 0, 1))#axis_to_rotation(optimal_hit_vector) # this might be wrong
            fake_car = Car(car)
            fake_car.rotation = optimal_rotation
            fake_car_front_center = fake_car.location + normalize(fake_car.forward()) * fake_car.hitbox().half_width[0] + normalize(fake_car.up()) * fake_car.hitbox().half_width[2]
            fake_car_translation_delta = optimal_hit_location - fake_car_front_center # try to position the car's front center directly on top of the best hit vector
            fake_car.location += fake_car_translation_delta
            euler = rotation_to_euler(optimal_rotation)
            # todo put some super precise trigonometry in here to find the max angle allowed at given height
            if fake_car.location[2] <= fake_car.hitbox().half_width[0]:
                euler.pitch = 0
            fake_car.rotation = euler_to_rotation(vec3(euler.pitch, euler.yaw, euler.roll))
            fake_car_front_center = fake_car.location + normalize(fake_car.forward()) * fake_car.hitbox().half_width[0] + normalize(fake_car.up()) * fake_car.hitbox().half_width[2]
            fake_car_translation_delta = optimal_hit_location - fake_car_front_center # try to position the car's front center directly on top of the best hit vector
            fake_car.location += fake_car_translation_delta

            # Adjust vertical position if it (still) intersects with ground
            if fake_car.location[2] < 17.0:
                fake_car.location[2] = 0
            fake_car_front_center = fake_car.location + normalize(fake_car.forward()) * fake_car.hitbox().half_width[0] + normalize(fake_car.up()) * fake_car.hitbox().half_width[2]
            # fake_car_translation_delta = optimal_hit_location - fake_car_front_center # try to position the car's front center directly on top of the best hit vector
            # fake_car.location += fake_car_translation_delta
            intercept.location = fake_car.location

            # Calculate jump time needed
            jump_height_time = JumpAnalysis().get_frame_by_height(fake_car.location[2]).time # or solve with motion equation
            car_euler = rotation_to_euler(c.rotation)
            jump_pitch_time = (euler.pitch - car_euler.pitch) / 5.5 + 0.2 # disregarding angular acceleration
            jump_yaw_time = (euler.yaw - car_euler.yaw) / 5.5 + 0.2 # disregarding angular acceleration
            jump_roll_time = (euler.roll - car_euler.roll) / 5.5 + 0.2 # disregarding angular acceleration
            jump_time = max(jump_height_time, jump_pitch_time, jump_yaw_time, jump_roll_time)

            analysis = analyzer.travel_distance(norm(intercept.location - c.location), norm(c.velocity))
            ball_index = int(round(analysis.time * 60))
            if ball_index >= len(ball_predictions):
                intercept.location = ball_predictions[-1]
                intercept.time = len(ball_predictions) / 60.0
                break
            ball_location = ball_predictions[ball_index]
            # print(f'Iteration {i} distance {norm(ball_location + vec3(optimal_hit_vector[0], optimal_hit_vector[1], 0) - intercept.location)}')
            if norm(ball_location - intercept_ball_position) <= 1:
                intercept.dodge = True
                intercept.jump_time = c.time + (ball_index / 60.0) - jump_time - 0.2
                intercept.dodge_preorientation = euler_to_rotation(vec3(euler.pitch, euler.yaw, euler.roll))
                intercept.dodge_delay = jump_time
                intercept.dodge_direction = vec2(optimal_hit_vector)
                print(f'Intercept convergence in {i} iterations')
                # print(f'desired roll {euler.roll}')
                # print(f'actual roll {rotation_to_euler(c.rotation).roll}')
                break

            intercept_ball_position = vec3(ball_location)
            # intercept.location = vec3(ball_location)
            # intercept.location[2] = 0
            intercept.time = c.time + (ball_index / 60.0)
            i += 1

        if i >= max_tries:
            print(f'Warning: max tries ({max_tries}) exceeded for calculating intercept')

        # Intercept is only meant for ground paths (and walls/cieling are only indirectly supported)
        # collision_radius = c.hitbox().half_width[2] * 2 + b.collision_radius + b.collision_radius * 8
        # on_ground = intercept.location[2] <= collision_radius
        # on_back_wall = abs(intercept.location[1]) >= 5120 - collision_radius
        # on_side_wall = abs(intercept.location[0]) >= 4096 - collision_radius
        # # on_cieling = intercept.location[2] >= 2044 - collision_radius
        # reachable = on_ground # or on_back_wall or on_side_wall # or on_cieling
        # if not reachable:
        #     return None

        return intercept
