from rlutilities.simulation import Car, Ball
from rlutilities.linear_algebra import *
from analysis.throttle import *
from analysis.boost import *
from analysis.jump import *
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import CarState
from util.drive import steer_toward_target
from util.vec import Vec3
from util.rlutilities import to_vec3, rotation_to_euler, closest_point_on_obb
from math import pi, atan, atan2, degrees

def get_car_front_center(car: Car):
    return car.location + normalize(car.forward()) * car.hitbox().half_width[0] + normalize(car.up()) * car.hitbox().half_width[2]

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
    def calculate_old(car: Car, ball: Ball, target: vec3, ball_predictions = None):
        # Init vars
        fake_car = Car(car)
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
            fake_car.rotation = optimal_rotation
            # print(f'fake_car.location {fake_car.location}')
            # print(f'get_car_front_center(fake_car) {get_car_front_center(fake_car)}')
            fake_car.location += optimal_hit_location - get_car_front_center(fake_car) # try to position the car's front center directly on top of the best hit vector
            euler = rotation_to_euler(optimal_rotation)
            # todo put some super precise trigonometry in here to find the max angle allowed at given height
            if fake_car.location[2] <= fake_car.hitbox().half_width[0]:
                euler.pitch = 0
            fake_car.rotation = euler_to_rotation(vec3(euler.pitch, euler.yaw, euler.roll))
            fake_car.location += optimal_hit_location - get_car_front_center(fake_car) # try to position the car's front center directly on top of the best hit vector

            # Adjust vertical position if it (still) intersects with ground
            if fake_car.location[2] < 17.0:
                fake_car.location[2] = 17.0
            intercept.location = get_car_front_center(fake_car)

            # Calculate jump time needed
            jump_height_time = JumpAnalysis().get_frame_by_height(intercept.location[2]).time # or solve with motion equation

            # car_euler = rotation_to_euler(car.rotation)
            # jump_pitch_time = (euler.pitch - car_euler.pitch) / 5.5 + 0.35 # disregarding angular acceleration
            # jump_yaw_time = (euler.yaw - car_euler.yaw) / 5.5 + 0.35 # disregarding angular acceleration
            # jump_roll_time = (euler.roll - car_euler.roll) / 5.5 + 0.35 # disregarding angular acceleration
            # jump_time = max(jump_height_time, jump_pitch_time, jump_yaw_time, jump_roll_time)
            jump_time = jump_height_time # todo revisit rotation time
            # print('jump_time', jump_time)

            # Calculate distance to drive before jumping (to arrive perfectly on target)
            total_translation = intercept.location - get_car_front_center(car)
            total_translation[2] = 0
            total_distance = norm(total_translation)
            start_index = analyzer.get_index_by_speed(norm(car.velocity))
            start_frame = analyzer.frames[start_index]
            custom_error_func = lambda frame : abs(total_distance - (frame.distance - start_frame.distance) - frame.speed * jump_time)
            drive_analysis = analyzer.get_frame_by_error(custom_error_func, start_index)
            arrival_time = drive_analysis.time - start_frame.time + jump_time
            # print('drive_analysis.time', drive_analysis.time)
            # print('drive_analysis', start_index)

            # arrival_time = analyzer.travel_distance(total_distance, norm(car.velocity)).time

            # drive_analysis = analyzer.travel_distance(norm(intercept.location - c.location), norm(c.velocity))
            ball_index = int(round(arrival_time * 60))
            if ball_index >= len(ball_predictions):
                intercept.location = ball_predictions[-1]
                intercept.time = len(ball_predictions) / 60.0
                break
            ball_location = ball_predictions[ball_index]
            # print(f'Iteration {i} distance {norm(ball_location + vec3(optimal_hit_vector[0], optimal_hit_vector[1], 0) - intercept.location)}')
            if norm(ball_location - intercept_ball_position) <= 1:
                # if norm(intercept_ball_position - get_car_front_center(fake_car)) > 100:
                #     intercept.location = ball_predictions[-1]
                #     intercept.time = len(ball_predictions) / 60.0
                #     return intercept

                intercept.dodge = True #jump_time > 0.2
                intercept.jump_time = car.time + arrival_time - jump_time
                intercept.dodge_preorientation = euler_to_rotation(vec3(euler.pitch, euler.yaw, euler.roll))
                intercept.dodge_delay = jump_time
                intercept.dodge_direction = normalize(vec2(optimal_hit_vector))
                # print(f'intercept_ball_position', intercept_ball_position)
                # print(f'intercept.location', intercept.location)
                # print(f'time until jump {drive_analysis.time}')
                # print(f'time now {car.time}')
                # print(f'distance until jump {drive_analysis.distance}')
                # print(f'total distance to target {total_distance}')
                # print(f'horiz speed @ jump {drive_analysis.speed}')
                # print(f'time intended to be in air {jump_time}')
                # print(f'distance travelled in air {jump_time * drive_analysis.speed}')
                # print(f'distance remaining to target @ jump {total_distance - drive_analysis.distance}')
                # print(f'Intercept convergence in {i} iterations')
                # print(f'desired roll {euler.roll}')
                # print(f'actual roll {rotation_to_euler(c.rotation).roll}')
                break

            intercept_ball_position = vec3(ball_location)
            # intercept.location = vec3(ball_location)
            # intercept.location[2] = 0
            intercept.time = arrival_time
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

    @staticmethod
    def calculate(car: Car, ball: Ball, target: vec3, ball_predictions = None):
        # Init vars
        b = Ball(ball)
        dt = 1.0 / 60.0

        # Generate predictions of ball path
        if ball_predictions is None:
            ball_predictions = []
            for i in range(60*5):
                b.step(dt)
                ball_predictions.append(vec3(b.location))

        # Gradually converge on ball location by aiming at a location, checking time to that location,
        # and then aiming at the ball's NEW position. Guaranteed to converge (typically in <10 iterations)
        # unless the ball is moving away from the car faster than the car's max boost speed
        intercept = Intercept(b.location)
        intercept.purpose = 'ball'
        intercept.boost = True
        intercept_ball_position = vec3(b.location)
        collision_achieved = False
        last_horizontal_error = None
        last_horizontal_offset = None
        i = 0
        max_tries = 101
        analyzer = BoostAnalysis() if intercept.boost else ThrottleAnalysis()
        while i < max_tries:
            i += 1
            fake_car = Car(car)
            direction = normalize(intercept.location - car.location)
            fake_car.rotation = look_at(direction, fake_car.up())
            
            for t in range(60*5):
                # Step car location with throttle/boost analysis data
                # Not super efficient but POITROAE
                frame = analyzer.travel_time(dt, norm(fake_car.velocity))
                # print('in 1 frame I travel', frame.time, frame.distance, frame.speed)
                fake_car.location += direction * frame.distance
                fake_car.velocity = direction * frame.speed
                fake_car.time += dt
                ball_location = ball_predictions[t]

                # Check for collision
                p = closest_point_on_obb(fake_car.hitbox(), ball_location)
                if norm(p - ball_location) <= ball.collision_radius:
                    direction_vector = p - (fake_car.location - normalize(fake_car.forward()) * 13.88) # octane center of mass
                    direction_vector[2] = 0
                    target_direction_vector = target - ball_location
                    target_direction_vector[2] = 0
                    intercept_ball_position = ball_location
                    direction = atan2(direction_vector[1], direction_vector[0])
                    ideal_direction = atan2(target_direction_vector[1], target_direction_vector[0])
                    horizontal_error = direction - ideal_direction

                    # intercept.location = vec3(ball_location)
                    # intercept.time = fake_car.time
                    # return intercept

                    # Now descend the hit direction gradient
                    # Kick off the gradient descent with an arbitrary seed value
                    if last_horizontal_error is None:
                        last_horizontal_error = horizontal_error
                        last_horizontal_offset = 0
                        if horizontal_error > 0:
                            horizontal_offset = 25
                        else:
                            horizontal_offset = 25
                        intercept.location = ball_location - normalize(fake_car.left()) * horizontal_offset
                        break

                    # Recursive case of gradient descent
                    if horizontal_offset == last_horizontal_offset:
                        gradient = 0
                    else:
                        gradient = (horizontal_error - last_horizontal_error) / (horizontal_offset - last_horizontal_offset)

                    if gradient == 0:
                        predicted_horizontal_offset = horizontal_offset
                    else:
                        predicted_horizontal_offset = horizontal_offset - horizontal_error / gradient

                    # Base case (convergence)
                    if abs(gradient) < 0.0005:
                        print(f'convergence in {i} iterations')
                        print(f'gradient = {gradient}')
                        print(f'last_horizontal_offset = {last_horizontal_offset}')
                        print(f'direction = {degrees(direction)}')
                        print(f'ideal direction = {degrees(ideal_direction)}')
                        print(f'target = {target}')
                        print(f'ball_location = {ball_location}')
                        return intercept

                    # Edge case exit: offset maxed out
                    max_horizontal_offset = car.hitbox().half_width[1] + ball.collision_radius
                    if predicted_horizontal_offset > max_horizontal_offset:
                        predicted_horizontal_offset = max_horizontal_offset
                    elif predicted_horizontal_offset < -max_horizontal_offset:
                        predicted_horizontal_offset = - max_horizontal_offset
                    last_horizontal_offset = horizontal_offset
                    last_horizontal_error = horizontal_error
                    horizontal_offset = predicted_horizontal_offset

                    # Return the latest intercept location and continue descending the gradient
                    intercept.location = ball_location - normalize(fake_car.left()) * predicted_horizontal_offset
                    print(f'iteration {i}')
                    print(f'gradient = {gradient}')
                    print(f'horizontal_offset = {horizontal_offset}')
                    print(f'horizontal_error = {degrees(horizontal_error)}')
                    # print(f'ideal direction = {degrees(ideal_direction)}')
                    break


                # Check for arrival
                if norm(fake_car.location - intercept.location) < ball.collision_radius / 2:
                    intercept.location = ball_location
                    break

        if i >= max_tries:
            print(f'Warning: max tries ({max_tries}) exceeded for calculating intercept')
        return intercept