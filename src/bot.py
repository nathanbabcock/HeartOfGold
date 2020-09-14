from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

from math import pi, sqrt, inf, cos, sin, tan, atan2
from random import randint

from rlutilities.simulation import Ball, Car, Field, Game, Input
from rlutilities.linear_algebra import vec3, mat3, dot, angle_between, rotation, rotation_to_axis, axis_to_rotation, euler_to_rotation, norm, vec2, look_at, cross

class HitData:
    def __init__(self, car_direction_before=None, ball_direction_after=None, car_speed_before=None, ball_speed_before=None):
        self.car_direction_before = car_direction_before
        self.ball_direction_after = ball_direction_after

        # not used yet
        self.car_speed_before = car_speed_before
        self.ball_speed_before = ball_speed_before

def veclen(vec: vec3) -> float:
    return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def setveclen(vec: vec3, new_len: float):
    return vec * new_len / veclen(vec)

def closest_point_on_obb(obb, vec):
    # get the local coordinates of b
    vec_local = dot(vec - obb.center, obb.orientation)

    # clip those coordinates to find the closest point (in local coordinates)
    closest_local = vec3(min(max(vec_local[0], -obb.half_width[0]), obb.half_width[0]), min(max(vec_local[1], -obb.half_width[1]), obb.half_width[1]), min(max(vec_local[2], -obb.half_width[2]), obb.half_width[2]))

    # transform back to world coordinates
    return dot(obb.orientation, closest_local) + obb.center
  
def angle_to_mat3(theta):
    return mat3(cos(theta), -sin(theta), 0,  sin(theta), cos(theta), 0, 0, 0, 0)

def Vec3_to_vec3(_Vec3: Vec3) -> vec3:
    return vec3(_Vec3.x, _Vec3.y, _Vec3.z)

def rotation_to_euler(theta: mat3) -> Rotator:
  return Rotator(
    atan2(theta[2, 0], norm(vec2(theta[0, 0], theta[1, 0]))),
    atan2(theta[1, 0], theta[0, 0]),
    atan2(-theta[2, 1], theta[2, 2])
  )

def project(u: vec3, v: vec3) -> vec3:
    return (dot(u, v) / norm(v)**2) * v

def get_closest_point_on_trajectory(b: Ball, target: vec3):
    closest = vec3(target)
    while True:
        b.step(1.0 / 120.0)
        if abs(veclen(target - b.location)) > abs(veclen(target - closest)):
            return closest
        closest = vec3(b.location)

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.iteration = 0
        self.last_touch_location = Vec3(0, 0, 0)
        self.hit_data = []
        self.pre_hit = HitData()
        self.ball_predictions = []
        self.not_hit_yet = True
        self.game = None

    def initialize_agent(self):
        print('> Alphabot: I N I T I A L I Z E D')

    def reset_gamestate(self):
        print('> reset_gamestate()')

        # Initialize inputs
        self.initial_ball_location = Vector3(randint(-3000, 3000), randint(-3000, 3000), 100)
        self.training_target_location = Vec3(randint(-3000, 3000), randint(-3000, 3000), 0)
        self.initial_car_location = Vector3(0, 0, 0) # gonna calculate...
        self.not_hit_yet = True
        self.ball_predictions = []
        self.last_dist = None
        self.ball_impact_target = Vec3(self.initial_ball_location)

        # vector from target to ball
        t = Vec3_to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        b.velocity = vec3(0,0,0)
        b.location = Vec3_to_vec3(self.initial_ball_location)
        c = Car(self.game.cars[self.index])

        # Set up car in line with ball
        translation_vector = vec3(b.location[0] - t[0], b.location[1] - t[1], 0)
        translation_vector = setveclen(translation_vector, 2000)
        c.location = vec3(b.location[0], b.location[1], 0) + translation_vector
        horizontal_axis = cross(t - b.location, vec3(0,0,1))
        c.location += setveclen(horizontal_axis, randint(-500, 500))
        self.initial_car_location = Vector3(c.location[0], c.location[1], c.location[2])

        # Initial velocity
        c.velocity = setveclen(translation_vector, -1410)

        # Point car at ball
        c.rotation = look_at(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), vec3(0, 0, 1))
        rotator = rotation_to_euler(c.rotation)

        # Set gamestate
        car_state = CarState(boost_amount=100, 
                     physics=Physics(location=self.initial_car_location, velocity=Vector3(c.velocity[0], c.velocity[1], c.velocity[2]), rotation=rotator,
                     angular_velocity=Vector3(0, 0, 0)))
        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=Vector3(0, 0, 0), rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        car_direction = car_velocity.ang_to(Vec3(1, 0, 0)) if car_velocity.length() > 0 else 0
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        ball_direction = ball_velocity.ang_to(Vec3(1, 0, 0)) if ball_velocity.length() > 0 else 0
        self.car_rotation = my_car.physics.rotation
        reset = False

        # Initialize simulation game model
        if self.game == None:
            Game.set_mode('soccar')
            self.game = Game(self.index, self.team)
            self.game.read_game_information(packet, self.get_rigid_body_tick(), self.get_field_info())
            self.reset_gamestate()
            return SimpleControllerState()

        # Update simulation
        self.game.read_game_information(packet, self.get_rigid_body_tick(), self.get_field_info())

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Check for car hit ball
        if self.last_touch_location != packet.game_ball.latest_touch.hit_location:
            self.pre_hit.ball_direction_after = ball_direction
            self.hit_data.append(self.pre_hit)
            self.iteration += 1
            self.last_touch_location = Vec3(packet.game_ball.latest_touch.hit_location)
            print(f'> Car hit ball')
            self.pre_hit = HitData()
            self.not_hit_yet = False
        else:
            self.pre_hit.car_direction_before = car_direction

        # Reset if ball hits target
        cur_dist = ball_location.dist(self.training_target_location)
        if self.last_dist != None and cur_dist > self.last_dist:
            reset = True
            print('> Ball hit target')
        self.last_dist = cur_dist

        # Prepare simulation of future hit
        t = Vec3_to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        impact_target = vec3(self.ball_impact_target.x, self.ball_impact_target.y, self.ball_impact_target.z)

        # Move car to touch the ball
        translation_vector = vec3(impact_target[0] - c.location[0], impact_target[1] - c.location[1], 0)
        translation_vector = setveclen(translation_vector, veclen(translation_vector) - (c.hitbox().half_width[0] + b.collision_radius))
        c.location += translation_vector

        # Rotate car and car velocity to point at target
        # c.rotation = look_at(vec3(impact_target[0] - c.location[0], impact_target[1] - c.location[1], 0), vec3(0, 0, 1))
        # c.velocity = setveclen(impact_target - c.location, veclen(c.velocity))

        # Collide with ball
        c.location += c.velocity * (1.0 / 120.0)
        b.step(1.0 / 120.0, c)

        # Record predicted path
        self.ball_predictions = []
        first_bounce = None
        for i in range(360):
            b.step(1.0 / 120.0, c)
            self.ball_predictions.append(vec3(b.location))
            if b.velocity[2] < 0 and first_bounce == None and b.location[2] <= 100:
                first_bounce = vec3(b.location)

        # Measure error
        ball_start = vec3(self.initial_ball_location.x, self.initial_ball_location.y, self.initial_ball_location.z)
        if first_bounce == None:
            first_bounce = ball_start
        horizontal_axis = cross(t - ball_start, vec3(0,0,1))
        distance_error = veclen(t - ball_start) - veclen(project(first_bounce - ball_start, t - ball_start))
        horizontal_error_vector = project(first_bounce - ball_start, horizontal_axis)
        horizontal_error = veclen(horizontal_error_vector)
        horizontal_error_angle = angle_between(horizontal_error_vector, horizontal_axis)
        if not (abs(horizontal_error_angle) < 0.01):
            horizontal_error *= -1

        # Adjust ball target to minimize horizontal error
        ball_loc = vec3(ball_location.x, ball_location.y, ball_location.z)
        h_step = 1
        if horizontal_error > -50 and veclen(impact_target + setveclen(horizontal_axis, h_step) - ball_loc) < 100:
            impact_target += setveclen(horizontal_axis, h_step)
        elif horizontal_error < 50 and veclen(impact_target - setveclen(horizontal_axis, h_step) - ball_loc) < 100:
            impact_target -= setveclen(horizontal_axis, h_step)

        # ball_loc = vec3(ball_location.x, ball_location.y, ball_location.z)
        # if impact_target - ball_loc
        self.ball_impact_target = Vec3(impact_target[0], impact_target[1], impact_target[2])

        # Rendering
        if len(self.ball_predictions) > 2:
            self.renderer.draw_polyline_3d(self.ball_predictions, self.renderer.red())
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        # self.renderer.draw_rect_3d(self.ball_impact_target, 8, 8, True, self.renderer.green(), centered=True)
        self.renderer.draw_line_3d(car_location, self.ball_impact_target, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())
        self.renderer.draw_string_2d(20, 20, 2, 2, f'Iteration: {self.iteration}', self.renderer.black())

        # Controller state
        controls = SimpleControllerState()
        if reset:
            self.reset_gamestate()
            return controls

        controls.steer = steer_toward_target(my_car, self.ball_impact_target)
        controls.throttle = 1.0
        if car_velocity.length() > 1411 and distance_error > 0:
            controls.boost = True
        elif car_velocity.length() < 1410 and distance_error < 0:
            controls.throttle = 0.0
            
        # controls.boost = True
        return controls

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        # self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
