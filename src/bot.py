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
from rlutilities.linear_algebra import vec3, mat3, dot, angle_between, rotation, rotation_to_axis, axis_to_rotation, euler_to_rotation, norm, vec2, look_at

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

        # vector from target to ball
        t = Vec3_to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        b.velocity = vec3(0,0,0)
        b.location = Vec3_to_vec3(self.initial_ball_location)
        c = Car(self.game.cars[self.index])

        # Line car up touching the ball
        translation_vector = vec3(b.location[0] - t[0], b.location[1] - t[1], 0)
        translation_vector = setveclen(translation_vector, b.collision_radius + c.hitbox().half_width[0])
        c.location = vec3(b.location[0], b.location[1], 0) + translation_vector
        start_location = vec3(c.location)
        real_location = vec3(b.location[0], b.location[1], 0) + translation_vector * 2
        self.initial_car_location = Vector3(real_location[0], real_location[1], real_location[2])

        # Point car at ball
        c.rotation = look_at(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), vec3(0, 0, 1))
        rotator = rotation_to_euler(c.rotation)

        last_error = 100 # arbitrary
        car_speed = 1410
        trials = 0
        while abs(last_error) > 10:
            # Check if we are stuck
            trials += 1
            if trials > 1000:
                print('Warning; trials exceeded!')
                break

            # Reset ball and car position/velocity
            b.location = Vec3_to_vec3(self.initial_ball_location)
            b.velocity = vec3(0,0,0)
            c.location = vec3(start_location)

            # Adjust in the direction of the error
            if last_error > 0:
                car_speed += 10
            elif last_error < 0:
                car_speed -= 10

            # Choose velocities...
            c.velocity = setveclen(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), car_speed) # Full speed, no boost, towards ball

            # Collide with ball
            c.location += c.velocity * (1.0 / 120.0)
            b.step(1.0 / 120.0, c)

            # Calculate target v_z to hit ball
            delta_x = veclen(t - b.location)
            g = -650.0
            # v_xy = dot(b.velocity, t - Vec3_to_vec3(self.initial_ball_location))
            v_xy = veclen(project(b.velocity, Vec3_to_vec3(self.initial_ball_location)))
            v_z = (delta_x * g) / (-2.0 * v_xy)
            last_error = v_z - b.velocity[2]
            # print('v_xy', v_xy)
            # print('v_z', v_z)
            # print('current initial v_z:', b.velocity[2])
            # print('target initial v_z:', v_z)
        self.initial_car_velocity = car_speed
        print('Found optimal car speed =', car_speed)
        print('Error =', last_error)

        # Record predicted path
        self.ball_predictions = []
        for i in range(360):
            b.step(1.0 / 120.0, c)
            # print('b.velocity.z', b.velocity[2])
            self.ball_predictions.append(vec3(b.location))

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

        # Reset if car passes ball
        # if car_location.y > ball_location.y:
        #     reset = True
        #     self.cur_tries = 999
        #     print('> Scrapping bad data')

        # Reset if ball passes target
        if abs(ball_location.x - self.training_target_location.x) < 40 and abs(ball_location.y - self.training_target_location.y) < 40:
            reset = True
            print('> Ball hit target')

        # Rendering
        # self.renderer.begin_rendering()
        if len(self.ball_predictions) > 2:
            self.renderer.draw_polyline_3d(self.ball_predictions, self.renderer.red())
        # self.renderer.draw_polyline_3d(car_predictions, self.renderer.red())
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        self.renderer.draw_line_3d(car_location, ball_location, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())
        self.renderer.draw_string_2d(20, 20, 2, 2, f'Iteration: {self.iteration}', self.renderer.black())
        # self.renderer.end_rendering()

        # Controller state
        controls = SimpleControllerState()
        if reset:
            self.reset_gamestate()
            return controls
        controls.steer = steer_toward_target(my_car, ball_location)
        controls.throttle = 1.0
        if car_velocity.length() > 1410 and car_velocity.length() < self.initial_car_velocity:
            controls.boost = True
        elif car_velocity.length() < 1410 and car_velocity.length() > self.initial_car_velocity:
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
