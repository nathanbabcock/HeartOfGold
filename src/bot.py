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
from rlutilities.mechanics import Aerial
from rlutilities.linear_algebra import *
from util.rlutilities import *

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.ball_predictions = []
        self.not_hit_yet = True
        self.game = None
        self.aerial = None

    def initialize_agent(self):
        print('> Alphabot: I N I T I A L I Z E D')

    def reset_gamestate(self):
        print('> reset_gamestate()')

        # Initialize inputs
        self.initial_ball_location = Vector3(0, 0, 100)
        self.initial_ball_velocity = Vector3(0, 0, 650 * 2)
        self.initial_car_location = Vector3(randint(-3000, 3000), randint(-3000, 3000), 0)
        self.initial_car_velocity = Vector3(0, 0, 0)
        self.training_target_location = Vec3(randint(-3000, 3000), randint(-3000, 3000), randint(0, 3000))
        self.not_hit_yet = True
        self.ball_predictions = []
        self.last_dist = None
        self.last_touch_location = Vec3(0, 0, 0)
        t = to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        b.location = to_vec3(self.initial_ball_location)
        b.velocity = to_vec3(self.initial_ball_velocity)
        c.location = to_vec3(self.initial_car_location)

        # Point car at ball
        c.rotation = look_at(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), vec3(0, 0, 1))
        rotator = rotation_to_euler(c.rotation)

        # Helpful push in the right direction
        # c.velocity = setveclen(c.forward(), 1000)
        # self.initial_car_velocity = to_Vector3(c.velocity)

        # Initialize aerial
        self.aerial = Aerial(c)
        for i in range(1200):
            b.step(1.0 / 120.0)
            if b.location[2] <= 500:
                continue
            self.aerial.target = b.location
            self.aerial.arrival_time = b.time

            # check if we can reach it by an aerial
            simulation = self.aerial.simulate()
            if norm(simulation.location - self.aerial.target) < 100:
                b.step(0.016666)
                b.step(0.016666)
                self.aerial.target = b.location
                self.aerial.arrival_time = b.time
                # self.target_ball = Ball(prediction)
                break

        # Set gamestate
        car_state = CarState(boost_amount=100, 
                     physics=Physics(location=self.initial_car_location, velocity=self.initial_car_velocity, rotation=rotator,
                     angular_velocity=Vector3(0, 0, 0)))
        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=self.initial_ball_velocity, rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
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

        # Check for car hit ball
        if self.last_touch_location != packet.game_ball.latest_touch.hit_location:
            self.last_touch_location = Vec3(packet.game_ball.latest_touch.hit_location)
            print(f'> Car hit ball')
            self.not_hit_yet = False

        # Reset if ball is no longer heading towards target (either from a miss or after a hit)
        cur_dist = ball_location.dist(self.training_target_location)
        if ball_location.z < 500 and ball_velocity.z < 0:
            reset = True
        self.last_dist = cur_dist

        # Prepare simulation of future hit
        t = to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])

        # Rendering
        if len(self.ball_predictions) > 2:
            self.renderer.draw_polyline_3d(self.ball_predictions, self.renderer.red())
        if self.aerial and self.aerial.target:
            self.renderer.draw_rect_3d(self.aerial.target, 8, 8, True, self.renderer.green(), centered=True)
            self.renderer.draw_line_3d(car_location, self.aerial.target, self.renderer.white())
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        # self.renderer.draw_line_3d(car_location, ball_location, self.renderer.white())
        # self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())

        # Controller state
        if reset:
            self.reset_gamestate()
            return SimpleControllerState()
        
        # Just do an aerial :4head:
        if self.aerial is not None:
            print('target', self.aerial.arrival_time)
            self.aerial.step(self.game.time_delta)
            # self.aerial.step(1.0 / 120.0)
            return self.aerial.controls

        return SimpleControllerState()