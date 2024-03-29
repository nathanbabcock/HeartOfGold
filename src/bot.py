from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from math import pi, sqrt, inf, cos, sin, tan, atan2
from random import randint
import time
import random
# random.seed(42)

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from util.rlutilities import *

from rlutilities.simulation import Ball, Car, Field, Game, Input
from rlutilities.mechanics import Aerial
from rlutilities.linear_algebra import *

from mechanics.drive import *
from mechanics.intercept import *
from mechanics.dodge import *
from mechanics.aerial import *
# from mechanics.path import *

from analysis.throttle import *

class HeartOfGold(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.ball_predictions = []
        self.not_hit_yet = True
        self.game = None
        self.aerial = None
        self.timer = 0.0
        self.action = None

        self.intercept = None
        self.target = None

        self.dodge = None
        self.dodge_started = False
        # self.dodge_error = None
        self.best_dodge = None
        self.best_dodge_sims = 0
        self.best_dodge_start_dist = 0

        self.time_estimate = None
        self.speed_estimate = None

    def initialize_agent(self):
        print('> Alphabot: I N I T I A L I Z E D')

    def reset_for_aerial(self):
        self.initial_ball_location = Vector3(0, 2000, 100)
        self.initial_ball_velocity = Vector3(randint(-250, 250), randint(-250, 250), 650 * 2)
        self.initial_car_location = Vector3(randint(-2000, 2000), 0, 0)
        self.initial_car_velocity = Vector3(0, 0, 0)
        self.not_hit_yet = True
        self.ball_predictions = []
        self.last_dist = None
        self.last_touch_location = Vec3(0, 0, 0)

    def reset_for_ground_shots(self):
        self.initial_ball_location = Vector3(0, 2000, 100)
        # self.initial_ball_velocity = Vector3(randint(-800, 800), randint(-800, 800), 0)
        self.initial_ball_velocity = Vector3(randint(-800, 800), randint(-800, 800), 0)
        self.initial_car_location = Vector3(randint(-800, 800), 0, 0)
        self.initial_car_velocity = Vector3(0, 0, 0)
        self.not_hit_yet = True
        self.ball_predictions = []
        self.last_dist = None
        self.last_touch_location = Vec3(0, 0, 0)

    def reset_gamestate(self):
        print('> reset_gamestate()')

        # Initialize inputs
        self.reset_for_ground_shots()
        t = self.target
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        b.location = to_vec3(self.initial_ball_location)
        b.velocity = to_vec3(self.initial_ball_velocity)
        c.location = to_vec3(self.initial_car_location)
        c.velocity = to_vec3(self.initial_car_velocity)

        # Point car at ball
        c.rotation = look_at(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), vec3(0, 0, 1))
        rotator = rotation_to_euler(c.rotation)

        # Reset
        self.aerial = None
        self.dodge = None
        self.rotation_input = None
        self.timer = 0.0

        # Set gamestate
        car_state = CarState(boost_amount=100, 
                     physics=Physics(location=self.initial_car_location, velocity=self.initial_car_velocity, rotation=rotator,
                     angular_velocity=Vector3(0, 0, 0)))
        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=self.initial_ball_velocity, rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

    def plan(self):
        self.intercept = Intercept.calculate(self.game.my_car, self.game.ball, self.target)

        # Clean up old dodge
        # if self.dodge is not None:
        #     if self.dodge.finished: self.dodge = None
        #     else: return

        # Calculate new intercept
        # not_ahead_of_ball = norm(self.game.my_car.location - self.target) > norm(self.game.ball.location - self.target)
        # on_ground = self.game.my_car.location[2] < 18
        # if on_ground and not_ahead_of_ball:
        #     self.intercept = Intercept.calculate(self.game.my_car, self.game.ball, self.target)
        #     if self.intercept is not None: return

        # # Otherwise, try to get in position
        # waypoint = vec3(self.game.ball.location)
        # waypoint[1] -= 2500
        # waypoint[2] = 0
        # self.intercept = Intercept(waypoint)
        # self.intercept.boost = False
        # self.intercept.purpose = 'position'

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Record start time
        self.tick_start = time.time()

        # Gather some information about our car and the ball
        my_car: CarState = packet.game_cars[self.index]
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
            self.target = vec3(0, 5120 + 880/2 if self.team is 0 else -(5120 + 880/2), 642.775 / 2) # Opposing net
            self.reset_gamestate()
            print('TEAM', self.team)
            return SimpleControllerState()

        # Update simulation
        self.game.read_game_information(packet, self.get_rigid_body_tick(), self.get_field_info())

        # Check for car hit ball
        if self.last_touch_location != packet.game_ball.latest_touch.hit_location:
            self.last_touch_location = Vec3(packet.game_ball.latest_touch.hit_location)
            print(f'> Car hit ball')
            self.not_hit_yet = False

        # Recalculate intercept every frame
        self.plan()

        # Re-simulate the aerial every frame
        if self.aerial is not None and not self.aerial.finished:
            simulate_aerial(self)
            simulate_alternate_aerials(self)

        # Update dodge (init or clean up old)
        # try_init_dodge(self)

        # Rendering
        if len(self.ball_predictions) > 2:
            self.renderer.draw_polyline_3d(self.ball_predictions, self.renderer.red())
        if self.aerial != None and self.aerial.target:
            self.renderer.draw_rect_3d(self.aerial.target, 8, 8, True, self.renderer.green(), centered=True)
            self.renderer.draw_line_3d(car_location, self.aerial.target, self.renderer.white())
            self.renderer.draw_line_3d(vec3_to_Vec3(self.target), self.target + self.avg_aerial_error, self.renderer.cyan())
        if self.intercept != None:
            self.renderer.draw_rect_3d(self.intercept.location, 8, 8, True, self.renderer.green(), centered=True)
        self.renderer.draw_rect_3d(vec3_to_Vec3(self.target), 8, 8, True, self.renderer.green(), centered=True)

        # Controller state
        if reset:
            self.reset_gamestate()
            return SimpleControllerState()
        # "Do a flip!"
        elif self.dodge is not None:
            if self.dodge.finished:
                self.dodge = None
                return SimpleControllerState()
            self.dodge.step(self.game.time_delta)
            return self.dodge.controls
        # Just do an aerial :4head:
        elif self.aerial is not None:
            aerial_step(self.aerial, Car(self.game.my_car), self.rotation_input, self.game.time_delta)
            return self.aerial.controls
        # Just hit the ball :4head:
        elif self.intercept is not None:
            if self.intercept.dodge and abs(self.game.time - self.intercept.jump_time) <= self.game.time_delta:
                print('im gonna nut')
                self.dodge = Dodge(self.game.my_car)
                self.dodge.duration = 0.2
                self.dodge.preorientation = self.intercept.dodge_preorientation
                self.dodge.delay = self.intercept.dodge_delay + 0.1
                self.dodge.direction = self.intercept.dodge_direction
                self.dodge.step(self.game.time_delta)
                return self.dodge.controls
            return self.intercept.get_controls(my_car, self.game.my_car) #drive_at(self, my_car, self.intercept.location)

        return SimpleControllerState()