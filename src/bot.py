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
import time

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
        self.timer = 0.0

    def initialize_agent(self):
        print('> Alphabot: I N I T I A L I Z E D')

    def reset_gamestate(self):
        print('> reset_gamestate()')

        # Initialize inputs
        self.initial_ball_location = Vector3(0, 0, 100)
        self.initial_ball_velocity = Vector3(0, 0, 650 * 2)
        # self.initial_ball_velocity = Vector3(randint(-500, 500), randint(-500, 500), 650 * 2)
        # self.initial_car_location = Vector3(randint(-3000, 3000), randint(1000, 2000), 0)
        # self.initial_car_location = Vector3(1500, 1500, 0)
        self.initial_car_location = Vector3(randint(-2000, 2000), 1000, 0)
        self.initial_car_velocity = Vector3(0, 0, 0)
        self.training_target_location = Vec3(0, -4000, 1000)
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
        c.velocity = to_vec3(self.initial_car_velocity)

        # Point car at ball
        c.rotation = look_at(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), vec3(0, 0, 1))
        # c.rotation = look_at(vec3(randint(-3000, 3000), randint(-3000, 3000), 0), vec3(0, 0, 1))
        rotator = rotation_to_euler(c.rotation)

        # Helpful push in the right direction
        # c.velocity = setveclen(c.forward(), 1000)
        # self.initial_car_velocity = to_Vector3(c.velocity)

        # Wait
        self.aerial = None
        self.timer = 0.0

        # Set gamestate
        car_state = CarState(boost_amount=100, 
                     physics=Physics(location=self.initial_car_location, velocity=self.initial_car_velocity, rotation=rotator,
                     angular_velocity=Vector3(0, 0, 0)))
        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=self.initial_ball_velocity, rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

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

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Record start time
        tick_start = time.time()

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

        # Wait
        self.timer += self.game.time_delta
        if self.timer >= 0.5 and self.aerial == None:
            self.init_aerial()

        # Re-simulate the aerial every frame
        closest_dist = None
        closest_point = None
        closest_vec = None
        t = to_vec3(self.training_target_location)
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        dt = 1.0 / 60.0
        if self.aerial is not None and not self.aerial.finished:
            # Simulate current aerial
            self.ball_predictions = [vec3(b.location)]
            aerial_copy = copy_aerial(self.aerial, c)
            for i in range(60*5):
                # Simulate
                aerial_copy.step(dt)
                c.step(aerial_copy.controls, dt)
                b.step(dt, c)

                # Measure dist from target
                dist = norm(t - b.location)
                if closest_dist == None or dist < closest_dist:
                    closest_dist = dist
                    closest_point = vec3(b.location)
                    closest_vec = b.location - t

                # Record trajectory
                self.ball_predictions.append(vec3(b.location))

            if self.avg_aerial_error == None:
                self.avg_aerial_error = closest_vec
                self.cur_aerial_sims = 1
            else:
                self.avg_aerial_error = self.avg_aerial_error + (closest_vec - self.avg_aerial_error) / (self.cur_aerial_sims + 1)
                self.cur_aerial_sims += 1

            avg_sim_time = 0
            num_sims = 0
            tick_deadline = tick_start + (1.0 / 120.0)
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
                perturbator = vec3(randint(-int(2 * b.collision_radius), int(2 * b.collision_radius)), randint(-int(2 * b.collision_radius), int(2 * b.collision_radius)), randint(-int(2 * b.collision_radius), int(2 * b.collision_radius)))
                aerial_copy.target = self.original_aerial_target + perturbator
                for i in range(60*5):
                    # Simulate
                    aerial_copy.step(dt)
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
                        # alt_closest_dist = norm((closest_point + alt_closest_point) / 2)
                        alt_closest_vec = b.location - t
                        # average this delta with the previous average
                        # alt_closest_dist = norm(self.avg_aerial_error + (alt_closest_vec - self.avg_aerial_error) / (self.cur_aerial_sims + 1))
                        alt_closest_dist = norm((self.avg_aerial_error + alt_closest_vec) / 2)

                    # Record trajectory
                    alt_ball_predictions.append(vec3(b.location))

                # We found a better aerial!
                if alt_closest_dist < closest_dist:
                    self.ball_predictions = alt_ball_predictions
                    self.aerial.target = aerial_copy.target
                    # closest_point = alt_closest_point
                    closest_dist = alt_closest_dist

                # Update tick time estimation
                time_this_sim = time.time() - sim_start
                avg_sim_time = avg_sim_time + (time_this_sim - avg_sim_time) / (num_sims + 1)
                num_sims += 1
        
            # print('Alternate realities visited during this hundreth of a second:', num_sims)
            # print('avg sim time:', avg_sim_time)
            # print('tick time:', 1.0 / 120.0)

        # Rendering
        if len(self.ball_predictions) > 2:
            self.renderer.draw_polyline_3d(self.ball_predictions, self.renderer.red())
        if self.aerial != None and self.aerial.target:
            self.renderer.draw_rect_3d(self.aerial.target, 8, 8, True, self.renderer.green(), centered=True)
            self.renderer.draw_line_3d(car_location, self.aerial.target, self.renderer.white())
            self.renderer.draw_line_3d(self.training_target_location, to_vec3(self.training_target_location) + self.avg_aerial_error, self.renderer.cyan())
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        # self.renderer.draw_line_3d(car_location, ball_location, self.renderer.white())
        # self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())

        # Controller state
        if reset:
            self.reset_gamestate()
            return SimpleControllerState()
        
        # Just do an aerial :4head:
        if self.aerial != None:
            self.aerial.step(self.game.time_delta)
            return self.aerial.controls

        return SimpleControllerState()