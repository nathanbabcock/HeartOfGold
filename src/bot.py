from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from math import pi, sqrt, inf, cos, sin, tan, atan2
from random import randint
import time
import random

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

from analysis.throttle import *

import csv
import os

class HeartOfGold(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.ball_predictions = []
        self.not_hit_yet = True
        self.game = None
        self.timer = 0.0

        self.intercept = None
        self.aerial = None

        self.dodge = None
        self.dodge_started = False
        self.dodge_start_time = None

        self.start_time = None
        self.start_recording = False
        self.done_recording = False

    def initialize_agent(self):
        print('> Alphabot: I N I T I A L I Z E D')

    def reset_for_data_collection(self):
        self.initial_ball_location = Vector3(2000, 2000, 100)
        self.initial_ball_velocity = Vector3(0, 0, 0)
        self.initial_car_location = Vector3(0, 0, 0)
        self.initial_car_velocity = Vector3(0, 0, 0)
        self.not_hit_yet = True
        self.ball_predictions = []
        self.last_dist = None
        self.last_touch_location = Vec3(0, 0, 0)
        self.start_time = self.game.time
        self.start_recording = False

    def reset_gamestate(self):
        print('> reset_gamestate()')

        # Initialize inputs
        self.reset_for_data_collection()
        t = self.target
        b = Ball(self.game.ball)
        c = Car(self.game.cars[self.index])
        b.location = to_vec3(self.initial_ball_location)
        b.velocity = to_vec3(self.initial_ball_velocity)
        c.location = to_vec3(self.initial_car_location)
        c.velocity = to_vec3(self.initial_car_velocity)

        # Point car at ball
        # c.rotation = look_at(vec3(b.location[0] - c.location[0], b.location[1] - c.location[1], 0), vec3(0, 0, 1))
        # rotator = rotation_to_euler(c.rotation)

        # Reset
        self.aerial = None
        self.dodge = None
        self.rotation_input = None
        self.timer = 0.0

        # Set gamestate
        car_state = CarState(boost_amount=100, 
                     physics=Physics(location=self.initial_car_location, velocity=self.initial_car_velocity, rotation=Rotator(0,pi/2,0),
                     angular_velocity=Vector3(0, 0, 0)))
        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=self.initial_ball_velocity, rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

    def write_json(self):
        filename = 'analysis/data/turn-right-boost.json'
        import json

        data = {}
        data['deltaTime'] = 0.008333333333333333
        data['frames'] = self.dodge_frames

        with open(os.path.join(os.path.dirname(__file__), filename), 'w') as outfile:
            json.dump(data, outfile,  indent=2)
            print(f'Wrote {len(self.dodge_frames)} frames to {filename}')

    def write_csv(self):
        filename = 'analysis/data/turn-right-boost.csv'
        with open(os.path.join(os.path.dirname(__file__), filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                'time',
                'pos_x',
                'pos_y',
                #'pos_z',
                'vel_x',
                'vel_y',
                #'vel_z',
                'speed',
                #'pitch',
                #'yaw',
                #'roll',
                #'angvel_x',
                #'angvel_y',
                #'angvel_z',
            ])
            for row in self.dodge_frames:
                writer.writerow([
                    row.time,
                    row.car_pos[0],
                    row.car_pos[1],
                    # row.car_pos[2],
                    row.car_vel[0],
                    row.car_vel[1],
                    # row.car_vel[2],
                    row.car_speed,
                    # row.car_rotator[0],
                    # row.car_rotator[1],
                    # row.car_rotator[2],
                    # row.car_angvel[0],
                    # row.car_angvel[1],
                    # row.car_angvel[2]
                ])
            print(f'Wrote {len(self.dodge_frames)} frames to {filename}')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Record start time
        self.tick_start = time.time()
        self.timer += 1.0 / 120.0

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

        if self.dodge_start_time is not None and self.game.time > self.dodge_start_time + 6.0 and not self.done_recording:
            #self.write_csv()
            self.write_json()
            self.dodge_start_time = None
            self.done_recording = True

        if self.dodge is None and not self.done_recording and self.game.time > self.start_time + 1.0:
            self.start_recording = True
            self.dodge = True #Dodge(self.game.my_car)
            self.dodge_start_time = self.game.time
            self.start_time = self.game.time
            # self.dodge.direction = vec2(1, 1)
            # self.dodge.duration = 0.15
            # self.dodge.delay = 0.2
            self.dodge_frames = []

        if self.start_recording and not self.done_recording:
            self.dodge_frames.append({
                'time': self.game.time - self.dodge_start_time,
                'pos_x': self.game.my_car.location[0],
                'pos_y': self.game.my_car.location[1],
                #'pos_z': self.game.my_car.location[2],
                'vel_x': self.game.my_car.velocity[0],
                'vel_y': self.game.my_car.velocity[1],
                # 'vel_z': self.game.my_car.velocity[2],
                'speed': veclen(self.game.my_car.velocity),
                # 'pitch': my_car.physics.rotation.pitch,
                # 'yaw': my_car.physics.rotation.yaw,
                # 'roll': my_car.physics.rotation.roll,
                # 'angvel_x': self.game.my_car.angular_velocity[0],
                # 'angvel_y': self.game.my_car.angular_velocity[1],
                # 'angvel_z': self.game.my_car.angular_velocity[2]
            })
            print(self.dodge_frames[-1])

        
        # Controller state
        if reset:
            self.reset_gamestate()
            return SimpleControllerState()
        elif self.start_recording:
            controls = SimpleControllerState()
            controls.throttle = 1
            controls.steer = 1
            controls.boost = True
            return controls
        else:
            return SimpleControllerState()