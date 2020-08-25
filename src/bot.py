from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

from math import pi
from random import randint
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras import Input


class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.iteration = 0
        self.waiting_for_reset = False
        self.skip_train_ticks = 0

        # define the keras model
        self.model = Sequential()
        # self.model.add(Dense(10, input_dim=1, activation='relu'))
        # self.model.add(Dense(6, activation='relu'))
        self.model.add(Dense(1, input_dim=1))

        # compile the keras model
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        self.reset_gamestate()

    def reset_gamestate(self):
        self.initial_ball_location = Vector3(0, 0, 100)
        # self.initial_car_location = Vector3(randint(-3000, 3000), -4000, 0)
        # self.initial_car_location = Vector3(-3000, -4000, 0)
        self.training_target_location = Vec3(x=randint(-3000, 3000), y=3000, z=0)
        # self.training_target_location = Vec3(x=1000, y=3000, z=0)

        if (self.iteration > 25):
            # inputs = np.array([self.initial_car_location.x, self.initial_car_location.y, self.initial_ball_location.x, self.initial_ball_location.y, self.training_target_location.x, self.training_target_location.y]).reshape(1, 6)
            inputs = np.array([self.training_target_location.x])
            prediction = self.model.predict(inputs)[0][0]
            print(f'> Prediction Input: {inputs}')
            print(f'> Prediction output: {prediction}')
            # self.intermediate_destination = Vec3(x=prediction[0], y=prediction[1], z=0)
            self.initial_car_location = Vector3(prediction, -3000, 0)
        else:
            # self.intermediate_destination = Vec3(x=randint(-2000, 2000), y=randint(-4000,0), z=0)
            self.initial_car_location = Vector3(randint(-3000, 3000), -3000, 0)
        # self.cur_destination = 'intermediate'
        self.cur_destination = 'ball'

        ang = (Vec3(self.initial_ball_location) - Vec3(self.initial_car_location)).ang_to(Vec3(1,0,0))
        # print(ang)

        car_state = CarState(jumped=False, double_jumped=False, boost_amount=0, 
                     physics=Physics(location=self.initial_car_location, velocity=Vector3(0, 0, 0), rotation=Rotator(0, ang, 0),
                     angular_velocity=Vector3(0, 0, 0)))

        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=Vector3(0, 0, 0), rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        # game_info_state = GameInfoState(world_gravity_z=700, game_speed=0.8)
        self.set_game_state(game_state)
        self.waiting_for_reset = False

        return None

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """
        if self.skip_train_ticks > 0:
            self.skip_train_ticks -= 1

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        reset = False
        train = True

        # Experiments constraints violated (going past the ball)
        if(car_location.y > ball_location.y):
            reset = True
            train = False
            print('> Scrapping bad data')
        
        # Check if it hit the target
        # if ball_location.dist(self.training_target_location) < 200:
        #     reset = True
        if ball_location.y > self.training_target_location.y:
            reset = True
        if reset and self.iteration == 0:
            train = False
            self.iteration += 1
            self.skip_train_ticks = 10
            print('Skipping first iteration')
        if reset and train and self.skip_train_ticks <= 0 :
            # inputs = np.array([self.initial_car_location.x, self.initial_car_location.y, self.initial_ball_location.x, self.initial_ball_location.y, ball_location.x, ball_location.y])
            # inputs = np.array([ball_location.x, ball_location.y])
            # inputs = np.array([ball_location.x, ball_location.y])
            # outputs = np.array([self.intermediate_destination.x, self.intermediate_destination.y])
            print(f'>>> TRAINING ITERATION {self.iteration}')
            inputs = np.array([ball_location.x])
            outputs = np.array([self.initial_car_location.x])
            self.model.train_on_batch(inputs, outputs)
            print(f'> Training Input: {inputs}')
            print(f'> Training Output: {outputs}')
            self.iteration += 1
            self.waiting_for_reset = True
            self.skip_train_ticks = 10

        # We're far away from the ball, let's try to lead it a little bit
        # if car_location.dist(ball_location) > 1500:
        #     ball_prediction = self.get_ball_prediction_struct() # This can predict bounces, etc
        #     ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 2)
        #     target_location = Vec3(ball_in_future.physics.location)
        #     self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())
        # else:
        #     target_location = ball_location

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        # self.renderer.draw_rect_3d(self.intermediate_destination, 8, 8, True, self.renderer.cyan(), centered=True)
        if self.cur_destination == 'ball':
            self.renderer.draw_line_3d(car_location, ball_location, self.renderer.white())
        else:
            self.renderer.draw_line_3d(car_location, self.intermediate_destination, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())
        self.renderer.draw_string_2d(20, 20, 2, 2, f'Iteration: {self.iteration}', self.renderer.black())
        # self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        # self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        # self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)
        # self.renderer.draw_rect_3d(training_target_location, 8, 8, True, self.renderer.cyan(), centered=True)
        # self.renderer.draw_line_3d(ball_location, training_target_location, self.renderer.white())
        # self.renderer.draw_string_3d(training_target_location, 1, 1, f'Distance: {training_target_location.dist(ball_location):.1f}', self.renderer.white())

        # if 750 < car_velocity.length() < 800:
        #     # We'll do a front flip if the car is moving at a certain speed.
        #     return self.begin_front_flip(packet)

        controls = SimpleControllerState()
        if reset:
            self.reset_gamestate()
            return controls
        if self.cur_destination == 'intermediate' and self.intermediate_destination.dist(car_location) <= 200:
            self.cur_destination = 'ball'
        controls.steer = steer_toward_target(my_car, ball_location if self.cur_destination=='ball' else self.intermediate_destination)
        controls.throttle = 1.0
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
