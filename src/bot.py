from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

from math import pi, sqrt, inf
from random import randint
import numpy as np

# from keras.models import Sequential
# from keras.layers import Dense
# from keras import Input
# import keras.backend as K

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TrainingState:
    def __init__(self, ballX=None, ballY=None, carX=None, carY=None, targetX=None, targetY=None):
        self.ballX = ballX
        self.ballY = ballY
        self.carX = carX
        self.carY = carY
        self.targetX = targetX
        self.targetY = targetY

    def dist(self, other):
        dist = 0
        if self.ballX and self.ballY and other.ballX and other.ballY:
            dist += sqrt((other.ballY - self.ballY)**2 + (other.ballX - self.ballX)**2)
        if self.carX and self.carY and other.carX and other.carY:
            dist += sqrt((other.carY - self.carY)**2 + (other.carX - self.carX)**2)
        if self.targetX and self.targetY and other.targetX and other.targetY:
            dist += sqrt((other.targetY - self.targetY)**2 + (other.targetX - self.targetX)**2)
        return dist

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.iteration = 0
        self.skip_train_ticks = 0
        self.inputs = []
        self.outputs = []
        self.plot = None
        self.fig = None
        self.axes = []
        self.coefs = []
        self.accuracies = []
        self.accuracy_labels = [
            'Calculation accuracy',
            'Regression accuracy',
            'Model mean squared error'
        ]
        self.training_states = []

        # init sklearn multi linear regression model
        self.model = LinearRegression()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        self.randomize_input_state()
        self.reset_gamestate()
        print('> Alphabot: I N I T I A L I Z E D')

    def is_line_possible(self):
        if not self.initial_ball_location or not self.training_target_location:
            return False

        # Point-point form to calculate correct predicted input
        x_1 = self.initial_ball_location.x
        y_1 = self.initial_ball_location.y
        x_2 = self.training_target_location.x
        y_2 = self.training_target_location.y
        self.output_calculated = (self.initial_car_y - y_1) * ((x_2 - x_1) / (y_2 - y_1)) + x_1

        return -4000 <= self.output_calculated <= 4000

    def randomize_input_state(self):
        # self.initial_ball_location = Vector3(randint(-4000, 4000), randint(-2000, 2000), 100)
        # self.training_target_location = Vec3(randint(-4000, 4000), randint(2000, 4000),  0)
        self.cur_tries = 0
        self.initial_car_y = -4000
        self.initial_ball_location = Vector3(1000, 0, 100)
        self.training_target_location = Vec3(randint(-3000, 3000), 3000, 0)
        while not self.is_line_possible():
            return self.randomize_input_state()
        print(f'> Calculation output: {self.output_calculated}')
        # self.training_target_location = Vec3(-250, 3000,  0)

    def get_cur_state(self):
        return TrainingState(
            ballX = self.initial_ball_location.x,
            ballY = self.initial_ball_location.y,
            carX = self.initial_car_location.x,
            carY = self.initial_car_location.y,
            targetX = self.training_target_location.x,
            targetY = self.training_target_location.y,
        )

    def get_closest_sample(self, state):
        samples = map(lambda x: [x, x.dist(state)], self.training_states)
        samples = sorted(samples, key=lambda x: x[1])
        bestDist = samples[0][1]
        print(f'> closest state distance: {bestDist}')

        # POC assumption: sample.targetX is used for narrowing search

        highSample = None
        lowSample = None

        for sample in samples:
            if sample[0].targetX > self.training_target_location.x:
                highSample = sample
                break
        for sample in samples:
            if sample[0].targetX < self.training_target_location.x:
                lowSample = sample
                break
        
        # If it's close enough, return the best match
        if bestDist < 100:
            return samples[0][0]

        if highSample is not None and lowSample is not None:
            averageSample = TrainingState(
                ballX = (highSample[0].ballX + lowSample[0].ballX)/2,
                ballY = (highSample[0].ballY + lowSample[0].ballY)/2,
                carX = (highSample[0].carX + lowSample[0].carX)/2,
                carY = (highSample[0].carY + lowSample[0].carY)/2,
                targetX = (highSample[0].targetX + lowSample[0].targetX)/2,
                targetY = (highSample[0].targetY + lowSample[0].targetY)/2,
            )
            return averageSample
        
        return samples[0][0]

    def reset_gamestate(self):     
        # Get a set of inputs that has a possible output
        # self.randomize_input_state()
        # while not self.is_line_possible():
        #     self.randomize_input_state()
        # print(f'> Calculation output: {self.output_calculated}')

        # Predict
        if self.iteration > 2:
            # inputs = [[self.training_target_location.x,  self.training_target_location.y, self.initial_ball_location.x, self.initial_ball_location.y]]
            prediction: TrainingState = self.get_closest_sample(state=TrainingState(targetX=self.training_target_location.x, targetY=self.training_target_location.y, ballX=self.initial_ball_location.x, ballY=self.initial_ball_location.y))
            # print(f'> Prediction input: {inputs}')
            print(f'> Prediction output: {prediction.carX}')
            self.initial_car_location = Vector3(prediction.carX, self.initial_car_y, 0)
        elif self.iteration == 2:
            self.initial_car_location = Vector3(-3500, self.initial_car_y, 0)
        elif self.iteration == 1:
            self.initial_car_location = Vector3(3500, self.initial_car_y, 0)
        else:
            self.initial_car_location = Vector3(randint(-4000, 4000), self.initial_car_y, 0)
        self.cur_destination = 'ball'

        # line car up with ball to avoid error from turning
        ang = (Vec3(self.initial_ball_location) - Vec3(self.initial_car_location)).ang_to(Vec3(1,0,0))

        # Set gamestate
        car_state = CarState(boost_amount=100, 
                     physics=Physics(location=self.initial_car_location, velocity=Vector3(0, 0, 0), rotation=Rotator(0, ang, 0),
                     angular_velocity=Vector3(0, 0, 0)))
        ball_state = BallState(Physics(location=self.initial_ball_location, velocity=Vector3(0, 0, 0), rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)
        return None

    def init_plot(self):
        return

        # Initialize plot
        print('> Initializing plot...')
        plt.clf()
        plt.close()
        plt.ion()
        self.fig, self.axes = plt.subplots(2)

        # Graph inputs
        labels = [
            'target x',
            'target y',
            'ball x',
            'ball y'
        ]
        self.plot = [ self.axes[0].scatter(list(zip(*self.inputs))[i], self.outputs, label=labels[i]) for i in range(len(self.inputs[0])) ]
        self.axes[0].set_xlim(-3000,3000)
        self.axes[0].set_ylim(-3000,3000)
        self.axes[0].legend()

        # Coefs
        self.coef_plot = [ self.axes[1].plot(i, list(zip(*self.coefs))[i], label=f'{labels[i]} coef')[0] for i in range(len(self.coefs[0])) ]
        self.axes[1].set_ylim(-3, 3)
        self.axes[1].legend()

        # Accuracies
        # accuracy_labels = [
        #     'distance from target',
        #     'prediction from calculation',
        # ]
        # self.acc_plot = [ self.axes[2].plot(i, list(zip(*self.accuracies))[i], label=accuracy_labels[i])[0] for i in range(len(self.accuracies[0])) ]
        # self.axes[2].set_ylim(-1,1)
        # self.axes[2].legend()

        # show
        plt.grid(True, axis='both')
        plt.show()
        return self.plot

    def draw_plot(self):
        return

        # init plot
        if not self.plot:
            self.init_plot()
        if not self.plot:
            return

        # update input plot
        for i in range(len(self.inputs[0])):
            self.plot[i].set_offsets(np.c_[list(zip(*self.inputs))[i], self.outputs])

        # update coef plot
        for i in range(len(self.coefs[0])):
            self.coef_plot[i].set_xdata(range(self.iteration))
            self.coef_plot[i].set_ydata([coef[i] for coef in self.coefs])
        self.axes[1].set_xlim(0, self.iteration)

        # update acc plot
        # for i in range(len(self.accuracies[0])):
        #     self.acc_plot[i].set_xdata(range(self.iteration))
        #     self.acc_plot[i].set_ydata([acc[i] for acc in self.accuracies])
        # self.axes[2].set_xlim(0, self.iteration)
        
        # draw
        plt.draw()
        plt.pause(0.01)

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
        if car_location.y > ball_location.y:
            reset = True
            train = False
            print('> Scrapping bad data')

        if self.cur_tries > 5:
            reset = True
            train = False
            self.randomize_input_state()
            print('> Maximum tries exceeded; resetting')
        
        # Episode completed sucessfully
        if ball_location.y > self.training_target_location.y:
            reset = True
            # print(f'checking reset with dist {ball_location.dist(self.training_target_location)}')
            if abs(ball_location.x - self.training_target_location.x) < 100 and abs(ball_location.y - self.training_target_location.y) < 100:
                print('Task succesfully learned; picking new target')
                self.randomize_input_state()

        # Skip first iteration
        if reset and self.iteration == 0:
            train = False
            self.iteration += 1
            self.skip_train_ticks = 10
            print('Skipping iteration')

        # Train
        if reset and train and self.skip_train_ticks <= 0 :
            print(f'>>> TRAINING ITERATION {self.iteration}')
            # inputs = [ball_location.x, ball_location.y, self.initial_ball_location.x, self.initial_ball_location.y]
            # outputs = [self.initial_car_location.x]
            # print(f'> Training Input: {inputs}')
            # print(f'> Training Output: {outputs}')
            # self.inputs.append(inputs)
            # self.outputs.append(outputs)

            # # Fit model
            # self.model.fit(self.inputs, self.outputs)

            # # Metrics: coefficients
            # self.coefs.append(self.model.coef_[0])

            # # Metrics: arbitrary accuracy
            # target_accuracy = ball_location.dist(self.training_target_location) / self.training_target_location.dist(self.initial_car_location)
            # prediction_accuracy = (self.initial_car_location.x - self.output_calculated) / self.output_calculated
            # self.accuracies.append([target_accuracy, prediction_accuracy])

            # # Plot
            # self.draw_plot()
            curState = TrainingState(
                ballX = self.initial_ball_location.x,
                ballY = self.initial_ball_location.y,
                carX = self.initial_car_location.x,
                carY = self.initial_car_location.y,
                targetX = ball_location.x,
                targetY = ball_location.y,
            )
            self.training_states.append(curState)

            # Increment & delay until next iteration
            self.iteration += 1
            self.cur_tries += 1
            self.skip_train_ticks = 10

        # Rendering
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        if self.cur_destination == 'ball':
            self.renderer.draw_line_3d(car_location, ball_location, self.renderer.white())
        else:
            self.renderer.draw_line_3d(car_location, self.intermediate_destination, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())
        self.renderer.draw_string_2d(20, 20, 2, 2, f'Iteration: {self.iteration}', self.renderer.black())

        # Controller state
        controls = SimpleControllerState()
        if reset:
            self.reset_gamestate()
            return controls
        if self.cur_destination == 'intermediate' and self.intermediate_destination.dist(car_location) <= 200:
            self.cur_destination = 'ball'
        controls.steer = steer_toward_target(my_car, ball_location if self.cur_destination=='ball' else self.intermediate_destination)
        controls.throttle = 1.0
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
