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

class HitData:
    def __init__(self, car_direction_before=None, ball_direction_after=None, car_speed_before=None, ball_speed_before=None):
        self.car_direction_before = car_direction_before
        self.ball_direction_after = ball_direction_after

        # not used yet
        self.car_speed_before = car_speed_before
        self.ball_speed_before = ball_speed_before

class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.iteration = 0
        self.last_touch_location = Vector3(0, 0, 0)
        self.hit_data = []
        self.pre_hit = HitData()

    def initialize_agent(self):
        self.reset_gamestate()
        print('> Alphabot: I N I T I A L I Z E D')

    def reset_gamestate(self):
        self.initial_ball_location = Vector3(0, 0, 100)
        self.initial_car_location = Vector3(3000, -3000, 0)
        self.training_target_location = Vec3(1000, 3000, 0)

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


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
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
        car_direction = car_velocity.ang_to(Vec3(1, 0, 0)) if car_velocity.length() > 0 else 0
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        ball_direction = ball_velocity.ang_to(Vec3(1, 0, 0)) if ball_velocity.length() > 0 else 0
        reset = False
        train = True

        # Check for car hit ball
        if self.last_touch_location != packet.game_ball.latest_touch.hit_location:
            self.pre_hit.ball_direction_after = ball_direction
            self.hit_data.append(self.pre_hit)
            self.last_touch_location = Vec3(packet.game_ball.latest_touch.hit_location)
            print(f'> BALL HIT')
            self.pre_hit = HitData()
        else:
            self.pre_hit.car_direction_before = car_direction

        # Reset if car passes ball
        if car_location.y > ball_location.y:
            reset = True
            self.cur_tries = 999
            print('> Scrapping bad data')

        # Reset if ball passes target
        if ball_location.y > self.training_target_location.y:
            reset = True
            if abs(ball_location.x - self.training_target_location.x) < 100 and abs(ball_location.y - self.training_target_location.y) < 100:
                print('Task succesfully learned; picking new target')

        # Skip first iteration
        # if reset and self.iteration == 0:
        #     train = False
        #     self.iteration += 1
        #     self.skip_train_ticks = 10
        #     print('Skipping iteration')

        # Rendering
        self.renderer.draw_rect_3d(self.training_target_location, 8, 8, True, self.renderer.green(), centered=True)
        self.renderer.draw_line_3d(car_location, ball_location, self.renderer.white())
        self.renderer.draw_line_3d(ball_location, self.training_target_location, self.renderer.green())
        self.renderer.draw_string_2d(20, 20, 2, 2, f'Iteration: {self.iteration}', self.renderer.black())

        # Controller state
        controls = SimpleControllerState()
        if reset:
            self.reset_gamestate()
            return controls
        controls.steer = steer_toward_target(my_car, ball_location)
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
