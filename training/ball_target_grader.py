from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

from rlbot.training.training import Grade, Pass, Fail
from rlbot.utils.structures.game_data_struct import Vector3

from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.common_graders.timeout import FailOnTimeout
from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.grading.grader import Grader

class BallTargetGrader(CompoundGrader):
    """
    Checks that the ball passes within the threshold of the target position.
    """
    def __init__(self, timeout_seconds=10.0, min_dist_to_pass=200, target=Vector3(x=0, y=0, z=0)):
        super().__init__([
            PassOnBallNearTarget(min_dist_to_pass=min_dist_to_pass, target=target),
            FailOnBallMissedTarget(target=target),
            FailOnTimeout(timeout_seconds),
        ])

@dataclass
class PassOnBallNearTarget(Grader):
    """
    Returns a Pass grade once the ball is sufficiently close to the target.
    """

    min_dist_to_pass: float = 200
    car_index: int = 0
    target: Vector3 = Vector3(x=0, y=0, z=0)

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        target = self.target
        ball = tick.game_tick_packet.game_ball.physics.location

        dist = sqrt(
            (target.x - ball.x) ** 2 +
            (target.y - ball.y) ** 2
        )
        if dist <= self.min_dist_to_pass:
            return Pass()
        return None

@dataclass
class FailOnBallMissedTarget(Grader):
    """
    Returns a Fail grade once the ball is sufficiently past the target's x-y plane.
    """

    min_dist_to_fail: float = 100
    car_index: int = 0
    target: Vector3 = Vector3(x=0, y=0, z=0)

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        target = self.target
        ball = tick.game_tick_packet.game_ball.physics.location

        if ball.y > target.y + self.min_dist_to_fail:
            return Fail()
        return None
