import csv
import os
from util.singleton import Singleton

class JumpFrame:
    def __init__(self, time: float, height: float, velocity_z: float):
        self.time = float(time)
        self.height = float(height)
        self.velocity_z = float(velocity_z)

    def copy(self):
        return JumpFrame(self.time, self.height, self.velocity_z)


@Singleton
class JumpAnalysis():
    # Read frames from csv
    def __init__(self):
        self.frames = []
        filename = 'data/jump.csv'
        with open(os.path.join(os.path.dirname(__file__), filename), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.frames.append(JumpFrame(row['time'], row['height'], row['velocity_z']))
            print(f'Read {len(self.frames)} frames from {filename}')

    # TODO could be improved with a binary search or gradient descent
    def get_frame_by_height(self, height: float, ascending = True) -> JumpFrame:
        closest = None
        for frame in self.frames:
            if (closest == None or abs(frame.height - height) < abs(closest.height - height)) and ((frame.velocity_z > 0) == ascending):
                closest = frame
        return closest.copy()

    # def jump_by_height(self, height: float):
    #     start = self.get_frame_by_speed(initial_speed)
    #     end_dist = start.distance + distance
    #     end = self.get_frame_by_distance(end_dist)

    #     # Handle speeds greater than than max throttle (e.g. throttle, without boost, while supersonic)
    #     if initial_speed > end.speed:
    #         end.speed = initial_speed

    #     # Interpolate any remaining distance using constant velocity
    #     if end_dist > end.distance:
    #         dist_left = end_dist - end.distance
    #         end.time += dist_left / end.speed
    #         end.distance = end_dist

    #     return ThrottleFrame(end.time - start.time, end.distance - start.distance, end.speed)

