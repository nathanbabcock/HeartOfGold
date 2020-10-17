import csv
import os
from analysis.throttle import ThrottleFrame
from util.singleton import Singleton

@Singleton
class BoostAnalysis():
    # Read boost self.frames from CSV
    def __init__(self):
        self.frames = []
        filename = 'data/boost.csv'
        with open(os.path.join(os.path.dirname(__file__), filename), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.frames.append(ThrottleFrame(row['time'], row['distance'], row['speed']))
            print(f'Read {len(self.frames)} self.frames from {filename}')

    # TODO could be improved with a binary search or gradient descent
    def get_frame_by_speed(self, speed: float) -> ThrottleFrame:
        closest = None
        for frame in self.frames:
            if closest == None or abs(frame.speed - speed) < abs(closest.speed - speed):
                closest = frame
        return closest.copy()

    def get_index_by_speed(self, speed: float) -> ThrottleFrame:
        closest = None
        closest_index = 0
        index = 0
        for frame in self.frames:
            if closest == None or abs(frame.speed - speed) < abs(closest.speed - speed):
                closest = frame
                closest_index = index
            index += 1
        return closest_index

    # TODO obvious repitition with above, could probably be generalized later
    def get_frame_by_distance(self, distance: float) -> ThrottleFrame:
        closest = None
        for frame in self.frames:
            if closest == None or abs(frame.distance - distance) < abs(closest.distance - distance):
                closest = frame
        return closest.copy()

    # TODO obvious repitition with above, could probably be generalized later
    def get_frame_by_time(self, time: float) -> ThrottleFrame:
        closest = None
        for frame in self.frames:
            if closest == None or abs(frame.time - time) < abs(closest.time - time):
                closest = frame
        return closest.copy()

    def get_frame_by_error(self, error_func, start_index = 0):
        # print('get_frame_by_error')
        closest = None
        index = 0
        for frame in self.frames:
            # if index >= start_index and closest is not None:
            #     # print(f'index >= start_index {index >= start_index}')
            #     print(f'error_func(frame) {error_func(frame)}')
            #     print(f'error_func(closest) {error_func(closest)}')
            if index >= start_index and (closest == None or error_func(frame) < error_func(closest)):
                closest = frame
            index += 1

        # at the end, explore past the last recorded ThrottleFrame by extrapolating at constant velocity
        while closest.time >= self.frames[-1].time:
            extrapolated_frame = ThrottleFrame(closest.time + 1.0/120.0, closest.distance + closest.speed * 1.0/120.0, closest.speed)
            if error_func(extrapolated_frame) > error_func(closest):
                return closest.copy()
            closest = extrapolated_frame

        return closest.copy()

    def travel_distance(self, distance: float, initial_speed: float = 0):
        start = self.get_frame_by_speed(initial_speed)
        end_dist = start.distance + distance
        end = self.get_frame_by_distance(end_dist)

        # Handle speeds greater than than max boost (e.g. boost, without boost, while supersonic)
        if initial_speed > end.speed:
            end.speed = initial_speed

        # Interpolate any remaining distance using constant velocity
        if end_dist > end.distance:
            dist_left = end_dist - end.distance
            end.time += dist_left / end.speed
            end.distance = end_dist

        return ThrottleFrame(end.time - start.time, end.distance - start.distance, end.speed)

    def travel_time(self, time: float, initial_speed: float = 0):
        start = self.get_frame_by_speed(initial_speed)
        end_time = start.time + time
        end = self.get_frame_by_time(end_time)

        # Handle speeds greater than than max boost (e.g. boost, without boost, while supersonic)
        if initial_speed > end.speed:
            end.speed = initial_speed

        # Interpolate any remaining distance using constant velocity
        if end_time > end.time:
            time_left = end_time - end.time
            end.distance += time_left * end.speed
            end.time = end_time

        return ThrottleFrame(end.time - start.time, end.distance - start.distance, end.speed)

