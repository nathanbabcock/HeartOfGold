import csv
import os

class ThrottleFrame:
    def __init__(self, time: float, distance: float, speed: float):
        self.time = float(time)
        self.distance = float(distance)
        self.speed = float(speed)

    def copy(self):
        return ThrottleFrame(self.time, self.distance, self.speed)

class ThrottleAnalysis:
    def __init__(self):
        self.frames = []
        filename = 'data/throttle.csv'
        with open(os.path.join(os.path.dirname(__file__), filename), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.frames.append(ThrottleFrame(row['time'], row['distance'], row['speed']))
            print(f'Read {len(self.frames)} frames from {filename}')

    # TODO could be improved with a binary search or gradient descent
    def get_frame_by_speed(self, speed: float) -> ThrottleFrame:
        closest = None
        for frame in self.frames:
            if closest == None or abs(frame.speed - speed) < abs(closest.speed - speed):
                closest = frame
        return closest.copy()

    # TODO obvious repitition with above, could probably be generalized later
    def get_frame_by_distance(self, distance: float) -> ThrottleFrame:
        closest = None
        for frame in self.frames:
            if closest == None or abs(frame.distance - distance) < abs(closest.distance - distance):
                closest = frame
        return closest.copy()

    def travel_distance(self, distance: float, initial_speed: float = 0):
        start = self.get_frame_by_speed(initial_speed)
        end_dist = start.distance + distance
        end = self.get_frame_by_distance(end_dist)

        # Handle speeds greater than than max throttle (e.g. throttle, without boost, while supersonic)
        if initial_speed > end.speed:
            end.speed = initial_speed

        # Interpolate any remaining distance using constant velocity
        if end_dist > end.distance:
            dist_left = end_dist - end.distance
            end.time += dist_left / end.speed
            end.distance = end_dist

        return ThrottleFrame(end.time - start.time, end.distance - start.distance, end.speed)

