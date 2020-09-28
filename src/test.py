# from rlutilities_v2.simulation import Ball, Car, Field, Game, Input
# from rlutilities_v2.linear_algebra import vec3

from rlutilities.python.rlutilities.simulation import Ball, Car, Field, Game, Input
from rlutilities.python.rlutilities.linear_algebra import vec3

# from rlutilities.simulation import Ball, Car, Field, Game, Input
# from rlutilities.linear_algebra import vec3

import multiprocessing as mp

Game.set_mode('soccar')
game = Game(0, 0)
car = game.cars[0]

car.location = vec3(1000, 1000, 0)

car.on_ground = True

print('onground', car.on_ground)
print('ngvel', car.angular_velocity)
print('start', car.location)

# car.velocity = vec3(-1000, -1000, 0)

input = Input()
input.jump = True

print('*JUMP*')

for i in range(120):
    car.step(input, 1.0 / 120.0)

print('end', car.location)

input.pitch = -1

print('*DODGE*')

for i in range(60):
    car.step(input, 1.0 / 120.0)

print('ngvel', car.angular_velocity)
print('end2', car.location)

# print(car.location)
# print(game.ball.location)

print('i got cpus = ', mp.cpu_count())