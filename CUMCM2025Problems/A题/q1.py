from func import *


direction = fake_target - fys[0]
theta = np.arctan2(direction[1], direction[0])
set_fy_mission(0, 120, theta)
interval = get_time_interval(0, 0, 1.5, 5.1)
period = get_lebesgue(interval)
print(interval, period)

