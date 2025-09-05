from func import *
import scipy


SPEED = 0
THETA = 1
T_DROP = 2
T_DETONATE = 3
params = np.array([70, 10, 10, 11])


set_fy_mission(0, params[SPEED], params[THETA])
interval = get_time_interval(0, 0, params[T_DROP], params[T_DETONATE])
period = get_lebesgue(interval)
print(interval, period)