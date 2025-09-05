import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm



# content = {"missile": [], "fy": [], "smoke bomb": [], "smoke": []}
class Content:
    def __init__(self):
        self.w = False
        self.data = {"missile": [], "fy": [], "smoke bomb": [], "smoke": [], "angle": []}

    def clear(self):
        for value in self.data.values():
            value.clear()

    def append(self, key, value):
        if self.w:
            self.data[key].append(value)

    def __enter__(self):
        self.clear()
        self.w = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.w = False
content = Content()


# ======================================================================================================================
smoke_velocity = np.array([0, 0, -3])
smoke_radius = 10
smoke_period = 20


# ======================================================================================================================
fake_target = np.array([0, 0, 0])


real_target_radius = 7
real_target_top = np.array([0, 200, 10])
real_target_button = np.array([0, 200, 0])


N = 40
real_target_samples = np.zeros([N * 2, 3])
for i in range(N):
    angle = 2 * np.pi * i / N
    real_target_samples[i] = [np.cos(angle), np.sin(angle), 0]
real_target_samples[N:2*N] = real_target_samples[0:N]
real_target_samples *= real_target_radius
real_target_samples[0:N] += real_target_button
real_target_samples[N:2*N] += real_target_top


# ======================================================================================================================
missile_speed = 300
missiles = np.array([
    [20000, 0, 2000],
    [19000, 600, 2100],
    [18000, -600, 1900]
])


# ======================================================================================================================
fys = np.array([
    [17800, 0, 1800],
    [12000, 1400, 1400],
    [6000, -3000, 700],
    [11000, 2000, 1800],
    [13000, -2000, 1300]
])
fy_v_theta = np.zeros([len(fys), 2])
fy_period = 1
fy_speed_min = 70
fy_speed_max = 140


def set_fy_mission(fy_id, speed, theta):
    fy_v_theta[fy_id, :] = speed, theta


