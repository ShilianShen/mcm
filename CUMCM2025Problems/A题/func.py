import matplotlib.pyplot as plt

from config import *


# \mu
def get_missile_position(m_id: int, t_now: float):
    start = missiles[m_id]
    direction = fake_target - start
    velocity = missile_speed / norm(direction) * direction
    missile_position = start + velocity * t_now
    content.append("missile", missile_position)
    return missile_position


# \phi
def get_fy_position(fy_id: int, t_now: float):
    fy_start = fys[fy_id]
    v, theta = fy_v_theta[fy_id]
    fy_velocity = np.array([np.cos(theta), np.sin(theta), 0]) * v
    fy_position = fy_start + fy_velocity * t_now
    content.append("fy", fy_position)
    return fy_position


# \sigma
def get_smoke_position(fy_id: int, t_drop: float, t_detonate: float, t_now: float):
    # [0, t_drop, t_drop + t_detonate, t_drop + t_detonate + smoke_period, infty)
    fy_start = fys[fy_id]
    v, theta = fy_v_theta[fy_id]
    fy_velocity = np.array([np.cos(theta), np.sin(theta), 0]) * v

    if t_drop <= t_now < t_drop + t_detonate:
        smoke_bomb_position = fy_start + fy_velocity * t_now
        smoke_bomb_position[2] -= 0.5 * 9.8 * (t_now - t_drop) ** 2
        content.append("smoke bomb", smoke_bomb_position)
        return smoke_bomb_position

    elif t_drop + t_detonate <= t_now < t_drop + t_detonate + smoke_period:
        detonate_position = fy_start + fy_velocity * (t_drop + t_detonate)
        detonate_position[2] -= 0.5 * 9.8 * t_detonate ** 2
        smoke_position = detonate_position + smoke_velocity * (t_now - t_detonate - t_drop)
        content.append("smoke", smoke_position)
        return smoke_position

    return None


# \alpha
def get_angle(missile_position: np.ndarray, smoke_position: np.ndarray):
    if not (missile_position.ndim == 1 and missile_position.shape[0] == 3):
        raise ValueError('missile_position must be 3D')
    if not (smoke_position.ndim == 1 and smoke_position.shape[0] == 3):
        raise ValueError('smoke_position must be 3D')

    vec_MS = smoke_position - missile_position
    if norm(vec_MS) <= smoke_radius:
        return 0
    cone_theta = np.arcsin(smoke_radius / norm(vec_MS))

    angles = [0]
    for point in real_target_samples:
        vec_MP = point - missile_position
        point_angle = np.arccos(np.dot(vec_MS, vec_MP) / (norm(vec_MS) * norm(vec_MP)))
        if point_angle > cone_theta:
            angles.append(point_angle - cone_theta)
    angle = np.mean(angles)
    content.append("angle", angle)
    return angle


# g
def get_countermeasure(missile_position: np.ndarray, smoke_position: np.ndarray):
    if not (missile_position.ndim == 1 and missile_position.shape[0] == 3):
        raise ValueError('missile_position must be 3D')
    if not (smoke_position.ndim == 1 and smoke_position.shape[0] == 3):
        raise ValueError('smoke_position must be 3D')

    return get_angle(missile_position, smoke_position) == 0


# \tau
def get_time_interval(m_id, fy_id, t_drop, t_detonate, res: float = 1e-2):
    result = []

    if isinstance(m_id, np.ndarray) and len(m_id) > 1:
        for m in m_id:
            result.append(get_time_interval(m, fy_id, t_drop, t_detonate, res))
        return np.array(result)

    if isinstance(fy_id, np.ndarray) and len(fy_id) >= 1:
        for j in range(len(fy_id)):
            result.append(get_time_interval(m_id, fy_id[j], t_drop[j], t_detonate[j], res))
        return np.array(result)

    a = b = -1
    T = min(t_drop + t_detonate + smoke_period, missile_life[m_id])
    for t in np.arange(0, T, res):
        missile_position = get_missile_position(m_id, t)
        fy_position = get_fy_position(fy_id, t)
        smoke_position = get_smoke_position(fy_id, t_drop, t_detonate, t)
        if smoke_position is None:
            continue
        countermeasure = get_countermeasure(missile_position, smoke_position)
        if countermeasure and a < 0:
            a = t
        if not countermeasure and b < 0 <= a:
            b = t
    if b < a:
        b = T
    return np.array([a, b])


# \lambda
def get_lebesgue(intervals: np.ndarray):
    if intervals.ndim == 1 and intervals.shape[0] == 2:
        return intervals[1] - intervals[0]
    if not (intervals.ndim == 2 and intervals.shape[1] == 2):
        raise ValueError("输入数组必须是N×2的形状")

    if intervals.size == 0:
        return 0.0

    # 移除无效区间（a > b的情况）
    valid_mask = intervals[:, 0] <= intervals[:, 1]
    intervals = intervals[valid_mask]

    if intervals.size == 0:
        return 0.0

    # 按区间起点排序
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]

    # 合并重叠区间
    merged = []
    current_start, current_end = sorted_intervals[0]

    for i in range(1, len(sorted_intervals)):
        start, end = sorted_intervals[i]

        if start <= current_end:  # 区间重叠或相邻
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    # 计算总测度
    total_measure = 0.0
    for start, end in merged:
        total_measure += (end - start)

    return total_measure


def draw_3d_scatter(vertices: np.ndarray):
    if not (vertices.ndim == 2 and vertices.shape[1] == 3):
        raise ValueError("vertices must be 3D")
    plt.style.use('_mpl-gallery')

    xs = vertices[:, 0]
    ys = vertices[:, 1]
    zs = vertices[:, 2]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(xs, ys, zs)

    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    plt.show()


def draw_content():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for key in ["missile", "fy", "smoke bomb", "smoke"]:
        if len(content.data[key]) <= 0:
            continue
        arr = np.array(content.data[key])
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2])
        ax.scatter(*arr[0], label=key)
    # ax.scatter(*real_target_top, label="real target")
    plt.legend()
    plt.show()

