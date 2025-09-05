import matplotlib.pyplot as plt

from config import *

# \phi
def get_missile_position(m_id: int, t_now: float):
    start = missiles[m_id]
    direction = fake_target - start
    velocity = missile_speed / norm(direction) * direction
    return start + velocity * t_now

# \mu
def get_smoke_position(fy_id: int, t_drop: float, t_detonate: float, t_now: float):
    # [0, t_drop, t_detonate, t_detonate + smoke_period, infty)
    if not (t_detonate < t_now < t_detonate + smoke_period):
        return None
    fy_position = fys[fy_id]
    v, theta = fy_v_theta[fy_id]
    fy_velocity = np.array([np.cos(theta), np.sin(theta), 0]) * v

    detonate_position = fy_position + fy_velocity * t_detonate
    detonate_position[2] -= 0.5 * 9.8 * (t_detonate - t_drop) ** 2

    smoke_position = detonate_position + smoke_velocity * (t_now - t_detonate)
    return smoke_position

# g
def get_countermeasure(missile_position: np.ndarray, smoke_position: np.ndarray):
    if not (missile_position.ndim == 1 and missile_position.shape[0] == 3):
        raise ValueError('missile_position must be 3D')
    if not (smoke_position.ndim == 1 and smoke_position.shape[0] == 3):
        raise ValueError('smoke_position must be 3D')

    vec_MS = smoke_position - missile_position
    if norm(vec_MS) <= smoke_radius:
        return True
    cone_theta = np.arcsin(smoke_radius / norm(vec_MS))
    for point in real_target_samples:
        vec_MP = point - missile_position
        point_theta = np.arccos(np.dot(vec_MS, vec_MP) / (norm(vec_MS) * norm(vec_MP)))
        if cone_theta < point_theta:
            return False
    return True

# \sigma
def get_time_interval(m_id, fy_id, t_drop, t_detonate, res = 0.1):
    result = []

    if isinstance(m_id, np.ndarray) and len(m_id) > 1:
        for m in m_id:
            result.append(get_time_interval(m, fy_id, t_drop, t_detonate))
        return np.array(result)

    if isinstance(fy_id, np.ndarray) and len(fy_id) >= 1:
        for fy in fy_id:
            result.append(get_time_interval(m_id, fy, t_drop, t_detonate))
        return np.array(result)

    a = b = -1
    for t in np.arange(t_detonate, t_detonate + smoke_period, res):
        missile_position = get_missile_position(m_id, t)
        smoke_position = get_smoke_position(fy_id, t_drop, t_detonate, t)
        if smoke_position is None:
            continue
        countermeasure = get_countermeasure(missile_position, smoke_position)
        if countermeasure and a < 0:
            a = t
        if not countermeasure and b < 0 <= a:
            b = t
    if b < a:
        b = a + smoke_period
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


if __name__ == '__main__':
    pass