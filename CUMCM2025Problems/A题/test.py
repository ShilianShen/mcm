import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 创建动画
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10))
    return line,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

# 保存为文件查看
ani.save('animation.gif', writer='pillow', fps=20)
print("动画已保存为 animation.gif")