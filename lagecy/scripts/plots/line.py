import matplotlib.pyplot as plt
import numpy as np

import scienceplots


# 设置图像大小
# plt.figure(figsize=(20, 2))

# 横坐标的标签
image_sizes = ['1024×1024', '4096×4096']
lenged = ['PipeFusion (PCIe)', 'PipeFusion (NVLink)', 'DistriFusion (PCIe)', 'DistriFusion (NVLink)']

# 假设我们有5组数据，每组数据对应一个image size
# 这里用随机数模拟内存大小
ori = [2.085, 183.6] # Original
memory_sizes = [
    [
        [1.235, 0.866], # PipeFusion (PCIe)
        [1.358, 0.9633, 0.969], # PipeFusion (NVLink)
        [1.946, 2.331], # DistriFusion (PCIe)
        [1.187, 0.795, 0.765] # DistriFusion (NVLink)
    ],
    [
        [101.7, 60.15],
        [101.0, 60.10, 42.79], # PipeFusion (NVLink)
        [98.29, 68.35], # DistriFusion (PCIe)
        [92.72, 48.49, 25.24], # DistriFusion (NVLink)
    ], 
]

# 计算每个image size的x轴位置
# bar_width = 0.14 # 柱子的宽度
# index = np.arange(1)  # 因为有5个柱子，所以这里应该是5个位置
ncols = len(image_sizes)

with plt.style.context(['science', 'ieee']):
    fig, axes = plt.subplots(1, ncols)


    # axes[-1].xlabel('Image Size')
    # axes[-1].ylabel('Memory (GB)')

    # colors = ['#6495ED', '#7FFF00', '#FFA500', '#D8BFD8', '#00FFFF']
    # colors = ['#feb24c', '#f03b20', '#ffeda0', '#43a2ca', '#a8ddb5', '#e0f3db']
    colors = ['#009392', '#39B185', '#9CCB86', '#E9E29C', '#EEB479', '#E88471']
    colors = ['#009392', '#9CCB86', '#E9E29C', '#f7f5dc', '#EEB479', '#E88471', '#CF597E']
    # colors = ['#e9e29c', '#beb880', '#959065', '#6e6a4c', '#494734']
    # colors = ['#e9e29c', '#c2cbcf', '#86b6ff']
    # colors = ['#e9e29c', '#ff99cc', '#86b6ff']
    colors = ['#045275', '#089099', '#7ccba2', '#fcde9c']

    for i, image in enumerate(image_sizes):
        for j, _ in enumerate(lenged):
            for k in range(len(memory_sizes[i][j])):
                if memory_sizes[i][j][k] == 0:
                    memory_sizes[i][j][k] = 1 
                else:
                    memory_sizes[i][j][k] = ori[i] / memory_sizes[i][j][k]
            

    for i, image in enumerate(image_sizes):

        for j, _ in enumerate(lenged):
            axes[i].plot(memory_sizes[i][j], label=lenged[j], color=colors[j], linewidth=2, marker='o')
    
        axes[i].set_xticks(np.arange(3), labels=['2', '4', '8'], fontsize=11)
        axes[i].set_xlabel(f'Devices for {image} Images', fontsize=11)
        axes[i].tick_params(axis='y', labelsize=13)

    axes[0].set_ylabel('Speed Up', fontsize=15)

    plt.legend(loc="upper left", frameon=False, ncols=2, 
                bbox_to_anchor=(-1.15, 1.3, 0, 0), fontsize=13)
    fig.set_figwidth(7)
    fig.set_figheight(3)

    # 显示图形
    # plt.show()
    plt.savefig('line.png', dpi=500, bbox_inches='tight')