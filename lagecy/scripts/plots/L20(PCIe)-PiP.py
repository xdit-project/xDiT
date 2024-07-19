import matplotlib.pyplot as plt
import numpy as np

import scienceplots


# 设置图像大小
# plt.figure(figsize=(20, 2))

# 横坐标的标签
image_sizes = ['1024×1024', '2048×2048', '4096×4096']
lenged = ['2 Devices', '4 Devices', '8 Devices']

# 假设我们有5组数据，每组数据对应一个image size
# 这里用随机数模拟内存大小
memory_sizes = [
    [2.085, 16.22, 183.6, 2728], # Original
    [4.915, 19.45, 105.7, 966.0], # Tensor Parallel
    [1.091, 7.029, 80.34, 1183], # PipeFusion (4-Step Warm-up)
    [0.866, 5.370, 60.15, 880.2], # PipeFusion (1-Step Warm-up)
    [2.331, 9.622, 68.35, 0], # DistriFusion
    [1.741, 7.932, 59.01, 0], # Seq Parallel (Ulysses)
    [2.633, 11.05, 65.85, 0], # Seq Parallel (Ring)
]

memory_sizes = [
    [1.96122, 2.10357, 2.4689, 3.21097, 6.31111],
    [14.62203, 14.83501, 15.13784, 15.5393, 19.20736],
    [172.1636, 169.45748, 170.96689, 177.3009217, 179.63923],
    [2.34505, 1.2355, 1.41505, 1.88827, 3.56416],
    [16.29876, 8.82932, 9.00101, 9.19651, 11.12511],
    [190.6757, 101.81423, 101.03711, 104.20725, 105.4911],
    [3.12597, 1.98875, 1.54057, 1.35587, 2.42538], # 1k 8 Devices
    [14.40754, 8.78011, 6.34505, 6.48518, 7.65064],
    [167.39098, 100.44529, 71.13361, 72.90846, 73.57951],
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
    colors = ['#045275', '#089099', '#7ccba2']

    for i, image in enumerate(image_sizes):

        for j, _ in enumerate(lenged):
            axes[i].plot(memory_sizes[j * 3 + i], label=lenged[j], color=colors[j], linewidth=2, marker='o')
    
        axes[i].set_xticks(np.arange(5), labels=['2', '4', '8', '16', '32'], fontsize=11)
        axes[i].set_xlabel(f'Patch Number for {image} Images', fontsize=11)
        axes[i].tick_params(axis='y', labelsize=13)

    axes[0].set_ylabel('Latency (s)', fontsize=17)

    plt.legend(loc="upper left", frameon=False, ncols=3, 
                bbox_to_anchor=(-2.15, 1.24, 0, 0), fontsize=20)
    fig.set_figwidth(11)
    fig.set_figheight(3)

    # 显示图形
    # plt.show()
    plt.savefig('latency-L20-PiP.png', dpi=500, bbox_inches='tight')