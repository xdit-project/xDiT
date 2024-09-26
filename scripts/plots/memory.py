import matplotlib.pyplot as plt
import numpy as np

import scienceplots

# 设置图像大小

with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(10, 1.7))

    # 横坐标的标签
    image_sizes = ["1024×1024", "2048×2048", "4096×4096", "8192×8192"]
    lenged = [
        "Original",
        "Tensor Parallel",
        "PipeFusion",
        "DistriFusion",
        "Seq Parallel (Ulysses)",
    ]

    # 假设我们有5组数据，每组数据对应一个image size
    # 这里用随机数模拟内存大小
    memory_sizes = [
        [12.28, 12.92, 15.48, 25.70],  # Original
        [11.56, 11.89, 13.16, 18.30],  # Tensor Parallel
        [11.76, 13.23, 19.14, 42.78],  # PipeFusion
        [14.25, 20.42, 45.07, 0],  # DistriFusion
        [14.23, 17.64, 31.26, 0],  # Ulysses Seq Parallel
    ]

    # 计算每个image size的x轴位置
    bar_width = 0.325  # 柱子的宽度
    index = np.arange(4) * 2  # 因为有5个柱子，所以这里应该是5个位置

    # colors = ['#6495ED', '#7FFF00', '#FFA500', '#D8BFD8', '#00FFFF']
    # colors = ['#feb24c', '#f03b20', '#ffeda0', '#43a2ca', '#a8ddb5', '#e0f3db']
    colors = ["#009392", "#39B185", "#9CCB86", "#E9E29C", "#EEB479", "#E88471"]
    colors = ["#009392", "#9CCB86", "#E9E29C", "#EEB479", "#E88471"]

    # 绘制每个image size对应的5个柱子
    for i, size in enumerate(lenged):
        plt.bar(
            index + bar_width * (i - 1),
            memory_sizes[i],
            width=bar_width - 0.04,
            label=size,
            color=colors[i],
            edgecolor="black",  # , alpha=0.1
        )

    for i, t in enumerate(memory_sizes):
        for j, test in enumerate(t):
            if test == 0:
                continue
            plt.text(
                index[j] + bar_width * (i - 1),
                test,
                str(test),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    plt.ylim(0, 55)

    plt.bar(
        index[3] + bar_width * 3,
        60,
        width=bar_width - 0.02,
        color=colors[4],
        edgecolor="black",
        alpha=0.1,
    )
    plt.bar(
        index[3] + bar_width * 2,
        60,
        width=bar_width - 0.02,
        color=colors[3],
        edgecolor="black",
        alpha=0.1,
    )
    plt.text(
        index[3] + bar_width * 2.5,
        10,
        "OOM",
        ha="center",
        va="baseline",
        color="red",
        fontsize=16,
    )

    # plt.axhline(y=3, color='gray', linestyle='dashed')

    # 设置横坐标和纵坐标的标签
    plt.xlabel("Image Size", fontsize=12)
    plt.ylabel("Memory (GB)", fontsize=12)

    # 设置图例
    plt.legend(fontsize=10, ncols=2)

    # 设置横坐标的标签，这里我们使用range(5)来匹配5个柱子
    plt.xticks(index + bar_width, [image for image in image_sizes])

    # 显示图形
    # plt.show()
    plt.savefig("memory.png", dpi=500, bbox_inches="tight")
