import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# 设置图像大小
# plt.figure(figsize=(20, 2))

# 横坐标的标签
image_sizes = ["1024×1024", "2048×2048", "4096×4096", "8192×8192"]
memory_sizes = [
    [2.085, 16.22, 183.6, 2728],  # Original
    [4.915, 19.45, 105.7, 966.0],  # Tensor Parallel
    # [1.091, 7.029, 80.34, 1183], # PipeFusion (4-Step Warm-up)
    [0.866, 5.370, 60.15, 880.2],  # PipeFusion (1-Step Warm-up)
    [2.331, 9.622, 68.35, 0],  # DistriFusion
    [1.741, 7.932, 59.01, 0],  # Seq Parallel (Ulysses)
    [2.633, 11.05, 65.85, 0],  # Seq Parallel (Ring)
]


legend = [
    "Original (1 Device)",
    "Tensor Parallel",
    "PipeFusion",
    "DistriFusion",
    "Seq Parallel (Ulysses)",
    "Seq Parallel (Ring)",
]

# 假设我们有5组数据，每组数据对应一个image size
# 这里用随机数模拟内存大小


# 计算每个image size的x轴位置
bar_width = 0.14  # 柱子的宽度
index = np.arange(1)  # 因为有5个柱子，所以这里应该是5个位置
ncols = len(image_sizes)

with plt.style.context(["science", "ieee"]):

    fig, axes = plt.subplots(1, ncols)

    # axes[-1].xlabel('Image Size')
    # axes[-1].ylabel('Memory (GB)')

    # colors = ['#6495ED', '#7FFF00', '#FFA500', '#D8BFD8', '#00FFFF']
    # colors = ['#feb24c', '#f03b20', '#ffeda0', '#43a2ca', '#a8ddb5', '#e0f3db']
    colors = ["#009392", "#39B185", "#9CCB86", "#E9E29C", "#EEB479", "#E88471"]
    colors = ["#009392", "#9CCB86", "#E9E29C", "#EEB479", "#E88471", "#CF597E"]

    import matplotlib.patches as mpatches

    for i, t in enumerate(memory_sizes):
        for j, val in enumerate(t):
            axes[j].bar(
                bar_width * (i - 2.5),
                val,
                width=bar_width - 0.02,
                label=legend[i],
                color=colors[i],
                edgecolor="black",
            )
            # arrow = mpatches.Arrow(bar_width * (i - 3), memory_sizes[0][j], 0, val - memory_sizes[0][j], linestyle='dashed', width=0.01, color='black')
            # axes[j].add_patch(arrow)
            if val == 0:
                continue
            axes[j].text(
                bar_width * (i - 2.5),
                val,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for i in range(ncols):
        axes[i].set_xticks(index, labels=[image_sizes[i]], fontsize=14)
        axes[i].tick_params(axis="y", labelsize=13)
        # axes[i].set_xticklabels(image_sizes[i])
    axes[0].set_ylabel("Latency (s)", fontsize=20)

    # 绘制每个image size对应的5个柱子
    # for i, size in enumerate(lenged):
    #     plt.bar(index + bar_width * (i - 1), memory_sizes[i], width=bar_width - 0.04,
    #             label=size, color=colors[i], edgecolor='black' #, alpha=0.1
    #             )

    # for i, t in enumerate(memory_sizes):
    #     for j, test in enumerate(t):
    #         if test == 0:
    #             continue
    #         plt.text(index[j] + bar_width * (i - 1), test, str(test), ha='center', va='bottom', fontsize=7.5)

    # plt.ylim(0, 55)

    # plt.bar(index[3] + bar_width * 3, 60, width=bar_width - 0.02,
    #             color=colors[4], edgecolor='black', alpha=0.1
    #             )
    # plt.bar(index[3] + bar_width * 2, 60, width=bar_width - 0.02,
    #             color=colors[3], edgecolor='black', alpha=0.1
    #            )
    axes[-1].bar(
        bar_width * 0.5,
        3000,
        width=bar_width - 0.02,
        color=colors[-3],
        edgecolor="black",
        alpha=0.1,
    )
    axes[-1].bar(
        bar_width * 1.5,
        3000,
        width=bar_width - 0.02,
        color=colors[-2],
        edgecolor="black",
        alpha=0.1,
    )
    axes[-1].bar(
        bar_width * 2.5,
        3000,
        width=bar_width - 0.02,
        color=colors[-1],
        edgecolor="black",
        alpha=0.1,
    )
    axes[-1].text(
        bar_width * 1.5,
        500,
        "OOM",
        ha="center",
        va="baseline",
        color="red",
        fontsize=25,
    )
    axes[-1].set_ylim(0, 2900)

    axes[-2].set_ylim(0, 199)
    axes[-3].set_ylim(0, 21)
    axes[-4].set_ylim(0, 5.3)
    # plt.axhline(y=3, color='gray', linestyle='dashed')

    # 设置横坐标和纵坐标的标签

    # 设置图例

    # 设置横坐标的标签
    # plt.xticks(index + bar_width, [image for image in image_sizes])

    plt.legend(
        loc="upper left",
        frameon=False,
        ncols=7,
        bbox_to_anchor=(-3.75, 1.2, 0, 0),
        columnspacing=1,
        labelspacing=0,
        fontsize=14,
    )

    fig.set_figwidth(15)
    fig.set_figheight(3)

    # 显示图形
    # plt.show()
    plt.savefig("latency-A100-PCIe.png", dpi=500, bbox_inches="tight")
