import os
from sklearn.metrics import *
import pandas as pd
from matplotlib import pyplot as plt

# plt中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径兼容处理
current_path = os.path.dirname(__file__)
excel_path = os.path.relpath(
    os.path.join(current_path, r"./metrics", "validation_metrics.xlsx")
)


def report():
    # 读取Excel数据

    excel_data = pd.read_excel(excel_path)
    label = excel_data["label"].values
    predict = excel_data["predict"].values
    # 整体报表
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
              27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]
    matrix = confusion_matrix(label, predict)
    print(matrix)
    # 下面的代码就是简单的plt绘制过程
    plt.matshow(matrix, cmap=plt.cm.Greens)
    # 显示颜色条
    plt.colorbar()
    # 显示具体的数字的过程
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.annotate(
                matrix[i, j],
                xy=(j, i),
                horizontalalignment="center",
                verticalalignment="center",
            )
    # 美化的东西
    plt.xlabel("Pred labels")
    plt.ylabel("True labels")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title("训练结果混淆矩阵视图")

    plt.show()


if __name__ == "__main__":
    report()
