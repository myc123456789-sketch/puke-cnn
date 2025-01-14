import os
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

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
    
    # 确保 target_names 包含所有类别名称
    num_classes = 53
    target_names = [f"class_{i}" for i in range(num_classes)]

    # 整体报表
    class_report = classification_report(
        label,
        predict,
        target_names=target_names
    )
    print(class_report)

    # 准确度
    accuracy = accuracy_score(label, predict)
    print("准确度：%.8f" % accuracy)

    # 精确度
    precision = precision_score(label, predict, average="macro")
    print("精确度：%.8f" % precision)

    # 召回率
    recall = recall_score(label, predict, average="macro")
    print("召回率：%.8f" % recall)

    # F1 分数
    f1 = f1_score(label, predict, average="macro")
    print("F1 分数：%.8f" % f1)


if __name__ == "__main__":
    report()