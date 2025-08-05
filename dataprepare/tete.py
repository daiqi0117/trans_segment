import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score

def compute_metrics(pred, gt, threshold=0.5):
    """
    计算AP, AP25, AP50, mCov, mWCov, mPre, mRec, mF1, oAcc, mAcc, mIoU。

    参数：
    pred: numpy array, shape (n, k), 预测的概率矩阵
    gt: numpy array, shape (n, k), 真实的标签矩阵（值为0或1）
    threshold: float, 将预测概率映射为二值的阈值，默认值为0.5

    返回：
    一个包含所有目标指标的字典
    """
    n, k = pred.shape

    # 将预测概率矩阵转为二值矩阵（默认阈值0.5）
    pred_binary = (pred >= threshold).astype(int)

    # AP: 对于每个类别计算平均精度，然后取平均值
    AP_per_class = []
    for i in range(k):
        try:
            ap = average_precision_score(gt[:, i], pred[:, i])
            AP_per_class.append(ap)
        except:
            AP_per_class.append(0)  # 如果某类没有正例或负例，设置AP为0
    AP = np.mean(AP_per_class)

    # AP25: 阈值0.25的二值化精度
    pred_binary_25 = (pred >= 0.25).astype(int)
    AP25 = np.mean([accuracy_score(gt[:, i], pred_binary_25[:, i]) for i in range(k)])

    # AP50: 阈值0.5的二值化精度
    AP50 = np.mean([accuracy_score(gt[:, i], pred_binary[:, i]) for i in range(k)])

    # Mean Coverage (mCov): 预测中有多少分类与真实标签匹配
    mCov = np.mean([np.sum((gt[i] & pred_binary[i]) > 0) > 0 for i in range(n)])

    # Mean Weighted Coverage (mWCov): mCov的加权实现
    mWCov = np.mean([
        np.sum(gt[i] & pred_binary[i]) / (np.sum(gt[i]) + 1e-8 if np.sum(gt[i]) > 0 else 1)
        for i in range(n)
    ])

    # mPre: 每个样本的平均precision
    mPre = precision_score(gt, pred_binary, average='samples', zero_division=0)

    # mRec: 每个样本的平均recall
    mRec = recall_score(gt, pred_binary, average='samples', zero_division=0)

    # mF1: 每个样本的平均F1-score
    mF1 = f1_score(gt, pred_binary, average='samples', zero_division=0)

    # oAcc: Overall Accuracy
    oAcc = np.mean(gt == pred_binary)

    # mAcc: 每个类的准确率的平均值
    mAcc = np.mean([accuracy_score(gt[:, i], pred_binary[:, i]) for i in range(k)])

    # mIoU: Mean Intersection over Union
    mIoU = np.mean([jaccard_score(gt[:, i], pred_binary[:, i]) for i in range(k)])

    # 返回结果
    return {
        "AP": AP,
        "AP25": AP25,
        "AP50": AP50,
        "mCov": mCov,
        "mWCov": mWCov,
        "mPre": mPre,
        "mRec": mRec,
        "mF1": mF1,
        "oAcc": oAcc,
        "mAcc": mAcc,
        "mIoU": mIoU
    }


# 示例例子
if __name__ == "__main__":
    # 生成随机预测矩阵和真实矩阵
    np.random.seed(42)
    n, k = 100, 5  # 100个点，5个类
    pred = np.random.rand(n, k)  # 随机生成预测概率矩阵 (n, k)
    gt = (np.random.rand(n, k) > 0.5).astype(int)  # 随机生成真值矩阵，值0或1

    # 计算指标
    metrics = compute_metrics(pred, gt)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")