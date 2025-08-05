import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import random


def compute_semantic_metrics_torch(results, num_classes):
    """
    计算语义分割指标：oAcc, mAcc, mIoU, 每类的 Acc 和 IoU（支持 PyTorch Tensor 格式）。

    :param results: 一个列表，每个元素是 [gt_sem, gt_inst, pred_sem, pred_inst]，每个都是 (N, 1) 的 Tensor
    :param num_classes: 语义类别的总数
    :return: oAcc, mAcc, mIoU, 每类的 Acc 和 IoU
    """
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # 构建混淆矩阵
    for gt_sem, _, pred_sem, _ in results:
        gt_sem = gt_sem.view(-1).cpu().numpy()
        pred_sem = pred_sem.view(-1).cpu().numpy()
        conf_matrix += torch.tensor(
            confusion_matrix(gt_sem, pred_sem, labels=list(range(num_classes)))
        )

        # Overall Accuracy
    oAcc = torch.sum(torch.diag(conf_matrix)) / torch.sum(conf_matrix)

    # Per-class Accuracy
    acc_per_class = torch.diag(conf_matrix).float() / (torch.sum(conf_matrix, dim=1) + 1e-6)
    mAcc = torch.mean(acc_per_class)

    # IoU per class
    iou_per_class = torch.diag(conf_matrix).float() / (
            torch.sum(conf_matrix, dim=1) + torch.sum(conf_matrix, dim=0) - torch.diag(conf_matrix) + 1e-6
    )
    mIoU = torch.mean(iou_per_class)

    return oAcc.item(), mAcc.item(), mIoU.item(), acc_per_class.numpy(), iou_per_class.numpy()


def compute_instance_metrics_torch(results):
    """
    计算实例分割指标：MCov, MWCov, mPre, mRec, F1-score（支持 PyTorch Tensor 格式）。

    :param results: 一个列表，每个元素是 [gt_sem, gt_inst, pred_sem, pred_inst]，每个都是 (N, 1) 的 Tensor
    :return: MCov, MWCov, mPre, mRec, F1-score
    """

    def assign_instances(true_instances, pred_instances):
        """
        为真值实例和预测实例计算 IoU 并进行匹配。

        :param true_instances: 一个包含每个真值实例 mask 的列表（Tensor 格式）
        :param pred_instances: 一个包含每个预测实例 mask 的列表（Tensor 格式）
        :return: IoU 矩阵 (num_true_instances, num_pred_instances)
        """
        num_true = len(true_instances)
        num_pred = len(pred_instances)
        iou_matrix = torch.zeros((num_true, num_pred))

        for i, true_mask in enumerate(true_instances):
            for j, pred_mask in enumerate(pred_instances):
                intersection = torch.sum(true_mask & pred_mask).item()
                union = torch.sum(true_mask | pred_mask).item() + 1e-6
                iou_matrix[i, j] = intersection / union

        return iou_matrix

    total_gt_instances = 0
    total_pred_instances = 0
    total_matched = 0
    total_weighted_matched = 0

    precision_list = []
    recall_list = []

    for gt_sem, gt_inst, pred_sem, pred_inst in results:
        gt_inst = gt_inst.view(-1)
        pred_inst = pred_inst.view(-1)

        # 获取唯一实例 ID（排除 -1）
        gt_instance_ids = gt_inst.unique().tolist()
        pred_instance_ids = pred_inst.unique().tolist()
        if -1 in gt_instance_ids:
            gt_instance_ids.remove(-1)
        if -1 in pred_instance_ids:
            pred_instance_ids.remove(-1)

            # 构建实例 mask 列表
        gt_instances = [(gt_inst == inst_id) for inst_id in gt_instance_ids]
        pred_instances = [(pred_inst == inst_id) for inst_id in pred_instance_ids]

        # 计算 IoU 矩阵
        iou_matrix = assign_instances(gt_instances, pred_instances)

        # 按阈值 0.5 匹配实例
        matched = (iou_matrix > 0.5).sum().item()
        weight_matched = iou_matrix.max(dim=1)[0].sum().item()  # 按真值实例加权

        # Precision and Recall
        precision = matched / (len(pred_instances) + 1e-6)
        recall = matched / (len(gt_instances) + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)

        total_gt_instances += len(gt_instances)
        total_pred_instances += len(pred_instances)
        total_matched += matched
        total_weighted_matched += weight_matched

        # MCov 和 MWCov
    MCov = total_matched / (total_gt_instances + 1e-6)
    MWCov = total_weighted_matched / (total_gt_instances + 1e-6)

    # 平均 Precision 和 Recall
    mPre = np.mean(precision_list)
    mRec = np.mean(recall_list)

    # F1-score
    F1 = 2 * mPre * mRec / (mPre + mRec + 1e-6)

    return MCov, MWCov, mPre, mRec, F1


def evaluate_point_cloud_segmentation_torch(results, num_classes):
    """
    评估点云的语义和实例分割指标（支持 PyTorch Tensor）。

    :param results: 一个列表，每个元素是 [gt_sem, gt_inst, pred_sem, pred_inst]
    :param num_classes: 语义类别数量
    :return: 一个包含所有指标的字典
    """
    # 计算语义分割指标
    semantic_metrics = compute_semantic_metrics_torch(results, num_classes)

    # 计算实例分割指标
    instance_metrics = compute_instance_metrics_torch(results)

    metrics = {
        "Semantic Metrics": {
            "oAcc": semantic_metrics[0],
            "mAcc": semantic_metrics[1],
            "mIoU": semantic_metrics[2],
            "Acc Per Class": semantic_metrics[3],
            "IoU Per Class": semantic_metrics[4],
        },
        "Instance Metrics": {
            "MCov": instance_metrics[0],
            "MWCov": instance_metrics[1],
            "mPre": instance_metrics[2],
            "mRec": instance_metrics[3],
            "F1": instance_metrics[4],
        }
    }
    return metrics





def generate_point_cloud_data(num_clouds=50, num_points=6000, num_sem_classes=9, num_instances=400):
    """
    生成模拟的点云数据，共 `num_clouds` 个点云，每个点云 `num_points` 点。
    每个点云有 `num_sem_classes` 个语义类别，最多 `num_instances` 个实例。

    :param num_clouds: 点云数量
    :param num_points: 每个点云的点数
    :param num_sem_classes: 语义类别数
    :param num_instances: 每个点云的实例数
    :return: 一个列表，每个元素是 [gt_sem, gt_inst, pred_sem, pred_inst]
    """
    results = []
    for _ in range(num_clouds):
        # 生成真值语义标签和实例标签
        gt_sem = torch.randint(0, num_sem_classes, (num_points, 1))  # N×1 的语义标签
        gt_inst = torch.randint(0, num_instances, (num_points, 1))  # N×1 的实例标签

        # 模拟预测的语义标签与实例标签
        # 预测语义标签
        pred_sem = gt_sem.clone()  # 初始拷贝真值语义标签
        # 加噪声：随机改变部分语义标签
        for i in range(int(num_points * 0.1)):  # 修改 10% 的点
            idx = random.randint(0, num_points - 1)
            new_label = random.randint(0, num_sem_classes - 1)
            pred_sem[idx] = new_label

            # 预测实例标签
        pred_inst = gt_inst.clone()  # 初始拷贝真值实例标签
        # 加噪声：随机改变部分实例标签
        for i in range(int(num_points * 0.1)):  # 修改 10% 的点
            idx = random.randint(0, num_points - 1)
            new_label = random.randint(0, num_instances - 1)
            pred_inst[idx] = new_label

            # 将生成的数据添加到结果中
        results.append([gt_sem, gt_inst, pred_sem, pred_inst])

    return results


num_clouds = 50       # 点云数量
num_points = 6000     # 每个点云的点数
num_sem_classes = 9   # 语义类别数
num_instances = 400   # 最多实例数

# 调用函数生成数据
results = generate_point_cloud_data(num_clouds, num_points, num_sem_classes, num_instances)

# 假设语义类别数为 9
metrics = evaluate_point_cloud_segmentation_torch(results, num_sem_classes)
print(metrics)