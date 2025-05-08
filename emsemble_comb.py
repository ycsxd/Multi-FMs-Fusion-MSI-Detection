import argparse
import glob
import json
import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import csv
import math
import itertools

from utils.yaml_utils import read_yaml
from utils.loop_utils import cal_scores
from utils.wsi_utils import WSI_Dataset, CDP_MIL_WSI_Dataset, LONG_MIL_WSI_Dataset
from utils.model_utils import get_model_from_yaml, get_criterion

warnings.filterwarnings('ignore')


def softmax(x, axis=None):
    """
    自定义实现softmax函数
    
    参数：
      x: 输入数组
      axis: 计算softmax的轴
    
    返回：
      softmax后的数组
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def ensemble_predictions(input_tensors, models, method="average", labels=None):
    """
    对于每个样本，使用每个模型各自对应的输入进行预测，然后集成各模型的预测结果。

    参数：
      input_tensors: 一个列表，每个元素为对应模型的输入张量（batch_size=1）
      models: 包含多个已训练模型的列表，顺序与 input_tensors 对应
      method: 集成方法，"average" 表示概率平均，"voting" 表示投票多数模型的平均预测概率
      labels: 一个列表，每个元素为对应模型的真实标签张量（可选）

    返回：
      集成后的预测概率，形状为 (num_classes,)
    """
    preds = []
    for idx, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            inp = input_tensors[idx]
            lb = labels[idx] if labels is not None else None
            forward_return = model(inp, label=lb)
            logits = forward_return['logits'].squeeze(0)
            pred_prob = torch.softmax(logits, 0)
            pred_prob = pred_prob.to('cuda:0')
            preds.append(pred_prob)
    preds_tensor = torch.stack(preds)  # 形状为 (num_models, num_classes)
    
    if method == "average":
        avg_preds = torch.mean(preds_tensor, dim=0)
        avg_preds = torch.softmax(avg_preds, dim=0)
        return avg_preds
    elif method == "voting":
        _, pred_each = torch.max(preds_tensor, dim=1)  # 形状为 (num_models,)
        counts = torch.bincount(pred_each)
        majority_label = torch.argmax(counts)
        majority_indices = (pred_each == majority_label).nonzero(as_tuple=False).squeeze(1)
        selected_probs = preds_tensor[majority_indices]
        avg_prob = torch.mean(selected_probs, dim=0)
        return avg_prob
    else:
        raise ValueError("未知的集成方法，请选择 'average' 或 'voting'.")


def ensemble_test(args):
    """
    使用 folds_dir 中每个 fold 子文件夹中的模型、配置文件及 CSV 文件进行集成预测。
    如果指定了 --search_combinations，则对所有子文件夹进行两两组合探索，寻找最佳两模型集成组合。
    """
    # 使用 os.walk 递归查找所有 fold 文件夹
    fold_dirs = []
    for root, dirs, _ in os.walk(args.folds_dir):
        for d in dirs:
            fold_dirs.append(os.path.join(root, d))
    fold_dirs = sorted(fold_dirs)
    if not fold_dirs:
        raise RuntimeError("在指定的 folds_dir 中未找到 fold 文件夹。")
    
    models = []
    dataloaders = []
    devices = []
    num_classes = None
    fold_names = []  # 保存每个 fold 文件夹的标识
    # 遍历每个 fold 文件夹，并加载对应的模型、YAML 以及 CSV 文件
    for fold in fold_dirs:
        fold_path = fold
        fold_names.append(os.path.basename(fold_path))
        # 获取 YAML 配置文件
        yaml_files = glob.glob(os.path.join(fold_path, "*.yaml")) + glob.glob(os.path.join(fold_path, "*.yml"))
        if len(yaml_files) != 1:
            raise RuntimeError(f"{fold_path} 中必须有且只有一个 YAML 配置文件。")
        yaml_file = yaml_files[0]
        yaml_args = read_yaml(yaml_file)
        
        # 获取 CSV 文件
        csv_files = glob.glob(os.path.join(fold_path, "*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(f"{fold_path} 中必须有且只有一个 CSV 文件。")
        csv_file = csv_files[0]
        
        # 获取模型权重文件，优先寻找 "Best_EPOCH_*.pth"，否则匹配任意 .pth 文件
        weight_files = glob.glob(os.path.join(fold_path, "Best_EPOCH_*.pth"))
        if len(weight_files) == 0:
            weight_files = glob.glob(os.path.join(fold_path, "*.pth"))
        if len(weight_files) != 1:
            raise RuntimeError(f"{fold_path} 中必须有且只有一个模型权重文件。")
        model_weight_path = weight_files[0]
        print(f"{fold} 使用模型权重文件: {model_weight_path}")
        
        # 加载 YAML 配置和模型
        model_name = yaml_args.General.MODEL_NAME
        if num_classes is None:
            num_classes = yaml_args.General.num_classes
        device = torch.device(f'cuda:{yaml_args.General.device}' if torch.cuda.is_available() else 'cpu')
        devices.append(device)
        
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))
        models.append(mil_model)
        
        # 构建测试数据集
        if model_name == 'CDP_MIL':
            test_ds = CDP_MIL_WSI_Dataset(csv_file, yaml_args.Dataset.BeyesGuassian_pt_dir, 'test')
        elif model_name == 'LONG_MIL':
            test_ds = LONG_MIL_WSI_Dataset(csv_file, yaml_args.Dataset.h5_csv_path, 'test')
        else:
            test_ds = WSI_Dataset(csv_file, 'test')
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
        dataloaders.append(test_dataloader)
    
    if not args.search_combinations:
        # 原有集成预测逻辑
        dataset_lengths = [len(dl.dataset) for dl in dataloaders]
        if len(set(dataset_lengths)) != 1:
            raise RuntimeError("各个 fold 的 CSV 数据集长度不一致，无法进行集成预测。")
        n_samples = dataset_lengths[0]
        
        ensemble_preds = []
        all_labels = []
        iter_list = [iter(dl) for dl in dataloaders]
        for sample_tuple in zip(*iter_list):
            input_tensors = []
            labels_sample = []
            for i, (inp, lb) in enumerate(sample_tuple):
                input_tensor = inp.to(devices[i]).float()
                label = lb.to(devices[i]).long()
                input_tensors.append(input_tensor)
                labels_sample.append(label)
            for lb in labels_sample[1:]:
                if not torch.equal(labels_sample[0].cpu(), lb.cpu()):
                    raise RuntimeError("不同 fold 中对应样本的标签不一致。")
            avg_prob = ensemble_predictions(input_tensors, models, method=args.ensemble_method, labels=labels_sample)
            ensemble_preds.append(avg_prob.cpu().numpy())
            all_labels.append(labels_sample[0].cpu().numpy())
        
        all_preds = np.stack(ensemble_preds, axis=0)
        all_labels = np.concatenate(all_labels)
        
        test_metrics = cal_scores(all_preds, all_labels, num_classes)
        
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        
        # 新增代码开始：进行 bootstrap 实验以获得95%置信区间
        n_bootstrap = 1000
        bootstrap_results = {}
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_preds = all_preds[indices]
            boot_labels = all_labels[indices]
            boot_metrics = cal_scores(boot_preds, boot_labels, num_classes)
            for key, value in boot_metrics.items():
                if key not in bootstrap_results:
                    bootstrap_results[key] = []
                bootstrap_results[key].append(value)
        ci_results = {}
        for key, values in bootstrap_results.items():
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            ci_results[key] = (lower, upper)
        print(f'{FAIL}Bootstrap 95% CI: {ENDC}{ci_results}\n')
        # 新增代码结束
        
        # 保存集成测试结果日志
        ensemble_result_dir = args.test_result_dir if args.test_result_dir else os.path.join(args.folds_dir, f'{args.ensemble_method}_ensemble_test_result')
        os.makedirs(ensemble_result_dir, exist_ok=True)
        ensemble_log_path = os.path.join(ensemble_result_dir, 'Ensemble_Test_Log.txt')
        log_to_save = {'test_metrics': test_metrics,
                       'bootstrap_ci': ci_results}
        with open(ensemble_log_path, 'w') as f:
            f.write(str(log_to_save))
        print(f"集成测试日志保存在: {ensemble_log_path}")
        
        # 绘制混淆矩阵图
        preds_labels = np.argmax(all_preds, axis=1)
        cm = confusion_matrix(all_labels, preds_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        cm_filename = os.path.join(ensemble_result_dir, 'confusion_matrix.png')
        plt.savefig(cm_filename)
        print(f"混淆矩阵图已保存在: {cm_filename}")
        plt.close()

        # 修改后的 ROC 曲线绘制（针对二分类任务）
        if num_classes == 2:
            # all_preds = np.array(all_preds)
            # # 如果预测概率是一维或只有一列，则转换为两列形式
            # if all_preds.ndim == 1 or (all_preds.ndim == 2 and all_preds.shape[1] == 1):
            #     all_preds = all_preds.reshape(-1, 1)
            #     all_preds = np.hstack((1 - all_preds, all_preds))
            #     print("已将一维预测概率转换为两列形式用于二分类 ROC 计算。")
            # 直接计算正类（第二列）的 ROC 曲线
            fpr, tpr, thresholds = roc_curve(all_labels, all_preds[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label='ROC curve (area = {0:0.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            roc_filename = os.path.join(ensemble_result_dir, 'roc_curve.png')
            plt.savefig(roc_filename)
            print(f"ROC 曲线图已保存在: {roc_filename}")
            plt.close()

            precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_preds[:, 1])
            auprc = auc(recall, precision)
            plt.figure()
            plt.plot(recall, precision, color='green', lw=2, label='AUPRC curve (area = {0:0.2f})'.format(auprc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='upper right')
            prc_filename = os.path.join(ensemble_result_dir, 'prc_curve.png')
            plt.savefig(prc_filename)
            print(f"AUPRC 曲线图已保存在: {prc_filename}")
            plt.close()
        else:
            # 如果任务不是二分类任务，则提示信息，或根据需要扩展多分类 ROC 实现
            print("当前任务不是二分类任务，未实现多分类 ROC 曲线绘制。")
        return
    
    # --- 进入组合探索模式 ---
    print("正在探索不同子文件夹的两两组合集成效果...")
    predictions_by_fold = []
    all_labels = None
    # 分别计算每个模型对测试集的预测结果
    for i, (model, dataloader, device) in enumerate(zip(models, dataloaders, devices)):
        model.eval()
        fold_predictions = []
        for inp, lb in dataloader:
            inp = inp.to(device).float()
            lb = lb.to(device)
            with torch.no_grad():
                forward_return = model(inp, label=lb)
                logits = forward_return['logits'].squeeze(0)
                pred_prob = torch.softmax(logits, 0)
            fold_predictions.append(pred_prob.cpu().numpy())
            if i == 0:
                if all_labels is None:
                    all_labels = []
                all_labels.append(lb.cpu().numpy())
        fold_predictions = np.stack(fold_predictions, axis=0)
        predictions_by_fold.append(fold_predictions)
    all_labels = np.concatenate(all_labels)
    n_samples = len(all_labels)
    
    results = []
    n_models = len(predictions_by_fold)
    # 修改为只计算两两组合的数量
    total_combinations = math.comb(n_models, 2) if n_models >= 2 else 0
    counter = 0
    threshold = 10 if total_combinations >= 50 else 1
    
    # 修改循环，只考虑两两组合(r=2)
    if n_models >= 2:
        for combo in itertools.combinations(range(n_models), 2):
            counter += 1
            if counter % threshold == 0 or counter == total_combinations:
                print(f"已处理 {counter} / {total_combinations} 个组合")
            selected_preds = np.array([predictions_by_fold[idx] for idx in combo])
    
            if args.ensemble_method == 'average':
                avg_preds = np.mean(selected_preds, axis=0)
                ensemble_preds = softmax(avg_preds, axis=1)
            elif args.ensemble_method == 'voting':
                ensemble_preds = []
                for j in range(n_samples):
                    sample_preds = selected_preds[:, j, :]
                    pred_labels = np.argmax(sample_preds, axis=1)
                    counts = np.bincount(pred_labels)
                    majority_label = np.argmax(counts)
                    indices = np.where(pred_labels == majority_label)[0]
                    if len(indices) > 0:
                        avg_prob = np.mean(sample_preds[indices, :], axis=0)
                        exp_prob = np.exp(avg_prob - np.max(avg_prob))
                        norm_prob = exp_prob / np.sum(exp_prob)
                    else:
                        norm_prob = sample_preds[0]
                    ensemble_preds.append(norm_prob)
                ensemble_preds = np.stack(ensemble_preds, axis=0)
            else:
                raise ValueError("未知的集成方法，请选择 'average' 或 'voting'.")
            
            test_metrics = cal_scores(ensemble_preds, all_labels, num_classes)
            roc_auc = test_metrics.get("auc", None)
            results.append({
                "combination": combo,
                "fold_names": [fold_names[idx] for idx in combo],
                "test_metrics": test_metrics,
                "roc_auc": roc_auc,
            })

    # 添加边界条件处理
    if not results:
        print("没有找到任何有效组合（至少需要2个模型才能进行两两组合）。")
        return
        
    results_sorted = sorted(results, key=lambda x: x["roc_auc"] if x["roc_auc"] is not None else -1, reverse=True)
    best_result = results_sorted[0] if results_sorted else None
    print("----------------两两组合探索结果----------------")
    if best_result:
        print(f"最佳组合: 子文件夹索引 {best_result['combination']} (对应 {best_result['fold_names']}), 指标: {best_result['test_metrics']}")
    else:
        print("没有找到有效组合结果。")
    
    ensemble_result_dir = args.test_result_dir if args.test_result_dir else os.path.join(args.folds_dir, f'{args.ensemble_method}_pairwise_ensemble_results')
    os.makedirs(ensemble_result_dir, exist_ok=True)
    csv_file_path = os.path.join(ensemble_result_dir, 'Pairwise_Ensemble_Log.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Combination', 'ROC_AUC'])
        for res in results_sorted:
            combination_str = ','.join(res['fold_names'])
            roc_auc = res['roc_auc'] if res['roc_auc'] is not None else 'N/A'
            csv_writer.writerow([combination_str, roc_auc])
    print(f"两两组合测试结果已保存为CSV文件: {csv_file_path}")
    
    best_combo = best_result['combination']
    selected_preds = np.array([predictions_by_fold[idx] for idx in best_combo])
    if args.ensemble_method == 'average':
        avg_preds = np.mean(selected_preds, axis=0)
        ensemble_preds = softmax(avg_preds, axis=1)
    elif args.ensemble_method == 'voting':
        ensemble_preds = []
        for j in range(n_samples):
            sample_preds = selected_preds[:, j, :]
            pred_labels = np.argmax(sample_preds, axis=1)
            counts = np.bincount(pred_labels)
            majority_label = np.argmax(counts)
            indices = np.where(pred_labels == majority_label)[0]
            if len(indices) > 0:
                avg_prob = np.mean(sample_preds[indices, :], axis=0)
                exp_prob = np.exp(avg_prob - np.max(avg_prob))
                norm_prob = exp_prob / np.sum(exp_prob)
            else:
                norm_prob = sample_preds[0]
            ensemble_preds.append(norm_prob)
        ensemble_preds = np.stack(ensemble_preds, axis=0)
    preds_labels = np.argmax(ensemble_preds, axis=1)
    cm = confusion_matrix(all_labels, preds_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("最佳两两组合 混淆矩阵")
    cm_filename = os.path.join(ensemble_result_dir, 'best_pairwise_confusion_matrix.png')
    plt.savefig(cm_filename)
    print(f"最佳两两组合混淆矩阵图已保存在: {cm_filename}")
    plt.close()
    
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, ensemble_preds[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('最佳两两组合 ROC 曲线')
        plt.legend(loc="lower right")
        roc_filename = os.path.join(ensemble_result_dir, 'best_pairwise_roc_curve.png')
        plt.savefig(roc_filename)
        print(f"最佳两两组合 ROC 曲线图已保存在: {roc_filename}")
        plt.close()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds_dir', type=str, default='',
                        help='包含 fold 文件夹的目录，每个 fold 文件夹中包含 YAML、模型权重及 CSV 文件')
    parser.add_argument('--test_result_dir', type=str, default='',
                        help='测试结果保存目录；若不指定，则默认在 folds_dir 下生成相应结果文件夹')
    parser.add_argument('--ensemble_method', type=str, default='average',
                        help='集成方法，可选值为 "average" 或 "voting"')
    parser.add_argument('--search_combinations', action='store_true',
                        help='是否进入两两组合探索模式（仅考虑两个模型的组合）')
    args = parser.parse_args()
    
    ensemble_test(args)

