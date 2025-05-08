import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop,ds_val_loop,dtfd_val_loop,cal_scores
import warnings
from utils.wsi_utils import WSI_Dataset,CDP_MIL_WSI_Dataset,LONG_MIL_WSI_Dataset
import torch
import shutil
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix

from utils.model_utils import get_model_from_yaml,get_criterion
from process.CLAM_MB_MIL.process_clam_mb_mil_ensemble import ensemble_val_loop
from process.CLAM_MB_MIL.process_clam_mb_mil_fusion import fusion_val_loop
warnings.filterwarnings('ignore')
import glob
import json
import numpy as np
def cal_scores(logits, labels, num_classes=2):
    if num_classes != 2:
        raise ValueError("该函数现在仅支持2分类任务")
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    probs = torch.softmax(logits, dim=1)
    auc_score = roc_auc_score(labels.numpy(), probs[:,1].numpy())
    binary_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='binary')
    binary_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='binary')
    binary_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='binary')
    baccuracy = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())
    quadratic_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')
    linear_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='linear')
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())
    metrics = {
        'acc': accuracy,
        'bacc': baccuracy,
        'auc': auc_score,
        'f1': binary_f1,
        'recall': binary_recall,
        'precision': binary_precision,
        'quadratic_kappa': quadratic_kappa,
        'linear_kappa': linear_kappa,
        'confusion_mat': confusion_mat
    }
    return metrics

def test(args):
    yaml_path = args.yaml_path
    print(f"MIL-model-yaml path: {yaml_path}")
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    # CDP_MIL and LONG_MIL models have different dataset pipeline
    if model_name == 'CDP_MIL':
        test_ds = CDP_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.BeyesGuassian_pt_dir,'test')
    elif model_name == 'LONG_MIL':
        LONG_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
    test_ds = WSI_Dataset(test_dataset_csv,'test')
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False)
    model_weight_path = args.model_weight_path
    print(f"Model weight path: {model_weight_path}")
    device = torch.device(f'cuda:{yaml_args.General.device}')
    criterion = get_criterion(yaml_args.Model.criterion)
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier,attention,dimReduction,attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(model_weight_path,weights_only=True)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier,attention,dimReduction,attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path,weights_only=True))

    
    # CLAM_SB_MIL and CLAM_MB_MIL models have different val loop pipeline (has instance loss)
    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
    elif yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
        test_loss,test_metrics =  ds_val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        test_loss,test_metrics =  dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,yaml_args.Model.num_Group,yaml_args.Model.grad_clipping,yaml_args.Model.distill,yaml_args.Model.total_instance)
    else:
        test_loss,test_metrics =  val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print('----------------INFO----------------\n')
    print(f'{FAIL}Test_Loss:{ENDC}{test_loss}\n')
    print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
    
    test_log_dir = args.test_log_dir
    os.makedirs(test_log_dir,exist_ok=True)
    new_yaml_path = os.path.join(test_log_dir,f'Test_{model_name}.yaml')
    shutil.copyfile(yaml_path,new_yaml_path)
    new_test_dataset_csv_path = os.path.join(test_log_dir,f'Test_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
    shutil.copyfile(test_dataset_csv,new_test_dataset_csv_path)
    test_log_path = os.path.join(test_log_dir,f'Test_Log_{model_name}.txt')
    log_to_save = {'test_loss':test_loss,'test_metrics':test_metrics}
    with open(test_log_path,'w') as f:
        f.write(str(log_to_save))
    print(f"Test log saved at: {test_log_path}")
     
def test_all_folds(args):
    """
    一次性评测所有折交叉验证的fold文件夹。

    输入参数:
      args.folds_dir: 包含fold1, fold2, ...等子文件夹的目录，每个子文件夹包含该fold的模型权重 (.pth 文件) 和 YAML 配置文件；
      args.test_dataset_csv: 测试数据集 CSV 文件路径。
      args.test_result_dir: 可选，测试结果存储的目录。如果未提供，则默认生成在 args.folds_dir 下的 'test_result' 文件夹中。

    对于每个fold，将先在fold目录下查找 YAML 文件（必须唯一，否则报错），
    然后按照 test 函数的逻辑评测，并保存对应 fold 的日志及拷贝 YAML、CSV 文件以及 best model pth 文件；
    最后生成一个 merge_5_fold_metrics.json 文件，汇总各折的测试指标及其均值和标准差。
    """

    # 如果未提供 test_result_dir，则默认为 folds_dir 下的 test_result 文件夹
    if not args.test_result_dir:
        test_result_dir = os.path.join(args.folds_dir, 'test_result')
    else:
        test_result_dir = args.test_result_dir
    os.makedirs(test_result_dir, exist_ok=True)

    # 收集所有 fold 的测试指标
    all_fold_metrics = {}

    # 遍历文件夹名以 "fold" 开头的子文件夹
    fold_dirs = sorted([d for d in os.listdir(args.folds_dir) if os.path.isdir(os.path.join(args.folds_dir, d)) and d.startswith("fold")])
    for fold in fold_dirs:
        fold_path = os.path.join(args.folds_dir, fold)
        
        # 在 fold 目录下查找 YAML 文件（支持 .yaml 和 .yml 格式）
        yaml_files = glob.glob(os.path.join(fold_path, "*.yaml")) + glob.glob(os.path.join(fold_path, "*.yml"))
        if len(yaml_files) == 0:
            print(f"在 {fold_path} 中未找到 YAML 文件，跳过该 fold。")
            raise RuntimeError(f"在 {fold_path} 中未找到 YAML 文件，请确保有一个 YAML 文件。")
        elif len(yaml_files) > 1:
            raise RuntimeError(f"在 {fold_path} 中找到多个 YAML 文件，请确保只有一个 YAML 文件。")
        yaml_file_path = yaml_files[0]
        yaml_args = read_yaml(yaml_file_path)
        model_name = yaml_args.General.MODEL_NAME
        num_classes = yaml_args.General.num_classes
        device = torch.device(f'cuda:{yaml_args.General.device}')
        criterion = get_criterion(yaml_args.Model.criterion)

        # 查找 Best_EPOCH_*.pth 模型权重文件
        weight_files = glob.glob(os.path.join(fold_path, "Best_EPOCH_*.pth"))
        if not weight_files:
            print(f"未在 {fold_path} 中找到模型权重文件，跳过该 fold。")
            continue
        # 如果有多个，则按照 epoch 数降序排序，选择 epoch 数最大的权重文件
        def extract_epoch(fp):
            basename = os.path.basename(fp)
            try:
                epoch_str = basename.split("Best_EPOCH_")[1].split(".pth")[0]
                return int(epoch_str)
            except:
                return -1
        weight_files.sort(key=extract_epoch, reverse=True)
        model_weight_path = weight_files[0]
        print(f"评测 {fold}，使用模型权重文件: {model_weight_path}")

        # 构建测试数据集和 DataLoader
        if model_name == 'CDP_MIL':
            test_ds = CDP_MIL_WSI_Dataset(args.test_dataset_csv, yaml_args.Dataset.BeyesGuassian_pt_dir, 'test')
        elif model_name == 'LONG_MIL':
            test_ds = LONG_MIL_WSI_Dataset(args.test_dataset_csv, yaml_args.Dataset.h5_csv_path, 'test')
        else:
            test_ds = WSI_Dataset(args.test_dataset_csv, 'test')
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # 加载模型权重
        if model_name == 'DTFD_MIL':
            classifier, attention, dimReduction, attCls = get_model_from_yaml(yaml_args)
            state_dict = torch.load(model_weight_path, weights_only=True)
            classifier.load_state_dict(state_dict['classifier'])
            attention.load_state_dict(state_dict['attention'])
            dimReduction.load_state_dict(state_dict['dimReduction'])
            attCls.load_state_dict(state_dict['attCls'])
            model_list = [classifier, attention, dimReduction, attCls]
            model_list = [model.to(device).eval() for model in model_list]
        else:
            mil_model = get_model_from_yaml(yaml_args)
            mil_model = mil_model.to(device)
            # print(mil_model) 
            mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))

           

        # 根据模型类型选择相应的验证（测试）循环
        if model_name == 'CLAM_MB_MIL':
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif model_name == 'CLAM_MB_MIL_ENSEMBLE':
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = ensemble_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif model_name == 'CLAM_MB_MIL_FUSION':
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = fusion_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif model_name == 'CLAM_SB_MIL':
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif model_name == 'DS_MIL':
            test_loss, test_metrics = ds_val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        elif model_name == 'DTFD_MIL':
            test_loss, test_metrics = dtfd_val_loop(device, num_classes, model_list, test_dataloader, criterion,
                                                    yaml_args.Model.num_Group, yaml_args.Model.grad_clipping,
                                                    yaml_args.Model.distill, yaml_args.Model.total_instance)
        else:
            test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)

        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}Test_Loss:{ENDC}{test_loss}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')

        # 保存该 fold 的测试结果到独立文件夹中
        fold_result_dir = os.path.join(test_result_dir, fold)
        os.makedirs(fold_result_dir, exist_ok=True)
        
        # 将 fold 下的 YAML 文件复制到结果文件夹中
        new_yaml_filename = f"Test_{model_name}{os.path.splitext(yaml_file_path)[1]}"
        new_yaml_path = os.path.join(fold_result_dir, new_yaml_filename)
        shutil.copyfile(yaml_file_path, new_yaml_path)
        
        # 同时拷贝测试数据集csv文件
        new_test_dataset_csv_path = os.path.join(fold_result_dir, f'Test_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
        shutil.copyfile(args.test_dataset_csv, new_test_dataset_csv_path)
        
        # 新增：保存使用的 best model pth 文件
        new_model_weight_filename = os.path.basename(model_weight_path)
        new_model_weight_path = os.path.join(fold_result_dir, new_model_weight_filename)
        shutil.copyfile(model_weight_path, new_model_weight_path)
        print(f"Fold {fold} 的 best model 权重保存于: {new_model_weight_path}")
        
        # 保存测试日志
        test_log_path = os.path.join(fold_result_dir, f'Test_Log_{model_name}.txt')
        log_to_save = {'test_loss': test_loss, 'test_metrics': test_metrics}
        with open(test_log_path, 'w') as f:
            f.write(str(log_to_save))
        print(f"Fold {fold} 测试日志保存在: {test_log_path}")

        # 将结果添加到汇总字典中
        all_fold_metrics[fold] = {'test_loss': test_loss, 'test_metrics': test_metrics}

    # 计算所有fold测试指标的均值和标准差
    losses = []
    # 初始化一个字典，用于存储每个指标的所有fold数据
    metrics_accum = {}
    for fold, results in all_fold_metrics.items():
        losses.append(results['test_loss'])
        for key, value in results['test_metrics'].items():
            if key not in metrics_accum:
                metrics_accum[key] = []
            metrics_accum[key].append(value)

    summary = {}
    # 计算 loss 的均值和标准差
    summary['test_loss_mean'] = round(float(np.mean(losses)), 3)
    summary['test_loss_std'] = round(float(np.std(losses)), 3)

    # 计算每个指标的均值和标准差
    for key, values in metrics_accum.items():
        summary[f"{key}_mean"] = round(float(np.mean(values)), 3)
        summary[f"{key}_std"] = round(float(np.std(values)), 3)
    
    # 将所有fold的指标和汇总指标写入 JSON 文件
    merge_json_path = os.path.join(test_result_dir, "merge_5_fold_metrics.json")
    merge_data = {"folds": all_fold_metrics, "summary": summary}
    with open(merge_json_path, 'w') as f:
        json.dump(merge_data, f, indent=4, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
    print(f"5折测试汇总指标及均值标准差保存在: {merge_json_path}")


def ensemble_predictions(input_tensor, models, method="average", label=None):
    """
    集成多个模型的预测结果。

    参数：
      input_tensor: 待预测的数据（单样本），模型输出经过 squeeze 后为 (num_classes)
      models: 包含多个训练好的模型列表
      method: 集成方法，"average" 表示概率平均，"voting" 表示投票多数的模型的平均预测概率
      label: 真实标签，用于传递给模型的 forward 函数（主要针对 CLAM 模型）

    返回：
      若 method 为 "average"，返回所有模型预测 logits 求均值并经过 softmax 后的概率（形状为 (num_classes,)）；
      若 method 为 "voting"，返回投票多数的模型对应的预测概率均值（形状为 (num_classes,)）。
    """
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            forward_return = model(input_tensor, label=label)
            val_logits = forward_return['logits'].squeeze(0)
            preds.append(torch.softmax(val_logits, 0))
    # 得到 preds_tensor 形状为 (num_models, num_classes)
    preds_tensor = torch.stack(preds)
    
    if method == "average":
        # 对各模型的预测结果求平均，再通过 softmax 计算概率
        avg_preds = torch.mean(preds_tensor, dim=0)  # 形状为 (num_classes,)
        avg_preds = torch.softmax(avg_preds, dim=0)
        return avg_preds
    elif method == "voting":
        # 首先将 logits 转为概率，每个模型的概率分布为一行
        prob_tensor = torch.softmax(preds_tensor, dim=1)  # 形状为 (num_models, num_classes)
        # 每个模型的预测类别
        _, pred_each = torch.max(prob_tensor, dim=1)  # 形状为 (num_models,)
        # 统计各类别票数，并获得投票多数的类别
        counts = torch.bincount(pred_each)
        majority_label = torch.argmax(counts)
        # 筛选出预测为 majority_label 的模型索引
        majority_indices = (pred_each == majority_label).nonzero(as_tuple=False).squeeze(1)
        # 选出这些模型对应的预测概率
        selected_probs = prob_tensor[majority_indices]  # 形状为 (selected_num, num_classes)
        # 计算这些模型预测概率的平均值
        avg_prob = torch.mean(selected_probs, dim=0)  # 形状为 (num_classes,)
        return avg_prob
    else:
        raise ValueError("未知的集成方法，请选择 'average' 或 'voting'.")
    
def ensemble_test(args):
    """
    使用五个fold的 Best_EPOCH_*.pth 模型进行集成预测，
    得到一个集成的测试结果，而不输出每个单独模型的预测效果。

    参数：
      args.folds_dir: 包含 fold1, fold2, ... 的文件夹，每个 fold 下包含一个 YAML 配置文件和对应的 Best_EPOCH_*.pth 权重文件；
      args.test_dataset_csv: 测试数据集 CSV 文件路径；
      args.test_result_dir: 可选，集成测试结果的保存目录；若不指定，则默认保存在 folds_dir 下的 'ensemble_test_result' 文件夹中。

    注意：
      本函数假设所有 fold 中 YAML 配置一致，且不支持 DTFD_MIL 类型模型的直接集成（如需扩展需要额外处理）。
    """

    # 收集所有 fold 文件夹
    fold_dirs = sorted([d for d in os.listdir(args.folds_dir)
                        if os.path.isdir(os.path.join(args.folds_dir, d)) and d.startswith("fold")])
    if not fold_dirs:
        raise RuntimeError("在指定的 folds_dir 中未找到 fold 文件夹。")
    
    models = []
    for fold in fold_dirs:
        fold_path = os.path.join(args.folds_dir, fold)
        # 获取唯一的 YAML 文件（要求各fold中仅有一个YAML配置文件）
        yaml_files = glob.glob(os.path.join(fold_path, "*.yaml")) + glob.glob(os.path.join(fold_path, "*.yml"))
        if len(yaml_files) != 1:
            raise RuntimeError(f"{fold_path} 中必须有且只有一个YAML配置文件。")
        yaml_file = yaml_files[0]
        yaml_args = read_yaml(yaml_file)
        first_yaml_args = yaml_args
        model_name = yaml_args.General.MODEL_NAME
        num_classes = yaml_args.General.num_classes
        device = torch.device(f'cuda:{yaml_args.General.device}')
        criterion = get_criterion(yaml_args.Model.criterion)
        # 查找该 fold 下的 Best_EPOCH 权重文件
        weight_files = glob.glob(os.path.join(fold_path, "Best_EPOCH_*.pth"))
        if not weight_files:
            print(f"在 {fold_path} 中未找到 Best_EPOCH 权重文件，跳过该 fold。")
            continue
        def extract_epoch(fp):
            basename = os.path.basename(fp)
            try:
                epoch_str = basename.split("Best_EPOCH_")[1].split(".pth")[0]
                return int(epoch_str)
            except:
                return -1
        weight_files.sort(key=extract_epoch, reverse=True)
        model_weight_path = weight_files[0]
        print(f"{fold} 使用模型权重文件: {model_weight_path}")

        if model_name == 'DTFD_MIL':
            # 暂不支持 DTFD_MIL 模型的集成，需额外封装其各模块以便统一调用
            raise NotImplementedError("DTFD_MIL 类型模型的集成预测暂未实现。")
        else:
            mil_model = get_model_from_yaml(yaml_args)
            mil_model = mil_model.to(device)
            mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))
            models.append(mil_model)

    if len(models) == 0:
        raise RuntimeError("未能加载任何模型，无法进行集成预测。")

    # 构建测试数据集（若为CDP_MIL或LONG_MIL则需要对应的路径）
    if model_name == 'CDP_MIL':
        test_ds = CDP_MIL_WSI_Dataset(args.test_dataset_csv, first_yaml_args.Dataset.BeyesGuassian_pt_dir, 'test')
    elif model_name == 'LONG_MIL':
        test_ds = LONG_MIL_WSI_Dataset(args.test_dataset_csv, first_yaml_args.Dataset.h5_csv_path, 'test')
    else:
        test_ds = WSI_Dataset(args.test_dataset_csv, 'test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 开始集成预测（使用 ensemble_predictions 函数；此处采用概率平均的方式）
    ensemble_preds = []
    labels = []
    for _, data in enumerate(test_dataloader):
        inputs = data[0].to(device).float()
        label = data[1].to(device).long()
        labels.append(label.cpu().numpy())

        # 使用集成函数进行预测
        avg_probs = ensemble_predictions(inputs, models, method=args.ensemble_method, label=label)
        ensemble_preds.append(avg_probs.cpu().numpy())
    
    all_preds = np.stack(ensemble_preds, axis=0)
    all_labels = np.concatenate(labels)
    
    # 计算测试指标（假设 cal_scores 要求 logits 为 (batch, num_classes)）
    test_metrics = cal_scores(all_preds, all_labels, num_classes)
    
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print('----------------INFO----------------\n')
    print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')

    # 保存集成测试结果日志
    ensemble_result_dir = args.test_result_dir if args.test_result_dir else os.path.join(args.folds_dir, f'{args.ensemble_method}_ensemble_test_result')
    os.makedirs(ensemble_result_dir, exist_ok=True)
    ensemble_log_path = os.path.join(ensemble_result_dir, f'Ensemble_Test_Log_{args.ensemble_method}.txt')
    log_to_save = {'test_metrics': test_metrics}
    with open(ensemble_log_path, 'w') as f:
        f.write(str(log_to_save))
    print(f"集成测试日志保存在: {ensemble_log_path}")


# 修改主入口，根据传入参数选择不同的测试方式
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='/path/to/your/config-yaml', help='MIL模型对应的YAML配置路径')
    parser.add_argument('--test_dataset_csv', type=str, default='/path/to/your/ds-csv-path', help='测试数据集CSV文件路径')
    parser.add_argument('--model_weight_path', type=str, default='/path/to/your/model-weight', help='单个模型的权重文件路径')
    parser.add_argument('--folds_dir', type=str, default='', help='包含 fold1, fold2, ... 的文件夹路径（用于交叉验证或集成预测）')
    parser.add_argument('--test_result_dir', type=str, default='', help='测试结果保存目录；若不指定，则默认在 folds_dir 下生成相应结果文件夹')
    parser.add_argument('--all_folds', action='store_true', help='是否对所有 fold 进行单独评测')
    parser.add_argument('--ensemble', action='store_true', help='是否使用 folds_dir 中的模型进行集成预测')
    parser.add_argument('--ensemble_method', type=str, default='average', help='集成方法，可选值为 "average" 或 "voting"')
    args = parser.parse_args()
    
    if args.ensemble:
        if args.folds_dir == "":
            print("必须提供 --folds_dir 参数 来指定包含 fold 目录的文件夹")
        else:
            ensemble_test(args)
    elif args.all_folds:
        if args.folds_dir == "":
            print("必须提供 --folds_dir 参数 来指定包含 fold 目录的文件夹")
        else:
            test_all_folds(args)
    else:
        test(args)

