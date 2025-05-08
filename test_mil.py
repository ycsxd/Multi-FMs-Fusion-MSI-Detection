"""
python test_mil.py --test_dataset_csv ./testdata/test/zhe1msi.csv  --all_folds --folds_dir ./testdata/test/uni-w9235/
"""
import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop,ds_val_loop,dtfd_val_loop
import warnings
from utils.wsi_utils import WSI_Dataset,CDP_MIL_WSI_Dataset,LONG_MIL_WSI_Dataset
import torch
import shutil
import os
from utils.model_utils import get_model_from_yaml,get_criterion
warnings.filterwarnings('ignore')

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
    然后按照 test 函数的逻辑评测，并保存对应 fold 的日志及拷贝 YAML、CSV 文件；
    最后生成一个 merge_5_fold_metrics.json 文件，汇总各折的测试指标及其均值和标准差。
    """
    import glob
    import json
    import numpy as np

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
            mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))

        # 根据模型类型选择相应的验证（测试）循环
        if model_name == 'CLAM_MB_MIL':
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
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
        
        new_test_dataset_csv_path = os.path.join(fold_result_dir, f'Test_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
        shutil.copyfile(args.test_dataset_csv, new_test_dataset_csv_path)
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
    # 计算loss的均值和标准差
    summary['test_loss_mean'] = float(np.mean(losses))
    summary['test_loss_std'] = float(np.std(losses))

    # 计算每个指标的均值和标准差
    for key, values in metrics_accum.items():
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    
    # 将所有fold的指标和汇总指标写入JSON文件
    merge_json_path = os.path.join(test_result_dir, "merge_5_fold_metrics.json")
    merge_data = {"folds": all_fold_metrics, "summary": summary}
    with open(merge_json_path, 'w') as f:
        json.dump(merge_data, f, indent=4, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
    print(f"5折测试汇总指标及均值标准差保存在: {merge_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='/path/to/your/config-yaml', help='path to MIL-model-yaml file')
    parser.add_argument('--test_dataset_csv', type=str, default='/path/to/your/ds-csv-path', help='path to dataset csv')
    parser.add_argument('--model_weight_path', type=str, default='/path/to/your/model-weight', help='path to model weights')
    parser.add_argument('--test_log_dir', type=str, default='', help='测试日志保存目录；若不指定，默认在folds_dir下生成test_log文件夹')
    parser.add_argument('--folds_dir', type=str, default='', help='包含fold1,...,fold5的文件夹路径（用于5折评测）')
    parser.add_argument('--test_result_dir', type=str, default='', help='测试结果保存目录；若不指定，默认在folds_dir下生成test_result文件夹')
    parser.add_argument('--all_folds', action='store_true', help='是否一次性评测所有fold，若设置则使用folds_dir下的各折进行评测')
    args = parser.parse_args()
    if args.all_folds:
        if args.folds_dir == "":
            print("必须提供 --folds_dir 参数 来指定包含fold目录的文件夹")
        else:
            test_all_folds(args)
    else:
        test(args)
