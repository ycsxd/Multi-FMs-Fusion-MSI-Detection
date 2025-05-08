import os
import json
import pandas as pd
from pathlib import Path

def find_metrics_files(root_dir):
    """查找所有merge_5_fold_metrics.json文件"""
    metrics_files = []
    for root, dirs, files in os.walk(root_dir):
        if "merge_5_fold_metrics.json" in files:
            metrics_files.append(os.path.join(root, "merge_5_fold_metrics.json"))
    return metrics_files

def extract_metrics_from_file(file_path):
    """从单个文件中提取指标"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 使用完整的文件路径作为model名称
    model_path = file_path
    
    # 提取每个fold的指标
    fold_metrics = []
    for fold_name, fold_data in data['folds'].items():
        metrics = fold_data['test_metrics']
        try:
            fold_metrics.append({
                'model': model_path,
                'fold': fold_name,
                'acc': metrics['acc'],
                'bacc': metrics['bacc'],
                'auc': metrics['auc'],
                'f1': metrics['f1']
            })
        except KeyError as e:
            print(f"错误：文件 {file_path} 的 fold {fold_name} 中缺少键 {e}")
            # 可以选择跳过这个fold或者使用默认值
            fold_metrics.append({
                'model': model_path,
                'fold': fold_name,
                'acc': metrics.get('acc', 0),
                'bacc': metrics.get('bacc', 0),
                'auc': metrics.get('auc', 0),
                'f1': metrics.get('f1', 0)
            })
    
    # 提取summary指标
    summary = data['summary']
    try:
        summary_metrics = {
            'model': model_path,
            'fold': 'summary',
            'acc': summary['acc_mean'],
            'bacc': summary['bacc_mean'],
            'auc': summary['auc_mean'],
            'f1': summary['f1_mean']
        }
    except KeyError as e:
        print(f"错误：文件 {file_path} 的 summary 中缺少键 {e}")
        summary_metrics = {
            'model': model_path,
            'fold': 'summary',
            'acc': summary.get('acc_mean', 0),
            'bacc': summary.get('bacc_mean', 0),
            'auc': summary.get('auc_mean', 0),
            'f1': summary.get('f1_mean', 0)
        }
    
    return fold_metrics + [summary_metrics]

def main():
    # 让用户输入要搜索的目录路径
    search_dir = input("请输入要搜索的目录路径：").strip()
    if not search_dir:
        print("未输入目录路径")
        return
    
    # 检查目录是否存在
    if not os.path.exists(search_dir):
        print(f"错误：目录 '{search_dir}' 不存在")
        return
    
    # 查找所有metrics文件
    metrics_files = find_metrics_files(search_dir)
    
    if not metrics_files:
        print(f"在目录 '{search_dir}' 中未找到任何merge_5_fold_metrics.json文件")
        return
    
    print(f"找到 {len(metrics_files)} 个merge_5_fold_metrics.json文件")
    
    # 提取所有指标
    all_metrics = []
    for file_path in metrics_files:
        try:
            all_metrics.extend(extract_metrics_from_file(file_path))
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")
    
    # 创建DataFrame并保存为CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # 将输出文件保存在搜索目录下
        output_file = os.path.join(search_dir, 'extract_all_model_metrics.csv')
        df.to_csv(output_file, index=False)
        print(f"指标已保存到 {output_file}")
    else:
        print("未成功提取任何指标数据")

if __name__ == "__main__":
    main() 