import os
import json
import pandas as pd
import re
from pathlib import Path

def find_ensemble_files(root_dir):
    """查找所有以Ensemble_Test开头的txt文件"""
    ensemble_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("Ensemble_Test") and file.endswith(".txt"):
                ensemble_files.append(os.path.join(root, file))
    return ensemble_files

def extract_metrics_from_file(file_path):
    """从单个Ensemble_Test文件中提取指标"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 尝试查找并解析JSON格式的测试指标
        metrics_match = re.search(r"'test_metrics': (\{.*?\})", content, re.DOTALL)
        if metrics_match:
            # 替换单引号为双引号以便JSON解析
            metrics_str = metrics_match.group(1).replace("'", "\"")
            # 将NumPy数组格式替换成可解析的格式
            metrics_str = re.sub(r'array\(\[(.*?)\]\)', r'[\1]', metrics_str)
            # 将dtype=int64等替换掉
            metrics_str = re.sub(r', dtype=\w+', '', metrics_str)
            
            try:
                # 尝试直接解析JSON
                metrics = json.loads(metrics_str)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试使用正则表达式提取各个指标
                acc = float(re.search(r"'acc': ([\d\.]+)", metrics_str).group(1))
                bacc = float(re.search(r"'bacc': ([\d\.]+)", metrics_str).group(1))
                auc = float(re.search(r"'auc': ([\d\.]+)", metrics_str).group(1))
                f1 = float(re.search(r"'f1': ([\d\.]+)", metrics_str).group(1))
                recall = float(re.search(r"'recall': ([\d\.]+)", metrics_str).group(1))
                precision = float(re.search(r"'precision': ([\d\.]+)", metrics_str).group(1))
                
                metrics = {
                    'acc': acc,
                    'bacc': bacc,
                    'auc': auc,
                    'f1': f1,
                    'recall': recall,
                    'precision': precision
                }
                
                # 尝试提取kappa指标，可能不是所有文件都有
                quadratic_kappa_match = re.search(r"'quadratic_kappa': ([\d\.]+)", metrics_str)
                if quadratic_kappa_match:
                    metrics['quadratic_kappa'] = float(quadratic_kappa_match.group(1))
                
                linear_kappa_match = re.search(r"'linear_kappa': ([\d\.]+)", metrics_str)
                if linear_kappa_match:
                    metrics['linear_kappa'] = float(linear_kappa_match.group(1))
        else:
            # 如果没有找到测试指标，返回空字典
            print(f"警告：在文件 {file_path} 中未找到测试指标")
            return None
        
        # 提取文件名作为模型标识
        model_name = os.path.basename(file_path)
        
        # 返回指标和文件信息
        result = {
            'model': model_name,
            'file_path': file_path,
            'acc': metrics.get('acc', 0),
            'bacc': metrics.get('bacc', 0),
            'auc': metrics.get('auc', 0),
            'f1': metrics.get('f1', 0),
            'recall': metrics.get('recall', 0),
            'precision': metrics.get('precision', 0)
        }
        
        # 添加kappa指标（如果有的话）
        if 'quadratic_kappa' in metrics:
            result['quadratic_kappa'] = metrics['quadratic_kappa']
        if 'linear_kappa' in metrics:
            result['linear_kappa'] = metrics['linear_kappa']
            
        return result
        
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
        return None

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
    
    # 查找所有Ensemble_Test文件
    ensemble_files = find_ensemble_files(search_dir)
    
    if not ensemble_files:
        print(f"在目录 '{search_dir}' 中未找到任何以Ensemble_Test开头的txt文件")
        return
    
    print(f"找到 {len(ensemble_files)} 个Ensemble_Test文件")
    
    # 提取所有指标
    all_metrics = []
    for file_path in ensemble_files:
        metrics = extract_metrics_from_file(file_path)
        if metrics:
            all_metrics.append(metrics)
    
    # 创建DataFrame并保存为CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # 将输出文件保存在搜索目录下
        output_file = os.path.join(search_dir, 'ensemble_test_metrics.csv')
        df.to_csv(output_file, index=False)
        print(f"指标已保存到 {output_file}")
        
        # 显示一些基本统计
        print("\n基本统计信息:")
        print(f"总文件数: {len(all_metrics)}")
        print(f"平均准确率 (ACC): {df['acc'].mean():.4f}")
        print(f"平均平衡准确率 (BACC): {df['bacc'].mean():.4f}")
        print(f"平均AUC: {df['auc'].mean():.4f}")
        print(f"平均F1分数: {df['f1'].mean():.4f}")
    else:
        print("未成功提取任何指标数据")

if __name__ == "__main__":
    main() 