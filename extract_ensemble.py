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
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # 尝试查找并解析JSON格式的测试指标
        metrics_match = re.search(r"'test_metrics': (\{.*?\})", content, re.DOTALL)
        if not metrics_match:
            # 尝试其他可能的格式
            # 尝试寻找test_metrics后跟的字典格式（可能没有用引号括起来）
            metrics_match = re.search(r"test_metrics:?\s*(\{.*?\})", content, re.DOTALL)
            
            if not metrics_match:
                # 尝试另一种格式：可能是在文件的某处直接列出了指标
                direct_metrics = extract_direct_metrics(content, file_path)
                if direct_metrics:
                    result = {
                        'model': os.path.basename(file_path),
                        'file_path': file_path,
                        **direct_metrics
                    }
                    return result
                
                print(f"警告：在文件 {file_path} 中未找到测试指标格式'test_metrics': {{...}}")
                return None
            
        # 替换单引号为双引号以便JSON解析
        metrics_str = metrics_match.group(1).replace("'", "\"")
        # 将NumPy数组格式替换成可解析的格式
        metrics_str = re.sub(r'array\(\[(.*?)\]\)', r'[\1]', metrics_str)
        # 将dtype=int64等替换掉
        metrics_str = re.sub(r', dtype=\w+', '', metrics_str)
        
        metrics = {}
        try:
            # 尝试直接解析JSON
            metrics = json.loads(metrics_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败，尝试使用正则表达式提取 {file_path}")
            # 使用安全的正则表达式提取方法
            metrics = extract_metrics_with_regex(metrics_str, file_path)
            if not metrics:
                # 如果从字典格式中提取失败，尝试从文件内容中直接提取
                metrics = extract_direct_metrics(content, file_path)
                if not metrics:
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

def extract_metrics_with_regex(metrics_str, file_path):
    """使用安全的正则表达式方法提取指标"""
    metrics = {}
    
    # 定义要提取的指标和对应的正则表达式
    metric_patterns = {
        'acc': r"'acc':\s*([\d\.]+)",
        'bacc': r"'bacc':\s*([\d\.]+)",
        'auc': r"'auc':\s*([\d\.]+)",
        'f1': r"'f1':\s*([\d\.]+)",
        'recall': r"'recall':\s*([\d\.]+)",
        'precision': r"'precision':\s*([\d\.]+)",
        'quadratic_kappa': r"'quadratic_kappa':\s*([\d\.]+)",
        'linear_kappa': r"'linear_kappa':\s*([\d\.]+)"
    }
    
    # 尝试提取每个指标
    found_any = False
    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, metrics_str)
        if match:
            found_any = True
            try:
                metrics[metric_name] = float(match.group(1))
            except (ValueError, IndexError):
                print(f"提取指标 {metric_name} 时出现错误，无法转换为浮点数")
    
    if not found_any:
        print(f"在字典中未找到任何可识别的指标，将尝试其他格式")
        return None
        
    return metrics

def extract_direct_metrics(content, file_path):
    """直接从文本内容中提取指标，适用于非标准格式"""
    metrics = {}
    
    # 更宽松的模式，直接在文本中查找指标，不要求在JSON字典中
    patterns = {
        'acc': [r"accuracy:?\s*([\d\.]+)", r"ACC:?\s*([\d\.]+)", r"Accuracy:?\s*([\d\.]+)"],
        'bacc': [r"balanced[_\s]*accuracy:?\s*([\d\.]+)", r"BACC:?\s*([\d\.]+)", r"Balanced[_\s]*Accuracy:?\s*([\d\.]+)"],
        'auc': [r"auc:?\s*([\d\.]+)", r"AUC:?\s*([\d\.]+)", r"Area[_\s]*Under[_\s]*Curve:?\s*([\d\.]+)"],
        'f1': [r"f1:?\s*([\d\.]+)", r"F1:?\s*([\d\.]+)", r"F1[_\s]*score:?\s*([\d\.]+)"],
        'recall': [r"recall:?\s*([\d\.]+)", r"Recall:?\s*([\d\.]+)", r"Sensitivity:?\s*([\d\.]+)"],
        'precision': [r"precision:?\s*([\d\.]+)", r"Precision:?\s*([\d\.]+)"],
        'quadratic_kappa': [r"quadratic[_\s]*kappa:?\s*([\d\.]+)", r"Quadratic[_\s]*Kappa:?\s*([\d\.]+)"],
        'linear_kappa': [r"linear[_\s]*kappa:?\s*([\d\.]+)", r"Linear[_\s]*Kappa:?\s*([\d\.]+)"]
    }
    
    found_any = False
    for metric_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                found_any = True
                try:
                    metrics[metric_name] = float(match.group(1))
                    break  # 找到一个匹配就跳出该指标的模式循环
                except (ValueError, IndexError):
                    continue  # 尝试下一个模式
    
    if not found_any:
        print(f"警告：在文件 {file_path} 中未找到任何可识别的指标")
        # 最后尝试查找数值本身，寻找类似"= 0.8765"这样的模式
        value_patterns = [
            r"=\s*(0\.\d+)",  # = 0.xxxx
            r":\s*(0\.\d+)",  # : 0.xxxx
            r"是\s*(0\.\d+)"   # 是 0.xxxx
        ]
        
        metric_names = ['acc', 'bacc', 'auc', 'f1', 'recall', 'precision']
        values = []
        
        for pattern in value_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    values.append(float(match))
                except (ValueError, IndexError):
                    continue
        
        # 如果找到了一些值，假设它们按顺序对应指标
        if values:
            found_any = True
            for i, value in enumerate(values):
                if i < len(metric_names):
                    metrics[metric_names[i]] = value
                else:
                    break
                    
        if not found_any:
            return None
                    
    return metrics

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
    failed_files = []
    for file_path in ensemble_files:
        metrics = extract_metrics_from_file(file_path)
        if metrics:
            all_metrics.append(metrics)
        else:
            failed_files.append(file_path)
    
    # 创建DataFrame并保存为CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # 将输出文件保存在搜索目录下
        output_file = os.path.join(search_dir, 'ensemble_test_metrics.csv')
        df.to_csv(output_file, index=False)
        print(f"指标已保存到 {output_file}")
        
        # 显示一些基本统计
        print("\n基本统计信息:")
        print(f"总文件数: {len(ensemble_files)}")
        print(f"成功提取指标的文件数: {len(all_metrics)}")
        print(f"失败文件数: {len(failed_files)}")
        
        if 'acc' in df.columns and not df['acc'].empty and df['acc'].mean() > 0:
            print(f"平均准确率 (ACC): {df['acc'].mean():.4f}")
        if 'bacc' in df.columns and not df['bacc'].empty and df['bacc'].mean() > 0:
            print(f"平均平衡准确率 (BACC): {df['bacc'].mean():.4f}")
        if 'auc' in df.columns and not df['auc'].empty and df['auc'].mean() > 0:
            print(f"平均AUC: {df['auc'].mean():.4f}")
        if 'f1' in df.columns and not df['f1'].empty and df['f1'].mean() > 0:
            print(f"平均F1分数: {df['f1'].mean():.4f}")
            
        # 如果有失败的文件，打印它们的列表
        if failed_files:
            print("\n未能提取指标的文件:")
            for file in failed_files:
                print(f"- {file}")
    else:
        print("未成功提取任何指标数据")
        if failed_files:
            print("\n所有失败的文件:")
            for file in failed_files:
                print(f"- {file}")

if __name__ == "__main__":
    main() 