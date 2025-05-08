import os
import shutil
import pandas as pd
from pathlib import Path

def copy_files_from_csv(csv_path, source_dir, target_dir):
    """
    从CSV文件中读取文件路径，并将文件从源目录复制到目标目录
    
    参数:
        csv_path: CSV文件路径
        source_dir: 源文件目录 (例如 "I:\\sgw\\tcga\\masks")
        target_dir: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 检查是否存在slide_path列
    if 'slide_path' not in df.columns:
        print(f"CSV文件中不存在'slide_path'列")
        return
    
    # 获取源目录中的所有文件
    try:
        source_files = os.listdir(source_dir)
    except Exception as e:
        print(f"读取源目录时出错: {e}")
        return
    
    # 创建源目录文件名到完整文件路径的映射
    # 对每个文件，我们存储无扩展名的版本作为键
    file_mapping = {}
    for filename in source_files:
        name_without_ext = os.path.splitext(filename)[0]
        file_mapping[name_without_ext] = filename
    
    # 计数器
    total_files = len(df['slide_path'])
    copied_files = 0
    missing_files = 0
    
    # 遍历CSV中的所有文件名
    for idx, file_name in enumerate(df['slide_path']):
        # 显示进度
        if (idx + 1) % 10 == 0 or idx == 0 or idx == total_files - 1:
            print(f"处理: {idx+1}/{total_files}")
        
        # 获取基本文件名(不含路径)
        base_name = os.path.basename(file_name)
        
        # 检查不含扩展名的文件名是否匹配源目录中的文件
        if base_name in file_mapping:
            # 找到了匹配的文件(带扩展名)
            full_filename = file_mapping[base_name]
            source_path = os.path.join(source_dir, full_filename)
            target_path = os.path.join(target_dir, full_filename)
            try:
                shutil.copy2(source_path, target_path)
                copied_files += 1
            except Exception as e:
                print(f"复制文件 {full_filename} 时出错: {e}")
        else:
            print(f"文件未找到: {base_name}")
            missing_files += 1
    
    print(f"\n复制完成! 总共处理 {total_files} 个文件")
    print(f"成功复制: {copied_files} 个文件")
    print(f"未找到: {missing_files} 个文件")

if __name__ == "__main__":
    # 这些路径可以根据需要修改
    csv_file = input("请输入CSV文件的路径 (例如: path/to/tcga_train_ds.csv): ")
    source_directory = input("请输入源文件夹的路径 (例如: I:\\sgw\\tcga\\masks): ")
    target_directory = input("请输入目标文件夹的路径: ")
    
    print(f"\n开始复制文件...")
    copy_files_from_csv(csv_file, source_directory, target_directory) 