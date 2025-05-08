import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm

def concat_features(csv_path, output_dir=None):
    """
    将CSV文件中的性别和年龄特征拼接到对应的PT文件张量中，
    并生成新的CSV文件，将slide_path替换为新生成的PT文件路径
    
    参数:
    - csv_path: CSV文件路径，包含slide_path, label, gender, age列
    - output_dir: 输出目录，如果为None则保存在原PT文件的相同位置，添加_with_meta后缀
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 确保CSV包含必要的列
    required_cols = ['test_slide_path', 'gender', 'age']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 创建输出目录（如果指定了）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    failed_files = []
    skipped_files = []
    
    # 创建新的DataFrame来存储更新后的信息
    new_df = df.copy()
    new_paths = []
    
    # 处理每个PT文件
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理PT文件"):
        slide_path = row['test_slide_path']
        gender = float(row['gender'])
        age = float(row['age'])
        
        # 检查PT文件是否存在
        if not os.path.exists(slide_path):
            print(f"警告: 找不到PT文件: {slide_path}")
            failed_files.append(slide_path)
            new_paths.append(slide_path)  # 保持原路径
            continue
        
        try:
            # 确定输出路径
            if output_dir:
                base_name = os.path.basename(slide_path)
                output_path = os.path.join(output_dir, base_name)
            else:
                file_name, file_ext = os.path.splitext(slide_path)
                output_path = f"{file_name}_with_meta{file_ext}"
                
            # 检查输出文件是否已存在，如果存在则跳过
            if os.path.exists(output_path):
                print(f"跳过: 输出文件已存在: {output_path}")
                skipped_files.append(slide_path)
                skipped_count += 1
                new_paths.append(output_path)  # 使用已存在的输出文件路径
                continue
            
            # 加载PT文件
            features = torch.load(slide_path)
            
            # 检查维度
            if len(features.shape) != 2:
                print(f"警告: PT文件 {slide_path} 不是二维张量，形状为 {features.shape}")
                failed_files.append(slide_path)
                new_paths.append(slide_path)  # 保持原路径
                continue
            
            n, d = features.shape
            
            # 创建性别和年龄张量并扩展为与特征相同的n
            gender_tensor = torch.full((n, 1), gender, dtype=features.dtype)
            age_tensor = torch.full((n, 1), age, dtype=features.dtype)
            
            # 拼接特征
            new_features = torch.cat([features, gender_tensor, age_tensor], dim=1)
            
            # 保存新的PT文件
            torch.save(new_features, output_path)
            processed_count += 1
            
            # 保存新的文件路径
            new_paths.append(output_path)
            
        except Exception as e:
            print(f"处理PT文件 {slide_path} 时出错: {str(e)}")
            failed_files.append(slide_path)
            new_paths.append(slide_path)  # 保持原路径
    
    # 更新DataFrame中的slide_path列
    new_df['test_slide_path'] = new_paths
    
    # 保存新的CSV文件
    if output_dir:
        output_csv_path = os.path.join(output_dir, "clinic_test.csv")
    else:
        output_dir = os.path.dirname(csv_path)
        output_csv_path = os.path.join(output_dir, "clinic_test.csv")
    
    new_df.to_csv(output_csv_path, index=False)
    print(f"更新后的CSV文件已保存至: {output_csv_path}")
    
    print(f"处理完成! 成功处理 {processed_count} 个文件，跳过 {skipped_count} 个文件，失败 {len(failed_files)} 个文件。")
    if skipped_files:
        print("跳过的文件:")
        for file in skipped_files[:10]:  # 只打印前10个跳过的文件
            print(f"  - {file}")
        if len(skipped_files) > 10:
            print(f"  ...以及其他 {len(skipped_files) - 10} 个文件")
    if failed_files:
        print("失败的文件:")
        for file in failed_files[:10]:  # 只打印前10个失败的文件
            print(f"  - {file}")
        if len(failed_files) > 10:
            print(f"  ...以及其他 {len(failed_files) - 10} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将性别和年龄特征拼接到PT文件")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV文件路径，包含slide_path, gender, age列")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认在原位置添加后缀")
    
    args = parser.parse_args()
    concat_features(args.csv_path, args.output_dir) 