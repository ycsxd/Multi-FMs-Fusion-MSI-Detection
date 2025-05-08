import os
import torch
from pathlib import Path
import glob
from tqdm import tqdm  # 添加进度条库

def merge_features_from_dirs(feat_dirs, output_dir):
    """
    合并指定文件夹下的同名pt文件中的特征矩阵，目前将各个文件夹的特征保存为列表，
    可以通过调用列表中对应的元素获得具体特征，必要时再进行合并操作。
    
    Args:
        feat_dirs: 包含pt文件的文件夹路径列表
        output_dir: 输出保存特征列表的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if not feat_dirs:
        print("没有提供任何文件夹路径")
        return
    
    # 获取第一个文件夹中的所有pt文件（基于第一个文件夹的文件名）
    first_dir = feat_dirs[0]
    pt_files = sorted(glob.glob(os.path.join(first_dir, '*.pt')))
    
    if not pt_files:
        print(f"在 {first_dir} 中没有找到pt文件")
        return
    
    print(f"找到 {len(pt_files)} 个pt文件需要处理")
    
    # 使用 tqdm 进度条遍历 pt_files
    for pt_file in tqdm(pt_files, desc="处理pt文件", unit="file"):
        # 获取文件名，如 1.pt、2.pt 等
        file_name = os.path.basename(pt_file)
        print(f"\n处理文件: {file_name}")
        
        # 定义一个列表用来存储各个文件夹对应的特征 tensor
        features_list = []
        
        # 遍历所有文件夹，依次加载相同名称的pt文件
        for feat_dir in feat_dirs:
            file_path = os.path.join(feat_dir, file_name)
            if os.path.exists(file_path):
                feature = torch.load(file_path)
                print(f"加载文件: {file_path}，特征维度: {feature.shape}")
                features_list.append(feature)
            else:
                print(f"警告: 在 {feat_dir} 中没有找到 {file_name}")
        
        # 保存特征列表
        output_file = os.path.join(output_dir, file_name)
        torch.save(features_list, output_file)
        print(f"已保存特征列表到: {output_file}")

if __name__ == "__main__":
    # 示例：手动指定要合并的文件夹路径
    feat_dirs = [
        "/share/home/guoweis/MIL_BASELINE-main/MSIfeat/gigapath/featdir/pt_files/",  # 请替换为第一个文件夹的实际路径
        "/share/home/guoweis/MIL_BASELINE-main/MSIfeat/uni2/featdir/pt_files/",       # 请替换为第二个文件夹的实际路径
        # 可以添加更多文件夹路径
    ]
    output_dir = "/share/home/guoweis/MIL_BASELINE-main/MSIfeat/merged_feat/giga_uni2/"  # 请替换为实际的输出目录路径
    
    # 使用新的函数处理手动指定的文件夹
    merge_features_from_dirs(feat_dirs, output_dir)