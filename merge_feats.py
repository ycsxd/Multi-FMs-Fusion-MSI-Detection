import os
import torch
import glob


def merge_features_from_dirs(feat_dirs, output_dir):
    """
    合并指定文件夹下的同名pt文件中的特征矩阵，
    将各个文件夹的特征按列合并（即在 dimension=1 拼接），
    然后保存合并后的特征 tensor。

    Args:
        feat_dirs: 包含pt文件的文件夹路径列表
        output_dir: 输出保存特征tensor的目录
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
    
    # 遍历pt_files，记录处理进度
    for idx, pt_file in enumerate(pt_files):
        # 获取文件名，例如 1.pt、2.pt 等
        file_name = os.path.basename(pt_file)
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            print(f"文件: {file_name} 已存在于输出目录，跳过")
            continue
        print(f"\n处理文件: {file_name}")
        
        # 定义一个列表存储各个文件夹对应的特征 tensor
        features_list = []
        
        for feat_dir in feat_dirs:
            file_path = os.path.join(feat_dir, file_name)
            if os.path.exists(file_path):
                feature = torch.load(file_path)
                print(f"加载文件: {file_path}，特征维度: {feature.shape}")
                features_list.append(feature)
            else:
                print(f"警告: 在 {feat_dir} 中没有找到 {file_name}")
        
        # 检查是否至少加载了一个特征
        if features_list:
            # 检查所有 tensor 的第一个维度是否一致
            base_first_dim = features_list[0].shape[0]
            if any(feature.shape[0] != base_first_dim for feature in features_list):
                print(f"错误: 文件 {file_name} 中的部分特征的第一个维度大小不一致，无法合并")
                continue

            # 列合并所有加载的特征，即在 dimension=1 拼接
            try:
                merged_features = torch.cat(features_list, dim=1)
                print(f"合并特征后维度: {merged_features.shape}")
            except Exception as e:
                print(f"合并特征时出现错误: {e}")
                continue
            
            # 保存合并后的特征 tensor
            output_file = os.path.join(output_dir, file_name)
            torch.save(merged_features, output_file)
            print(f"已保存合并特征到: {output_file}")
        else:
            print(f"未找到任何文件进行合并: {file_name}")
        
        # 每处理100个文件打印一次综述
        if (idx + 1) % 100 == 0:
            print(f"\n=============\nSummary: 已处理 {idx + 1} 个文件，共 {len(pt_files)} 个文件。\n")
            
if __name__ == "__main__":
    # 示例：手动指定要合并的文件夹路径
    feat_dirs = [
        "/share/home/guoweis/MIL_BASELINE-main/MSIfeat/gigapath/featdir/pt_files/",  # 请替换为第一个文件夹的实际路径
        "/share/home/guoweis/MIL_BASELINE-main/MSIfeat/uni2/featdir/pt_files/",       # 请替换为第二个文件夹的实际路径
        "/share/home/guoweis/MIL_BASELINE-main/msidata/featdir/pt_files/",
        # 可继续添加更多路径
    ]
    output_dir = "/share/home/guoweis/MIL_BASELINE-main/MSIfeat/merged_feat/giga_uni2_virchow2_merged/"  # 请替换为实际的输出目录路径
    
    merge_features_from_dirs(feat_dirs, output_dir)