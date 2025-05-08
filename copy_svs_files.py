import os
import shutil
import sys

def copy_svs_files(source_dir, target_dir):
    """
    遍历源目录及其子目录，将所有SVS格式的文件复制到目标目录
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")
    
    # 计数器
    total_files = 0
    copied_files = 0
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            total_files += 1
            # 检查文件是否为SVS格式
            if file.lower().endswith('.svs'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # 复制文件
                try:
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    print(f"已复制: {source_file} -> {target_file}")
                except Exception as e:
                    print(f"复制文件失败: {source_file}")
                    print(f"错误信息: {str(e)}")
    
    print(f"\n复制完成。共处理文件 {total_files} 个，其中复制了 {copied_files} 个SVS文件到目标目录。")

if __name__ == "__main__":
    source_directory = r"I:\GasMsi-sy\20250328"
    target_directory = r"I:\GasMsi-sy\20250328-svs\svs-1"
    
    if not os.path.exists(source_directory):
        print(f"错误: 源目录 '{source_directory}' 不存在!")
        sys.exit(1)
    
    print(f"开始从 '{source_directory}' 复制SVS文件到 '{target_directory}'")
    copy_svs_files(source_directory, target_directory) 