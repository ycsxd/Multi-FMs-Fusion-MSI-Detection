import os
import csv

def list_files_to_csv(folder_path, output_csv):
    """
    遍历指定文件夹，将所有文件名保存到CSV文件中
    
    参数:
    folder_path: 要遍历的文件夹路径
    output_csv: 输出的CSV文件路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return False
    
    # 获取所有文件
    file_list = []
    
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取相对于给定文件夹的路径
            relative_path = os.path.relpath(os.path.join(root, file), folder_path)
            file_list.append(relative_path)
    
    # 将文件名写入CSV文件
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入标题行
            writer.writerow(['文件名'])
            # 写入文件名
            for file_name in file_list:
                writer.writerow([file_name])
        
        print(f"成功将文件列表保存到 '{output_csv}'")
        print(f"共找到 {len(file_list)} 个文件")
        return True
    
    except Exception as e:
        print(f"保存CSV时出错: {e}")
        return False

if __name__ == "__main__":
    folder_path = input("请输入要遍历的文件夹路径: ")
    # 在指定文件夹下生成CSV文件
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_csv = os.path.join(folder_path, f"{folder_name}_files.csv")
    print(f"将在文件夹 '{folder_path}' 下生成CSV文件: '{os.path.basename(output_csv)}'")
    list_files_to_csv(folder_path, output_csv)