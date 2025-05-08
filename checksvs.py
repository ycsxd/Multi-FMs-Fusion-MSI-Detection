import os
import openslide
import csv

def save_svs_levels_to_csv(folder_path, csv_filename="svs_info.csv"):
    """
    traverse all svs files in the specified folder, and print the level information of each file
    also show the level0 mpp information
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入CSV表头
        csvwriter.writerow(["文件路径", "Level数量", "Level尺寸", "mpp-x", "mpp-y", "错误信息"])
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.svs'):
                    file_path = os.path.join(root, file)
                    print(f"正在处理: {file_path}")
                    # 初始化各项信息
                    level_count = ""
                    dimensions = ""
                    mpp_x = ""
                    mpp_y = ""
                    error_msg = ""
                    
                    try:
                        slide = openslide.OpenSlide(file_path)
                        level_count = slide.level_count
                        dimensions = slide.level_dimensions  # 直接以元组列表的形式展示, 如有需要可转换为字符串
                        
                        # 获取 level0 的 mpp 信息
                        mpp_x = slide.properties.get('openslide.mpp-x', "")
                        mpp_y = slide.properties.get('openslide.mpp-y', "")
                        
                        # 关闭slide以释放资源
                        slide.close()
                    except Exception as e:
                        error_msg = str(e)
                        print(f"处理文件 {file} 时出错: {error_msg}")
                    
                    csvwriter.writerow([file_path, level_count, dimensions, mpp_x, mpp_y, error_msg])


    print("处理完成")

if __name__ == "__main__":
    folder_path = input("please input the folder path of svs files: ").strip()
    if not os.path.exists(folder_path):
        print("error: folder path not exists!")
    else:
        save_svs_levels_to_csv(folder_path)
        print("CSV file has been saved as svs_info.csv")