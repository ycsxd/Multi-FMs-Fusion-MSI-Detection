import os
import cv2
import numpy as np
from openslide import OpenSlide
from tqdm import tqdm  # 添加tqdm模块
import concurrent.futures  # 导入并行处理模块

def get_slide_mpp(slide):
    """
    根据幻灯片的MPP信息获取当前的像素间距。

    Args:
        slide (OpenSlide): 打开的SVS文件。

    Returns:
        tuple: 返回幻灯片的级别和 (mpp_x, mpp_y) 信息。如果信息缺失，则返回 (None, (None, None))
    """
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    
    if mpp_x is None or mpp_y is None:
        return None, (None, None)
    
    current_mpp = float(mpp_x)  # 假设 mpp-x 和 mpp-y 数值相同
    return 0, (current_mpp, current_mpp)

def process_svs_file(input_path, output_folder, target_mpp=1.0):
    """
    处理单个svs文件，将其 resize 到目标MPP后保存为tif格式。
    如果对应名称的tif文件已经存在则跳过处理。

    Args:
        input_path (str): svs文件的路径。
        output_folder (str): 输出tif图片保存的文件夹路径。
        target_mpp (float): 目标微米每像素（默认1.0）。
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.tif")
    
    # 如果对应的tif文件已存在，则跳过处理
    if os.path.exists(output_path):
        print(f"文件 {output_path} 已存在，跳过。")
        return
    
    try:
        with OpenSlide(input_path) as slide:
            level, mpp = get_slide_mpp(slide)
            if mpp[0] is None or mpp[1] is None:
                print(f"跳过裁剪，因为 {base_name}.svs 的MPP信息未知。")
                return

            # 计算缩放因子：当前 mpp 与目标 mpp 的比值
            scaling_factor = mpp[0] / float(target_mpp)
            print(f"开始处理: {base_name}, 当前 mpp: {mpp[0]}, 目标 mpp: {target_mpp}, 缩放比例: {scaling_factor}")
            
            # 读取整张幻灯片区域（级别0）
            region = slide.read_region((0, 0), level, slide.level_dimensions[level])
            img = np.array(region)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # 图像resize
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
            
            # 保存tif图片
            cv2.imwrite(output_path, img)
            print(f"保存图片: {output_path}")
            
    except Exception as e:
        print(f"处理 {base_name}.svs 时发生错误: {e}")

def process_folder(input_folder, target_mpp=1.0):
    """
    遍历指定文件夹下所有svs图片，并使用多进程进行并行处理，保存到输出文件夹。

    Args:
        input_folder (str): 包含svs图片的文件夹路径。
        target_mpp (float): 目标微米每像素（默认1.0）。
    """
    output_folder = os.path.join(input_folder, f'tif_{target_mpp}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有扩展名为 .svs 的文件
    svs_files = [file for file in os.listdir(input_folder) if file.lower().endswith(".svs")]
    
    # 使用ProcessPoolExecutor实现多进程并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for file in svs_files:
            input_path = os.path.join(input_folder, file)
            future = executor.submit(process_svs_file, input_path, output_folder, target_mpp)
            futures.append(future)
        
        # 使用tqdm跟踪进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理svs文件"):
            try:
                future.result()
            except Exception as e:
                print(f"处理文件时发生异常: {e}")

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='将SVS文件转换为TIFF格式')
    parser.add_argument('--input_folder', type=str, help='输入SVS文件夹路径')
    parser.add_argument('--target_mpp', type=float, default=1.0, help='目标微米每像素')
    args = parser.parse_args()

    process_folder(args.input_folder, args.target_mpp)
        