# python checktif.py folder_path   
import os
import sys
from PIL import Image, ImageFile
import multiprocessing  # 导入多进程模块

# 关闭图像像素数量限制（防止处理超大tif图像时报错）
Image.MAX_IMAGE_PIXELS = None
# 加载截断图像（防止图像损坏时读取错误）
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_thumbnail(image_path, thumbnail_path, thumbnail_size=(300, 300)):
    """
    为给定的tif图像生成缩略图，并保存为JPEG格式。

    参数：
        image_path: 原始tif图像路径
        thumbnail_path: 要保存的缩略图路径
        thumbnail_size: 缩略图的尺寸（默认宽300, 高300）
    """
    try:
        with Image.open(image_path) as im:
            print(f"-------------------processing image: {image_path}--------------------------------")
            im.thumbnail(thumbnail_size)
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            im.save(thumbnail_path, "JPEG")
            print(f"thumbnail generated successfully: {thumbnail_path}")
    except Exception as e:
        print(f"failed to generate thumbnail (image_path: {image_path})")

def main(folder):
    """
    遍历目标文件夹，查找所有tif文件并生成缩略图
    """
    thumb_dir = os.path.join(folder, 'thumb')
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
    
    # 获取所有tif文件的完整路径
    tif_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.tif', '.tiff'))]
    
    # 逐个处理生成缩略图
    for file in tif_files:
        generate_thumbnail(file, os.path.join(thumb_dir, os.path.basename(file).replace('.tif', '_thumbnail.jpg')))

if __name__ == '__main__':
    
    folder_path = input("please input the folder path of tif files: ").strip()
    if not os.path.exists(folder_path):
        print("error: folder path not exists!")
    else:
        main(folder_path)