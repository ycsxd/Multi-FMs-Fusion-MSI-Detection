'''
Segment TIFF 和 SVS 文件生成特征矩阵
支持 tif 和 svs 格式，通过 --file_format 参数指定
'''
import warnings
warnings.filterwarnings("ignore")
import torch
import timm
import os
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
import argparse
import time
from PIL import Image
import tifffile as tiff
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from openslide import OpenSlide
from torch.utils.data import Dataset, DataLoader

Image.MAX_IMAGE_PIXELS = None

def parse_arguments():
    """
    解析命令行参数，设置输入、输出路径、各类阈值、模型选择以及文件格式等参数。
    """
    parser = argparse.ArgumentParser(
        description='基于文件名与标签，将TIFF或SVS图像分割为图块，并生成特征矩阵。'
    )
    parser.add_argument('--label_txt', type=str, default='',
                        help='标签 TXT 文件路径。')
    parser.add_argument('--input_folder', type=str, default='',
                        help='包含图像文件（TIFF 或 SVS）的输入文件夹。')
    parser.add_argument('--output_folder', type=str, default='',
                        help='存储生成特征矩阵的输出文件夹。')
    parser.add_argument('--threshold', type=int, default=200,
                        help='前景与背景分离的阈值。')
    parser.add_argument('--patch_size', type=int, default=448,
                        help='图块的尺寸。')
    parser.add_argument('--edge_threshold', type=int, default=1000,
                        help='边缘检测的阈值。')
    parser.add_argument('--model_choice', type=str, default='uni',
                        help='模型选择：uni, uni2, virchow2, musk, ctranspath, gigapath, PathoDuetVits')
    parser.add_argument('--svs_level', type=int, choices=[0, 1, 2], default=0,
                        help='如果使用svs的level参数来裁剪，则需要指定level，mpp参数无效')
    parser.add_argument('--mpp', type=str, choices=['0.5', '1.0'], default='1.0',
                        help='mpp 参数：0.5um/px 对应20x；1.0um/px 对应10x')
    parser.add_argument('--file_format', type=str, choices=['tif', 'svs'], default='svs',
                        help='文件格式：tif 或 svs')
    return parser.parse_args()

def load_and_parallel_model(model_choice):
    # 根据传入的模型名称加载对应的模型（这里只举例部分，其它可按需扩展）
    if model_choice == 'PathoDuetVits':
        from models.PathoDuetVits import VisionTransformerMoCo
        model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
        model.load_state_dict(torch.load("/share/home/guoweis/checkpoint/checkpoint_HE.pth", weights_only=True), strict=True)
    elif model_choice == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath")
        model.load_state_dict(torch.load('/share/home/guoweis/checkpoint/pytorch_model_giga.bin', weights_only=True), strict=True)
    elif model_choice == 'uni':
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load("/share/home/guoweis/checkpoint/pytorch_model.bin", weights_only=True), strict=True)
    elif model_choice == 'ctranspath':
        from models.ctran import ctranspath
        model = ctranspath()  # ctranspath 需要安装 timm-0.5.4.tar 包
        model.load_state_dict(torch.load("/share/home/guoweis/checkpoint/ctranspath.pth", weights_only=True)['model'], strict=True)
    elif model_choice == 'virchow2':
        from timm.layers import SwiGLUPacked
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False,
                                  mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("/share/home/guoweis/checkpoint/virchow2/pytorch_model.bin", weights_only=True), strict=True)
    elif model_choice == 'resnet18':
        from models.resnet import ResNet18
        model = ResNet18()
    elif model_choice == 'uni2':
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", **timm_kwargs)
        model.load_state_dict(torch.load("/share/home/guoweis/checkpoint/uni2/pytorch_model.bin", weights_only=True), strict=True)
    elif model_choice == 'musk':
        from musk import utils, modeling
        model = timm.create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("/share/home/guoweis/checkpoint/musk/model.safetensors", model, 'model|module', 'weights_only=True')
    else:
        raise ValueError(f"未知的模型选择: {model_choice}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 如果有多块GPU，则用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    return model
   


def feature_extractor_adapter(model, batch,model_name):
	if model_name == 'plip':
		features = model.get_image_features(batch)
	elif model_name == 'conch':
		features = model.encode_image(batch)
	elif model_name == 'virchow':
		features = model(batch)
		class_token = features[:, 0]    
		patch_tokens = features[:, 1:]  
		features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1) 
	elif model_name == 'virchow_v2':
		features = model(batch)
		class_token = features[:, 0]    
		patch_tokens = features[:, 5:] 
		features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
	else:
		features = model(batch)
	return features
def get_slide_mpp(slide):
    """
    根据目标MPP，选择最接近的级别并获取MPP信息。

    Args:
        slide (OpenSlide): 打开的SVS文件。
        target_mpp (float): 目标微米每像素。

    Returns:
        tuple: 最接近目标MPP的级别和MPP信息。
    """
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    
    if mpp_x is None or mpp_y is None:
        return None, (None, None)
    
    current_mpp = float(mpp_x)  # 假设mpp-x和mpp-y相同

    return 0, (current_mpp, current_mpp)

def process_file(args):
    """
    处理单个文件（TIFF 或 SVS）：按照阈值和边缘检测条件分割图像为图块，
    然后通过预训练模型提取特征，最终保存特征矩阵及返回对应标签。
    """
    (base_name, labels, input_folder, output_folder, THRESHOLD, PATCH_SIZE,
     EDGE_THRESHOLD, model_choice, target_mpp, file_format, svs_level) = args

    model_output_folder = os.path.join(output_folder, f"{model_choice}")
    feature_path = os.path.join(model_output_folder, f"{base_name}.pt")
    if os.path.exists(feature_path):
        print(f"{feature_path} 已存在，跳过处理 {base_name}")
        return None

    # 在当前进程中加载模型，而非使用全局变量
    local_model = load_and_parallel_model(model_choice)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_choice in ['uni', 'uni2', 'gigapath', 'ctranspath', 'virchow2']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
        ])

    all_filename = os.listdir(input_folder)
    expected_filename = f'{base_name}.{file_format}'
    if expected_filename not in all_filename:
        print(f"未找到文件: {expected_filename}")
        return None
    input_path = os.path.join(input_folder, expected_filename)

    # 读取并处理图像
    if file_format == 'tif':
        img = tiff.imread(input_path)
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

    elif file_format == 'svs':
        if svs_level in [0, 1, 2]:
            print(f"使用svs的level参数来裁剪: {svs_level}")
            with OpenSlide(input_path) as slide:
                img = slide.read_region((0, 0), svs_level, slide.level_dimensions[svs_level])
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            print(f"使用mpp参数来裁剪: {target_mpp}")
            with OpenSlide(input_path) as slide:
                level, mpp = get_slide_mpp(slide)
                if mpp[0] is None or mpp[1] is None:
                    print(f"跳过裁剪，因为 {base_name}.svs 的MPP信息未知。")
                    return None
                scaling_factor = mpp[0] / float(target_mpp)
                print(f"开始裁剪: {base_name}, 标签: {labels[base_name]}, current_mpp: {mpp[0]}, target_mpp: {target_mpp} um/pixel, scaling_factor: {scaling_factor}")
                region = slide.read_region((0, 0), level, slide.level_dimensions[level])
                img = np.array(region)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("不支持的文件格式。")

    print(f"开始提取特征: {base_name}, 标签: {labels[base_name]}")
    height_img, width_img = img.shape[:2]
    if height_img < PATCH_SIZE or width_img < PATCH_SIZE:
        print(f"图像大小太小，无法提取特征: {base_name}")
        return None

    step_size = PATCH_SIZE
    features = []
    unfiltered_count = 0

    for y in range(0, height_img - PATCH_SIZE + 1, step_size):
        for x in range(0, width_img - PATCH_SIZE + 1, step_size):
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            _, binary_patch = cv2.threshold(gray_patch, THRESHOLD, 255, cv2.THRESH_BINARY)
            foreground_ratio = np.mean(binary_patch == 0)
            if foreground_ratio > 0.5:
                edges = cv2.Canny(gray_patch, 100, 200)
                edge_count = np.sum(edges > 0)
                if edge_count > EDGE_THRESHOLD:
                    patch_tensor = transform(patch).to(device).unsqueeze(0)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            patch_features = feature_extractor_adapter(local_model, patch_tensor, model_choice)
                    feature_vector = patch_features.squeeze(0).cpu().numpy().flatten()
                    features.append(feature_vector)
                else:
                    unfiltered_count += 1

    if features:
        feature_matrix = np.stack(features)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)
        torch.save(feature_matrix, feature_path)
        print(f"已保存{base_name}特征矩阵维度: {feature_matrix.shape}, label: {labels[base_name]}，过滤模糊图块数: {unfiltered_count}")
        if feature_matrix.shape[0] < 50:
            print(f"{base_name} 特征向量数量小于 50，不写入 CSV")
            return None
        else:
            return (base_name, labels[base_name])
    else:
        print(f"{base_name}未检测到符合条件的图块")
        return None


def main():
    """
    主函数：执行图像分割、特征提取，并将特征矩阵保存，同时输出一个 CSV 文件记录相应信息。
    """
    total_start_time = time.time()
    args = parse_arguments()
    print(args)
    label_txt = args.label_txt
    input_folder = args.input_folder
    output_folder = args.output_folder
    THRESHOLD = args.threshold
    PATCH_SIZE = args.patch_size
    EDGE_THRESHOLD = args.edge_threshold
    model_choice = args.model_choice
    mpp = args.mpp
    file_format = args.file_format
    svs_level = args.svs_level

    # 从 TXT 文件中读取标签
    labels = {}
    filenames = []
    with open(label_txt, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                fn, label = parts[0], parts[1]
                labels[fn] = label
                filenames.append(fn)
    print(f"共有 {len(filenames)} 个文件需要处理")

    args_list = []
    for fl in filenames:
        args_tuple = (fl, labels, input_folder, output_folder, THRESHOLD,
                      PATCH_SIZE, EDGE_THRESHOLD, model_choice, mpp, file_format, svs_level)
        args_list.append(args_tuple)

    # 确保输出文件夹存在
    model_output_folder = os.path.join(output_folder, model_choice)
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)
    # 提前构造 CSV 文件路径，并初始化 CSV 文件（如果还不存在的话）
    csv_path = os.path.join(model_output_folder, f"{model_choice}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("slide_path,label\n")

    with Pool(processes=cpu_count()//5) as pool:
        all_results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
    
    for result in all_results:
        if result:
            base_name, label = result
            feature_path = os.path.join(model_output_folder, f"{base_name}.pt")
            print(f"已保存特征矩阵: {feature_path}")
            with open(csv_path, 'a') as f:
                f.write(f"{feature_path},{label}\n")
        else:
            continue

    print(f"所有处理完成，保存{csv_path}... ")
    print(f"总运行时间: {(time.time() - total_start_time):.2f} 秒")

if __name__ == "__main__":
    main() 