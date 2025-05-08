from openslide import OpenSlide
import numpy as np
from math import ceil
import openslide
import os
import tifffile
import cv2
from tqdm import tqdm
import time
import glob
import copy
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 512

gfi = lambda img,ind : copy.deepcopy(img[ind[0]:ind[1], ind[2]:ind[3]])


def find_file(path,depth_down,depth_up=0,suffix='.xml'):
    ret = []
    for i in range(depth_up,depth_down):
        _path = os.path.join(path,'*/'*i+'*'+suffix)
        ret.extend(glob.glob(_path))
    ret.sort()
    return ret

def up_to16_manifi(hw):
    return int(ceil(hw[0]/TILE_SIZE)*TILE_SIZE), int(ceil(hw[1]/TILE_SIZE)*TILE_SIZE)

def gen_im(wsi, index):
    ind = 0
    while True:
        temp_img = gfi(wsi, index[ind])
        ind+=1
        yield temp_img
def get_name_from_path(file_path:str, ret_all:bool=False):
    dir, n = os.path.split(file_path)
    n, suffix = os.path.splitext(n)
    if ret_all:
        return dir, n, suffix
    return n

def gen_patches_index(ori_size, *, img_size=224, stride = 224,keep_last_size = False):

    """
        这个函数用来按照输入的size和patch大小，生成每个patch所在原始的size上的位置

        keep_last_size：表示当size不能整除patch的size的时候，最后一个patch要不要保持输入的img_size
        
        返回：
            一个np数组，每个成员表示当前patch所在的x和y的起点和终点如：
                [[x_begin,x_end,y_begin,y_end],...]
    """
    height, width = ori_size[:2]
    index = []
    if height<img_size or width<img_size: 
        print("input size is ({} {}), small than img_size:{}".format(height, width, img_size))
        return index
        
    for h in range(0, height+1, stride):
        xe = h+img_size
        if h+img_size>height:
            xe = height
            h = xe-img_size if keep_last_size else h

        for w in range(0, width+1, stride):
            ye = w+img_size
            if w+img_size>width:
                ye = width
                w = ye-img_size if keep_last_size else w
            index.append(np.array([h, xe, w, ye]))

            if ye==width:
                break
        if xe==height:
            break
    return index

def just_ff(path:str,*,file=False,floder=True,create_floder=False, info=True):
    """
    Check the input path status. Exist or not.

    Args:
        path (str): _description_
        file (bool, optional): _description_. Defaults to False.
        floder (bool, optional): _description_. Defaults to True.
        create_floder (bool, optional): _description_. Defaults to False.
        info (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if file:
        return os.path.isfile(path)
    elif floder:
        if os.path.exists(path):
            return True
        else:
            if create_floder:
                try:
                    os.makedirs(path) 
                    if info:
                        print(r"Path '{}' does not exists, but created ！！".format(path))
                    return True
                except ValueError:
                    if info:
                        print(r"Path '{}' does not exists, and the creation failed ！！".format(path))
                    pass
            else:
                if info:
                    print(r"Path '{}' does not exists！！".format(path))
                return False
                

def just_dir_of_file(file_path:str, create_floder:bool=True):
    """_summary_
    Check the dir of the input file. If donot exist, creat it!
    Args:
        file_path (_type_): _description_
        create_floder (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    _dir = os.path.split(file_path)[0]
    return just_ff(_dir, create_floder = create_floder)

def split_path(root_path:str, input_path:str):
    path_split = os.sep
    while(root_path[-1]==path_split):
        root_path = root_path[0:len(root_path)-1]
    ret_path = input_path[len(root_path):len(input_path)]
    if len(ret_path) == 0:
        return ''
    while(ret_path[0]==path_split):
        ret_path = ret_path[1:len(ret_path)]
    return ret_path

def gen_pyramid_tiff(in_file, out_file, thumb_out_dir, select_level=0):
    '''
    生成两层金字塔的svs文件：
      - 第一层为原始10x图像（mpp=1）
      - 第二层为下采样一半的5x图像
    '''
    svs_desc = 'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'
    
    # 打开tif文件
    odata = openslide.open_slide(in_file)
    if 'mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X' in odata.properties:
        mpp = float(odata.properties['mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X'])
    elif 'openslide.mpp-x' in odata.properties:
        mpp = float(odata.properties['openslide.mpp-x'])
    else:
        mpp = 1.0  # 默认设为1.0
    
    # 对于mpp为1的tif文件，我们固定放大倍数为10x
    mag = 10

    # 保持分辨率定义（实际内容中，SVS文件保存的分辨率依然反映原始MPP）
    resolution = (10000 / mpp, 10000 / mpp)  
    resolutionunit = 'CENTIMETER'           

    if odata.properties.get('aperio.Filename') is not None:
        filename = odata.properties['aperio.Filename']
    else:
        filename = get_name_from_path(in_file)

    print(f"loading '{in_file}'")
    start = time.time()
    dimensions = odata.level_dimensions[select_level]
    region = odata.read_region((0, 0), select_level, dimensions)
    image = np.array(region.convert('RGB'))
    print(f"finished loading '{in_file}', costing time: {time.time()-start:.2f}s")
    
    tile_hw = np.int64([TILE_SIZE, TILE_SIZE])
    width, height = image.shape[0:2]

    # 仅生成两层金字塔：
    # 第一层：10x（原始尺寸），第二层：5x（尺寸下采样为原图的一半）
    multi_hw = [(width, height), (width // 2, height // 2)]
    
    with tifffile.TiffWriter(out_file, bigtiff=True) as tif:
        thw = tile_hw.tolist()
        compression = ['JPEG', 95, dict(outcolorspace='YCbCr')]
        kwargs = dict(subifds=0, photometric='rgb', planarconfig='CONTIG',
                      compression=compression, dtype=np.uint8, metadata=None)

        for i, hw in enumerate(multi_hw):
            # 对尺寸进行向TILE_SIZE对齐
            hw_aligned = up_to16_manifi(hw)
            
            # 对图像进行缩放处理：第一层直接调整尺寸，第二层为原图下采样一半
            if i == 0:
                # 第一层（10x）：保持原始图像尺寸
                temp_img = cv2.resize(image, (hw_aligned[1], hw_aligned[0]))
            else:
                # 第二层（5x）：下采样原图
                temp_img = cv2.resize(image, (hw_aligned[1], hw_aligned[0]))

            # 若图像尺寸不足，对齐补白
            new_x, new_y = hw_aligned
            new_wsi = np.ones((new_x, new_y, 3), dtype=np.uint8) * 255
            new_wsi[0:temp_img.shape[0], 0:temp_img.shape[1], :] = temp_img[..., :3]

            # 生成按照TILE_SIZE切割的patch索引及对应的图像生成器
            index = gen_patches_index((new_x, new_y), img_size=TILE_SIZE, stride=TILE_SIZE)
            gen = gen_im(new_wsi, index)
            
            if i == 0:
                # 第一层描述信息
                desc = svs_desc.format(mag=mag, filename=filename, mpp=mpp)
                tif.write(data=gen, shape=(*hw_aligned, 3), tile=thw[::-1],
                          resolution=resolution, description=desc, **kwargs)
                # 同时生成缩略图，保存至thumb_out_dir
                thumb_save_path = os.path.join(thumb_out_dir, f'{filename}_thumbnail.jpg')
                os.makedirs(os.path.dirname(thumb_save_path), exist_ok=True)
                cv2.imwrite(thumb_save_path, new_wsi)
            else:
                # 第二层描述信息简单标识
                tif.write(data=gen, shape=(*hw_aligned, 3), tile=thw[::-1],
                          resolution=resolution, resolutionunit=resolutionunit,
                          description='5x layer', **kwargs)

# tif所在的目录
DATA_DIR = 'I:/sgw/shenyang_10xtif/'
#保存目录
SAVE_DIR = 'I:/sgw/shenyang_10xsvs/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 定义缩略图保存目录
THUMBNAIL_SAVE_DIR = 'I:/sgw/shenyang_10xsvs/shenyang_thumb/'
if not os.path.exists(THUMBNAIL_SAVE_DIR):
    os.makedirs(THUMBNAIL_SAVE_DIR)

def process_file(w_name):
    """处理单个WSI文件（tif转换为svs）"""
    t1 = time.perf_counter()
    wsi_name = get_name_from_path(w_name)
    diff_path = split_path(DATA_DIR, get_name_from_path(w_name, ret_all=True)[0])
    save_path = os.path.join(SAVE_DIR, diff_path, f'{wsi_name}.svs')
    # 如果 svs 文件已经存在，则跳过
    if just_ff(save_path, file=True):
        print(f"{wsi_name} 已存在，跳过。")
        return
    just_dir_of_file(save_path)
    # 调用时传入缩略图目录
    gen_pyramid_tiff(w_name, save_path, THUMBNAIL_SAVE_DIR)
    print(f'{wsi_name}:', time.perf_counter() - t1)

if __name__ == '__main__':
    # 查找所有tif文件
    wsi_list = find_file(DATA_DIR, 1, suffix='.tif')
    max_workers = 5  # 根据服务器核数进行调整

    # 使用ProcessPoolExecutor进行多进程并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, w_name) for w_name in wsi_list]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            # 可在此处添加对每个future的结果处理（目前不需要捕获返回值）
            pass