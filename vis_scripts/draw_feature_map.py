import umap.umap_ as umap
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(grandparent_dir)
import numpy as np
import argparse
from utils.model_utils import get_model_from_yaml
import matplotlib.pyplot as plt
import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop,ds_val_loop,dtfd_val_loop
import warnings
from utils.wsi_utils import WSI_Dataset,CDP_MIL_WSI_Dataset,LONG_MIL_WSI_Dataset
import torch
from utils.model_utils import get_model_from_yaml,get_criterion
import ast
import itertools
warnings.filterwarnings('ignore')

def draw_umap(feature_tensor, label_tensor, id2class, save_path, fig_size=(10, 8), seed=42, n_neighbors=None, min_dist=0.1, learning_rate=1.0, n_epochs=None, metric='euclidean'):
    """
    Draws a UMAP plot for the given feature tensor and labels, and saves it to the specified path.

    Parameters:
    feature_tensor (numpy.ndarray): An N x D tensor of features.
    label_tensor (numpy.ndarray): An N x 1 tensor of labels.
    id2class (str): str type dictionary mapping label ids to class names.
    save_path (str): The path where the plot will be saved.
    fig_size (tuple, optional): The size of the figure, default is (10, 8).
    seed (int, optional): Random seed for reproducibility.
    n_neighbors (int, optional): Number of neighbors for UMAP. If None, defaults to min(30, num_samples-1).
    min_dist (float, optional): Minimum distance for UMAP.
    learning_rate (float, optional): Learning rate for UMAP.
    n_epochs (int, optional): Number of epochs for UMAP training; if None, UMAP will choose a default.
    metric (str, optional): Distance metric for UMAP.

    Returns: 
    None
    """
    id2class = ast.literal_eval(id2class)
    if n_neighbors is None:
         n_neighbors = min(30, feature_tensor.shape[0] - 1)  # 默认 30，且保证小于样本数
    umap_inst = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=seed, learning_rate=learning_rate, n_epochs=n_epochs)
    umap_result = umap_inst.fit_transform(feature_tensor)

    plt.figure(figsize=fig_size)
    for label_id, class_name in id2class.items():
        indices = np.where(label_tensor == label_id)[0]
        plt.scatter(umap_result[indices, 0], umap_result[indices, 1], label=class_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
def main(args):
    yaml_path = args.yaml_path
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    mil_model = get_model_from_yaml(yaml_args)
    model_name = yaml_args.General.MODEL_NAME
    print(f"Model name: {model_name}")
    num_classes = yaml_args.General.num_classes
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    # CDP_MIL and LONG_MIL models have different dataset pipeline
    if model_name == 'CDP_MIL':
        raise NotImplementedError("CDP_MIL model is not supported for feature map visualization now.")
        test_ds = CDP_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.BeyesGuassian_pt_dir,'test')
    elif model_name == 'LONG_MIL':
        LONG_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
    test_ds = WSI_Dataset(test_dataset_csv,'test')
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False)
    model_weight_path = args.ckpt_path
    print(f"Model weight path: {model_weight_path}")
    device = torch.device(f'cuda:{yaml_args.General.device}')
    criterion = get_criterion(yaml_args.Model.criterion)
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier,attention,dimReduction,attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(model_weight_path,weights_only=True)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier,attention,dimReduction,attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path,weights_only=True))

    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL' or yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        WSI_features = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight,retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
        WSI_features =  ds_val_loop(device,num_classes,mil_model,test_dataloader,criterion,retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        WSI_features =  dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,yaml_args.Model.num_Group,yaml_args.Model.grad_clipping,yaml_args.Model.distill,yaml_args.Model.total_instance,retrun_WSI_feature=True)
    else:
        WSI_features =  val_loop(device,num_classes,mil_model,test_dataloader,criterion,retrun_WSI_feature=True)

    WSI_labels = np.array(test_ds.labels_list)
    # 如果传入的 fig_size 不是元组，则尝试转换
    if not isinstance(args.fig_size, tuple):
        try:
            args.fig_size = ast.literal_eval(args.fig_size)
        except Exception as e:
            raise ValueError("无法解析 fig_size 参数，请确保其格式为元组，如 '(10,8)'") from e

    # 解析新的 UMAP 参数列表
    try:
        neighbors_list = ast.literal_eval(args.neighbors_list)
        min_dist_list = ast.literal_eval(args.min_dist_list)
        learning_rate_list = ast.literal_eval(args.learning_rate_list)
        n_epochs_list = ast.literal_eval(args.n_epochs_list)
        seed_list = ast.literal_eval(args.seed_list)
    except Exception as e:
        raise ValueError("无法解析 UMAP 参数列表，请确保格式正确") from e

    task_id = 0
    for n, md, lr, ep, s in itertools.product(neighbors_list, min_dist_list, learning_rate_list, n_epochs_list, seed_list):
         task_id += 1
         print(f'------------- Task {task_id} Started -------------')
         base_save_path = args.save_path
         if base_save_path == '':
              base_save_path = f'./umap_n{n}_md{md}_lr{lr}_ep{ep}_seed{s}.png'
         else:
              if os.path.isdir(base_save_path):
                   base_save_path = os.path.join(base_save_path, f"umap_n{n}_md{md}_lr{lr}_ep{ep}_seed{s}.png")
              else:
                   base_name, ext = os.path.splitext(base_save_path)
                   base_save_path = f"{base_name}_n{n}_md{md}_lr{lr}_ep{ep}_seed{s}{ext}"
         draw_umap(WSI_features, WSI_labels, args.id2class, base_save_path, args.fig_size, s, n_neighbors=n, min_dist=md, learning_rate=lr, n_epochs=ep, metric=args.umap_metric)
         print(f"UMAP plot saved at {base_save_path}")

if __name__ == "__main__":    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--yaml_path', type=str, default='', help='path to yaml file')
    argparser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    argparser.add_argument('--ckpt_path', type=str, default='', help='path to pretrained weights')
    argparser.add_argument('--save_path', type=str, default='', help='path to save the model or directory to save multiple plots')
    argparser.add_argument('--fig_size', type=tuple, default=(10,8), help='size of the figure')
    argparser.add_argument('--id2class', type=str, default='', help='str type dictionary mapping label ids to class names')
    argparser.add_argument('--test_dataset_csv', type=str, default='', help='path to dataset csv file')
    argparser.add_argument('--neighbors_list', type=str, default='[8,15]', help='list of neighbors for UMAP, e.g., "[8,15]"')
    argparser.add_argument('--min_dist_list', type=str, default='[0.95,0.9]', help='list of min_dist values for UMAP, e.g., "[0.1,0.5,0.9]"')
    argparser.add_argument('--learning_rate_list', type=str, default='[0.5,0.1]', help='list of learning_rate values for UMAP, e.g., "[0.5,1.0]"')
    argparser.add_argument('--n_epochs_list', type=str, default='[50]', help='list of n_epochs values for UMAP, e.g., "[50,100]" (use None for default)')
    argparser.add_argument('--seed_list', type=str, default='[0, 7, 12, 19, 22]', help='list of seeds for UMAP, e.g., "[42,7]"')
    argparser.add_argument('--umap_metric', type=str, default='euclidean', choices=['euclidean', 'manhattan', 'cosine', 'hamming'], help='distance metric for UMAP. Choices: %(choices)s')
    args = argparser.parse_args()
    main(args)