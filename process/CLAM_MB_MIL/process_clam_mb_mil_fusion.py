import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from modules.CLAM_MB_MIL.clam_mb_mil_fusion import CLAM_MB_MIL_Fusion
from utils.process_utils import get_process_pipeline,get_act
from utils.general_utils import set_global_seed,init_epoch_info_log,add_epoch_info_log,early_stop
from utils.model_utils import get_optimizer,get_scheduler,get_criterion,save_last_model,save_log,model_select
from utils.loop_utils import cal_scores
from tqdm import tqdm
from collections import Counter
import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class WSI_Dataset(Dataset):
    def __init__(self,dataset_info_csv_path,group):
        assert group in ['train','val','test'], 'group must be in [train,val,test]'
        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group+'_slide_path'].dropna().to_list()
        self.labels_list = self.dataset_df[group+'_label'].dropna().to_list()

    def __len__(self):
        return len(self.slide_path_list)
    
    def __getitem__(self, idx):

        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path)
        if isinstance(feat, list):
            return feat,label
        else:
            if len(feat.shape) == 3:
                feat = feat.squeeze(0)
        return feat,label

    def is_None_Dataset(self):
        return (self.__len__() == 0)    
    
    def is_with_labels(self):
        return (len(self.labels_list) != 0)
def fusion_train_loop(device, model, loader, criterion, optimizer, scheduler, bag_weight):
    """
    fusion_train_loop 用于训练 fusion 模型。
    本版本要求数据集中返回的 bag 必须为 list，否则报错。
    计算 bag 级别 loss 时，同时融合 instance_loss（若有）。
    """
    start = time.time()
    model.train()
    train_loss_log = 0.0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        # 获取标签并转移到 device 上
        label = data[1].long().to(device)
        # 获取 bag 数据，检查是否为 list
        bag = data[0]
        if not isinstance(bag, list):
            raise ValueError("bag输入必须为list")
        feature_list = [b.to(device).float() for b in bag]

        # forward 时传入 feature_list
        forward_return = model(feature_list, label=label)
        instance_loss = forward_return.get('instance_loss', 0)
        logits = forward_return['logits']
        loss = criterion(logits, label)
        total_loss = loss * bag_weight + instance_loss * (1 - bag_weight)
        train_loss_log += total_loss.item()
        total_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    total_time = time.time() - start
    return train_loss_log, total_time

def fusion_val_loop(device, num_classes, model, loader, criterion, bag_weight,
                      return_WSI_feature=False, return_WSI_attn=False):
    """
    fusion_val_loop 用于验证 fusion 模型。
    与训练循环类似，要求 bag 输入必须为 list，否则报错。
    除了计算 bag 级别 loss 外，还可支持返回 WSI 的特征或注意力信息。
    """
    model.eval()
    val_loss_log = 0.0
    labels = []
    bag_predictions_after_normal = []
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0]
            if not isinstance(bag, list):
                raise ValueError("bag输入必须为list")
            feature_list = [b.to(device).float() for b in bag]

            if return_WSI_feature:
                out = model(feature_list, label=label, return_WSI_feature=True)
                WSI_feature = out.get('WSI_feature', None)
                if WSI_feature is not None:
                    WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                out = model(feature_list, label=label, return_WSI_attn=True)
                WSI_attn = out.get('WSI_attn', None)
                if WSI_attn is not None:
                    WSI_attns.append(WSI_attn)
                continue

            out = model(feature_list, label=label)
            instance_loss = out.get('instance_loss', 0)
            val_logits = out['logits']
            # squeeze 保证 logits 维度正确（例如 (1, num_classes) --> (num_classes,)）
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits, dim=0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            loss = criterion(val_logits, label)
            total_loss = loss * bag_weight + instance_loss * (1 - bag_weight)
            val_loss_log += total_loss.item()
    if return_WSI_feature:
        # 将各个 WSI 特征拼接后返回 numpy 数组
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    # 计算评价指标（cal_scores 方法未改动）
    val_metrics = cal_scores(bag_predictions_after_normal, labels, num_classes)
    val_loss_log /= len(loader)
    return val_loss_log, val_metrics

def make_weights_for_balanced_classes_split(dataset):
    """
    根据每个类别的样本数量计算每个样本的权重，达到类别平衡。
    对于每个样本 i，其权重计算为：N / count(label_i)
    其中 N 为样本总数，count(label_i) 为类别 label_i 的样本数。
    """
    N = len(dataset)
    # 将标签转换为整数，并统计出现次数
    labels = [int(label) for label in dataset.labels_list]
    label_counts = Counter(labels)
    # 为每个样本计算对应的权重
    weights = [N / label_counts[label] for label in labels]
    return torch.DoubleTensor(weights)
    
def process_CLAM_MB_MIL_FUSION(args):

    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'test')
    process_pipeline = get_process_pipeline(val_dataset,test_dataset) 
    args.General.process_pipeline = process_pipeline

    
    generator = torch.Generator()
    if args.General.seed is not None:
        generator.manual_seed(args.General.seed)
        set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    
    if args.General.weighted:
        weights = make_weights_for_balanced_classes_split(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), num_workers = num_workers,generator=generator)	
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    print('DataLoader Ready!')
    in_dim = args.Model.in_dim
    embed_dim = args.Model.embed_dim
    subtyping = args.Model.subtyping
    k_sample = args.Model.k_sample
    size_arg = args.Model.size_arg
    dropout = args.Model.dropout
    num_classes = args.General.num_classes
    gate = args.Model.gate
    act = args.Model.act
    instance_eval = args.Model.instance_eval
    device = torch.device(f'cuda:{args.General.device}')
    instance_loss_fn = args.Model.instance_loss_fn
    instance_loss_fn = get_criterion(instance_loss_fn)
    bag_weight = args.Model.bag_weight
    mil_model = CLAM_MB_MIL_Fusion(
        in_dims= in_dim,  # 适当构造 in_dims 列表
        gate=gate,
        size_arg=size_arg,
        dropout=dropout,
        k_sample=k_sample,
        num_classes=num_classes,
        instance_loss_fn=instance_loss_fn,
        subtyping=subtyping,
        embed_dim=embed_dim,  # 如果需要传递 embed_dim
        act=act,
        instance_eval=instance_eval
    )   
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer,base_lr = get_optimizer(args,mil_model)
    scheduler,warmup_scheduler = get_scheduler(args,optimizer,base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    '''
    begin training
    '''
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 9999
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:',process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs),colour='GREEN'):
        if epoch+1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler
        train_loss,cost_time = fusion_train_loop(device,mil_model,train_dataloader,criterion,optimizer,now_scheduler,bag_weight)
        if process_pipeline == 'Train_Val_Test':
            val_loss,val_metrics = fusion_val_loop(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)
            test_loss,test_metrics = fusion_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
        elif process_pipeline == 'Train_Val':
            val_loss,val_metrics = fusion_val_loop(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)
            test_loss,test_metrics = None,None
        elif process_pipeline == 'Train_Test':
            val_loss,val_metrics,test_loss,test_metrics = None,None,None,None
            if epoch+1 == args.General.num_epochs:
                test_loss,test_metrics = fusion_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)


        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)
        
        # model selection, it only works when process_pipeline is 'Train_Val_Test' or 'Train_Val'
        best_val_metric,best_epoch = model_select(REVERSE,args,mil_model.state_dict(),val_metrics,best_model_metric,best_val_metric,epoch,best_epoch)
        print(f"{best_val_metric}")
        '''
        early stop
        '''
        if early_stop(args,epoch_info_log,process_pipeline,epoch,mil_model.state_dict(),best_epoch):
            break

        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model.state_dict(),epoch+1)
            save_log(args,epoch_info_log,best_epoch,process_pipeline)

    
    return best_val_metric
