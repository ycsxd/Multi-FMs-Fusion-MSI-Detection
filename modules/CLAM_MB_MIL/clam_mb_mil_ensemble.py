import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL



class CLAM_MB_MIL_ENSEMBLE(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, num_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, act='relu', instance_eval=False,
                 in_dims = [1536,1536]):
        """
        参数：
            in_dims: 一个整数列表，每个元素对应输入列表中每个矩阵的特征维度 d
            gate, size_arg, dropout, k_sample, num_classes, instance_loss_fn, subtyping, embed_dim, act, instance_eval:
                      与 CLAM_MB_MIL 类中相应参数的意义一致。
        说明：
            该类接受一个列表，每个元素都是形状为 (n, d_i) 的特征矩阵，其中 n 必须相同，d_i 可以不同。
            通过为每个分支设置单独的特征投影层（将 d_i 映射到统一的 embed_dim）
            以及独立的 CLAM_MB_MIL 模块，最终对各个分支的预测结果（例如 logits）进行融合，
            得到最终的 bag 级别预测结果。
        """
        super(CLAM_MB_MIL_ENSEMBLE, self).__init__()
        # 使用原始输入的两个分支及其合并分支，因此branch_dims包含原始输入维度和合并后的维度
        self.branch_dims = in_dims + [sum(in_dims)]  # 例如 [1536, 1536, 3072]
        self.num_branches = len(self.branch_dims)
        self.num_classes = num_classes
        
        # 不需要投影层，因此移除projection_layers
        self.branches = nn.ModuleList([
            CLAM_MB_MIL(gate=gate, size_arg=size_arg, dropout=dropout, k_sample=k_sample,
                        num_classes=num_classes, instance_loss_fn=instance_loss_fn, subtyping=subtyping,
                        embed_dim=branch_dim, act=act, instance_eval=instance_eval)
            for branch_dim in self.branch_dims
        ])
    
    def forward(self, feature_list, label=None, instance_eval=False, return_features=False, attention_only=False,
                return_WSI_attn=False, return_WSI_feature=False):
        """
        参数：
            feature_list: 列表，每个元素形状为 (n, d_i)，其中所有 n 必须相同，d_i 可以不同。
            label: 标签信息（可选），用于计算 instance 级别 loss。
            instance_eval: 是否开启 instance 级别评估。
            return_features: 是否返回融合后的特征表示。
            attention_only: 是否仅返回注意力矩阵。
            return_WSI_attn: 是否返回 WSI 的注意力信息。
            return_WSI_feature: 是否返回 WSI 的特征信息。
        返回：
            一个字典，包含 bag 级别的预测结果（logits, Y_prob, Y_hat），以及（可选的）instance_loss、features、WSI_attn 和 WSI_feature 等。
        """
        branch_outputs = []
        instance_loss_total = 0.0
        inst_loss_count = 0
        features_agg = []
        logits_list = []
        
        if len(feature_list) != 2:
            raise ValueError("预期输入feature_list长度为2")
        # 根据输入张量的维度选择合适的拼接维度
        if feature_list[0].ndim == 3:  # 形状 [B, n, d]
            merged_feature = torch.cat(feature_list, dim=2)
        elif feature_list[0].ndim == 2:  # 形状 [n, d]
            merged_feature = torch.cat(feature_list, dim=1)
        else:
            raise ValueError("不支持的feature张量维度")
        # 构造新的特征列表，包括原始两个和合并的
        new_feature_list = feature_list + [merged_feature]

        # 遍历每个分支，直接使用原始特征，不经过投影变换
        for i in range(self.num_branches):
            x = new_feature_list[i]
            branch_out = self.branches[i](x, label=label, instance_eval=instance_eval, return_features=return_features,
                                            attention_only=attention_only, return_WSI_attn=return_WSI_attn, return_WSI_feature=return_WSI_feature)
            branch_outputs.append(branch_out)
            
            if not attention_only:
                logits_list.append(branch_out['logits'])
            
            if instance_eval and (label is not None) and ('instance_loss' in branch_out):
                instance_loss_total += branch_out['instance_loss']
            
            if return_features and ('features' in branch_out):
                features_agg.append(branch_out['features'])
        
        forward_return = {}
        if instance_eval and (label is not None):
            forward_return['instance_loss'] = instance_loss_total  # 使用所有分支loss的总和
        
        if return_features and len(features_agg) > 0:
            # 这里采用平均，各分支的特征加权融合；也可以选择拼接等其他策略
            feat_stack = torch.stack(features_agg, dim=0)
            forward_return['features'] = torch.mean(feat_stack, dim=0)
        
        if return_WSI_attn:
            # 这里暂取第一个分支的 WSI 注意力信息；也可以改为融合多分支结果
            forward_return['WSI_attn'] = branch_outputs[0].get('WSI_attn', None)
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = branch_outputs[0].get('WSI_feature', None)
        
        if not attention_only:
            stacked_logits = torch.stack(logits_list, dim=0)
            forward_return['logits'] = torch.mean(stacked_logits, dim=0)
        
        return forward_return 