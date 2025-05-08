import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL
# 3/28 10:00 运行模型备份

class CLAM_MB_MIL_Fusion(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, num_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=512, act='relu', instance_eval=False, use_att_gate=True,
                 in_dims = [1536,1536,2048,512]):
        """
        参数：
            in_dims: 一个整数列表，每个元素对应输入列表中每个矩阵的特征维度 d
            gate, size_arg, dropout, k_sample, num_classes, instance_loss_fn, subtyping, embed_dim, act, instance_eval:
                      与 CLAM_MB_MIL 类中相应参数的意义一致。
         """
        super(CLAM_MB_MIL_Fusion, self).__init__()
        self.num_branches = len(in_dims)
        self.num_classes = num_classes
        self.use_att_gate = use_att_gate
        if self.use_att_gate:
            self.attention_gate = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(self.num_branches)])
        
        self.proj_layers = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(in_dims[i], embed_dim), nn.ReLU()) for i in range(self.num_branches)])
        
        # 添加LayerNorm层，用于融合后的特征归一化
        self.layer_norm = nn.LayerNorm(embed_dim * self.num_branches)
        
        # 使用单一的 CLAM_MB_MIL 模块处理融合后的特征
        self.clam_mb_mil = CLAM_MB_MIL(gate=gate, size_arg=size_arg, dropout=dropout, k_sample=k_sample,
                                       num_classes=num_classes, instance_loss_fn=instance_loss_fn, subtyping=subtyping,
                                       embed_dim=embed_dim*self.num_branches, act=act, instance_eval=instance_eval)
    
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
            一个字典，包含 CLAM_MB_MIL 模块的预测结果（例如 logits, Y_prob, Y_hat 等）。
        """

        # 对每个分支分别进行线性投影，将不同维度的特征转换到512维
        projected_features = [self.proj_layers[i](feature_list[i]) for i in range(self.num_branches)]
        if self.use_att_gate:
            # 计算每个分支的注意力分数，结果形状为 (b, n, 1)
            scores = [self.attention_gate[i](projected_features[i]) for i in range(self.num_branches)]
            # 将各分支分数拼接，形状变为 (b, n, num_branches)
            attn_scores = torch.cat(scores, dim=2)
            # 沿分支维度应用softmax，得到注意力权重
            attn_weights = F.softmax(attn_scores, dim=2)
            # 对每个分支的特征按对应注意力权重加权
            gated_features = [projected_features[i] * attn_weights[:, :, i:i+1] for i in range(self.num_branches)]
            fused_features = torch.cat(gated_features, dim=2)
        else:
            # 直接拼接所有分支的特征
            fused_features = torch.cat(projected_features, dim=2)
        
        # 对融合后的特征应用LayerNorm
        fused_features = self.layer_norm(fused_features)

        # 将融合后的特征输入到单一的 CLAM_MB_MIL 模块
        return self.clam_mb_mil(fused_features, label=label, instance_eval=instance_eval,
                                  return_features=return_features, attention_only=attention_only,
                                  return_WSI_attn=return_WSI_attn, return_WSI_feature=return_WSI_feature) 