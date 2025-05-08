import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL

class AttentionProjectionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, dropout=0.):
        """
        参数：
            in_dim: 输入特征维度
            embed_dim: 投影后的目标维度
        说明：
            该模块首先对输入做线性投影，
            然后计算每个投影后特征的注意力系数，
            最后将投影特征进行加权。
        """
        super(AttentionProjectionLayer, self).__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_fc = nn.Linear(embed_dim, 1)  # 用于计算每个特征的注意力分数

    def forward(self, x):
        """
        参数：
            x: 张量，形状 (n, in_dim)，n为特征数
        返回：
            proj_attn: 加权后的投影特征，形状 (n, embed_dim)
        """
        proj = self.linear(x)  # (n, embed_dim)
        proj = self.dropout(proj)   
        # 计算注意力系数（可以用sigmoid或者softmax，这里采用sigmoid简单归一化到(0,1)）
        attn_scores = torch.sigmoid(self.attn_fc(proj))  # (n, 1)
        # 对投影后的特征进行加权
        proj_attn = proj * attn_scores  # 广播机制自动扩展，结果形状仍为 (n, embed_dim)
        return proj_attn

class MultiheadAttentionProjectionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads=4):
        """
        参数：
            in_dim: 输入特征的维度
            embed_dim: 投影后的目标维度
            num_heads: 多头自注意力中的头数
        说明：
            先将输入通过一个线性层映射到 embed_dim，再利用多头自注意力来对特征进行交互建模，提高注意力聚合的灵活性。
        """
        super(MultiheadAttentionProjectionLayer, self).__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        # 注意：nn.MultiheadAttention 默认输入形状为 (batch, seq_len, embed_dim)
        # 设置 batch_first=True，使得输入更直观
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        """
        参数：
            x: 张量，形状为 (n, in_dim) 或 (batch, n, in_dim)
        返回：
            out: 经过多头自注意力计算后的特征，形状为 (n, embed_dim) 或 (batch, n, embed_dim)
            attn_weights: 注意力权重，形状为 (batch, n, n) 或 (n, n)（如果无 batch）
        说明：
            若输入不含 batch 维度，则在计算前先增加 batch 维度，计算完后再去除该维度。
        """
        proj = self.linear(x)  # 映射到 (n, embed_dim) 或 (batch, n, embed_dim)
        if proj.dim() == 2:
            # 输入不含 batch 维度，增加
            proj = proj.unsqueeze(0)  # (1, n, embed_dim)
            attn_output, attn_weights = self.multihead_attn(proj, proj, proj)
            out = attn_output.squeeze(0)  # 恢复为 (n, embed_dim)
        elif proj.dim() == 3:
            # 输入已含 batch 维度，直接使用
            attn_output, attn_weights = self.multihead_attn(proj, proj, proj)
            out = attn_output
        else:
            raise ValueError("输入张量的维度应为 2 或 3，当前：{}.".format(proj.dim()))
        return out, attn_weights

class CrossAttentionProjectionLayer(nn.Module):
    def __init__(self, query_in_dim, key_in_dim, embed_dim, num_heads=4, dropout=0.0):
        """
        参数：
            query_in_dim: 查询向量的输入维度
            key_in_dim: 键和值向量的输入维度
            embed_dim: 投影后的目标维度
            num_heads: 多头注意力中的头数
            dropout: dropout 概率
        说明：
            该模块使用交叉注意力机制，将查询向量与键和值向量进行交叉注意力计算，
            输出加权后的特征表示。
        """
        super(CrossAttentionProjectionLayer, self).__init__()
        self.query_proj = nn.Linear(query_in_dim, embed_dim)
        self.key_proj = nn.Linear(key_in_dim, embed_dim)
        self.value_proj = nn.Linear(key_in_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
    
    def forward(self, query, key):
        """
        参数：
            query: 张量，形状为 (n, query_in_dim) 或 (batch, n, query_in_dim)
            key: 张量，形状为 (m, key_in_dim) 或 (batch, m, key_in_dim)
        返回：
            out: 交叉注意力后的特征，形状为 (n, embed_dim) 或 (batch, n, embed_dim)
            attn_weights: 注意力权重，形状由 nn.MultiheadAttention 决定
        """
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(key)
        if Q.dim() == 2:
            # 若无 batch 维度，添加 batch 维度
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
            out, attn_weights = self.multihead_attn(Q, K, V)
            out = out.squeeze(0)
        else:
            out, attn_weights = self.multihead_attn(Q, K, V)
        return out, attn_weights

class TanhProjectionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, dropout=0.):
        super(TanhProjectionLayer, self).__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        proj = self.linear(x)
        proj = self.dropout(proj)
        return torch.tanh(proj)


class CLAM_MB_MIL_Fusion(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, num_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, act='relu', instance_eval=False, 
                 in_dims = [1536,1536], fusion_method="add"):
        """
        参数：
            in_dims: 一个整数列表，每个元素对应输入列表中每个矩阵的特征维度 d
            gate, size_arg, dropout, k_sample, num_classes, instance_loss_fn, subtyping, embed_dim, act, instance_eval:
                      与 CLAM_MB_MIL 类中相应参数的意义一致。
            fusion_method: 指定融合方式，支持 'add'（元素加法）或 'mul'（元素乘法）
        """
        super(CLAM_MB_MIL_Fusion, self).__init__()
        self.num_branches = len(in_dims)
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        if self.fusion_method == "add":
            self.fusion_weights = nn.Parameter(torch.ones(self.num_branches))
        
        # 为每个分支构建特征投影层，将各输入矩阵投影到统一的 embed_dim，并进行 tanh 归一化
        self.projection_layers = nn.ModuleList([
            AttentionProjectionLayer(in_dim, embed_dim, dropout) for in_dim in in_dims
        ])
        
        # 使用单一的 CLAM_MB_MIL 模块处理融合后的特征
        self.clam_mb_mil = CLAM_MB_MIL(gate=gate, size_arg=size_arg, dropout=dropout, k_sample=k_sample,
                                       num_classes=num_classes, instance_loss_fn=instance_loss_fn, subtyping=subtyping,
                                       embed_dim=embed_dim, act=act, instance_eval=instance_eval)
    
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
        projected_features = []
        for i in range(self.num_branches):
            x = feature_list[i]  # 形状: (n, d_i)
            proj = self.projection_layers[i](x)  # 形状: (n, embed_dim)
            projected_features.append(proj)
        
        # 使用指定的融合方式融合各分支的投影特征，使用加权求和
        if self.fusion_method == "add":
            weights = F.softmax(self.fusion_weights, dim=0)  # 归一化权重，形状为 (num_branches,)
            fused_features = 0
            for i, feat in enumerate(projected_features):
                fused_features = fused_features + weights[i] * feat
        elif self.fusion_method == "mul":
            fused_features = projected_features[0]
            for feat in projected_features[1:]:
                fused_features = fused_features * feat
        else:
            raise ValueError("不支持的融合方法: " + self.fusion_method)
        
        # 将融合后的特征输入到单一的 CLAM_MB_MIL 模块
        return self.clam_mb_mil(fused_features, label=label, instance_eval=instance_eval,
                                  return_features=return_features, attention_only=attention_only,
                                  return_WSI_attn=return_WSI_attn, return_WSI_feature=return_WSI_feature) 