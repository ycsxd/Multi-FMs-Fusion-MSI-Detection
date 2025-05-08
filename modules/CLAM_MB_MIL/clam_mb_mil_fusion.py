import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL


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
        self.embed_dim = embed_dim  # 将embed_dim保存为实例属性
        
        if self.use_att_gate:
            self.attention_gate = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(self.num_branches)])
        
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dims[i], embed_dim), 
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(self.num_branches)
        ])
        
        # 添加LayerNorm层，用于融合后的特征归一化
        self.layer_norm = nn.LayerNorm(embed_dim * self.num_branches)
        
        # 使用单一的 CLAM_MB_MIL 模块处理融合后的特征
        self.clam_mb_mil = CLAM_MB_MIL(gate=gate, size_arg=size_arg, dropout=dropout, k_sample=k_sample,
                                       num_classes=num_classes, instance_loss_fn=instance_loss_fn, subtyping=subtyping,
                                       embed_dim=embed_dim*self.num_branches, act=act, instance_eval=instance_eval)
        
        # 添加全局上下文感知的注意力
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        self.branch_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.num_branches)])
    
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
        projected_features = []
        for i in range(self.num_branches):
            if i < len(feature_list):
                # 检查特征维度并处理
                x = feature_list[i]
                # 对特征应用投影层前检查形状
                if len(x.shape) == 2:  # 如果是(n, d_i)形状
                    proj_x = self.proj_layers[i](x)
                    projected_features.append(proj_x)
                elif len(x.shape) == 3:  # 如果是(batch_size, n, d_i)形状
                    batch_size, n_instances, feature_dim = x.shape
                    # 重塑为2D进行处理
                    x_reshaped = x.view(-1, feature_dim)
                    proj_x = self.proj_layers[i](x_reshaped)
                    # 恢复原始批次维度
                    proj_x = proj_x.view(batch_size, n_instances, -1)
                    projected_features.append(proj_x)
                else:
                    # 如果维度不匹配，进行必要的调整
                    raise ValueError(f"Branch {i}的特征形状不正确: {x.shape}，应为(n, d_i)或(batch_size, n, d_i)")
            else:
                # 如果特征列表长度小于分支数，使用零填充
                raise ValueError(f"Branch {i}的特征列表长度小于分支数")
        
        # 对每个分支的投影特征应用LayerNorm
        normalized_features = [self.branch_norms[i](projected_features[i]) for i in range(self.num_branches)]
        
        # 实现全局上下文感知的注意力机制
        batch_size, n_instances = normalized_features[0].shape[0], normalized_features[0].shape[1]
        
        # 准备用于计算注意力的特征
        all_features = torch.stack(normalized_features, dim=2)  # shape: (batch_size, n_instances, num_branches, embed_dim)
        
        # 计算全局上下文注意力
        # 1. 生成查询、键和值
        queries = self.query_proj(all_features)  # (batch_size, n_instances, num_branches, embed_dim)
        keys = self.key_proj(all_features)       # (batch_size, n_instances, num_branches, embed_dim)
        values = self.value_proj(all_features)   # (batch_size, n_instances, num_branches, embed_dim)
        
        # 2. 计算注意力分数 (点积注意力)
        # 为简化计算，我们对每个实例计算分支间的注意力
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))  # (batch_size, n_instances, num_branches, num_branches)
        
        # 3. 归一化注意力权重
        attn_weights = F.softmax(attn_scores / (self.embed_dim ** 0.5), dim=-1)  # 缩放点积注意力
        
        # 4. 加权聚合
        context_vectors = torch.matmul(attn_weights, values)  # (batch_size, n_instances, num_branches, embed_dim)
        
        # 结合原有的注意力门控机制
        if self.use_att_gate:
            # 计算每个分支的注意力分数，结果形状为 (batch_size, n_instances, 1)
            gate_scores = [self.attention_gate[i](normalized_features[i]) for i in range(self.num_branches)]
            
            # 将各分支分数拼接，形状变为 (batch_size, n_instances, num_branches)
            gate_scores = torch.cat(gate_scores, dim=2)
            
            # 沿分支维度应用softmax，得到注意力权重
            gate_weights = F.softmax(gate_scores, dim=2)
            
            # 将门控注意力与上下文注意力结合
            # 对每个分支，结合门控权重和上下文向量
            refined_features = []
            for i in range(self.num_branches):
                # 从上下文向量中获取特定分支的上下文表示
                branch_context = context_vectors[:, :, i, :]  # (batch_size, n_instances, embed_dim)
                # 应用门控注意力权重
                gated_feature = normalized_features[i] * gate_weights[:, :, i:i+1]  # (batch_size, n_instances, embed_dim)
                # 结合原始特征和上下文信息
                refined_feature = gated_feature + branch_context  # 残差连接
                refined_features.append(refined_feature)
            
            # 拼接所有分支的融合特征
            fused_features = torch.cat(refined_features, dim=2)  # (batch_size, n_instances, num_branches*embed_dim)
        else:
            # 如果不使用门控机制，直接使用上下文增强的特征
            context_enhanced = context_vectors.view(batch_size, n_instances, -1)  # 展平最后两个维度
            fused_features = torch.cat(normalized_features, dim=2) + context_enhanced  # 残差连接
        
        # 对融合后的特征应用LayerNorm
        fused_features = self.layer_norm(fused_features)
        
        # 将融合后的特征输入到单一的 CLAM_MB_MIL 模块
        return self.clam_mb_mil(fused_features, label=label, instance_eval=instance_eval,
                                  return_features=return_features, attention_only=attention_only,
                                  return_WSI_attn=return_WSI_attn, return_WSI_feature=return_WSI_feature) 