import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math
import copy
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 采样点
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化采样点的偏移量"""
        # thetas (n_heads,)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # grid_init (n_heads, 2)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 对每个坐标(x,y)，除以max{|x|,|y|}，将单位圆上的点拉到方形边框上
        # (n_heads, n_levels, n_points, 2)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # 每个head的每个level的第i个点，偏移量 * (i+1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 将这些偏移量赋值给线性层的偏置
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        """初始化采样点的权重"""
        # 将权重初始化为0
        # 这样模型第一次前向传播时，Linear(query) = weight * query + bias，输出完全等于bias
        constant_(self.sampling_offsets.weight.data, 0.)
        """初始化线性层 query->注意力"""
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        """初始化线性层 input->value"""
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        """初始化线性层 加权注意力和->最终输出"""
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    def forward(self, query, reference_points, input_flatten, input_space_shape, input_level_start_idx):
        """
        query (N, Len_q, C)
        reference_points (N, Len_q, n_levels, 2) range in [0,1]
        input_flatten (N, \sum(H*W), C)
        input_space_shape (n_levels, 2), [(H0,W0),(H1,W1),...,(H4,W4)]
        input_level_start_idx (n_levels, ), [0, H0*W0, H0*W0+H1*W1, H0*W0+H1*W1+H2*W2,...]
        
        return output (N, Len_q, C)
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 检查输入长度是否与4层的总像素数量相等
        assert (input_space_shape[:, 0] * input_space_shape[:, 1]).sum() == Len_in
        """线性层 input->value"""
        # (N, Len_in, C)
        value = self.value_proj(input_flatten)
        # (N, Len_in, n_heads, C / 8)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        """生成采样偏移与注意力权重"""
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # 每个N的每个query的每个头，共用一组权重，所以最后有self.n_levels * self.n_points = 16个元素
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 对这16个元素用softmax归一化，说明它们共用一组权重
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        """生成采样点坐标，归一化到[0,1]"""
        if reference_points.shape[-1] == 2:
            # (n_levels, 2), [[H0,W0],[H1,W1],[H2,W2],[H3,W3]]
            offset_normalizer = torch.stack([input_space_shape[..., 1], input_space_shape[..., 0]], -1)
            # (N, Len_q, n_heads, n_levels, n_points, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        """计算注意力"""
        # (N, Len_q, C)
        output = self.deformable_attn(value,input_space_shape, sampling_locations, attention_weights)
        # 再经过一个线性层得到输出
        # (N, Len_q, C)
        output = self.output_proj(output)
        return output
    def deformable_attn(self, value, space_shape, sampling_locations, attention_weights):
        N_ = value.shape[0] # batch_size
        M_ = self.n_heads # 注意力头数
        D_ = self.d_model // self.n_heads # 每个头处理的通道数256/8 = 32
        Len_q = sampling_locations.shape[1]
        # value_list是一个列表，共有n_levels = 4个元素
        # 每个元素形如(N, Hi*Wi, n_heads, C / 8)，也就是把展平的value还原回4层
        value_list = value.split([H_*W_ for H_,W_ in space_shape], dim=1)
        # 将采样点的相对坐标从[0,1]变化到[-1,1]，满足之后输入函数的需求
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        # i表示第i层
        for i, (H_,W_) in enumerate(space_shape):
            # 变化：(N, Hi*Wi, n_heads, C / 8)->(N, Hi*Wi, n_heads*C/8)->(N, n_heads*C/8, Hi*Wi)->(N*n_heads, C/8, Hi, Wi)
            # 最终：(N*n_heads, C/8, Hi, Wi)
            value_level = value_list[i].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
            # 变化：(N, Len_q, n_heads, n_points, 2)->(N, n_heads, Len_q, n_points, 2)->(N*n_heads, Len_q, n_points, 2)
            # 最终：(N*n_heads, Len_q, n_points, 2)
            sampling_grid_level = sampling_grids[:, :, :, i].transpose(1, 2).flatten(0, 1)
            """
            双线性插值
            F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            input - (Batch, Channels, Height, Width)
             grid - (Batch, H_out, W_out, 2)
            在这里，用N*n_heads来伪造一个batch
            """
            # 输出：(N_*M_, D_, Len_q, n_points)
            # 含义是：对于Len_q中的某一个query，在这层里需要n_points=4个采样特征值
            # 于是程序在每个采样点都进行双线性插值（上下左右距离加权），得到一个通道D_=32的采样特征值张量
            # 因此对于一个query，我们在第i层得到n_points = 4个通道数为D_=32的采样特征值
            sampling_value_level = F.grid_sample(value_level, sampling_grid_level,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
            # 每一层的结果运算完，放入sampling_value_list中
            # (4, N_*M_, D_, Len_q, n_points)
            sampling_value_list.append(sampling_value_level)
        """注意力计算"""
        #变化：(N, Len_q, n_heads, n_levels, n_points)->(N, n_heads, Len_q, n_levels, n_points)->(N*n_heads, 1, Len_q, n_levels*n_points)
        #最终：(N*n_heads, 1, Len_q, n_levels*n_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Len_q, self.n_levels*self.n_points)
        # stack->(N_*M_, D_, Len_q, n_levels, n_points)->(N_*M_, D_, Len_q, n_levels*n_points)
        # 相乘(N_*M_, D_, Len_q, n_levels*n_points) * (N_*M_, D_, Len_q, n_levels*n_points)->(N_*M_, D_, Len_q, n_levels*n_points)
        # .sum(-1) ->(N_*M_, D_, Len_q,1)
        # .view(N_, M_*D_, Len_q) -> (N_, M_*D_, Len_q)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Len_q)
        """return (N, Len_q, C)，与原query的维度一致"""
        return output.transpose(1,2).contiguous()

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
            d_model=256, d_ffn=1024,
            dropout=0.1, activation="relu",
            n_levels=4,n_heads=8,n_points=4):
        super().__init__()
        # self-attention自注意力
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    @staticmethod # 静态方法，不用到类中的self
    def with_pos_embed(tensor, pos):
         return tensor if pos is None else tensor + pos
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    def forward(self, src, pos, reference_points, space_shape, level_start_idx):
        # self-attention，每次进入之前可以先位置编码
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, space_shape, level_start_idx)
        # 残差连接
        src = src + self.dropout1(src2)
        # 归一化
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src
    
"""
Encoder:传入最原始的4层拼接特征图(N, \sum Hi*Wi, C)
        返回带有位置信息，语义信息，注意力信息的4层拼接特征图(N, \sum Hi*Wi, C)
"""
class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    @staticmethod
    def get_reference_points(space_shape, valid_ratios, device):
        reference_points_list = []
        for i, (H_,W_) in enumerate(space_shape):
            # Encoder中，每一层的每一个元素都是参考点
            # 它们的中心点坐标从0.5直到H-0.5.
            # ref_y : (Hi,Wi), ref_x : (Hi,Wi)
            # 注意这里的间隔选取是在整张padding过的图中
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_-0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_-0.5, W_, dtype=torch.float32, device=device))
            # .reshape(-1) -> (Hi*Wi,)
            # [None]等价于.unsqueeze(0) -> (1, Hi*Wi)
            # valid_ratios : (N, n_levels, 2)
            # valid_rations[:,None,i,1] * H_ -> (N,1)
            # (1,Hi*Wi)/(N,1) -> (N,Hi*Wi)
            # 实际作用是把该层每个像素的绝对坐标ref_y,ref_x，除以其真实有效占比valid_ratios，归到[0,1]
            # 但在整张图的尺度上，padding的偏移仍然存在
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, i, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, i, 0] * W_)
            # (N, Hi*Wi, 2)
            ref = torch.stack((ref_x,ref_y),-1)
            # 在列表中，把这一层的参考点整体输入进去
            reference_points_list.append(ref)
        #把列表中的4层ref前后连接起来
        # (N, \sum Hi*Wi, 2)
        reference_points = torch.cat(reference_points_list, 1)
        """
        注意\sum Hi*Wi 其实就是Len_q
        在可变形注意力中，要求输入reference_points (N, Len_q, n_levels, 2) range in [0,1]
        """
        # [:,:,None] -> (N, Len_q, 1, 2)
        # [:, None] -> (N, 1, n_levels, 2)
        # 相乘      -> (N, Len_q, n_levels, 2)
        # Len_q包含了所有的像素点，将它在n_levels维度广播的含义是：参考点在每层中相对坐标不应该改变
        # 乘以valid_ratios，是为了解除padding的影响
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def forward(self,src,space_shape,level_start_idx,valid_ratios, pos=None):
        output = src
        reference_points = self.get_reference_points(space_shape, valid_ratios, device=src.device)
        """
        传入DeformableTransformerEncoderLayer
        要求(src, pos, reference_points, space_shape, level_start_idx)
        """
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, space_shape, level_start_idx)
        return output
"""
===================
骨干网络-ResNet50
Input : (N, 3, H, W)
Output: List[(N,256,H/8,W/8),(N,256,H/16,W/16),(N,256,H/32,W/32),(N,256,H/64,W/64)]
===================
"""
class Backbone(nn.Module):
    def __init__(self,d_model=256):
        super().__init__()
        # 加载现成的 ResNet50 (不带全连接层和分类头)
        backbone = resnet50(pretrained=True)
        # 将要返回的三层映射为0,1,2
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # 用IntermediateLayerGetter直接拿取中间层
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # ResNet 中间层的通道数分别是 512, 1024, 2048，通过1x1卷积统一降维到 d_model
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, d_model, kernel_size=1)),
            nn.Sequential(nn.Conv2d(1024, d_model, kernel_size=1)),
            nn.Sequential(nn.Conv2d(2048, d_model, kernel_size=1)),
        ])
        # 特比地，对 layer4 的输出 (2048通道) 进行步长为 2 的 3x3 卷积，得到最高级语义的特征图
        self.top_down_proj = nn.Conv2d(2048, d_model, kernel_size=3, stride=2, padding=1)
    def forward(self,x): # x是最原始的图像
        features = self.body(x)
        srcs = []
        for i, feat in enumerate(features.values()):
            srcs.append(self.input_proj[i](feat))
        # 第 4 张特征图：单独用 features["2"] (即 layer4 的输出) 生成
        src_top_down = self.top_down_proj(features["2"])
        srcs.append(src_top_down)
        return srcs

"""
===================
Encoder固定位置编码
===================
"""
# class PositionEmbeddingSine(nn.Module):
#     def __init__():

if __name__ == '__main__':
    # 伪造一张 800x800 的图片
    dummy_image = torch.rand(2, 3, 800, 800)
    backbone = Backbone(d_model=256)
    out_features = backbone(dummy_image)
    for i, feat in enumerate(out_features):
        print(f"特征图 {i} 的形状: {feat.shape}")
    # print("开始测试 Deformable Transformer Encoder...\n")

    # # ==========================================
    # # 1. 设置基础超参数
    # # ==========================================
    # N = 2             # Batch size (模拟同时处理2张图片)
    # d_model = 256     # 隐藏层特征维度
    # n_levels = 4      # 多尺度特征图的层数
    # n_heads = 8       # 多头注意力的头数
    # n_points = 4      # 每个头在每层采样的点数
    # num_layers = 2    # Encoder层数 (为了测试跑得快点，这里设为2层，原版是6层)

    # # ==========================================
    # # 2. 实例化你的网络模型
    # # ==========================================
    # # 先实例化单层 Layer
    # encoder_layer = DeformableTransformerEncoderLayer(
    #     d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
    # )
    # # 再将单层传入，组装成完整的 Encoder
    # encoder = DeformableTransformerEncoder(encoder_layer, num_layers)

    # # ==========================================
    # # 3. 伪造多尺度特征图的形状参数
    # # ==========================================
    # # 假设 4 层特征图大小分别为 64x64, 32x32, 16x16, 8x8
    # space_shape = torch.tensor([[64, 64], [32, 32], [16, 16], [8, 8]])
    
    # # 计算每一层的像素面积: [4096, 1024, 256, 64]
    # spatial_sizes = space_shape[:, 0] * space_shape[:, 1]
    
    # # 计算每层特征在展平列表中的起始索引: [0, 4096, 5120, 5376]
    # level_start_idx = torch.cat((torch.zeros(1, dtype=torch.long), spatial_sizes.cumsum(0)[:-1]))

    # # 计算特征展平后的总长度 Len_q = \sum H*W = 5440
    # Len_q = spatial_sizes.sum().item()

    # # ==========================================
    # # 4. 伪造输入数据 (Input Tensors)
    # # ==========================================
    # device = torch.device("cuda") # 如果配置了CUDA，也可以改成 "cuda"
    # encoder.to(device)
    # space_shape = space_shape.to(device)
    # level_start_idx = level_start_idx.to(device)

    # # src: 展平后的图像特征 (N, Len_q, d_model)。注意要加上 requires_grad=True 来测试源头梯度
    # src = torch.rand(N, Len_q, d_model, device=device, requires_grad=True)
    
    # # pos: 位置编码，维度和src一样
    # pos = torch.rand(N, Len_q, d_model, device=device)
    
    # # valid_ratios: (N, n_levels, 2), 表示每张图在每层的有效宽高比例
    # # 用 rand 生成 0~1 的数，乘以 0.5 再加 0.5，确保有效比例在 [0.5, 1.0] 之间
    # valid_ratios = torch.rand(N, n_levels, 2, device=device) * 0.5 + 0.5

    # # ==========================================
    # # 5. 测试前向传播 (Forward Pass)
    # # ==========================================
    # print("--- 测试前向传播 ---")
    # try:
    #     output = encoder(src, space_shape, level_start_idx, valid_ratios, pos)
    #     print("✅ 前向传播成功！")
    #     print(f"预期输出形状: ({N}, {Len_q}, {d_model})")
    #     print(f"实际输出形状: {output.shape}")
    #     assert output.shape == (N, Len_q, d_model), "输出维度不符合预期！"
    # except Exception as e:
    #     print(f"❌ 前向传播失败，报错信息: {e}")
    #     exit()

    # # ==========================================
    # # 6. 测试反向传播 (Backward Pass)
    # # ==========================================
    # print("\n--- 测试反向传播 ---")
    # try:
    #     # 简单把整个输出张量求和当作 Loss
    #     loss = output.sum()
    #     # 反向传播，计算梯度
    #     loss.backward()
        
    #     # 检查网络内部的参数是否收到了梯度
    #     # 我们检查第 0 层 EncoderLayer 里 attention 层的采样偏移量偏置
    #     grad_check = encoder.layers[0].self_attn.sampling_offsets.bias.grad
        
    #     if grad_check is not None:
    #         print("✅ 反向传播成功！梯度已顺利流回网络内部的权重参数。")
    #     else:
    #         print("❌ 反向传播失败，内部参数未能计算出梯度。")
            
    #     # 检查最源头的输入特征是否收到了梯度
    #     if src.grad is not None:
    #         print("✅ 梯度流完全连通！梯度已成功回传至最源头的输入特征 src。")
    #     else:
    #         print("❌ 输入特征 src 未能接收到回传梯度，可能在计算图中发生了断裂。")
            
    # except Exception as e:
    #     print(f"❌ 反向传播失败，报错信息: {e}")


