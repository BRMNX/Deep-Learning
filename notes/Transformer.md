#深度学习 #DeepLearning #Transformer

# 1. Transformer Architecture
输入向量`X(batch_size, seq_len, d_model = 512)`送入多头注意力后被拆成`h = 8`份，也就是`X_i(batch_size, seq_len, 64)`.Token的数量是`seq_len`（行），**不是64个token！不是64个token！不是64个token！**`64`是每个token的特性（列）！！！
这种变量的命名方式真让人恶心！！！多头注意力分开的是特性空间，不是分开Token！！！所有的token都要注意到同一batch里的其他token，这是Transformer的设计初衷！
Transformer的输出仍然是`(batch_size, seq_len, 512)`，形状是不会变的！

## 1.1 Encoder and Decoder Stack 编码器与解码器堆叠

> **Embedding and Positional Encoding 词嵌入和位置编码**

对于一段自然语言，首先通过分词（Tokenization）将它们拆解为最小单元（Token），每个Token对应一个数字编号（ID）.通过词嵌入（Word Embedding）将ID转化为一个`512`维的语义向量.再利用位置编码（Positional Encoding），得到一个`512`维的位置向量.对于每个Token来说，它的语义向量+位置向量=最终该Token的输入向量（`512`维），它同时包含了语义信息和位置信息.在实际工作时,Encoder和Decoder的嵌入层共用一个权重矩阵$W$,并且在通过嵌入得到语义向量后,需要将此向量乘以$\sqrt{d_{\mathrm{model}}}$再和位置向量相加，以免语义信息被位置信息淹没.此外，在Decoder最终的softmax输出概率之前，有一个线性层，它的权重矩阵是$W^{\mathrm{T}}$.

在Transformer中，其Inputs的大小为`(batch_size, seq_len, d_model)`.其中`batch_size`是批量大小（一次送入模型的句子数），`seq_len`是每个句子的词数（Token数，若实际Token数小于`seq_len`，则会填充），`d_model`就是每个Token向量的维度，固定$d_{\mathrm{model}} = 512$.

词嵌入矩阵在Transformer中是可学习参数.

Transformer使用如下公式进行位置编码：$$PE_{(pos,2i)} = \sin(pos / 10000^{2i / d_{\mathrm{model}}})$$$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i / d_{\mathrm{model}}})$$其中$0\leqslant pos < seq\_len$，$0\leqslant i < d_{\mathrm{model}} / 2 -1$.其输出形状与词嵌入矩阵大小一样是`(batch_size, seq_len, d_model)`.

> **Encoder** **编码器**

*The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers.*
编码器由$N = 6$个相同的层堆叠而成，每个层都有2个子层.

*The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.*
第一个子层是多头自注意力机制，第二个子层是一个简单的全连接位置前馈网络.

*We employ a residual connection around each of the two sub-layers, followed by layer normalization. All sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_\mathrm{model} = 512$.*
我们在每一个子层都利用残差连接，然后接上层规范化.模型中所有子层，包括嵌入的部分，其产生的输出的维度都是$d_{\mathrm{model}} = 512$.

> **Attention 注意力**

*An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.*
一个注意力函数可以被描述为从查询、键-值对到输出的映射，其中查询、键-值对与输出都是向量.输出是值的加权和，每个值对应的权重由相应的查询和键通过一个兼容性函数计算得到.

>> **Scaled Dot-Product Attention 缩放点积注意力**

![[屏幕截图 2026-03-21 220948.png]]

*The input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$.We compute the dot products of the query with all keys, divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.*
输入包含$d_{k}$维的查询和键，以及$d_{v}$维的值.我们将每一个查询与键作点积，并将每一个结果都除以$\sqrt{d_{k}}$，然后应用softmax函数来获得每个值对应的权重.

*In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:*
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt{d_{k}}}\right)V$$
实际上，我们计算注意力函数时并行利用多组查询，它们被包装在一个矩阵$Q$中.键和值同样被包装在矩阵$K$和$V$中，我们利用如下的矩阵运算计算输出：$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt{d_{k}}}\right)V$$
*We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\sqrt{d_{k}}$ .*
我们设想当$d_{k}$取大值时，点积的绝对值将会变得非常大，导致softmax函数进入到梯度极小的趋于.为了消除这一影响，我们在点积中缩放了$\sqrt{d_{k}}$倍.

>>**Multi-Head Attention 多头注意力** 

*Instead of performing a single attention function with $d_{\mathrm{model}}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_{k}$, $d_{k}$ and $d_{v}$ dimensions, respectively.*
与其使用$d_{\mathrm{model}}$维的键、值和查询来执行单一的注意力函数，我们发现，对查询、键和值分别使用不同的、已学习的线性投影，分别投影$h$次到$d_{k}$，$d_{k}$和$d_{v}$维，会更有效.

*In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_{k}=d_{v} = d_{\mathrm{model}} / h = 64$.*
在本研究中，我们采用了$h = 8$个并行注意力层（或称头部）.对于每个注意力层，我们使用$d_{k}=d_{v} = d_{\mathrm{model}} / h = 64$.

*We then perform the attention function in parallel, yielding $d_{v}$-dimensional output values. These are concatenated and once again projected, resulting in the final values.*
随后，我们并行执行注意力函数，从而得到$d_{v}$维的输出值.这些输出值被拼接起来，并再次进行（线性）投影，最终得到结果值.
$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat(head_{1},\cdots,head_{h})}W^{O}$$$$\mathrm{where~head_{i}} = \mathrm{Attention}(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})$$其中参数矩阵$W_{i}^{Q}\in\mathbb{R}^{d_{\mathrm{model}}\times d_{k}},W_{i}^{K}\in\mathbb{R}^{d_{\mathrm{model}}\times d_{k}},W_{i}^{V}\in\mathbb{R}^{d_{\mathrm{model}}\times d_{v}}$以及$W^{O}\in \mathbb{R}^{hd_{v}\times d_{\mathrm{model}}}$.
**注意此处的$Q,K,V$指的是送入多头注意力模块的原始张量.在Encoder的第一个子层的多头注意力模块，$Q=K=V=X$，若记第一个子层的输出为$X^{(1)}$，则在第二个子层的多头注意力模块，$Q=K=V=X^{(1)}$.**
![[屏幕截图 2026-03-21 223235.png]]
*Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.*
多头注意力机制使模型能够同时关注来自不同位置的、来自不同表示子空间的信息.而使用单个注意力头时，平均效应会抑制这种能力.

>> **Applications of Attention in our Model 注意力在模型中的应用**

- *In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models.*
- 在“编码器-解码器注意力”层中，查询来自解码器的前一层，而具有记忆的键和值来自编码器的输出.这使得解码器中的每个位置都能关注输入序列中的所有位置.这种设计模拟了序列到序列模型中典型的编码器-解码器注意力机制.
- *The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.*
- 编码器包含自注意力层.在自注意力层中，所有键、值和查询都来自同一个来源，即编码器中上一层的输出.编码器中的每个位置都可以关注编码器上一层中的所有位置
- *Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.*
- 同样地，解码器中的自注意力层允许解码器中的每个位置关注解码器中截至该位置（含该位置）的所有位置.为了保持自回归特性，我们需要阻止解码器中的左向信息流.我们通过在缩放点积注意力中屏蔽（设置为 −∞）softmax函数输入中所有对应于非法连接的值来实现这一点.（也就是，在注意力分数矩阵$\frac{QK}{\sqrt{d_{k}}}$计算出来后，将严格上三角部分设为$-\infty$，这样经过softmax函数后，相应位置的权重变为0，确保模型学习到合理的条件概率）

> **Position-wise Feed-Forward Networks 位置前馈网络**

*This consists of two linear transformations with a ReLU activation in between.*
*位置前馈网络* 包含两个线性层以及它们中间的ReLU激活层.
$$\mathrm{FFN(x)} = \max(0,xW_{1}+b_{1})W_{2}+b_{2}$$

*While the linear transformations are the same across different positions, they use different parameters from layer to layer.*
在一个FFN中，输入序列（一个句子）的每一个位置（每一个Token）都共用相同的参数$W_{1},b_{1},W_{2},b_{2}$.但是Encoder的不同子层的FFN所用的参数各不相同.

# 2. Detection Transformer

```python
"""原始图像"""
images = torch.randn(B, 3, 800, 1066) 
# [B, C, H, W] # [实际] (B, 3, 800, 1066)

"""Backbone"""
features = backbone(images) # 输出layer4特征图 
# [实际] (B, 2048, 25, 34) # 800/32=25, 1066/32≈33.3→34

"""1x1 Conv降维"""
src = input_proj(features) # nn.Conv2d(2048, 256, 1) 
# [实际] (B, 256, 25, 34) # 通道→d_model=256

""""""
这里的25*34 = 850才是我们Token的个数(即seq_len)！通道数256是特性数.在多头注意力里，要划分的是256，不是850！！！
""""""

"""展平"""
# 展平空间维度 (H'×W') 
src_flat = src.flatten(2) 
# (B, 256, 25*34) = (B, 256, 850)

# 转置！满足 (seq_len, batch_size, d_model)
src_seq = src_flat.permute(2, 0, 1) 
# (850, B, 256)

"""位置编码"""
pos_embed = pos_encoding_generator(features) # 生成 (B, 256, 25, 34) 
pos_embed = pos_embed.flatten(2).permute(2, 0, 1) 
# (850, B, 256) # [实际] (850, B, 256) | 与src_seq严格对齐
src_with_pos = src_seq + pos_embed # (850, B, 256)

"""Encoder"""
memory = transformer.encoder(src_with_pos) # 6层 
# [实际] (850, B, 256) ← **全程保持 (seq_len, batch_size, d_model)**

"""Object queries"""
# 可学习参数 (num_queries=100, d_model=256) 
query_embed = nn.Embedding(100, 256) # .weight.shape = (100, 256)
queries = query_embed.weight.unsqueeze(1).repeat(1, B, 1) # (100, B, 256) 
query_pos = query_pos_embed.weight.unsqueeze(1).repeat(1, B, 1) # (100, B, 256) 
tgt = queries + query_pos # (100, B, 256) 
# [实际] (100, B, 256) ← **Decoder输入真实形状**

"""Decoder"""
# 输入: 
# tgt = (100, B, 256) ← Queries (Q来源) 
# memory = (850, B, 256) ← Encoder输出 (K/V来源) 
output = transformer.decoder(tgt, memory) # 6层 
# [实际] (100, B, 256) ← **输出形状与tgt一致**
# 详细的步骤：
# Q: 来自tgt (经自注意力后) → (100, B, 256) → 投影 → reshape → (100, B, nheads=8, 32) 
# K/V: 来自memory → (850, B, 256) → 投影 → reshape → (850, B, nheads=8, 32)
# B和nheads全程不参与计算！
# 点积: Q @ K^T → (nheads=8, 100, B, 850) ← **关键：100个query对850个位置打分** 
# softmax → (nheads=8, 100, B, 850) 
# 加权聚合V: (nheads=8, 100, B, 850) @ (nheads=8, 850, B, 32) → (8, 100, B, 32)
# 合并头 → (100, B, 256)
"""预测头"""
# 转置回 (B, 100, 256) 便于后续处理
hs = output.permute(1, 0, 2)  # (B, 100, 256)
# 分类头
outputs_class = class_embed(hs)  # Linear(256, 91) → (B, 100, 91)
# 框回归头
outputs_coord = bbox_embed(hs).sigmoid()  # MLP → (B, 100, 4) [cx,cy,w,h]
```

# 3. Deformabel DETR
## 1. 关于核心机制（Deformable Attention）

相比于原版 DETR 中让 Query 和特征图上所有像素交互的全局交叉注意力（Cross-Attention），Deformable DETR 到底是怎么把计算复杂度降下来的？请简述一下，在 Deformable Attention 中，一个 Object Query 是如何获取它需要的特征的？

>Deformable DETR利用Multi-Scale Deformable Attention模块替代传统的Cross-Attention模块以降低计算复杂度。具体来说，Deformable Attention先将特征图线性映射为Value矩阵，并将Query线性映射为注意力权重矩阵，每个Query包含一个起始的参考点，Deformable Attention在特征图中对参考点附近的k个点（原文k=4）进行采样，在每一个采样点（可能在原图中无法对应准确的一个像素），用双线性插值取周围4个像素的Value值作加权平均，得到这个采样点的Value值。一个参考点在须在每个注意力头内的4层特征图、每层4个采样点做注意力计算，也就是说一个头内的16个点的注意力权重的和为1，而注意力头共有8个。经过这样的计算，Query就和Encoder中包含高级特征信息的图片交互，获取所需的特征。另外，Query初始化时会生成位置编码，每当它进入Decoder，Query之间会先做Self-Attention，避免关注相同区域，然后再对Encoder的输出做Deformable Attention。

## 2. 关于迭代微调（Iterative Bounding Box Refinement）

在 Decoder 的多层传递中，第 $l$ 层预测出的边界框中心坐标，在被传递给第 $l+1$ 层作为新的“参考点（Reference Point）”时，为什么要先经过一个**逆 Sigmoid（Inverse Sigmoid）** 操作？如果不做这个操作，在数学逻辑上会产生什么冲突？

>如果不做处理，坐标的尺度会出问题导致无法预测。这里有一个**关键的数学细节**需要澄清。逆 Sigmoid（$\sigma^{-1}$）**并不是**把坐标变成“真实像素坐标”（比如从 0.5 变成 400 像素）。 实际上，逆 Sigmoid 是把 $[0, 1]$ 归一化区间的坐标，拉回到了**无界的实数空间（Logit 空间，$-\infty$ 到 $+\infty$）**。 为什么必须在实数空间？因为网络第 $l+1$ 层预测出的偏移量 $\Delta$ 是一个没有任何限制的实数（可正可负，可大可小）。只有把参考点也变回无界的实数，两者才能直接相加（$\text{Logit} + \Delta$）。加完之后，再套一个正向的 Sigmoid，把最终的坐标稳稳地压回 $[0, 1]$ 区间。如果不做逆运算直接拿 $[0, 1]$ 的值去加 $\Delta$，再经过一层 Sigmoid，坐标的几何意义就被彻底扭曲了。

## 3. 关于两阶段架构（Two-Stage）

在 Two-Stage 版本中，Encoder 兼职做了类似 RPN 的工作，并筛选出了 Top-K（例如 300 个）得分最高的候选框（Proposals）。请问，这 300 个候选框的具体信息（不仅是位置，还有特征），是如何直接“赋能”给 Decoder 的？（提示：想想 Decoder 最开始的那 300 个 Query 的两部分构成是如何被替换的）。

>Query包含语义信息和位置信息。这300个候选框的中心坐标被直接作为Query的参考点的坐标，这300个候选框在原特征图（Encoder的输出）中所对应的像素的256维特征向量，被作为Query的语义特征向量。