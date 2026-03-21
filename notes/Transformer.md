#深度学习 #DeepLearning #Transformer

# 1. Transformer Architecture

## 1.1 Encoder and Decoder Stack 编码器与解码器堆叠

> **Embedding and Positional Encoding 词嵌入和位置编码**

对于一段自然语言，首先通过分词（Tokenization）将它们拆解为最小单元（Token），每个Token对应一个数字编号（ID）.通过词嵌入（Word Embedding）将ID转化为一个`512`维的语义向量.再利用位置编码（Positional Encoding），得到一个`512`维的位置向量.对于每个Token来说，它的语义向量+位置向量=最终该Token的输入向量（`512`维），它同时包含了语义信息和位置信息.

在Transformer中，其Inputs的大小为`(batch_size, seq_len, d_model)`.其中`batch_size`是批量大小（一次送入模型的句子数），`seq_len`是每个句子的词数（Token数，若实际Token数小于`seq_len`，则会填充），`d_model`就是每个Token向量的维度，固定$d_{\mathrm{model}} = 512$.

词嵌入矩阵在Transformer中是可学习参数.

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

> **Position-wise Feed-Forward Networks 位置前馈网络**

*This consists of two linear transformations with a ReLU activation in between.*
*位置前馈网络* 包含两个线性层以及它们中间的ReLU激活层.
$$\mathrm{FFN(x)} = \max(0,xW_{1}+b_{1})W_{2}+b_{2}$$

*While the linear transformations are the same across different positions, they use different parameters from layer to layer.*
在一个FFN中，输入序列（一个句子）的每一个位置（每一个Token）都共用相同的参数$W_{1},b_{1},W_{2},b_{2}$.但是Encoder的不同子层的FFN所用的参数各不相同.



