注意力机制的特点是通过计算$Query$与$Key$的相关性为真值加权求和，从而拟合序列中每一个词同其他词的相关关系

注意力机制有三个核心变量：query（查询值），key（键值），value（真值）

query是我们需要查询的向量  key作为数据的一个向量  value是数据向量对应的一个值

$$value=v1*w1+v2*w2+v3*w3$$

不同的$key$所赋予的不同权重$weight$，就是所说的注意力分数。也就是为了查询到$Query$，我们应该赋予给每一个$Key$多少注意力。如何去计算出对应的注意力分数？   直观上讲， 我们可以认为$Key$与$Query$相关性越高，则其所应赋予的注意力权重越大

词向量能够表征语义信息，通过训练拟合，将语义相近的词在向量空间中距离更近，语义较远的词在向量空间中距离更远。用点积来计算欧氏距离————>衡量词向量的相似性:

$$v\cdot w=\sum_{i} v_iw_i$$

计算$Query$和每一个键的相似程度：

$x=qK^T$

此处的$K$即为将所有$Key$对应的词向量堆叠形成的矩阵.$x$反映了$Query$和每一个$Key$的相似程度，我们再通过一个$softmax$层将其转化为和为1的权重：

$softmax(x)_i=\frac{e^{xi}}{\sum_{j}e^{x_j}}$

得到的向量就能够反映$Query$和每一个$Key$的相似程度，同时又相加权重为1，也就是注意力分数。最后，将得到的注意力分数和值向量做对应乘积即可。注意力机制计算公式：

$attention(Q,K,V)=softmax(qK^T)v$

此时，这样的值还是一个标量，只是查询了一个$Query$,我们需要一次性查询多个$Query$，多个$Query$对应的词向量堆叠在一起形成矩阵$Q$,得到公式：

$attention(Q,K,V)=softmax(QK^T)V$

如果Q和K对应的维度$d_k$比较大，$softmax$会是一个“尖锐”的注意力分布。因此将Q和K乘积的结果做一个放缩：

$attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$

这就是注意力机制的核心计算公式

基于上文，可以简单用Pytorch实现注意力机制的代码：

```python
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    #获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1)
    #计算Q与K的内积并除以根号dk
    #transpose——相当于转置
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    #softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        #采样
    #根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

```
在上文代码中，输入的q，k，v是已经经过转化的词向量矩阵，也就是公式中的Q，K，V。

自注意力

注意力机制的本质，是对两段序列的元素依次进行相似度计算，寻找出一个序列的每个元素对另一个序列的每个元素的相关度，然后基于相关度加权，即分配注意力

```python
#attention 
attention(x,x,x)
```

### 掩码自注意力
Mask Self-Attention 遮蔽一些特定位置的token 只能使用历史信息进行预测而不是看到未来信息

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过full函数创建一个1 * seq_len * seq_len的矩阵
mask = torch.full((1,args.max_seq_len, args.max_seq_len), float("-inf"))
# triu函数的功能是创建一个上三角函数
mask = torch.triu(mask, diagonal=1)
```
生成的Mask矩阵会是一个上三角矩阵，上三角位置的元素均为-inf，其他位置的元素置为0.

在注意力计算时，我们将计算得到的注意力分数与这个掩码做和，在进行Softmax操作：

```python
# 此处的 scores 为计算得到的注意力分数， mask 为上文生成的掩码矩阵
scores = scores + mask[:,:seqlen,:seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```
-inf的值经过softmax之后会被置为0，从而或略上三角区域计算的注意力分数，从而实现注意力遮蔽

### 多头注意力

一次注意力计算只能拟合一种相关关系，单一的注意力机制很难全面拟合语句序列里的相关关系。因此Transformer使用了多头注意力机制(Multi-Head Attention),即同时对一个语料进行多次注意力计算。

事实上，所谓的多头注意力机制其实就是将原始的输入序列进行多组的自注意力处理;然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理，得到最终的输出。

$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^0where head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$

```Python
import torch.nn as nn
import torch

'''MultiHeadAttention Module'''
class MultiHeadAttention(nn.Module):
    def _init_(self, args:ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成多数个矩阵
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为n_embd * dim
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵， 维度为dim x dim（head_dim = dim/n_heads）
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意， 因为是多头注意力，Mask矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 获取批次大小和序列长度， [batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做Dropout
        scores = self.attn_dropout(scores)
        # V * Score, 维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

## 2.2 Encoder-Decoder

两个核心组件Encoder、Decoder , 使用了注意力机制

### 2.2.1 seq2seq
输入$input=(x_1, x_2, x_3...x_n)$,输出$output=(y_1, y_2, y_3...y_m)$.

一个Encoder(Decoder)由6个Encoder(Decoder) Layer 组成

Encoder 和 Decoder 内部结构： FNN前馈神经网络、层归一化 LayerNorm、 残差连接 Residual Connection

### 2.2.2前馈神经网络
每一层的神经元都和上下两层的每一个神经元完全连接的网络结构
```python
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
```
注意，Transformer 的前馈神经网络是由两个线性层中间加一个 RELU 激活函数组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合

### 2.2.3层归一化

层归一化，也就是 Layer Norm

首先计算样本的均值：
$$
\mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^i
$$

其中，$Z_j^i$ 是样本i在第j个维度上的值，m是mini-batch 大小。

再计算样本的方差：

$$
\sigma^2=\frac{1}{m}\sum^{m}_{i=1}{(Z_{j}^{i}-\mu_j)}^2
$$

最后，对每个样本的值减去均值再除以标准差来将这一个mini-batch的样本的分布转化为标准正态分布：

$$
\widetilde{Z_j}=\frac{Z_j-\mu_j}{\sqrt{\sigma^2+\epsilon}}
$$

此处加上 $\epsilon$ 这一极小值是为了避免分母为0.

但是，批归一化存在一些缺陷，例如：
- 当显存有限，mini-batch 较小时，Batch Norm 取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；
- 对于在时间维度展开的 RNN，不同句子的同一分布大概率不同，所以 Batch Norm 的归一化会失去意义；
- 在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；
- 应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力

因此，出现了在深度神经网络中更常用、效果更好的层归一化（Layer Norm）。相较于 Batch Norm 在每一层统计所有样本的均值和方差，Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计统计量的维度不同。

```python
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
        #线性矩阵做映射
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True) # std:[bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

```

### 2.2.4 残差链接

Encoder Layer，输入进入多头自注意力层的同时会直接传递到该层的输出，然后该层的输出会与原输入相加，再进行标准化。

$$
x=x+MultiHeadSelfAttention(LayerNorm(x))
$$
$$
output = x + FFN(LayerNorm(x))
$$
通过在层的 forward 计算中加上原值来实现残差连接:

```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```
在上文代码中，self.attention_norm 和 self.fnn_norm 都是 LayerNorm 层，self.attn 是注意力层，而 self.feed_forward 是前馈神经网络。

### 2.2.5 Encoder
在实现上述组件之后，我们可以搭建起 Transformer 的 Encoder。Encoder 由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。因此，我们可以首先实现一个 Encoder Layer:
```python
class EncoderLayer(nn.Module):
     '''Encoder层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.hidden_dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
```

然后我们搭建一个 Encoder，由 N 个 Encoder Layer 组成，在最后会加入一个 Layer Norm 实现规范化：
```python
class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super().__int__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

### 2.2.6 Decoder
类似的，我们也可以先搭建 Decoder Layer，再将 N 个 Decoder Layer 组装为 Decoder。但是和 Encoder 不同的是，Decoder 由两个注意力层和一个前馈神经网络组成。第一个注意力层是一个掩码自注意力层，即使用 Mask 的注意力计算，保证每一个 token 只能使用该 token 之前的注意力分数；第二个注意力层是一个多头注意力层，该层将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。最后，再经过前馈神经网络：
```python
class DecoderLayer(nn.Module):
    '''解码层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args.dim, args.hidden_dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h +self.feed_forward.forward(self.ffn_norm(h))
        return out
```
然后同样的，我们搭建一个 Decoder 块：
```python
class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super().__init__()
         # 一个 Decoder 由 N 个 Decoder Layer 组成
         self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
         self.norm = LayerNorm(args.n_embd)
    
    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```
完成上述 Encoder、Decoder 的搭建，就完成了 Transformer 的核心部分，接下来将 Encoder、Decoder 拼接起来再加入 Embedding 层就可以搭建出完整的 Transformer 模型啦。

## 2.3 搭建一个Transformer

深入剖析了 Attention 机制和 Transformer 的核心——Encoder、Decoder 结构

### 2.3.1 Embedding层

Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表。也就是说，在输入神经网络之前，我们往往会先让自然语言输入通过分词器tokenizer，分词器的作用是把自然语言输入切分成token并转化成一个固定的index。例如，我们将词表大小设为4，输入“我喜欢你”，那么，分词器可以将输入转化成：
```
input: 我
output: 0

input: 喜欢
output: 1

input: 你
output: 2
```

当然，在实际情况下，tokenizer的工作会比这更复杂。例如，分词有多种不同的方式，可以切分成词、切分成子词、切分成字符等，而词表大小则往往高达数万数十万。 tokenizer 在大模型的运行与训练是非常重要的。

因此，Embedding 层的输入往往是一个形状为(batch_size, seq_len, 1)的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。例如，对上述输入，Embedding 层的输入会是：
```
[[[0],[1],[2]]]
```
其 batch_size 为1，seq_len 为3，转化出来的 index 如上。

而 Embedding 内部其实是一个可训练的（Vocab_size，embedding_dim）的权重矩阵，词表里的每一个值，都对应一行维度为 embedding_dim 的向量。对于输入的值，会对应到这个词向量，然后拼接成（batch_size，seq_len，embedding_dim）的矩阵输出。

可以直接使用torch中的Embedding层：
```python
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
```

### 2.3.2 位置编码
在注意力机制的计算过程中，对于序列中的每一个 token，其他各个位置对其来说都是平等的，即“我喜欢你”和“你喜欢我”在注意力机制看来是完全相同的，但无疑这是注意力机制存在的一个巨大问题。因此，为使用序列顺序信息，保留序列中的相对位置信息，Transformer 采用了位置编码机制，该机制也在之后被多种模型沿用。

位置编码，即根据序列中 token 的相对位置对其进行编码，再将位置编码加入词向量编码中。Transformer 使用了正余弦函数来进行位置编码（绝对位置编码Sinusoidal），其编码方式为：

$$
PE(pos,2i)=sin(pos/10000^{2i/d_{model}})\\
PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}})
$$
pos 为 token 在句子中的位置，2i 和 2i+1 则是指示了 token 是奇数位置还是偶数位置，从上式中我们可以看出对于奇数位置的 token 和偶数位置的 token，Transformer 采用了不同的函数进行编码。

举一个例子，输入长度为4的句子"I like to code"，得到词向量矩阵$x$:
$$
\mathrm x =\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4\\
0.2 & 0.3 & 0.4 & 0.5\\
0.3 & 0.4 & 0.5 & 0.6\\
0.4 & 0.5 & 0.6 & 0.7\\
\end{bmatrix}
$$
其中，每一行代表一个词向量，$x_0 = [0.1,0.2,0.3,0.4]$对应"I"的词向量，pos为0

经过位置编码后的词向量为：
$$
\mathrm x_{PE} = \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4\\
0.2 & 0.3 & 0.4 & 0.5\\
0.3 & 0.4 & 0.5 & 0.6\\
0.4 & 0.5 & 0.6 & 0.7\\
\end{bmatrix}+\begin{bmatrix}
\sin(\frac{0}{10000^0}) & \cos(\frac{0}{10000^0}) & \sin(\frac{0}{10000^{2/4}}) & \cos(\frac{0}{10000^{2/4}}) \\
\sin(\frac{1}{10000^0}) & \cos(\frac{1}{10000^0}) & \sin(\frac{1}{10000^{2/4}}) & \cos(\frac{1}{10000^{2/4}}) \\
\sin(\frac{2}{10000^0}) & \cos(\frac{2}{10000^0}) & \sin(\frac{2}{10000^{2/4}}) & \cos(\frac{2}{10000^{2/4}}) \\
\sin(\frac{3}{10000^0}) & \cos(\frac{3}{10000^0}) & \sin(\frac{3}{10000^{2/4}}) & \cos(\frac{3}{10000^{2/4}}) \\
\end{bmatrix} = \begin{bmatrix}
0.1 & 1.2 & 0.3 & 1.4 \\
1.041 & 0.84 & 0.41 & 1.49 \\
1.209 & -0.016 & 0.52 & 1.59 \\
0.541 & -0.489 & 0.895 & 1.655
\end{bmatrix}
$$

使用代码来获取上述例子的位置编码：
```python
import numpy as np
import matplotlib.pyplot as plt
def PositionEncoding(seq_len, d_model, n=10000)
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

PositionEncoding(seq_len=4, d_model=4, n=100)
```

```python
[[ 0.          1.          0.          1.        ]
 [ 0.84147098  0.54030231  0.09983342  0.99500417]
 [ 0.90929743 -0.41614684  0.19866933  0.98006658]
 [ 0.14112001 -0.9899925   0.29552021  0.95533649]]
```

1. 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
2. 可以让模型容易地计算出相对位置，对于固定长度的间距 k，PE(pos+k) 可以用 PE(pos) 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。

原始的Transformer Embedding 可以表示为：
$$
\begin{equation}
f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_n,\cdots,\boldsymbol{x}_m,\cdots)
\end{equation}
$$

很明显，这样的函数是不具有不对称性的，也就是无法表征相对位置信息。我们想要得到这样一种编码方式：

$$
\begin{equation}
\tilde{f}(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_m + \boldsymbol{p}_m ,\cdots,\boldsymbol{x}_n + \boldsymbol{p}_n,\cdots)
\end{equation}
$$

这里加上的 $p_m$, $p_n$ 就是位置编码。接下来我们将 $f(\cdots, x_m+p_m,\cdots, x_n+p_n)$ 在 m,n 两个位置上做泰勒展开：
$$
\begin{equation}
\tilde{f}\approx f + \boldsymbol{p}_m^{\top}\frac{\partial f}{\partial \boldsymbol{x_m}} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x_n}} + \frac{1}{2}\boldsymbol{p}_m^{\top}\frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2} \boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}_n}_{\boldsymbol{p}_m^{\top}  \boldsymbol{\mathcal{H}} \boldsymbol{p}_n} 
\end{equation}
$$

位置编码层：
```python
class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super().__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算theta
        div_term = torch.exp(torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd))
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
```


### 2.3.3 Transformer

Transformer 模型图
<div align="center">
<img src="transformer.png" alt="Transformer" width="80%"/>
</div>


```python
class Transformer(nn.Module):
    '''整体模型'''
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False)：
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self. module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx", idx.size()) # (batch size, sequence length, 1)
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx) # (batch size, sequence length, n_embd)
        print("tok_emb", tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) # (batch size, sequence length, n_embd)
        # 再进行 Dropout
        x = self.transformer.drop(pos_enb)
        # 然后通过 Encoder
        print("x after wpe:", x.size) # (batch size, sequence length, n_embd)
        enc_out = self.transformer.encoder(x)
        print("enc_out:", enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:", x.size())

        if target is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
```

注意，上述代码除去搭建了整个 Transformer 结构外，我们还额外实现了三个函数：
- get_num_params：用于统计模型的参数量
- _init_weights：用于对模型所有参数进行随机初始化
- forward: 前向计算函数

