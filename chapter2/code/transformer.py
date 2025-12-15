import torch
from torch import nn
from dataclasses import dataclass
from transformer import BertTokenizer
import torch.nn.functional as F

@dataclass
class ModelArgs():
    n_embd: int # 嵌入维度
    n_heads: int # 头数
    dim : int # 模型维度
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int
    n_layer: int


class PositionalEncoding(nn.Module):
    def __init__(self, args: ModelArgs):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.n_embd, 2).float() * (-torch.log(torch.tensor(10000.0)) / args.n_embd))   
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, block_size, n_embd)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].require_grad_(False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim //args.n_heads
        self.n_heads = args.n_heads
        # 这里可以添加更多的多头注意力组件，例如线性层、注意力计算等
        self.wq = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 多头注意力的前向传播逻辑
        bsz, seqlen, _ = q.shape
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)  # (B, T, n_heads * head_dim)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim) # (B, T, n_heads, head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)  # (B, n_heads, T, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算 
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if self.is_causal:
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        # V * scores, 维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)  # (B, nh, T, hs)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (B, T, n_heads * head_dim)
        output = self.wo(output)  # (B, T, n_embd)
        output = self.resid_dropout(output)

        return output
    
class MLP(nn.Module):
    def __init__(self, dim: int, hiddden_dim: int, dropout: float):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, hiddden_dim, bias=False)
        self.fc2 = nn.Linear(hiddden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(torch.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super(EncoderLayer, self).__init__()
        self.args = args
        # 这里可以添加更多的编码器层组件，例如多头注意力、前馈网络等
        self.attention_norm = nn.LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_decoder=False)
        self.ffn_norm = nn.LayerNorm(args.n_embd)
        self.ffn = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # 编码器层的前向传播逻辑
        x = self.attention_norm(x)
        h = x + self.attention(x, x, x)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Encoder, self).__init__()
        self.args = args
        # 这里可以添加更多的编码器组件，例如多头注意力、前馈网络等
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = nn.LayerNorm(args.n_embd)

    def forward(self, x):
        # 编码器的前向传播逻辑
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super(DecoderLayer, self).__init__()
        self.args = args
        # 这里可以添加更多的解码器层组件，例如多头注意力、前馈网络等
        self.attention_norm_1 = nn.LayerNorm(args.n_embd)
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = nn.LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = nn.LayerNorm(args.n_embd)
        self.ffn = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # 解码器层的前向传播逻辑
        x = self.attention_norm_1(x)
        x = x + self.mask_attention(x, x, x)
        x = self.attention_norm_2(x)
        h = x + self.attention(x, enc_out, enc_out)
        out = h + self.ffn(self.ffn_norm(h))    
        return out


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Decoder, self).__init__()
        self.args = args
        # 这里可以添加更多的解码器组件，例如多头注意力、前馈网络等
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = nn.LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        # 解码器的前向传播逻辑
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Transformer, self).__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        # 这里可以添加更多的模型组件，例如嵌入层、注意力机制等
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),   
        ))
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.apply(self._init_weights)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None):
        # idx 和 targets 都是 (B, T) 的张量
        device = idx.device
        b, t = idx.size()
        
        # 获取词嵌入和位置嵌入
        token_embeddings = self.transformer.wte(idx) # (B, T, n_embd)
        position_embeddings = self.transformer.wpe(token_embeddings) # (B, T, n_embd)

        x = self.transformer.drop(position_embeddings) # (B, T, n_embd)

        # 通过编码器和解码器
        enc_out = self.transformer.encoder(x)
        x = self.transformer.decoder(x, enc_out)
        
        if targets is not None:
            # 训练阶段
            # 计算损失
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # 推理阶段
            logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)
            loss = None

        return logits, loss

def main():
    args = ModelArgs(100, 10, 100, 0.1, 512, 1000, 1000, 2)
    text = "我喜欢快乐地学习大模型"
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    inputs_token = tokenizer(
        text,
        return_tensors="pt",
        max_length=args.max_seq_len,
        truncation=True,
        padding="max_length",
    )
    args.vocab_size = tokenizer.vocab_size
    transformer = Transformer(args)
    inputs_id = inputs_token['input_ids']
    logits, loss = transformer(inputs_id)
    print(logits)
    predicted_ids = torch.argmax(logits, dim=-1).item()
    output = tokenizer.decode([predicted_ids])
    print(output)
    

if __name__ == "__main__":
    print("-------------start------------")
    main()    
