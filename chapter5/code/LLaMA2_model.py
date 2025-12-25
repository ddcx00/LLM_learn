import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms = norm_x / (self.hidden_size ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.weight * x_normed

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_length, _ = query.size()

        def shape(x):
            return x.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        query = shape(self.query(query))
        key = shape(self.key(key))
        value = shape(self.value(value))

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        output = self.out(attn_output)
        return output, attn_weights
    

class LLaMA2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attention = Attention(config.hidden_size, config.num_attention_heads)
        self.ffn = MLP(config.hidden_size, config.ffn_hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        x_norm = self.attn_norm(x)
        attn_output, _ = self.attention(x_norm, attn_mask=attention_mask)
        h = x + attn_output
        x = self.norm1(h)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class LLaMA2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LLaMA2Block(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.tok_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits