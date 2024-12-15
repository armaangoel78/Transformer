import torch

DTYPE = torch.float32
EPS = 1e-6

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(hidden_dim, dtype=DTYPE))
        self.beta = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=DTYPE))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + EPS) + self.beta

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Parameter(
            torch.randn(vocab_size, hidden_dim, dtype=DTYPE)
        )

    def forward(self, x):
        return self.embedding[x]

class Linear(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(input_dim, output_dim, dtype=DTYPE)
            * torch.sqrt(torch.tensor(2 / (input_dim + output_dim), dtype=DTYPE))
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(output_dim, dtype=DTYPE)
        )

    def forward(self, x):
        return x @ self.weights + self.bias
        
class ReLu(torch.nn.Module):
    def forward(self, x):
        return x.clamp(min=0)
    

class MLP(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = Linear(hidden_dim, hidden_dim)
        self.relu = ReLu()
        self.linear2 = Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        return output
    

class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.q = Linear(hidden_dim, hidden_dim)
        self.k = Linear(hidden_dim, hidden_dim)
        self.v = Linear(hidden_dim, hidden_dim)
        self.o = Linear(hidden_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        seq_len = x.shape[-2]

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Create attention matrix
        attn = Q @ torch.transpose(K, -1, -2)

        # Scale
        attn = attn / torch.sqrt(torch.tensor(seq_len, dtype=DTYPE))

        # Create mask of elements that should not be attended to
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=DTYPE, device=x.device), diagonal=1
        ).bool()

        # Set masked elements to -inf
        attn = attn.masked_fill(mask, float("-inf"))

        # Apply softmax
        attn = self.softmax(attn)

        output = attn @ V
        output = self.o(output)

        return output
    
class TransformerLayer(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(TransformerLayer, self).__init__()
        self.attention = Attention(hidden_dim)
        self.layernorm1 = LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim)
        self.layernorm2 = LayerNorm(hidden_dim)

    def forward(self, x):
        output = self.attention(self.layernorm1(x)) + x
        output = self.mlp(self.layernorm2(output)) + output
        return output

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        assert num_layers > 0, "Number of layers must be greater than 0"
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [TransformerLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.linear = Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)
        return x