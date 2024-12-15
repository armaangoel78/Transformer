from data import *
import wandb
import modal
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

image = modal.Image.debian_slim().pip_install("torch==2.3.0", "torchdata==0.8.0", "torchtext==0.18.0", "wandb", "datasets")
app = modal.App(
    "example-get-started", 
    image=image, 
    mounts=[modal.Mount.from_local_dir("data", remote_path="data")]
)

BATCH_SIZE = 16
MAX_LR = 0.0001
MIN_LR = 0.0000
STEPS = 10_000

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # PyTorch's built-in Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        # Create source mask (to prevent attention to padding tokens)
        src_mask = None  # For causal/autoregressive prediction, you'd want to use a causal mask here
        src_key_padding_mask = None  # If you have padding tokens, create a mask here
        
        # Embedding and positional encoding
        x = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoder(x)
        
        # Transformer encoder
        output = self.transformer_encoder(x, src_mask, src_key_padding_mask)
        
        # Final linear layer
        output = self.output(output)
        return output

def log_grad_norms(model):
    for i, param in enumerate(model.parameters()):
        wandb.log({
            f"grad_norms/{i}": param.grad.norm(),
        }, commit=False)

@app.function(gpu="T4")
def train(data_cache_path, device):
    print("Initializing wandb...")
    wandb.login(key="080a7a2df42b4715e1478f781fe50aa356656a17")
    wandb.init(project="transformer") 

    print("Initializing dataset...")
    dataset, token_to_id = get_or_create_dataset(data_cache_path, create=True, limit=1, char_limit=50)

    print("Initializing model...")
    model = TransformerModel(
        vocab_size=len(token_to_id),
        d_model=512,  # Standard transformer size
        nhead=8,      # Standard number of attention heads
        num_layers=6  # Standard number of transformer layers
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=MIN_LR)

    for i in range(STEPS):
        # Get batch
        x, y = get_batch(dataset, token_to_id, BATCH_SIZE)
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_hat = model(x)
        
        # Reshape tensors for loss computation
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)

        # Compute loss
        loss = criterion(y_hat, y)

        # Backward pass
        loss.backward()

        # Logging
        log_grad_norms(model)
        wandb.log({
            "loss/loss": loss.item(),
        }, commit=True)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if i % 1 == 0:
            print(f"Loss: {loss.item()}")

@app.local_entrypoint()
def main():
    train.remote(data_cache_path="/data/tokens_per_row.pkl", device="cuda")

if __name__ == "__main__":
    train.local(data_cache_path="data/tokens_per_row.pkl", device="cpu")
