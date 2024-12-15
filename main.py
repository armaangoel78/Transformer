from layers import *
from data import *
import wandb
import modal
from torch.optim.lr_scheduler import CosineAnnealingLR

image = modal.Image.debian_slim().pip_install("torch==2.3.0", "torchdata==0.8.0", "torchtext==0.18.0", "wandb", "datasets")
app = modal.App(
    "example-get-started", 
    image=image, 
    mounts=[modal.Mount.from_local_dir("data", remote_path="data")]
)

BATCH_SIZE = 1
MAX_LR = 0.0001
MIN_LR = 0.0000
STEPS = 10_000

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
    dataset, token_to_id = get_or_create_dataset(data_cache_path, create=True, char_limit=50)
    # dataset, token_to_id = get_dummy_dataset()

    print("Initializing model...")
    model = Transformer(len(token_to_id), 256, 4).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    # scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=MIN_LR)

    for i in range(STEPS):
        # Get batch
        x, y = get_batch(dataset, token_to_id, BATCH_SIZE)
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_hat = model(x)
        
        # Reshape for loss
        num_logits = y_hat.size(-1)
        y_hat = y_hat.view(-1, num_logits)
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
        # scheduler.step()

        if i % 1 == 0:
            print(f"Loss: {loss.item()}")


@app.local_entrypoint()
def main():
    train.remote(data_cache_path="/data/tokens_per_row.pkl", device="cuda")

if __name__ == "__main__":
    train.local(data_cache_path="data/tokens_per_row.pkl", device="cpu")
