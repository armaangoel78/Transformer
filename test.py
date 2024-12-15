from layers import *
from data import *

class NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.layers = torch.nn.ModuleList(
            [MLP(input_dim, input_dim) for _ in range(3)]
        )
        
    def forward(self, x):
        return self.relu(self.linear_projection(x))

BATCH_SIZE = 4

model = MLP(10, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(4, 10)
y = torch.tensor([0, 1, 0, 1])

for i in range(10000):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Loss: {loss.item()}")