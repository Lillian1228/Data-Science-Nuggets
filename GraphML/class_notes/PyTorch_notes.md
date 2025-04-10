
#### How to train a Neural Network

Graph convolutional network example:
1. Define the network architechture: initialization and forward function

```Python
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

# initialize the model object
model = GCN()
print(model)
```


2. Define a loss criterion and initialize a stochastic gradient optimizer


```Python
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.
```

3. Training scheme - Perform multiple rounds of optimization:
   
   - Each round consists of a forward and backward pass to compute the gradients of our model parameters w.r.t. to the loss derived from the forward pass.
   - Incrementally update model parameters based on fresh gradients obtained in each step.


```Python
# semi-supervised training example

def train(data):
    optimizer.zero_grad()  # clear gradients to avoid mixing them into the next batch
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass to compute node embeddings on all data.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update model parameters based on gradients from current epoch.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    # Visualize the node embeddings every 10 epochs
    if epoch % 10 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)
```

4. Test the network on the test data

not applicable for the GCN example.