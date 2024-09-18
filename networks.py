
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # First hidden layer
        self.fc2 = nn.Linear(1024, 512)         # Second hidden layer
        self.fc3 = nn.Linear(512, 1)           # Output layer (binary classification)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Sigmoid activation for binary classification
        return x

def forward_mlp(model, input):
    return torch.sigmoid(model(input))

def forward_siamese(model, input1, input2):
    # Pass both inputs through the same model (weight sharing)
    reward1 = model(input1)
    reward2 = model(input2)
    # Compute the difference between the two embeddings
    diff = reward1 - reward2

    # Use this difference for binary prediction (preference/ranking)
    return torch.sigmoid(diff)

def train_model(model, device, model_mode, X_train, y_train, X_test, y_test, epochs=3, lr=0.001, batch_size = 512, verbose=False):
    model = model.to(device)  # Move model to GPU
    criterion = nn.BCELoss()  # Binary cross-entropy loss with logits for numerical stability
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = min(batch_size, X_train.shape[0])
    for epoch in range(epochs):
        for n_batch in range(0, X_train.shape[0], batch_size):

            # randomly selecte batch
            batch_idx = torch.randperm(X_train.shape[0])[:batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            model.train()
            optimizer.zero_grad()

            # Forward pass
            if model_mode == 'clf':
                outputs = forward_mlp(model, X_batch)
            elif model_mode == 'siamese':
                outputs = forward_siamese(model, X_batch[:, 0], X_batch[:, 1])
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            if model_mode == 'clf':
                val_outputs = forward_mlp(model, X_test)
            elif model_mode == 'siamese':
                val_outputs = forward_siamese(model, X_test[:, 0], X_test[:, 1])
        # early stopping?
        val_loss = criterion(val_outputs, y_test)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.cpu().detach().item()}, Val Loss: {val_loss.cpu().detach().item()}")
        # accuracy
        val_outputs = val_outputs.cpu().detach().numpy()
        val_pred = (val_outputs > 0.5).astype(int)
        accuracy = (val_pred == y_test.cpu().detach().numpy()).mean()
        if verbose:
            print(f"Accuracy: {accuracy}")

def save_model(model, path):
    torch.save(model.state_dict(), path)