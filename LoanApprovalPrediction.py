import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = [x.strip(' ') for x in df.columns]
numerical_cols = [x.strip(' ') for x in df.describe().columns.drop("loan_id")]

ordinal_mapping = {
    " Approved": 1,
    " Rejected": 0,
}
df['loan_mapped'] = df['loan_status'].map(ordinal_mapping)

# After studying the data correlation with Loan_status
X = df[["cibil_score"]]
Y = df['loan_mapped']
print(type(df['loan_id']))
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    Y,
    df['loan_id'],
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class LoanApproval(nn.Module):
    def __init__(self, input_size):
        super(LoanApproval, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


input_size = X_train.shape[1]  # Number of features
model = LoanApproval(input_size)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

# Training loop
num_epochs = 100
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss = 0

for epoch in range(num_epochs):
    train_count = 0
    model.train()
    for batch_index, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(X_batch)
        # Compute loss
        loss = criterion(output, y_batch.float())
        train_loss += loss.item()
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        train_count += 1
    train_loss = train_loss / train_count
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

# Tensors
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_loss = 0
correct = 0
probability_prediction = list()
model.eval()
# Evaluation on test data
with torch.no_grad():
    for X_batch, y_batch in test_dataloader:
        y_prediction = model(X_batch)
        y_prediction_labels = (y_prediction > 0.50).float()
        loss = criterion(y_prediction, y_prediction_labels)
        test_loss += loss.item()
        correct += (y_batch == y_prediction_labels).sum().item()

        predicted_value = y_prediction * 100
        # Flatten 2-d tensor and convert to list
        flat_list = predicted_value.reshape(-1).tolist()
        probability_prediction = probability_prediction + flat_list

    results_df = pd.DataFrame({
        'Loan Id': id_test,
        'Prediction': probability_prediction
    })

    # Convert the DataFrame to a CSV file
    results_df.to_csv('Loan_Prediction.csv', index=False)
    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * correct / len(test_dataloader.dataset)
    print(f'Test Loss: {test_loss:.4f}%, Test Accuracy: {test_accuracy:.4f}%')
