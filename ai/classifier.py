# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# from sklearn.preprocessing import StandardScaler

# # --- 1. The Neural Network Class: Defining the AI's Brain ---
# class ClassifierNet(nn.Module):
#     def __init__(self, input_features=8):
#         super(ClassifierNet, self).__init__()
#         # Define the layers of the network
#         self.layer_1 = nn.Linear(input_features, 64)
#         self.layer_2 = nn.Linear(64, 32)
#         self.output_layer = nn.Linear(32, 1)
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.1)
        
#     def forward(self, inputs):
#         # Define the forward pass (how data flows through the network)
#         x = self.relu(self.layer_1(inputs))
#         x = self.dropout(x)
#         x = self.relu(self.layer_2(x))
#         x = self.dropout(x)
#         x = torch.sigmoid(self.output_layer(x)) # Sigmoid for final probability
#         return x

# # --- 2. Data Loading and Preparation ---
# print("Loading and preparing data...")
# try:
#     dataframe = pd.read_csv("data/ai_training_data.csv")
# except FileNotFoundError:
#     print("❌ Error: ai_training_data.csv not found.")
#     print("Please run ai/data_preparer.py first.")
#     exit()

# # Separate features (X) from the target label (y)
# X = dataframe.drop('label', axis=1).values
# y = dataframe['label'].values

# # Split data: 80% for training, 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Scale the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Convert numpy arrays to PyTorch Tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# # --- 3. Creating Datasets and DataLoaders ---
# # Create TensorDatasets
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# # Create DataLoaders to handle batching
# batch_size = 64
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# # --- 4. Initializing the Training Components ---
# print("Initializing model and optimizer...")
# model = ClassifierNet()
# criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # --- 5. The Training Loop: Teaching the AI ---
# print("Starting training...")
# num_epochs = 20
# for epoch in range(num_epochs):
#     model.train() # Set the model to training mode
#     running_loss = 0.0
    
#     for features, labels in train_loader:
#         # 1. Zero the gradients
#         optimizer.zero_grad()
#         # 2. Forward pass
#         outputs = model(features)
#         # 3. Calculate loss
#         loss = criterion(outputs, labels)
#         # 4. Backward pass
#         loss.backward()
#         # 5. Update weights
#         optimizer.step()
        
#         running_loss += loss.item()
        
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# # --- 6. The Evaluation Loop: Testing the AI ---
# print("\nStarting evaluation...")
# model.eval() # Set the model to evaluation mode
# all_preds = []
# all_labels = []

# with torch.no_grad(): # Disable gradient calculation for evaluation
#     for features, labels in test_loader:
#         outputs = model(features)
#         predicted = outputs.round() # Round probabilities to get 0 or 1
#         all_preds.extend(predicted.numpy())
#         all_labels.extend(labels.numpy())

# # Calculate metrics
# accuracy = accuracy_score(all_labels, all_preds)
# precision = precision_score(all_labels, all_preds)
# recall = recall_score(all_labels, all_preds)
# roc_auc = roc_auc_score(all_labels, all_preds)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")

# # --- 7. Saving the Model: Preserving the Brain ---
# model_save_path = "ai/models/z_classifier_v1.pth"
# torch.save(model.state_dict(), model_save_path)
# print(f"\n✅ Model saved to {model_save_path}")

















import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- 1. The UPGRADED Neural Network Class ---
class ClassifierNet(nn.Module):
    # This model is correctly defined to accept 9 input features
    def __init__(self, input_features=9):
        super(ClassifierNet, self).__init__()
        self.layer_1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer_2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer_3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, inputs):
        x = self.relu(self.bn1(self.layer_1(inputs)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer_3(x)))
        x = self.output_layer(x)
        return x

# --- 2. Data Loading and Preparation ---
print("Loading and preparing final dataset...")
try:
    # Point to the new, final data file
    dataframe = pd.read_csv("data/final_ai_data.csv") # <--- MODIFIED
except FileNotFoundError:
    print("❌ Error: final_ai_data.csv not found.")
    print("Please run the latest ai/data_preparer.py first.")
    exit()

# Use all columns except 'label' as features
X = dataframe.drop('label', axis=1).values # <--- MODIFIED
y = dataframe['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# --- 3. Creating Datasets and DataLoaders ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- 4. Initializing the Training Components ---
print("Initializing final model and optimizer...")
model = ClassifierNet()

# Calculate class weights for the final dataset
num_background = np.sum(y_train == 0)
num_signal = np.sum(y_train == 1)
# Add a small epsilon to prevent division by zero if a class is missing
weight_for_signal = num_background / (num_signal + 1e-6) 
class_weights = torch.tensor([weight_for_signal], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. The Training Loop: Teaching the AI ---
print("Starting final training...")
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# --- 6. The Evaluation Loop: Testing the AI ---
print("\nStarting final evaluation...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        probabilities = torch.sigmoid(outputs)
        predicted = probabilities.round()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# --- 7. Saving the Model: Preserving the Brain ---
model_save_path = "ai/models/z_classifier_v4_final.pth" # New model name
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Final model saved to {model_save_path}")