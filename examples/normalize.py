import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GazeDataset(Dataset):
    def __init__(self, data):
        self.samples = data
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Normalize inputs
        gaze_direction = np.array(sample['gaze_direction'])  # Already normalized (-1 to 1)
        sphere_center = np.array(sample['sphere_center']) / 20.0  # Normalize large values
        marker_position = np.array(sample['marker_position']) / 500.0  # Normalize pixel coordinates
        
        features = np.concatenate([gaze_direction, sphere_center])
        
        return torch.FloatTensor(features), torch.FloatTensor(marker_position)

class BranchedGazePredictionModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(BranchedGazePredictionModel, self).__init__()
        
        self.input_size = 6  # gaze_direction (3) + sphere_center (3)
        self.output_size = 2  # marker_position (2)
        
        # Separate processing branches for gaze direction and sphere center
        self.gaze_branch = nn.Sequential(
            nn.Linear(3, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2)
        )
        
        self.sphere_branch = nn.Sequential(
            nn.Linear(3, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2)
        )
        
        # Main network after feature concatenation
        self.main_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//4),
            
            nn.Linear(hidden_size//4, self.output_size)
        )
        
    def forward(self, x):
        # Split input into gaze and sphere components
        gaze_features = x[:, :3]
        sphere_features = x[:, 3:]
        
        # Process each branch
        gaze_processed = self.gaze_branch(gaze_features)
        sphere_processed = self.sphere_branch(sphere_features)
        
        # Concatenate processed features
        combined = torch.cat([gaze_processed, sphere_processed], dim=1)
        
        return self.main_network(combined)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gaze_model3.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print('-' * 50)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves2.png')
    plt.close()
    
    return train_losses, val_losses

def predict_gaze_point(model, gaze_direction, sphere_center):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Normalize inputs
        sphere_center = np.array(sphere_center) / 20.0
        
        input_features = torch.FloatTensor(
            np.concatenate([gaze_direction, sphere_center])
        ).unsqueeze(0).to(device)
        
        predicted_point = model(input_features)
        predicted_point = predicted_point.cpu().squeeze().numpy() * 500.0
        return predicted_point

def visualize_predictions(model, data, num_samples=5):
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        sample = data[i]
        predicted_point = predict_gaze_point(
            model,
            sample['gaze_direction'],
            sample['sphere_center']
        )
        actual_point = np.array(sample['marker_position'])
        
        plt.scatter(actual_point[0], actual_point[1], c='blue', label='Actual' if i == 0 else "")
        plt.scatter(predicted_point[0], predicted_point[1], c='red', label='Predicted' if i == 0 else "")
        
        # Draw line between actual and predicted points
        plt.plot([actual_point[0], predicted_point[0]], 
                [actual_point[1], predicted_point[1]], 
                'g--', alpha=0.3)
    
    plt.title('Actual vs Predicted Gaze Points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.savefig('predictions_visualization2.png')
    plt.close()

def main():
    # Load data
    with open('eye_tracking_data2s.json', 'r') as f:
        data = json.load(f)
    
    # Filter samples with confidence 1.0
    data = [sample for sample in data if sample['confidence'] > 0.9]
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
    
    # Create datasets
    train_dataset = GazeDataset(train_data)
    val_dataset = GazeDataset(val_data)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and training components
    model = BranchedGazePredictionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True
    )
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100
    )
    
    # Load the best model for prediction
    model.load_state_dict(torch.load('best_gaze_model3.pth'))
    
    # Visualize some predictions
    visualize_predictions(model, val_data)
    
    # Calculate and print average error on validation set
    print("\nCalculating average error on validation set...")
    total_error = 0
    count = 0
    
    for sample in val_data:
        predicted = predict_gaze_point(
            model,
            sample['gaze_direction'],
            sample['sphere_center']
        )
        actual = np.array(sample['marker_position'])
        error = np.linalg.norm(predicted - actual)
        total_error += error
        count += 1
    
    avg_error = total_error / count
    print(f"Average error on validation set: {avg_error:.2f} pixels")

if __name__ == "__main__":
    main()