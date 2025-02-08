import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

class GazeDataset(Dataset):
    def __init__(self, data, augment=False):
        self.samples = data
        self.augment = augment
        
    def __len__(self):
        return len(self.samples)
    
    def augment_data(self, gaze_direction, sphere_center):
        # Add small random noise for data augmentation
        gaze_noise = np.random.normal(0, 0.01, size=gaze_direction.shape)
        sphere_noise = np.random.normal(0, 0.01, size=sphere_center.shape)
        
        return gaze_direction + gaze_noise, sphere_center + sphere_noise
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        gaze_direction = np.array(sample['gaze_direction'])
        sphere_center = np.array(sample['sphere_center']) / 20.0
        marker_position = np.array(sample['marker_position']) / 500.0
        
        if self.augment:
            gaze_direction, sphere_center = self.augment_data(gaze_direction, sphere_center)
        
        features = np.concatenate([gaze_direction, sphere_center])
        return torch.FloatTensor(features), torch.FloatTensor(marker_position)

class ImprovedGazePredictionModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(ImprovedGazePredictionModel, self).__init__()
        
        self.input_size = 6
        self.output_size = 2
        
        # Improved gaze branch with residual connections
        self.gaze_branch = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size//2),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_size//2, hidden_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size//2),
                nn.Dropout(0.1)
            )
        ])
        
        # Improved sphere branch with attention mechanism
        self.sphere_branch = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size//2),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_size//2, hidden_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size//2),
                nn.Dropout(0.1)
            )
        ])
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=1)
        )
        
        # Main network with skip connections
        self.main_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size//2),
                nn.Dropout(0.2)
            ),
            nn.Linear(hidden_size//2, self.output_size)
        ])
        
    def forward(self, x):
        # Split input
        gaze_features = x[:, :3]
        sphere_features = x[:, 3:]
        
        # Process gaze branch with residual connection
        gaze_out = gaze_features
        for layer in self.gaze_branch:
            gaze_out = layer(gaze_out) + gaze_out if gaze_out.shape == layer(gaze_out).shape else layer(gaze_out)
            
        # Process sphere branch with residual connection
        sphere_out = sphere_features
        for layer in self.sphere_branch:
            sphere_out = layer(sphere_out) + sphere_out if sphere_out.shape == layer(sphere_out).shape else layer(sphere_out)
        
        # Combine features
        combined = torch.cat([gaze_out, sphere_out], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        weighted_features = torch.stack([gaze_out, sphere_out], dim=2) * attention_weights.unsqueeze(1)
        attentive_features = weighted_features.sum(dim=2)
        
        # Process through main network with skip connections
        main_out = combined
        for layer in self.main_network[:-1]:
            main_out = layer(main_out) + main_out if main_out.shape == layer(main_out).shape else layer(main_out)
        
        return self.main_network[-1](main_out)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.alpha = alpha
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        smooth_l1_loss = self.smooth_l1(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * smooth_l1_loss

def calculate_metrics(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    rmse = torch.sqrt(mse)
    return {'MSE': mse.item(), 'MAE': mae.item(), 'RMSE': rmse.item()}

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda'):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_metrics = {'MSE': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            batch_metrics = calculate_metrics(outputs, targets)
            running_loss += loss.item()
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key]
        
        train_loss = running_loss / len(train_loader)
        train_metrics = {k: v / len(train_loader) for k, v in running_metrics.items()}
        train_losses.append(train_loss)
        train_metrics_history.append(train_metrics)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'MSE': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                batch_metrics = calculate_metrics(outputs, targets)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_gaze_model2.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print('Training Metrics:', train_metrics)
            print('Validation Metrics:', val_metrics)
            print('-' * 50)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for metric in ['MSE', 'MAE', 'RMSE']:
        plt.plot([m[metric] for m in train_metrics_history], label=f'Train {metric}')
        plt.plot([m[metric] for m in val_metrics_history], label=f'Val {metric}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return train_losses, val_losses, train_metrics_history, val_metrics_history

def k_fold_cross_validation(data, k=5, hidden_size=256, batch_size=32, num_epochs=100):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f'FOLD {fold + 1}')
        print('-' * 50)
        
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        train_dataset = GazeDataset(train_data, augment=True)
        val_dataset = GazeDataset(val_data, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = ImprovedGazePredictionModel(hidden_size=hidden_size).to(device)
        criterion = CombinedLoss(alpha=0.8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        results = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            num_epochs=num_epochs, device=device
        )
        
        fold_results.append({
            'train_losses': results[0],
            'val_losses': results[1],
            'train_metrics': results[2],
            'val_metrics': results[3]
        })
    
    return fold_results

def visualize_predictions(model, data, num_samples=5):
    device = next(model.parameters()).device
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        sample = data[i]
        
        features = torch.FloatTensor(
            np.concatenate([
                sample['gaze_direction'],
                np.array(sample['sphere_center']) / 20.0
            ])
        ).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_point = model(features).cpu().squeeze().numpy() * 500.0
            
        actual_point = np.array(sample['marker_position'])
        
        plt.scatter(actual_point[0], actual_point[1], c='blue', label='Actual' if i == 0 else "")
        plt.scatter(predicted_point[0], predicted_point[1], c='red', label='Predicted' if i == 0 else "")
        plt.plot([actual_point[0], predicted_point[0]], 
                [actual_point[1], predicted_point[1]], 
                'g--', alpha=0.3)
    
    plt.title('Actual vs Predicted Gaze Points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.savefig('predictions_visualization3.png')
    plt.close()

def main():
    # Load data
    with open('eye_tracking_data2.json', 'r') as f:
        data = json.load(f)
    
    # Filter high confidence samples
    data = [sample for sample in data if sample['confidence'] > 0.9]
    
    # Perform k-fold cross validation
    fold_results = k_fold_cross_validation(data, k=5)
    
    # Calculate and display average results across folds
    avg_final_val_loss = np.mean([fold['val_losses'][-1] for fold in fold_results])
    print(f"\nAverage final validation loss across folds: {avg_final_val_loss:.4f}")
    
    # Plot average learning curves
    plt.figure(figsize=(10, 5))
    for i, fold in enumerate(fold_results):
        plt.plot(fold['val_losses'], alpha=0.3, label=f'Fold {i+1}')
    plt.plot(np.mean([fold['val_losses'] for fold in fold_results], axis=0), 
             'r-', linewidth=2, label='Average')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Learning Curves Across Folds')
    plt.legend()
    plt.savefig('cross_validation_curves.png')
    plt.close()

if __name__ == "__main__":
    main()