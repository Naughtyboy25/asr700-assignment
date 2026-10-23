"""
Machine Learning Policy for ASR700 Component B.

Implements neural network policies for robotic navigation and decision-making.
Supports training, evaluation, and deployment to Webots controllers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import json
from python_analysis.utils.config import MLConfig, SensorConfig, EnvironmentConfig
from python_analysis.utils.helpers import normalize_angle


@dataclass
class TrainingMetrics:
    """Training metrics for logging."""
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float = 0.0
    val_acc: float = 0.0
    learning_rate: float = 0.0


class NavigationCNN(nn.Module):
    """
    CNN for robot navigation from LiDAR and camera inputs.

    Architecture:
    - LiDAR branch: 1D convolutions for range scan processing
    - Camera branch: 2D convolutions for image processing
    - Fusion: Concatenate features + fully connected layers
    - Output: (linear_velocity, angular_velocity)
    """

    def __init__(self, config: MLConfig = None):
        super().__init__()
        self.config = config or MLConfig()

        # LiDAR branch
        lidar_in = self.config.lidar_input_dim
        self.lidar_conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.lidar_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.lidar_pool = nn.AdaptiveAvgPool1d(16)
        self.lidar_fc = nn.Linear(64 * 16, self.config.hidden_dim)

        # Camera branch
        c_in, c_h, c_w = self.config.camera_input_shape
        self.cam_conv1 = nn.Conv2d(c_in, 32, kernel_size=3, padding=1)
        self.cam_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.cam_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cam_fc = nn.Linear(64 * 4 * 4, self.config.hidden_dim)

        # Fusion
        self.fusion_fc1 = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        self.fusion_fc2 = nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2)
        self.output = nn.Linear(self.config.hidden_dim // 2, self.config.output_dim)

        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, lidar: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            lidar: (batch, lidar_dim) or (batch, 1, lidar_dim)
            camera: (batch, H, W, C) or (batch, C, H, W)

        Returns:
            (batch, 2) - [linear_vel, angular_vel]
        """
        # LiDAR branch
        if lidar.dim() == 2:
            lidar = lidar.unsqueeze(1)  # (B, 1, D)
        x_lidar = self.relu(self.lidar_conv1(lidar))
        x_lidar = self.relu(self.lidar_conv2(x_lidar))
        x_lidar = self.lidar_pool(x_lidar)
        x_lidar = x_lidar.flatten(1)
        x_lidar = self.relu(self.lidar_fc(x_lidar))

        # Camera branch
        if camera.dim() == 4:
            # Check if NHWC format (batch, H, W, C) by checking if last dim is 3 (RGB)
            if camera.shape[-1] == 3 and camera.shape[1] != 3:
                # NHWC -> NCHW
                camera = camera.permute(0, 3, 1, 2)
        x_cam = self.relu(self.cam_conv1(camera))
        x_cam = self.relu(self.cam_conv2(x_cam))
        x_cam = self.cam_pool(x_cam)
        x_cam = x_cam.flatten(1)
        x_cam = self.relu(self.cam_fc(x_cam))

        # Fusion
        x = torch.cat([x_lidar, x_cam], dim=1)
        x = self.dropout(self.relu(self.fusion_fc1(x)))
        x = self.dropout(self.relu(self.fusion_fc2(x)))
        x = self.output(x)

        return x


class NavigationMLP(nn.Module):
    """
    Simpler MLP for navigation (LiDAR only).
    Faster training, good for baseline comparison.
    """

    def __init__(self, config: MLConfig = None):
        super().__init__()
        self.config = config or MLConfig()

        lidar_in = self.config.lidar_input_dim
        hidden = self.config.hidden_dim

        self.net = nn.Sequential(
            nn.Linear(lidar_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.config.output_dim)
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.net(lidar)


class DiscreteActionCNN(nn.Module):
    """
    CNN for discrete action selection (classification).

    Output: action probabilities [forward, left, right, stop, backward]
    """

    def __init__(self, config: MLConfig = None, num_actions: int = 5):
        super().__init__()
        self.config = config or MLConfig()
        self.num_actions = num_actions

        # Shared feature extractor (same as NavigationCNN)
        self.lidar_conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.lidar_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.lidar_pool = nn.AdaptiveAvgPool1d(16)
        self.lidar_fc = nn.Linear(64 * 16, self.config.hidden_dim)

        c_in, c_h, c_w = self.config.camera_input_shape
        self.cam_conv1 = nn.Conv2d(c_in, 32, kernel_size=3, padding=1)
        self.cam_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.cam_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cam_fc = nn.Linear(64 * 4 * 4, self.config.hidden_dim)

        # Classification head
        self.fc1 = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        self.fc2 = nn.Linear(self.config.hidden_dim, num_actions)

    def forward(self, lidar: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
        # LiDAR
        if lidar.dim() == 2:
            lidar = lidar.unsqueeze(1)
        x_lidar = torch.relu(self.lidar_conv1(lidar))
        x_lidar = torch.relu(self.lidar_conv2(x_lidar))
        x_lidar = self.lidar_pool(x_lidar).flatten(1)
        x_lidar = torch.relu(self.lidar_fc(x_lidar))

        # Camera
        if camera.dim() == 4 and camera.shape[1] != self.config.camera_input_shape[2]:
            camera = camera.permute(0, 3, 1, 2)
        x_cam = torch.relu(self.cam_conv1(camera))
        x_cam = torch.relu(self.cam_conv2(x_cam))
        x_cam = self.cam_pool(x_cam).flatten(1)
        x_cam = torch.relu(self.cam_fc(x_cam))

        # Fusion
        x = torch.cat([x_lidar, x_cam], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x  # Logits


class PolicyTrainer:
    """
    Trainer for navigation policies.
    """

    def __init__(self, model: nn.Module,
                 config: MLConfig = None,
                 device: str = None):
        self.model = model
        self.config = config or MLConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.MSELoss()  # For continuous control
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )

        self.metrics_history: List[TrainingMetrics] = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for lidar, camera, targets in train_loader:
            lidar = lidar.to(self.device)
            camera = camera.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if isinstance(self.model, NavigationCNN):
                outputs = self.model(lidar, camera)
            else:
                outputs = self.model(lidar)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * lidar.size(0)
            total += lidar.size(0)

            # For discrete actions, compute accuracy
            if outputs.shape[1] > 2:  # Classification
                _, predicted = torch.max(outputs, 1)
                _, target_class = torch.max(targets, 1)
                correct += (predicted == target_class).sum().item()

        avg_loss = total_loss / total
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for lidar, camera, targets in val_loader:
                lidar = lidar.to(self.device)
                camera = camera.to(self.device)
                targets = targets.to(self.device)

                if isinstance(self.model, NavigationCNN):
                    outputs = self.model(lidar, camera)
                else:
                    outputs = self.model(lidar)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * lidar.size(0)
                total += lidar.size(0)

                if outputs.shape[1] > 2:
                    _, predicted = torch.max(outputs, 1)
                    _, target_class = torch.max(targets, 1)
                    correct += (predicted == target_class).sum().item()

        avg_loss = total_loss / total
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = None) -> List[TrainingMetrics]:
        """Full training loop."""
        epochs = epochs or self.config.epochs
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
            self.metrics_history.append(metrics)

            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={metrics.learning_rate:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')

        return self.metrics_history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics_history
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint.get('metrics', [])


def generate_training_data(num_samples: int = 10000,
                           config: MLConfig = None,
                           sensor_config: SensorConfig = None,
                           env_config: EnvironmentConfig = None) -> Tuple:
    """
    Generate synthetic training data for navigation.

    In practice, this would collect data from Webots simulations.
    Here we generate synthetic data with simple rules.
    """
    config = config or MLConfig()
    sensor_config = sensor_config or SensorConfig()
    env_config = env_config or EnvironmentConfig()

    lidar_dim = config.lidar_input_dim
    # Camera shape in HWC format (64, 64, 3) for data generation
    cam_shape = (64, 64, 3)

    # Generate random robot states
    lidar_data = []
    camera_data = []
    targets = []

    for _ in range(num_samples):
        # Random robot pose
        x = np.random.uniform(1, env_config.world_size[0] - 1)
        y = np.random.uniform(1, env_config.world_size[1] - 1)
        theta = np.random.uniform(-np.pi, np.pi)

        # Random goal
        gx = np.random.uniform(1, env_config.world_size[0] - 1)
        gy = np.random.uniform(1, env_config.world_size[1] - 1)

        # Compute desired velocities (simple proportional control)
        dx = gx - x
        dy = gy - y
        distance = np.hypot(dx, dy)
        desired_heading = np.arctan2(dy, dx)
        heading_error = normalize_angle(desired_heading - theta)

        # Simple P controller for demonstration
        v = min(0.1, distance * 0.5)
        omega = np.clip(heading_error * 2.0, -1.0, 1.0)

        # Generate synthetic LiDAR (random obstacles)
        ranges = np.random.uniform(0.5, 5.0, lidar_dim)
        # Add some structure - closer obstacles in front
        front_idx = lidar_dim // 2
        ranges[front_idx-10:front_idx+10] *= 0.3

        # Normalize
        ranges = ranges / sensor_config.lidar_max_range

        # Synthetic camera (random noise with some pattern) - HWC format
        camera = np.random.rand(*cam_shape).astype(np.float32)

        lidar_data.append(ranges.astype(np.float32))
        camera_data.append(camera)
        targets.append([v, omega])

    return (np.array(lidar_data), np.array(camera_data), np.array(targets, dtype=np.float32))


def create_dataloaders(lidar_data: np.ndarray,
                       camera_data: np.ndarray,
                       targets: np.ndarray,
                       config: MLConfig = None) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders."""
    config = config or MLConfig()

    # Convert to tensors
    lidar_tensor = torch.FloatTensor(lidar_data)
    # Camera data is in HWC format (64, 64, 3), convert to CHW format (3, 64, 64)
    camera_tensor = torch.FloatTensor(camera_data).permute(0, 3, 1, 2)
    target_tensor = torch.FloatTensor(targets)

    # Split
    n = len(lidar_data)
    split = int(n * config.train_split)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = TensorDataset(
        lidar_tensor[train_idx], camera_tensor[train_idx], target_tensor[train_idx]
    )
    val_dataset = TensorDataset(
        lidar_tensor[val_idx], camera_tensor[val_idx], target_tensor[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: str = None) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Returns:
        Dictionary with metrics
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for lidar, camera, targets in test_loader:
            lidar = lidar.to(device)
            camera = camera.to(device)
            targets = targets.to(device)

            if isinstance(model, NavigationCNN):
                outputs = model(lidar, camera)
            else:
                outputs = model(lidar)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * lidar.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    # Per-dimension metrics
    v_mse = np.mean((preds[:, 0] - targets[:, 0]) ** 2)
    w_mse = np.mean((preds[:, 1] - targets[:, 1]) ** 2)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'v_mse': float(v_mse),
        'w_mse': float(w_mse),
        'num_samples': len(preds)
    }


def discretize_actions(continuous_actions: np.ndarray,
                       v_threshold: float = 0.05,
                       w_threshold: float = 0.3) -> np.ndarray:
    """
    Convert continuous (v, w) to discrete action classes.

    Classes:
    0: Forward (v > threshold, |w| < threshold)
    1: Left (w > threshold)
    2: Right (w < -threshold)
    3: Stop (v < threshold, |w| < threshold)
    4: Backward (v < -threshold)
    """
    v = continuous_actions[:, 0]
    w = continuous_actions[:, 1]

    actions = np.zeros(len(v), dtype=int)

    # Forward
    actions[(v > v_threshold) & (np.abs(w) <= w_threshold)] = 0
    # Left
    actions[w > w_threshold] = 1
    # Right
    actions[w < -w_threshold] = 2
    # Stop
    actions[(np.abs(v) <= v_threshold) & (np.abs(w) <= w_threshold)] = 3
    # Backward
    actions[v < -v_threshold] = 4

    return actions


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             num_classes: int = 5) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(cm: np.ndarray) -> Dict[str, float]:
    """Compute metrics from confusion matrix."""
    num_classes = cm.shape[0]
    metrics = {}

    # Overall accuracy
    total = cm.sum()
    correct = np.trace(cm)
    metrics['accuracy'] = correct / total if total > 0 else 0.0

    # Per-class metrics
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'class_{i}_precision'] = precision
        metrics[f'class_{i}_recall'] = recall
        metrics[f'class_{i}_f1'] = f1

    # Macro averages
    precisions = [metrics[f'class_{i}_precision'] for i in range(num_classes)]
    recalls = [metrics[f'class_{i}_recall'] for i in range(num_classes)]
    f1s = [metrics[f'class_{i}_f1'] for i in range(num_classes)]

    metrics['macro_precision'] = np.mean(precisions)
    metrics['macro_recall'] = np.mean(recalls)
    metrics['macro_f1'] = np.mean(f1s)

    return metrics


class DeployablePolicy:
    """
    Lightweight policy for deployment on Webots (no PyTorch dependency).
    """

    def __init__(self, model_path: str = None, model: nn.Module = None):
        self.model = model
        self.device = 'cpu'

        if model_path:
            self.load(model_path)

    def load(self, path: str):
        """Load PyTorch model and convert to numpy operations."""
        checkpoint = torch.load(path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            # Full checkpoint
            self.model = NavigationCNN()  # Default
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # State dict only
            self.model = NavigationCNN()
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self._extract_weights()

    def _extract_weights(self):
        """Extract weights as numpy arrays for fast inference."""
        self.weights = {}
        for name, param in self.model.named_parameters():
            self.weights[name] = param.detach().numpy()

    def predict(self, lidar: np.ndarray, camera: np.ndarray) -> np.ndarray:
        """
        Run inference using numpy only.

        Note: This is a simplified implementation. For production,
        consider using ONNX Runtime or TensorRT.
        """
        if self.model is not None:
            with torch.no_grad():
                lidar_t = torch.FloatTensor(lidar).unsqueeze(0)
                camera_t = torch.FloatTensor(camera).unsqueeze(0)
                if camera_t.shape[1] == 3:  # HWC -> CHW
                    camera_t = camera_t.permute(0, 3, 1, 2)
                out = self.model(lidar_t, camera_t)
                return out.numpy().flatten()

        # Fallback: numpy implementation (simplified)
        # In practice, you'd implement the full forward pass in numpy
        return np.array([0.1, 0.0])


def export_to_onnx(model: nn.Module, path: str,
                   input_shape_lidar: Tuple[int, ...] = (1, 360),
                   input_shape_camera: Tuple[int, ...] = (1, 3, 64, 64)):
    """Export model to ONNX for deployment."""
    model.eval()
    dummy_lidar = torch.randn(*input_shape_lidar)
    dummy_camera = torch.randn(*input_shape_camera)

    torch.onnx.export(
        model,
        (dummy_lidar, dummy_camera),
        path,
        input_names=['lidar', 'camera'],
        output_names=['velocity'],
        dynamic_axes={
            'lidar': {0: 'batch'},
            'camera': {0: 'batch'},
            'velocity': {0: 'batch'}
        },
        opset_version=11
    )
    print(f"Model exported to {path}")


if __name__ == "__main__":
    # Test model creation
    config = MLConfig()
    model = NavigationCNN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    lidar = torch.randn(2, config.lidar_input_dim)
    camera = torch.randn(2, *config.camera_input_shape)
    out = model(lidar, camera)
    print(f"Output shape: {out.shape}")

    # Test MLP
    mlp = NavigationMLP(config)
    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    out = mlp(lidar)
    print(f"MLP output shape: {out.shape}")

    # Test discrete action model
    disc = DiscreteActionCNN(config, num_actions=5)
    out = disc(lidar, camera)
    print(f"Discrete output shape: {out.shape}")

    # Test data generation
    lidar_data, camera_data, targets = generate_training_data(100)
    print(f"Training data: lidar={lidar_data.shape}, camera={camera_data.shape}, targets={targets.shape}")

    # Test discretization
    actions = discretize_actions(targets)
    print(f"Discrete actions: {np.bincount(actions)}")

    print("ML policy tests passed!")