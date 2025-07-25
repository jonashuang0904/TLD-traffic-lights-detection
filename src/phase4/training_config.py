#!/usr/bin/env python3
"""
Traffic Light Detection - Training Configuration Manager
Phase 4 Implementation: Advanced Training Configuration and Hyperparameter Management
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import torch

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_size: str = "yolo11n"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    pretrained: bool = True
    freeze_backbone: bool = False
    freeze_layers: int = 0  # Number of layers to freeze from backbone

@dataclass 
class DataConfig:
    """Data configuration parameters"""
    imgsz: int = 640              # Input image size
    batch_size: int = 16          # Batch size for training
    workers: int = 8              # Number of data loading workers
    cache: bool = False           # Cache images for faster training
    rect: bool = False            # Rectangular training
    single_cls: bool = False      # Train as single-class dataset

@dataclass
class TrainingConfig:
    """Core training parameters"""
    epochs: int = 300             # Maximum training epochs
    patience: int = 50            # Early stopping patience
    save_period: int = 10         # Save checkpoint every N epochs
    val: bool = True              # Validate during training
    plots: bool = True            # Generate training plots
    save: bool = True             # Save checkpoints
    verbose: bool = True          # Verbose output
    seed: int = 42                # Random seed for reproducibility
    deterministic: bool = True    # Use deterministic algorithms
    amp: bool = True              # Automatic Mixed Precision training
    half: bool = False            # Use FP16 half-precision training

@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer: str = "SGD"        # Optimizer type (SGD, Adam, AdamW, etc.)
    lr0: float = 0.01            # Initial learning rate
    lrf: float = 0.01            # Final learning rate fraction
    momentum: float = 0.937       # SGD momentum/Adam beta1
    weight_decay: float = 0.0005  # Optimizer weight decay 5e-4
    warmup_epochs: float = 3.0    # Warmup epochs
    warmup_momentum: float = 0.8  # Warmup initial momentum
    warmup_bias_lr: float = 0.1   # Warmup initial bias lr
    cos_lr: bool = False          # Use cosine learning rate scheduler

@dataclass
class AugmentationConfig:
    """Data augmentation parameters optimized for traffic lights"""
    hsv_h: float = 0.015         # HSV-Hue augmentation (±h_gain)
    hsv_s: float = 0.7           # HSV-Saturation augmentation (±s_gain)
    hsv_v: float = 0.4           # HSV-Value augmentation (±v_gain)
    degrees: float = 10.0        # Image rotation (±degrees)
    translate: float = 0.1       # Image translation (±translation)
    scale: float = 0.5           # Image scale (+/- gain)
    shear: float = 0.0           # Image shear (±shear) - disabled for small objects
    perspective: float = 0.0     # Image perspective (±perspective) - disabled
    flipud: float = 0.0          # Image flip up-down (probability) - disabled
    fliplr: float = 0.5          # Image flip left-right (probability)
    mosaic: float = 1.0          # Image mosaic (probability)
    mixup: float = 0.1           # Image mixup (probability)
    copy_paste: float = 0.0      # Segment copy-paste (probability) - disabled
    close_mosaic: int = 10       # Disable mosaic augmentation for final N epochs

@dataclass
class LossConfig:
    """Loss function configuration"""
    box: float = 7.5             # Box loss gain
    cls: float = 0.5             # Class loss gain  
    dfl: float = 1.5             # Distribution focal loss gain
    fl_gamma: float = 0.0        # Focal loss gamma (efficientDet default gamma=1.5)
    label_smoothing: float = 0.0 # Label smoothing epsilon

@dataclass
class TrainingEnvironmentConfig:
    """Training environment configuration"""
    device: str = "auto"         # Training device (auto, cpu, cuda, cuda:0, etc.)
    project: str = "experiments" # Project directory
    name: str = "traffic_light_detection"  # Experiment name
    exist_ok: bool = True        # Whether existing project/name ok
    resume: bool = False         # Resume training from last checkpoint
    nosave: bool = False         # Only save final checkpoint
    noval: bool = False          # Only validate final epoch
    overlap_mask: bool = True    # Use overlap mask for training
    mask_ratio: int = 4          # Mask downsample ratio
    dropout: float = 0.0         # Use dropout regularization

class TrafficLightTrainingConfig:
    """
    Comprehensive training configuration manager for traffic light detection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training configuration
        
        Args:
            config_path: Path to custom configuration file
        """
        # Initialize default configurations
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.optimizer = OptimizerConfig()
        self.augmentation = AugmentationConfig()
        self.loss = LossConfig()
        self.environment = TrainingEnvironmentConfig()
        
        # Device setup
        self._setup_device()
        
        # Load custom configuration if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def _setup_device(self):
        """Setup optimal training device"""
        if self.environment.device == "auto":
            if torch.cuda.is_available():
                # Use the best available GPU
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    # Use first GPU if multiple available
                    self.environment.device = "cuda:0"
                else:
                    self.environment.device = "cuda"
            else:
                self.environment.device = "cpu"
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get complete training configuration as dictionary
        
        Returns:
            Dictionary containing all training parameters
        """
        config = {}
        
        # Merge all configuration sections
        config.update(asdict(self.model))
        config.update(asdict(self.data))
        config.update(asdict(self.training))
        config.update(asdict(self.optimizer))
        config.update(asdict(self.augmentation))
        config.update(asdict(self.loss))
        config.update(asdict(self.environment))
        
        # Handle special mappings
        config['imgsz'] = self.data.imgsz
        config['batch'] = self.data.batch_size
        config['device'] = self.environment.device
        
        return config
    
    def get_optimized_config_for_model(self, model_size: str) -> Dict[str, Any]:
        """
        Get optimized configuration for specific model size
        
        Args:
            model_size: YOLO model size (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
            
        Returns:
            Optimized configuration dictionary
        """
        # Update model size
        self.model.model_size = model_size
        
        # Adjust batch size and learning rate based on model complexity
        model_adjustments = {
            'yolo11n': {
                'batch_size': 32,    # Smaller model, can use larger batches
                'lr0': 0.01,         # Standard learning rate
                'epochs': 300,       # Standard epochs
                'warmup_epochs': 3.0
            },
            'yolo11s': {
                'batch_size': 24,    # Slightly smaller batch
                'lr0': 0.01,         # Standard learning rate
                'epochs': 300,       # Standard epochs  
                'warmup_epochs': 3.0
            },
            'yolo11m': {
                'batch_size': 16,    # Medium batch size
                'lr0': 0.008,        # Slightly lower LR
                'epochs': 400,       # More epochs for convergence
                'warmup_epochs': 5.0
            },
            'yolo11l': {
                'batch_size': 12,    # Smaller batch for larger model
                'lr0': 0.006,        # Lower learning rate
                'epochs': 500,       # More epochs
                'warmup_epochs': 8.0
            },
            'yolo11x': {
                'batch_size': 8,     # Smallest batch size
                'lr0': 0.005,        # Lowest learning rate
                'epochs': 600,       # Most epochs
                'warmup_epochs': 10.0
            }
        }
        
        if model_size in model_adjustments:
            adjustments = model_adjustments[model_size]
            self.data.batch_size = adjustments['batch_size']
            self.optimizer.lr0 = adjustments['lr0']
            self.training.epochs = adjustments['epochs']
            self.optimizer.warmup_epochs = adjustments['warmup_epochs']
        
        return self.get_training_config()
    
    def get_traffic_light_optimized_config(self) -> Dict[str, Any]:
        """
        Get configuration specifically optimized for traffic light detection
        
        Returns:
            Traffic light optimized configuration
        """
        # Optimize for small object detection
        self.augmentation.degrees = 10.0      # Conservative rotation
        self.augmentation.scale = 0.3         # Conservative scaling
        self.augmentation.shear = 0.0         # No shearing
        self.augmentation.perspective = 0.0   # No perspective changes
        self.augmentation.flipud = 0.0        # No vertical flipping
        self.augmentation.hsv_h = 0.015       # Conservative hue changes
        
        # Optimize loss weights for detection
        self.loss.box = 7.5                   # Higher box loss weight
        self.loss.cls = 0.5                   # Standard class loss weight
        self.loss.dfl = 1.5                   # Distribution focal loss
        
        # Training optimizations
        self.training.patience = 50           # Early stopping patience
        
        return self.get_training_config()
    
    def save_config(self, path: Union[str, Path]):
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration file
        """
        config = self.get_training_config()
        path = Path(path)
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            # Default to JSON
            with open(path.with_suffix('.json'), 'w') as f:
                json.dump(config, f, indent=2, default=str)
    
    def load_config(self, path: Union[str, Path]):
        """
        Load configuration from file
        
        Args:
            path: Path to configuration file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load based on file extension
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        # Update configurations
        self._update_from_dict(config)
    
    def _update_from_dict(self, config: Dict[str, Any]):
        """Update configuration from dictionary"""
        
        # Update model config
        model_keys = set(asdict(self.model).keys())
        for key, value in config.items():
            if key in model_keys:
                setattr(self.model, key, value)
        
        # Update data config
        data_keys = set(asdict(self.data).keys())
        for key, value in config.items():
            if key in data_keys:
                setattr(self.data, key, value)
            elif key == 'batch':  # Handle batch size alias
                setattr(self.data, 'batch_size', value)
        
        # Update training config
        training_keys = set(asdict(self.training).keys())
        for key, value in config.items():
            if key in training_keys:
                setattr(self.training, key, value)
        
        # Update optimizer config
        optimizer_keys = set(asdict(self.optimizer).keys())
        for key, value in config.items():
            if key in optimizer_keys:
                setattr(self.optimizer, key, value)
        
        # Update augmentation config
        aug_keys = set(asdict(self.augmentation).keys())
        for key, value in config.items():
            if key in aug_keys:
                setattr(self.augmentation, key, value)
        
        # Update loss config
        loss_keys = set(asdict(self.loss).keys())
        for key, value in config.items():
            if key in loss_keys:
                setattr(self.loss, key, value)
        
        # Update environment config
        env_keys = set(asdict(self.environment).keys())
        for key, value in config.items():
            if key in env_keys:
                setattr(self.environment, key, value)
    
    def create_hyperparameter_search_configs(self, 
                                           num_configs: int = 5) -> List[Dict[str, Any]]:
        """
        Create multiple configurations for hyperparameter search
        
        Args:
            num_configs: Number of configurations to generate
            
        Returns:
            List of configuration dictionaries
        """
        import random
        
        configs = []
        base_config = self.get_traffic_light_optimized_config()
        
        # Define search ranges
        search_ranges = {
            'lr0': [0.005, 0.008, 0.01, 0.015, 0.02],
            'momentum': [0.9, 0.937, 0.95],
            'weight_decay': [0.0001, 0.0005, 0.001],
            'box': [5.0, 7.5, 10.0],
            'cls': [0.3, 0.5, 0.7],
            'dfl': [1.0, 1.5, 2.0],
            'hsv_s': [0.5, 0.7, 0.9],
            'hsv_v': [0.3, 0.4, 0.5],
            'mosaic': [0.8, 1.0],
            'mixup': [0.0, 0.1, 0.2]
        }
        
        for i in range(num_configs):
            config = base_config.copy()
            config['name'] = f"{config['name']}_search_{i+1}"
            
            # Randomly sample hyperparameters
            for param, values in search_ranges.items():
                config[param] = random.choice(values)
            
            configs.append(config)
        
        return configs
    
    def print_config_summary(self):
        """Print a formatted summary of the current configuration"""
        
        print("=" * 60)
        print("TRAFFIC LIGHT DETECTION - TRAINING CONFIGURATION")
        print("=" * 60)
        
        print(f"\n📱 MODEL CONFIGURATION:")
        print(f"  Model Size: {self.model.model_size}")
        print(f"  Pretrained: {self.model.pretrained}")
        print(f"  Freeze Backbone: {self.model.freeze_backbone}")
        
        print(f"\n📊 DATA CONFIGURATION:")
        print(f"  Image Size: {self.data.imgsz}x{self.data.imgsz}")
        print(f"  Batch Size: {self.data.batch_size}")
        print(f"  Workers: {self.data.workers}")
        print(f"  Cache Images: {self.data.cache}")
        
        print(f"\n🏋️ TRAINING CONFIGURATION:")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  Patience: {self.training.patience}")
        print(f"  Device: {self.environment.device}")
        print(f"  Mixed Precision: {self.training.amp}")
        
        print(f"\n⚡ OPTIMIZER CONFIGURATION:")
        print(f"  Optimizer: {self.optimizer.optimizer}")
        print(f"  Learning Rate: {self.optimizer.lr0}")
        print(f"  Momentum: {self.optimizer.momentum}")
        print(f"  Weight Decay: {self.optimizer.weight_decay}")
        
        print(f"\n🎭 AUGMENTATION CONFIGURATION:")
        print(f"  HSV (H/S/V): {self.augmentation.hsv_h}/{self.augmentation.hsv_s}/{self.augmentation.hsv_v}")
        print(f"  Rotation: ±{self.augmentation.degrees}°")
        print(f"  Scale: ±{self.augmentation.scale}")
        print(f"  Mosaic: {self.augmentation.mosaic}")
        print(f"  Mixup: {self.augmentation.mixup}")
        
        print(f"\n💥 LOSS CONFIGURATION:")
        print(f"  Box Loss: {self.loss.box}")
        print(f"  Class Loss: {self.loss.cls}")
        print(f"  DFL Loss: {self.loss.dfl}")
        
        print("=" * 60)

def create_default_configs() -> Dict[str, TrafficLightTrainingConfig]:
    """Create default configurations for different use cases"""
    
    configs = {}
    
    # Fast training configuration (for testing)
    fast_config = TrafficLightTrainingConfig()
    fast_config.model.model_size = "yolo11n"
    fast_config.training.epochs = 50
    fast_config.data.batch_size = 32
    fast_config.environment.name = "fast_training"
    configs['fast'] = fast_config
    
    # Balanced configuration (good performance vs speed)
    balanced_config = TrafficLightTrainingConfig()
    balanced_config.model.model_size = "yolo11s"
    balanced_config.training.epochs = 200
    balanced_config.data.batch_size = 24
    balanced_config.environment.name = "balanced_training"
    configs['balanced'] = balanced_config
    
    # High accuracy configuration
    accuracy_config = TrafficLightTrainingConfig()
    accuracy_config.model.model_size = "yolo11m"
    accuracy_config.training.epochs = 400
    accuracy_config.data.batch_size = 16
    accuracy_config.optimizer.lr0 = 0.008
    accuracy_config.environment.name = "high_accuracy_training"
    configs['accuracy'] = accuracy_config
    
    # Production configuration
    production_config = TrafficLightTrainingConfig()
    production_config.model.model_size = "yolo11l"
    production_config.training.epochs = 500
    production_config.data.batch_size = 12
    production_config.optimizer.lr0 = 0.006
    production_config.training.patience = 100
    production_config.environment.name = "production_training"
    configs['production'] = production_config
    
    return configs

if __name__ == "__main__":
    # Example usage
    print("Creating traffic light detection training configurations...")
    
    # Create default configuration
    config = TrafficLightTrainingConfig()
    config.print_config_summary()
    
    # Get optimized configuration for different models
    print("\n\nOptimized configurations:")
    for model_size in ['yolo11n', 'yolo11s', 'yolo11m']:
        print(f"\n{model_size.upper()} Configuration:")
        opt_config = config.get_optimized_config_for_model(model_size)
        print(f"  Batch Size: {opt_config['batch']}")
        print(f"  Learning Rate: {opt_config['lr0']}")
        print(f"  Epochs: {opt_config['epochs']}")
    
    # Save configuration
    config.save_config("training_configs/default_config.yaml")
    print(f"\nConfiguration saved to: training_configs/default_config.yaml")
    
    # Create preset configurations
    preset_configs = create_default_configs()
    for name, preset in preset_configs.items():
        preset.save_config(f"training_configs/{name}_config.yaml")
        print(f"Saved {name} configuration")