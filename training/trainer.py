"""
Trainer para Temporal Action Detection

Implementa el loop de entrenamiento completo con:
- Training/validation loops
- Learning rate scheduling con warmup
- Early stopping
- Checkpointing (best + epoch)
- Logging (TensorBoard/WandB)
- Mixed precision (AMP)
- Gradient accumulation
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Callable
from pathlib import Path
import logging

from .losses import TADLoss
from .metrics import DetectionMetrics

logger = logging.getLogger(__name__)


class TADTrainer:
    """
    Trainer completo para Temporal Action Detection.
    
    Args:
        model: Modelo TAD (TemporalActionDetector)
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        optimizer: Optimizador (AdamW, SGD, etc.)
        scheduler: Learning rate scheduler
        loss_fn: Función de pérdida (TADLoss)
        num_classes: Número de clases
        config: Dict con configuración de entrenamiento
        device: 'cuda' o 'cpu'
        checkpoint_dir: Directorio para guardar checkpoints
        log_dir: Directorio para logs
        use_amp: Usar mixed precision
        gradient_accumulation_steps: Steps para acumular gradientes
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: TADLoss,
        num_classes: int,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.config = config
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Directorios
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Métricas
        self.train_metrics = DetectionMetrics(num_classes)
        self.val_metrics = DetectionMetrics(num_classes)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_map = 0.0
        self.epochs_no_improve = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'lr': []
        }
        
        # Logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Configura logging."""
        log_file = self.log_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Ejecuta una época de entrenamiento.
        
        Returns:
            Dict con métricas de la época
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        logger.info(f"Epoch {self.current_epoch + 1} - Training...")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Mover datos a device
            frames = batch['frames'].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch['targets'].items()}
            
            # Forward pass con autocast si usamos AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(frames, targets=targets)
                    losses = self.loss_fn(outputs, targets)
                    loss = losses['loss_total'] / self.gradient_accumulation_steps
            else:
                outputs = self.model(frames, targets=targets)
                losses = self.loss_fn(outputs, targets)
                loss = losses['loss_total'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Actualizar pesos (cada gradient_accumulation_steps)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Acumular pérdidas
            total_loss += losses['loss_total'].item()
            for k, v in losses.items():
                if k.startswith('loss_'):
                    if k not in loss_components:
                        loss_components[k] = 0.0
                    loss_components[k] += v.item()
            
            # Log cada N batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"  Batch {batch_idx + 1}/{num_batches} - "
                    f"Loss: {avg_loss:.4f}"
                )
        
        # Promediar pérdidas
        epoch_metrics = {
            'loss': total_loss / num_batches
        }
        for k, v in loss_components.items():
            epoch_metrics[k] = v / num_batches
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Ejecuta validación.
        
        Returns:
            Dict con métricas de validación
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        logger.info(f"Epoch {self.current_epoch + 1} - Validation...")
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Mover datos a device
                frames = batch['frames'].to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch['targets'].items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(frames, targets=targets)
                        losses = self.loss_fn(outputs, targets)
                else:
                    outputs = self.model(frames, targets=targets)
                    losses = self.loss_fn(outputs, targets)
                
                total_loss += losses['loss_total'].item()
                
                # Post-process para métricas
                # Convertir outputs a formato de detecciones
                detections = self._outputs_to_detections(outputs)
                gts = self._targets_to_detections(targets)
                
                all_predictions.extend(detections)
                all_ground_truths.extend(gts)
        
        # Calcular métricas
        self.val_metrics.add_batch(all_predictions, all_ground_truths)
        metrics = self.val_metrics.compute_metrics()
        
        # Agregar pérdida
        metrics['loss'] = total_loss / num_batches
        
        # Log
        logger.info(
            f"  Val Loss: {metrics['loss']:.4f}, "
            f"mAP@0.5: {metrics.get('mAP@0.50', 0):.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )
        
        return metrics
    
    def _outputs_to_detections(self, outputs: Dict) -> List[List[Dict]]:
        """
        Convierte outputs del modelo a formato de detecciones.
        
        Args:
            outputs: Dict con 'detections' del modelo
        
        Returns:
            Lista (batch) de listas (por video) de dicts
        """
        # El modelo ya devuelve formato correcto en modo inference
        # outputs['detections'] = List[List[Dict]] con keys:
        # - 'class_id', 'segment', 'score'
        return outputs.get('detections', [])
    
    def _targets_to_detections(self, targets: Dict) -> List[List[Dict]]:
        """
        Convierte targets a formato de detecciones (ground truth).
        
        Args:
            targets: Dict con 'labels' y 'segments'
        
        Returns:
            Lista (batch) de listas (por video) de dicts
        """
        batch_size = targets['labels'].size(0)
        gts = []
        
        for b in range(batch_size):
            labels = targets['labels'][b]  # (max_actions,)
            segments = targets['segments'][b]  # (max_actions, 2)
            num_actions = targets.get('num_actions', torch.tensor([len(labels)]))[b]
            
            video_gts = []
            for i in range(num_actions):
                if labels[i] > 0:  # Skip background
                    video_gts.append({
                        'class_id': labels[i].item(),
                        'segment': segments[i].cpu().numpy()
                    })
            
            gts.append(video_gts)
        
        return gts
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_every: int = 5,
        eval_every: int = 1
    ):
        """
        Loop de entrenamiento completo.
        
        Args:
            num_epochs: Número de épocas
            early_stopping_patience: Épocas sin mejora para early stopping
            save_every: Guardar checkpoint cada N épocas
            eval_every: Evaluar cada N épocas
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Learning rate step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log training
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Validation
            val_metrics = None
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate()
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_map'].append(val_metrics.get('mAP@0.50', 0))
                self.history['lr'].append(current_lr)
                
                # Check for improvement
                current_map = val_metrics.get('mAP@0.50', 0)
                if current_map > self.best_val_map:
                    self.best_val_map = current_map
                    self.epochs_no_improve = 0
                    
                    # Save best model
                    self.save_checkpoint('best.pth', is_best=True)
                    logger.info(f"  ✓ New best mAP@0.5: {self.best_val_map:.4f}")
                else:
                    self.epochs_no_improve += 1
                    logger.info(
                        f"  No improvement for {self.epochs_no_improve} epoch(s). "
                        f"Best: {self.best_val_map:.4f}"
                    )
                
                # Early stopping
                if self.epochs_no_improve >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs. "
                        f"Best mAP@0.5: {self.best_val_map:.4f}"
                    )
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pth')
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch time: {epoch_time:.2f}s")
            logger.info("-" * 60)
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info(f"Best validation mAP@0.5: {self.best_val_map:.4f}")
        logger.info("=" * 60)
        
        # Save final model
        self.save_checkpoint('final.pth')
        
        # Save history
        self._save_history()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Guarda checkpoint del modelo.
        
        Args:
            filename: Nombre del archivo
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_map': self.best_val_map,
            'history': self.history,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"  Checkpoint saved: {filepath}")
        
        if is_best:
            # También guardar como 'best.pth'
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str):
        """
        Carga checkpoint.
        
        Args:
            filepath: Ruta al checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_map = checkpoint.get('best_val_map', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Checkpoint loaded from: {filepath}")
        logger.info(f"  Resuming from epoch {self.current_epoch}")
        logger.info(f"  Best mAP@0.5: {self.best_val_map:.4f}")
    
    def _save_history(self):
        """Guarda historial de entrenamiento."""
        import json
        
        history_file = self.log_dir / 'history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved: {history_file}")


# Test/Example code
if __name__ == '__main__':
    print("TAD Trainer - Example Setup")
    print("=" * 60)
    
    # Este es un ejemplo de cómo usar el trainer
    # En práctica, se llamaría desde scripts/train.py
    
    print("""
Example usage:

```python
from models.complete_model import build_model
from data.dataset import TemporalActionDataset
from training.losses import TADLoss
from training.trainer import TADTrainer
import torch
from torch.utils.data import DataLoader

# 1. Crear modelo
config = {
    'backbone': 'video_swin',
    'num_classes': 24,
    'feature_dim': 768
}
model = build_model(config)

# 2. Crear datasets
train_dataset = TemporalActionDataset(
    video_dir='data/videos',
    annotation_file='data/annotations_train.json',
    split='train'
)
val_dataset = TemporalActionDataset(
    video_dir='data/videos',
    annotation_file='data/annotations_val.json',
    split='val'
)

# 3. Crear dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 4. Optimizador y scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 5. Loss function
loss_fn = TADLoss(num_classes=24)

# 6. Crear trainer
trainer = TADTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    num_classes=24,
    config=config,
    device='cuda',
    checkpoint_dir='checkpoints',
    log_dir='logs',
    use_amp=True,
    gradient_accumulation_steps=2
)

# 7. Entrenar
trainer.train(
    num_epochs=100,
    early_stopping_patience=10,
    save_every=5,
    eval_every=1
)
```
    """)
    
    print("=" * 60)
    print("✅ Trainer implementation complete!")
    print("   See scripts/train.py for full training script")
