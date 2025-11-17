"""
Script principal de entrenamiento para TAD

Usage:
    # Entrenar desde cero
    python scripts/train.py --config config.yaml

    # Reanudar desde checkpoint
    python scripts/train.py --config config.yaml --resume checkpoints/epoch_10.pth
    
    # Entrenamiento con configuración personalizada
    python scripts/train.py --config config.yaml --epochs 150 --batch_size 8 --lr 2e-4
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import logging

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from models.complete_model import build_model
from data.dataset import TemporalActionDataset, collate_fn
from data.augmentation import get_train_transforms, get_val_transforms
from training.losses import TADLoss
from training.trainer import TADTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Carga configuración desde YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, args) -> tuple:
    """
    Crea dataloaders de entrenamiento y validación.
    
    Returns:
        (train_loader, val_loader)
    """
    logger.info("Creando datasets...")
    
    # Configuración de datos
    data_config = config['data']
    
    # Transforms
    train_transforms = get_train_transforms(
        target_size=tuple(data_config['target_size']),
        augmentation_config=config.get('augmentation', {})
    )
    val_transforms = get_val_transforms(
        target_size=tuple(data_config['target_size'])
    )
    
    # Datasets
    train_dataset = TemporalActionDataset(
        annotations_file=data_config['train_annotations'],
        videos_root=data_config['video_dir'],
        clip_length=data_config['clip_length'],
        sampling_rate=data_config.get('sampling_rate', 1),
        mode='train',
        transform=train_transforms
    )
    
    val_dataset = TemporalActionDataset(
        annotations_file=data_config['val_annotations'],
        videos_root=data_config['video_dir'],
        clip_length=data_config['clip_length'],
        sampling_rate=data_config.get('sampling_rate', 1),
        mode='val',
        transform=val_transforms
    )
    
    logger.info(f"  Train dataset: {len(train_dataset)} clips")
    logger.info(f"  Val dataset: {len(val_dataset)} clips")
    
    # DataLoaders
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def create_model(config: dict, device: str):
    """Crea el modelo."""
    logger.info("Creando modelo...")
    
    model = build_model(config)
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Total parámetros: {total_params / 1e6:.2f}M")
    logger.info(f"  Parámetros entrenables: {trainable_params / 1e6:.2f}M")
    
    return model


def create_optimizer(model: nn.Module, config: dict, args):
    """Crea el optimizador."""
    training_config = config['training']
    
    lr = args.lr if args.lr else training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.01)
    
    # Separar parámetros del backbone y detection heads
    # Aplicar menor LR al backbone si está pre-entrenado
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    # Optimizer groups con diferentes LR
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},  # 10x menor para backbone
        {'params': other_params, 'lr': lr}
    ]
    
    optimizer_type = training_config.get('optimizer', 'adamw')
    
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=training_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_type}")
    
    logger.info(f"  Optimizer: {optimizer_type.upper()}")
    logger.info(f"  Learning rate (detection heads): {lr}")
    logger.info(f"  Learning rate (backbone): {lr * 0.1}")
    logger.info(f"  Weight decay: {weight_decay}")
    
    return optimizer


def create_scheduler(optimizer, config: dict, args):
    """Crea el learning rate scheduler."""
    training_config = config['training']
    scheduler_type = training_config.get('scheduler', 'cosine')
    
    num_epochs = args.epochs if args.epochs else training_config['epochs']
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=training_config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.get('lr_decay_step', 30),
            gamma=training_config.get('lr_decay_gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        milestones = training_config.get('milestones', [60, 80])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=training_config.get('lr_decay_gamma', 0.1)
        )
    else:
        logger.warning(f"Scheduler '{scheduler_type}' no reconocido, usando None")
        scheduler = None
    
    if scheduler:
        logger.info(f"  Scheduler: {scheduler_type}")
    
    return scheduler


def create_loss_function(config: dict):
    """Crea la función de pérdida."""
    model_config = config['model']
    loss_config = config.get('loss_weights', {})
    
    loss_fn = TADLoss(
        num_classes=model_config['num_classes'],
        lambda_cls=loss_config.get('classification', 1.0),
        lambda_boundary=loss_config.get('boundary', 1.0),
        lambda_iou=loss_config.get('iou', 0.5),
        lambda_reg=loss_config.get('regression', 0.3),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0)
    )
    
    logger.info("  Loss: TADLoss")
    logger.info(f"    λ_cls={loss_config.get('classification', 1.0)}")
    logger.info(f"    λ_boundary={loss_config.get('boundary', 1.0)}")
    logger.info(f"    λ_iou={loss_config.get('iou', 0.5)}")
    logger.info(f"    λ_reg={loss_config.get('regression', 0.3)}")
    
    return loss_fn


def main(args):
    """Función principal."""
    logger.info("=" * 60)
    logger.info("TAD Training Script")
    logger.info("=" * 60)
    
    # Cargar configuración
    config = load_config(args.config)
    logger.info(f"Config cargada: {args.config}")
    
    # Device
    if args.device:
        device = args.device
    elif config.get('device') and config['device'] != 'auto':
        device = config['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Device: {device}")
    
    if device == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
    
    # Crear componentes
    train_loader, val_loader = create_dataloaders(config, args)
    model = create_model(config, device)
    optimizer = create_optimizer(model, config, args)
    scheduler = create_scheduler(optimizer, config, args)
    loss_fn = create_loss_function(config)
    
    # Directorios
    checkpoint_dir = args.checkpoint_dir or 'checkpoints'
    log_dir = args.log_dir or 'logs'
    
    # Crear trainer
    training_config = config['training']
    
    trainer = TADTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_classes=config['model']['num_classes'],
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        use_amp=training_config.get('use_amp', True),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1)
    )
    
    # Cargar checkpoint si se especifica
    if args.resume:
        logger.info(f"Resumiendo desde: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Entrenar
    num_epochs = args.epochs if args.epochs else training_config['epochs']
    early_stopping = training_config.get('early_stopping_patience', 10)
    
    logger.info("=" * 60)
    logger.info("Iniciando entrenamiento...")
    logger.info("=" * 60)
    
    trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping,
        save_every=args.save_every,
        eval_every=args.eval_every
    )
    
    logger.info("=" * 60)
    logger.info("✅ Entrenamiento completado!")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo TAD')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Ruta al checkpoint para reanudar entrenamiento'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu). Auto-detect si no se especifica'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Número de épocas (override config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (override config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (override config)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directorio para checkpoints'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directorio para logs'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=5,
        help='Guardar checkpoint cada N épocas'
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        default=1,
        help='Evaluar cada N épocas'
    )
    
    args = parser.parse_args()
    main(args)
