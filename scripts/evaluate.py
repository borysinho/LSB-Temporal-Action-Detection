"""
Script de evaluación para modelo TAD entrenado

Usage:
    python scripts/evaluate.py \
        --config config.yaml \
        --checkpoint checkpoints/best.pth \
        --split test
"""

import argparse
import yaml
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))

from models.complete_model import build_model, load_pretrained
from data.dataset import TemporalActionDataset, collate_fn
from data.augmentation import get_val_transforms
from training.metrics import DetectionMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    dataloader,
    device,
    num_classes,
    save_predictions=False,
    output_file=None
):
    """
    Evalúa el modelo en un dataset.
    
    Returns:
        Dict con métricas
    """
    model.eval()
    metrics = DetectionMetrics(num_classes=num_classes)
    
    all_predictions = []
    all_ground_truths = []
    
    logger.info("Evaluando modelo...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)
            targets = batch['targets']
            
            # Inference
            outputs = model(frames)
            
            # Convertir a formato de detecciones
            detections = outputs.get('detections', [])
            
            # Ground truths
            gts = []
            for b in range(frames.size(0)):
                labels = targets['labels'][b]
                segments = targets['segments'][b]
                num_actions = targets.get('num_actions', [len(labels)])[b]
                
                video_gts = []
                for i in range(num_actions):
                    if labels[i] > 0:
                        video_gts.append({
                            'class_id': labels[i].item(),
                            'segment': segments[i].cpu().numpy()
                        })
                gts.append(video_gts)
            
            all_predictions.extend(detections)
            all_ground_truths.extend(gts)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Procesado {batch_idx + 1}/{len(dataloader)} batches")
    
    # Calcular métricas
    metrics.add_batch(all_predictions, all_ground_truths)
    results = metrics.compute_metrics()
    
    # Guardar predicciones si se solicita
    if save_predictions and output_file:
        predictions_data = {
            'predictions': all_predictions,
            'ground_truths': all_ground_truths
        }
        
        with open(output_file, 'w') as f:
            # Convertir numpy arrays a listas para JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(predictions_data), f, indent=2)
        
        logger.info(f"Predicciones guardadas en: {output_file}")
    
    return results, metrics


def main(args):
    logger.info("=" * 60)
    logger.info("TAD Evaluation Script")
    logger.info("=" * 60)
    
    # Cargar config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Cargar modelo
    logger.info(f"Cargando modelo desde: {args.checkpoint}")
    model = build_model(config)
    model = load_pretrained(model, args.checkpoint, device=device)
    model = model.to(device)
    model.eval()
    
    # Crear dataset
    data_config = config['data']
    
    if args.split == 'val':
        annotation_file = data_config['val_annotations']
    elif args.split == 'test':
        annotation_file = data_config['test_annotations']
    else:
        raise ValueError(f"Split inválido: {args.split}")
    
    logger.info(f"Evaluando en split: {args.split}")
    
    transforms = get_val_transforms(tuple(data_config['target_size']))
    
    dataset = TemporalActionDataset(
        video_dir=data_config['video_dir'],
        annotation_file=annotation_file,
        clip_length=data_config['clip_length'],
        sampling_rate=data_config.get('sampling_rate', 1),
        split=args.split,
        transform=transforms,
        overlap=0.0
    )
    
    logger.info(f"Dataset: {len(dataset)} clips")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Evaluar
    results, metrics_obj = evaluate_model(
        model,
        dataloader,
        device,
        num_classes=config['model']['num_classes'],
        save_predictions=args.save_predictions,
        output_file=args.output_file
    )
    
    # Mostrar resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS DE EVALUACIÓN")
    logger.info("=" * 60)
    
    logger.info(f"\nmAP:")
    for key in ['mAP@0.5:0.95', 'mAP@0.50', 'mAP@0.75', 'mAP@0.95']:
        if key in results:
            logger.info(f"  {key}: {results[key]:.4f}")
    
    logger.info(f"\nMétricas de clasificación:")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1 Score:  {results['f1']:.4f}")
    
    logger.info(f"\nEstadísticas:")
    logger.info(f"  Total predicciones:  {results['total_predictions']}")
    logger.info(f"  Total ground truths: {results['total_ground_truths']}")
    logger.info(f"  Clases detectadas:   {results['num_classes_detected']}")
    
    # Métricas por clase si se solicita
    if args.per_class:
        logger.info("\n" + "=" * 60)
        logger.info("MÉTRICAS POR CLASE")
        logger.info("=" * 60)
        
        per_class_metrics = metrics_obj.compute_per_class_metrics()
        
        # Cargar mapeo de clases
        class_mapping_file = Path(data_config.get('class_mapping', 'data/annotations/class_mapping.json'))
        if class_mapping_file.exists():
            with open(class_mapping_file, 'r') as f:
                class_mapping = json.load(f)
            # Invertir mapeo
            id_to_name = {v: k for k, v in class_mapping.items()}
        else:
            id_to_name = {}
        
        for class_id, class_metrics in sorted(per_class_metrics.items()):
            class_name = id_to_name.get(class_id, f"Class_{class_id}")
            logger.info(f"\n{class_name} (ID: {class_id}):")
            logger.info(f"  AP@0.5:  {class_metrics.get('AP@0.50', 0):.4f}")
            logger.info(f"  Precision: {class_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {class_metrics['recall']:.4f}")
            logger.info(f"  F1:        {class_metrics['f1']:.4f}")
            logger.info(f"  Predicciones: {class_metrics['num_predictions']}")
            logger.info(f"  Ground truth: {class_metrics['num_ground_truths']}")
    
    # Guardar resultados
    if args.output_file:
        output_path = Path(args.output_file).parent / f"metrics_{args.split}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nMétricas guardadas en: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Evaluación completada!")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar modelo TAD')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Archivo de configuración'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint del modelo'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Split a evaluar'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Guardar predicciones'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='evaluation_results/predictions.json',
        help='Archivo de salida para predicciones'
    )
    parser.add_argument(
        '--per_class',
        action='store_true',
        help='Mostrar métricas por clase'
    )
    
    args = parser.parse_args()
    main(args)
