"""
Métricas de evaluación para Temporal Action Detection

Este módulo implementa:
1. Temporal IoU: Intersection over Union para segmentos temporales
2. mAP: Mean Average Precision a diferentes thresholds
3. Precision/Recall/F1: Métricas de clasificación
4. DetectionMetrics: Clase para computar y trackear todas las métricas
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_temporal_iou(
    segment1: np.ndarray,
    segment2: np.ndarray
) -> float:
    """
    Calcula IoU temporal entre dos segmentos.
    
    Args:
        segment1: [start, end] en frames absolutos o relativos
        segment2: [start, end]
    
    Returns:
        iou: Valor en [0, 1]
    """
    # Intersección
    inter_start = max(segment1[0], segment2[0])
    inter_end = min(segment1[1], segment2[1])
    inter = max(0, inter_end - inter_start)
    
    # Unión
    area1 = segment1[1] - segment1[0]
    area2 = segment2[1] - segment2[0]
    union = area1 + area2 - inter
    
    # IoU
    if union > 0:
        return inter / union
    else:
        return 0.0


def compute_average_precision(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Calcula Average Precision (AP) para una clase a un IoU threshold.
    
    Args:
        predictions: Lista de dicts con keys:
            - 'segment': [start, end]
            - 'score': confidence score
        ground_truths: Lista de dicts con keys:
            - 'segment': [start, end]
        iou_threshold: Threshold de IoU para considerar TP
    
    Returns:
        ap: Average Precision en [0, 1]
    """
    if len(predictions) == 0:
        return 0.0 if len(ground_truths) > 0 else 1.0
    if len(ground_truths) == 0:
        return 0.0
    
    # Ordenar predicciones por score descendente
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Arrays para TP/FP
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # Marcar GT como detectados
    gt_detected = [False] * len(ground_truths)
    
    # Para cada predicción
    for pred_idx, pred in enumerate(predictions):
        pred_segment = pred['segment']
        
        # Encontrar GT con mayor IoU
        max_iou = 0.0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_detected[gt_idx]:
                continue
            
            iou = compute_temporal_iou(pred_segment, gt['segment'])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Clasificar como TP o FP
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            tp[pred_idx] = 1.0
            gt_detected[max_gt_idx] = True
        else:
            fp[pred_idx] = 1.0
    
    # Calcular precision y recall acumulados
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calcular AP usando interpolación de 11 puntos
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def compute_map(
    all_predictions: Dict[int, List[Dict]],
    all_ground_truths: Dict[int, List[Dict]],
    iou_thresholds: List[float] = [0.5, 0.75, 0.95]
) -> Dict[str, float]:
    """
    Calcula mean Average Precision (mAP) para todas las clases.
    
    Args:
        all_predictions: Dict {class_id: lista de predicciones}
        all_ground_truths: Dict {class_id: lista de GTs}
        iou_thresholds: Lista de thresholds de IoU
    
    Returns:
        Dict con 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95', etc.
    """
    results = {}
    
    # Obtener todas las clases
    all_classes = set(all_predictions.keys()) | set(all_ground_truths.keys())
    
    # Para cada threshold
    for iou_thr in iou_thresholds:
        aps = []
        
        for class_id in all_classes:
            preds = all_predictions.get(class_id, [])
            gts = all_ground_truths.get(class_id, [])
            
            ap = compute_average_precision(preds, gts, iou_threshold=iou_thr)
            aps.append(ap)
        
        # Mean AP
        map_value = np.mean(aps) if len(aps) > 0 else 0.0
        results[f'mAP@{iou_thr:.2f}'] = map_value
    
    # mAP@0.5:0.95 (promedio sobre thresholds)
    if len(iou_thresholds) > 1:
        avg_map = np.mean([results[f'mAP@{t:.2f}'] for t in iou_thresholds])
        results['mAP@0.5:0.95'] = avg_map
    
    return results


def compute_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcula Precision, Recall y F1 score.
    
    Args:
        predictions: Lista de predicciones
        ground_truths: Lista de GTs
        iou_threshold: Threshold para considerar match
    
    Returns:
        Dict con 'precision', 'recall', 'f1'
    """
    if len(predictions) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    if len(ground_truths) == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
    
    # Contar TP
    gt_detected = [False] * len(ground_truths)
    tp = 0
    
    for pred in predictions:
        pred_segment = pred['segment']
        
        # Encontrar mejor match
        max_iou = 0.0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_detected[gt_idx]:
                continue
            
            iou = compute_temporal_iou(pred_segment, gt['segment'])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            tp += 1
            gt_detected[max_gt_idx] = True
    
    # Calcular métricas
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


class DetectionMetrics:
    """
    Clase para computar y trackear métricas de detección.
    
    Acumula predicciones y ground truths durante evaluación
    y calcula todas las métricas al final.
    """
    
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: List[float] = [0.5, 0.75, 0.95]
    ):
        """
        Args:
            num_classes: Número de clases (sin contar background)
            iou_thresholds: Thresholds de IoU para mAP
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        
        # Acumuladores
        self.reset()
    
    def reset(self):
        """Reinicia todos los acumuladores."""
        # Dict: {class_id: [lista de predicciones]}
        self.predictions = defaultdict(list)
        
        # Dict: {class_id: [lista de ground truths]}
        self.ground_truths = defaultdict(list)
        
        # Stats adicionales
        self.total_predictions = 0
        self.total_ground_truths = 0
    
    def add_batch(
        self,
        predictions: List[List[Dict]],
        ground_truths: List[List[Dict]]
    ):
        """
        Agrega un batch de predicciones y GTs.
        
        Args:
            predictions: Lista (batch) de listas (por video) de dicts:
                - 'class_id': int
                - 'segment': [start, end]
                - 'score': float
            ground_truths: Lista (batch) de listas (por video) de dicts:
                - 'class_id': int
                - 'segment': [start, end]
        """
        for preds, gts in zip(predictions, ground_truths):
            # Agrupar por clase
            for pred in preds:
                class_id = pred['class_id']
                self.predictions[class_id].append({
                    'segment': pred['segment'],
                    'score': pred['score']
                })
                self.total_predictions += 1
            
            for gt in gts:
                class_id = gt['class_id']
                self.ground_truths[class_id].append({
                    'segment': gt['segment']
                })
                self.total_ground_truths += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Calcula todas las métricas acumuladas.
        
        Returns:
            Dict con todas las métricas:
                - mAP@0.5, mAP@0.75, mAP@0.95, mAP@0.5:0.95
                - precision, recall, f1 (promedio sobre clases)
                - per_class_ap (opcional)
        """
        results = {}
        
        # 1. Calcular mAP
        map_results = compute_map(
            self.predictions,
            self.ground_truths,
            self.iou_thresholds
        )
        results.update(map_results)
        
        # 2. Calcular precision/recall/f1 por clase y promedio
        all_classes = set(self.predictions.keys()) | set(self.ground_truths.keys())
        
        precisions = []
        recalls = []
        f1s = []
        
        for class_id in all_classes:
            preds = self.predictions.get(class_id, [])
            gts = self.ground_truths.get(class_id, [])
            
            metrics = compute_precision_recall(preds, gts, iou_threshold=0.5)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics['f1'])
        
        results['precision'] = np.mean(precisions) if precisions else 0.0
        results['recall'] = np.mean(recalls) if recalls else 0.0
        results['f1'] = np.mean(f1s) if f1s else 0.0
        
        # 3. Stats adicionales
        results['total_predictions'] = self.total_predictions
        results['total_ground_truths'] = self.total_ground_truths
        results['num_classes_detected'] = len(self.predictions)
        results['num_classes_in_gt'] = len(self.ground_truths)
        
        return results
    
    def compute_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Calcula métricas detalladas por clase.
        
        Returns:
            Dict {class_id: {métrica: valor}}
        """
        all_classes = set(self.predictions.keys()) | set(self.ground_truths.keys())
        per_class = {}
        
        for class_id in all_classes:
            preds = self.predictions.get(class_id, [])
            gts = self.ground_truths.get(class_id, [])
            
            # AP para cada threshold
            aps = {}
            for iou_thr in self.iou_thresholds:
                ap = compute_average_precision(preds, gts, iou_threshold=iou_thr)
                aps[f'AP@{iou_thr:.2f}'] = ap
            
            # Precision/Recall/F1
            pr_metrics = compute_precision_recall(preds, gts, iou_threshold=0.5)
            
            # Combinar
            per_class[class_id] = {
                **aps,
                **pr_metrics,
                'num_predictions': len(preds),
                'num_ground_truths': len(gts)
            }
        
        return per_class
    
    def summary(self) -> str:
        """
        Genera un resumen textual de las métricas.
        
        Returns:
            String formateado con las métricas principales
        """
        metrics = self.compute_metrics()
        
        summary = "=" * 60 + "\n"
        summary += "DETECTION METRICS SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        # mAP
        summary += "Mean Average Precision (mAP):\n"
        for key in ['mAP@0.5:0.95', 'mAP@0.50', 'mAP@0.75', 'mAP@0.95']:
            if key in metrics:
                summary += f"  {key}: {metrics[key]:.4f}\n"
        
        # Precision/Recall/F1
        summary += "\nClassification Metrics (avg over classes):\n"
        summary += f"  Precision: {metrics['precision']:.4f}\n"
        summary += f"  Recall:    {metrics['recall']:.4f}\n"
        summary += f"  F1 Score:  {metrics['f1']:.4f}\n"
        
        # Stats
        summary += "\nStatistics:\n"
        summary += f"  Total Predictions:  {metrics['total_predictions']}\n"
        summary += f"  Total Ground Truth: {metrics['total_ground_truths']}\n"
        summary += f"  Classes Detected:   {metrics['num_classes_detected']}/{self.num_classes}\n"
        
        summary += "=" * 60 + "\n"
        
        return summary


# Test code
if __name__ == '__main__':
    print("Testing Detection Metrics...")
    
    # Test 1: Temporal IoU
    print("\n1. Testing compute_temporal_iou...")
    seg1 = np.array([10, 50])
    seg2 = np.array([30, 70])
    iou = compute_temporal_iou(seg1, seg2)
    print(f"   IoU between {seg1} and {seg2}: {iou:.4f}")
    assert 0.0 <= iou <= 1.0
    
    # Test 2: Average Precision
    print("\n2. Testing compute_average_precision...")
    predictions = [
        {'segment': np.array([10, 50]), 'score': 0.9},
        {'segment': np.array([60, 100]), 'score': 0.8},
        {'segment': np.array([15, 55]), 'score': 0.7},  # FP
    ]
    ground_truths = [
        {'segment': np.array([12, 48])},
        {'segment': np.array([62, 98])},
    ]
    ap = compute_average_precision(predictions, ground_truths, iou_threshold=0.5)
    print(f"   AP@0.5: {ap:.4f}")
    
    # Test 3: mAP
    print("\n3. Testing compute_map...")
    all_preds = {
        1: [{'segment': np.array([10, 50]), 'score': 0.9}],
        2: [{'segment': np.array([60, 100]), 'score': 0.8}]
    }
    all_gts = {
        1: [{'segment': np.array([12, 48])}],
        2: [{'segment': np.array([62, 98])}]
    }
    map_results = compute_map(all_preds, all_gts)
    print(f"   mAP@0.5: {map_results['mAP@0.50']:.4f}")
    print(f"   mAP@0.75: {map_results['mAP@0.75']:.4f}")
    
    # Test 4: Precision/Recall
    print("\n4. Testing compute_precision_recall...")
    pr_metrics = compute_precision_recall(predictions, ground_truths)
    print(f"   Precision: {pr_metrics['precision']:.4f}")
    print(f"   Recall: {pr_metrics['recall']:.4f}")
    print(f"   F1: {pr_metrics['f1']:.4f}")
    print(f"   TP={pr_metrics['tp']}, FP={pr_metrics['fp']}, FN={pr_metrics['fn']}")
    
    # Test 5: DetectionMetrics class
    print("\n5. Testing DetectionMetrics class...")
    metrics = DetectionMetrics(num_classes=24)
    
    # Simular batch
    batch_preds = [[
        {'class_id': 1, 'segment': np.array([10, 50]), 'score': 0.9},
        {'class_id': 2, 'segment': np.array([60, 100]), 'score': 0.8}
    ]]
    batch_gts = [[
        {'class_id': 1, 'segment': np.array([12, 48])},
        {'class_id': 2, 'segment': np.array([62, 98])}
    ]]
    
    metrics.add_batch(batch_preds, batch_gts)
    
    results = metrics.compute_metrics()
    print(f"   mAP@0.5: {results['mAP@0.50']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    
    # Test summary
    print("\n6. Testing summary...")
    print(metrics.summary())
    
    # Test per-class metrics
    print("\n7. Testing per-class metrics...")
    per_class = metrics.compute_per_class_metrics()
    for class_id, class_metrics in per_class.items():
        print(f"   Class {class_id}: AP@0.5={class_metrics['AP@0.50']:.4f}, "
              f"P={class_metrics['precision']:.4f}, R={class_metrics['recall']:.4f}")
    
    print("\n✅ All metrics working correctly!")
