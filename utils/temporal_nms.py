"""
Temporal Non-Maximum Suppression (NMS) utilities

Implementa:
1. temporal_nms: NMS estándar para detecciones temporales
2. soft_nms: Variante que reduce scores en vez de eliminar
3. compute_iou_batch: IoU eficiente para batches
"""

import numpy as np
import torch
from typing import List, Dict, Union


def compute_temporal_iou(segment1: np.ndarray, segment2: np.ndarray) -> float:
    """
    Calcula IoU entre dos segmentos temporales.
    
    Args:
        segment1: [start, end]
        segment2: [start, end]
    
    Returns:
        iou: float en [0, 1]
    """
    inter_start = max(segment1[0], segment2[0])
    inter_end = min(segment1[1], segment2[1])
    inter = max(0, inter_end - inter_start)
    
    area1 = segment1[1] - segment1[0]
    area2 = segment2[1] - segment2[0]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def compute_iou_batch(
    segments1: Union[np.ndarray, torch.Tensor],
    segments2: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calcula IoU entre todos los pares de segmentos de forma eficiente.
    
    Args:
        segments1: (N, 2) array/tensor
        segments2: (M, 2) array/tensor
    
    Returns:
        iou: (N, M) array/tensor
    """
    if isinstance(segments1, torch.Tensor):
        return _compute_iou_batch_torch(segments1, segments2)
    else:
        return _compute_iou_batch_numpy(segments1, segments2)


def _compute_iou_batch_numpy(segments1: np.ndarray, segments2: np.ndarray) -> np.ndarray:
    """Versión NumPy de compute_iou_batch."""
    N, M = len(segments1), len(segments2)
    
    # Expandir para broadcasting
    s1 = segments1[:, np.newaxis, :]  # (N, 1, 2)
    s2 = segments2[np.newaxis, :, :]  # (1, M, 2)
    
    # Intersección
    inter_start = np.maximum(s1[:, :, 0], s2[:, :, 0])
    inter_end = np.minimum(s1[:, :, 1], s2[:, :, 1])
    inter = np.maximum(0, inter_end - inter_start)
    
    # Áreas
    area1 = s1[:, :, 1] - s1[:, :, 0]
    area2 = s2[:, :, 1] - s2[:, :, 0]
    union = area1 + area2 - inter
    
    # IoU
    iou = inter / (union + 1e-10)
    
    return iou


def _compute_iou_batch_torch(segments1: torch.Tensor, segments2: torch.Tensor) -> torch.Tensor:
    """Versión PyTorch de compute_iou_batch."""
    N, M = segments1.size(0), segments2.size(0)
    
    # Expandir
    s1 = segments1.unsqueeze(1).expand(N, M, 2)
    s2 = segments2.unsqueeze(0).expand(N, M, 2)
    
    # Intersección
    inter_start = torch.maximum(s1[:, :, 0], s2[:, :, 0])
    inter_end = torch.minimum(s1[:, :, 1], s2[:, :, 1])
    inter = torch.clamp(inter_end - inter_start, min=0)
    
    # Áreas
    area1 = s1[:, :, 1] - s1[:, :, 0]
    area2 = s2[:, :, 1] - s2[:, :, 0]
    union = area1 + area2 - inter
    
    # IoU
    iou = inter / (union + 1e-10)
    
    return iou


def temporal_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> List[Dict]:
    """
    Temporal Non-Maximum Suppression.
    
    Elimina detecciones duplicadas basándose en IoU temporal.
    
    Args:
        detections: Lista de dicts con keys:
            - 'segment': [start, end]
            - 'score': confidence
            - 'class_id': ID de clase
        iou_threshold: Threshold de IoU para considerar duplicado
        score_threshold: Score mínimo para considerar detección
    
    Returns:
        Lista de detecciones filtradas
    """
    if len(detections) == 0:
        return []
    
    # Filtrar por score
    detections = [d for d in detections if d['score'] >= score_threshold]
    
    if len(detections) == 0:
        return []
    
    # Separar por clase
    per_class_detections = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in per_class_detections:
            per_class_detections[class_id] = []
        per_class_detections[class_id].append(det)
    
    # NMS por clase
    final_detections = []
    
    for class_id, class_dets in per_class_detections.items():
        # Ordenar por score descendente
        class_dets = sorted(class_dets, key=lambda x: x['score'], reverse=True)
        
        # Array de segmentos
        segments = np.array([d['segment'] for d in class_dets])
        
        # NMS
        keep = []
        suppressed = np.zeros(len(class_dets), dtype=bool)
        
        for i in range(len(class_dets)):
            if suppressed[i]:
                continue
            
            keep.append(i)
            
            # Calcular IoU con detecciones restantes
            seg_i = segments[i:i+1]
            ious = compute_iou_batch(seg_i, segments[i+1:])
            
            # Suprimir detecciones con IoU alto
            suppress_indices = np.where(ious[0] > iou_threshold)[0] + i + 1
            suppressed[suppress_indices] = True
        
        # Agregar detecciones mantenidas
        for idx in keep:
            final_detections.append(class_dets[idx])
    
    # Ordenar por score
    final_detections = sorted(final_detections, key=lambda x: x['score'], reverse=True)
    
    return final_detections


def soft_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    sigma: float = 0.5,
    method: str = 'gaussian'
) -> List[Dict]:
    """
    Soft Non-Maximum Suppression.
    
    En vez de eliminar detecciones, reduce sus scores basándose en IoU.
    Útil cuando hay detecciones válidas que se solapan.
    
    Args:
        detections: Lista de detecciones
        iou_threshold: Threshold de IoU
        score_threshold: Score mínimo final
        sigma: Parámetro para decay gaussiano
        method: 'linear' o 'gaussian'
    
    Returns:
        Lista de detecciones con scores ajustados
    """
    if len(detections) == 0:
        return []
    
    # Copiar para no modificar original
    detections = [d.copy() for d in detections]
    
    # Separar por clase
    per_class_detections = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in per_class_detections:
            per_class_detections[class_id] = []
        per_class_detections[class_id].append(det)
    
    # Soft-NMS por clase
    final_detections = []
    
    for class_id, class_dets in per_class_detections.items():
        # Ordenar por score
        class_dets = sorted(class_dets, key=lambda x: x['score'], reverse=True)
        segments = np.array([d['segment'] for d in class_dets])
        
        for i in range(len(class_dets)):
            if class_dets[i]['score'] < score_threshold:
                continue
            
            # Calcular IoU con detecciones posteriores
            seg_i = segments[i:i+1]
            ious = compute_iou_batch(seg_i, segments[i+1:])
            
            # Ajustar scores
            for j, iou in enumerate(ious[0]):
                idx = i + 1 + j
                
                if iou > iou_threshold:
                    if method == 'linear':
                        # Decay lineal
                        class_dets[idx]['score'] *= (1 - iou)
                    else:  # gaussian
                        # Decay gaussiano
                        class_dets[idx]['score'] *= np.exp(-(iou ** 2) / sigma)
        
        # Filtrar por threshold final
        class_dets = [d for d in class_dets if d['score'] >= score_threshold]
        final_detections.extend(class_dets)
    
    # Ordenar por score
    final_detections = sorted(final_detections, key=lambda x: x['score'], reverse=True)
    
    return final_detections


# Test code
if __name__ == '__main__':
    print("Testing Temporal NMS...")
    
    # Test detections (con overlaps)
    detections = [
        {'segment': np.array([10, 50]), 'score': 0.9, 'class_id': 1},
        {'segment': np.array([15, 55]), 'score': 0.7, 'class_id': 1},  # Overlap con anterior
        {'segment': np.array([60, 100]), 'score': 0.8, 'class_id': 1},
        {'segment': np.array([65, 95]), 'score': 0.6, 'class_id': 1},  # Overlap con anterior
        {'segment': np.array([20, 40]), 'score': 0.85, 'class_id': 2},  # Otra clase
    ]
    
    print(f"\nDetecciones originales: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  {i}: class={det['class_id']}, "
              f"segment={det['segment']}, score={det['score']:.2f}")
    
    # Test NMS estándar
    print("\n1. Testing temporal_nms...")
    nms_dets = temporal_nms(detections, iou_threshold=0.5)
    print(f"   Después de NMS: {len(nms_dets)} detecciones")
    for i, det in enumerate(nms_dets):
        print(f"   {i}: class={det['class_id']}, "
              f"segment={det['segment']}, score={det['score']:.2f}")
    
    # Test Soft-NMS
    print("\n2. Testing soft_nms...")
    soft_dets = soft_nms(detections, iou_threshold=0.5, method='gaussian')
    print(f"   Después de Soft-NMS: {len(soft_dets)} detecciones")
    for i, det in enumerate(soft_dets):
        print(f"   {i}: class={det['class_id']}, "
              f"segment={det['segment']}, score={det['score']:.2f}")
    
    # Test IoU batch
    print("\n3. Testing compute_iou_batch...")
    segments1 = np.array([[10, 50], [60, 100]])
    segments2 = np.array([[15, 55], [65, 95], [120, 160]])
    ious = compute_iou_batch(segments1, segments2)
    print(f"   IoU matrix shape: {ious.shape}")
    print(f"   IoU values:\n{ious}")
    
    print("\n✅ All temporal NMS functions working!")
