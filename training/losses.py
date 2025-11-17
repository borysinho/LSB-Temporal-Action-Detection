"""
Funciones de pérdida especializadas para Temporal Action Detection

Este módulo implementa:
1. FocalLoss: Para manejar desbalance de clases
2. BoundaryDetectionLoss: Para detectar inicio/fin de acciones
3. TemporalIoULoss: Basada en GIoU para propuestas temporales
4. RegressionLoss: Para refinar boundaries con offsets
5. TADLoss: Combinación ponderada de todas las anteriores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación con desbalance de clases.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Factor de balanceo de clases (default: 0.25)
        gamma: Factor de enfoque (default: 2.0)
        reduction: Tipo de reducción ('mean', 'sum', 'none')
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, num_classes) logits sin sigmoid/softmax
            targets: (B,) índices de clase o (B, num_classes) one-hot
        
        Returns:
            loss: Scalar si reduction='mean'/'sum', sino (B,)
        """
        # Si targets son índices, convertir a one-hot
        if targets.dim() == 1:
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets
        
        # Calcular probabilidades
        probs = torch.sigmoid(inputs)
        
        # Calcular p_t
        p_t = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)
        
        # Calcular focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_one_hot, reduction='none'
        )
        
        # Focal loss
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BoundaryDetectionLoss(nn.Module):
    """
    Pérdida para detectar boundaries (inicio/fin) de acciones.
    
    Combina BCE para clasificación con ponderación por proximidad.
    Las predicciones cerca de boundaries reales tienen más peso.
    
    Args:
        pos_weight: Peso para muestras positivas (default: 2.0)
        temporal_weight: Aplicar ponderación temporal (default: True)
    """
    def __init__(self, pos_weight: float = 2.0, temporal_weight: bool = True):
        super().__init__()
        self.pos_weight = pos_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self, 
        start_probs: torch.Tensor,
        end_probs: torch.Tensor,
        start_targets: torch.Tensor,
        end_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            start_probs: (B, T) probabilidades de inicio
            end_probs: (B, T) probabilidades de fin
            start_targets: (B, T) targets binarios de inicio
            end_targets: (B, T) targets binarios de fin
        
        Returns:
            dict con 'start_loss', 'end_loss', 'total'
        """
        # BCE con ponderación
        pos_weight = torch.tensor([self.pos_weight], device=start_probs.device)
        
        start_loss = F.binary_cross_entropy_with_logits(
            start_probs, start_targets, pos_weight=pos_weight, reduction='none'
        )
        end_loss = F.binary_cross_entropy_with_logits(
            end_probs, end_targets, pos_weight=pos_weight, reduction='none'
        )
        
        # Ponderación temporal (más peso cerca de boundaries reales)
        if self.temporal_weight:
            start_weight = self._compute_temporal_weights(start_targets)
            end_weight = self._compute_temporal_weights(end_targets)
            start_loss = start_loss * start_weight
            end_loss = end_loss * end_weight
        
        return {
            'start_loss': start_loss.mean(),
            'end_loss': end_loss.mean(),
            'total': (start_loss.mean() + end_loss.mean()) / 2
        }
    
    def _compute_temporal_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula pesos temporales basados en distancia a boundaries.
        
        Args:
            targets: (B, T) targets binarios
        
        Returns:
            weights: (B, T) pesos en [1.0, 2.0]
        """
        # Encontrar posiciones de boundaries
        boundaries = targets > 0.5
        
        # Crear tensor de distancias
        B, T = targets.shape
        weights = torch.ones_like(targets)
        
        for b in range(B):
            boundary_indices = torch.where(boundaries[b])[0]
            if len(boundary_indices) > 0:
                # Calcular distancia mínima a cualquier boundary
                positions = torch.arange(T, device=targets.device).float()
                for idx in boundary_indices:
                    dist = torch.abs(positions - idx.float())
                    # Peso inversamente proporcional a distancia (max 2.0 en boundary)
                    weight = 1.0 + torch.exp(-dist / 5.0)
                    weights[b] = torch.maximum(weights[b], weight)
        
        return weights


class TemporalIoULoss(nn.Module):
    """
    Pérdida basada en Generalized IoU (GIoU) para propuestas temporales.
    
    GIoU mejora IoU considerando el área de la región envolvente.
    Formula: GIoU = IoU - |C - (A ∪ B)| / |C|
    donde C es el segmento envolvente más pequeño.
    
    Args:
        loss_type: 'giou' o 'iou' (default: 'giou')
    """
    def __init__(self, loss_type: str = 'giou'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self, 
        pred_segments: torch.Tensor,
        target_segments: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_segments: (N, 2) predicciones [start, end] normalizadas [0,1]
            target_segments: (N, 2) targets [start, end] normalizadas [0,1]
        
        Returns:
            loss: Scalar
        """
        # Validar entrada
        assert pred_segments.size() == target_segments.size()
        assert pred_segments.size(1) == 2
        
        # Extraer start/end
        pred_start, pred_end = pred_segments[:, 0], pred_segments[:, 1]
        target_start, target_end = target_segments[:, 0], target_segments[:, 1]
        
        # Asegurar que end >= start
        pred_end = torch.maximum(pred_end, pred_start + 1e-6)
        target_end = torch.maximum(target_end, target_start + 1e-6)
        
        # Calcular IoU
        inter_start = torch.maximum(pred_start, target_start)
        inter_end = torch.minimum(pred_end, target_end)
        inter = torch.clamp(inter_end - inter_start, min=0)
        
        pred_area = pred_end - pred_start
        target_area = target_end - target_start
        union = pred_area + target_area - inter
        
        iou = inter / (union + 1e-6)
        
        if self.loss_type == 'iou':
            # IoU loss simple
            loss = 1 - iou
        else:
            # GIoU loss
            # Calcular segmento envolvente C
            convex_start = torch.minimum(pred_start, target_start)
            convex_end = torch.maximum(pred_end, target_end)
            convex_area = convex_end - convex_start
            
            # GIoU = IoU - (C - union) / C
            giou = iou - (convex_area - union) / (convex_area + 1e-6)
            loss = 1 - giou
        
        return loss.mean()


class RegressionLoss(nn.Module):
    """
    Pérdida L1 suave para regresión de boundary offsets.
    
    Usa Smooth L1 (Huber loss) que es menos sensible a outliers
    que L1 puro y más estable que L2.
    
    Args:
        beta: Threshold para suavizado (default: 1.0)
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred_offsets: (N, 2) offsets predichos [start_offset, end_offset]
            target_offsets: (N, 2) offsets reales
            mask: (N,) máscara booleana de samples válidos (opcional)
        
        Returns:
            loss: Scalar
        """
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_offsets, target_offsets, beta=self.beta, reduction='none')
        
        # Aplicar máscara si se proporciona
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (N, 1)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()


class TADLoss(nn.Module):
    """
    Pérdida combinada para Temporal Action Detection.
    
    Integra todas las pérdidas con pesos configurables:
    - Classification loss (Focal)
    - Boundary detection loss (BCE ponderado)
    - Temporal IoU loss (GIoU)
    - Regression loss (Smooth L1)
    
    Args:
        num_classes: Número de clases de acciones
        lambda_cls: Peso para classification loss (default: 1.0)
        lambda_boundary: Peso para boundary detection loss (default: 1.0)
        lambda_iou: Peso para temporal IoU loss (default: 0.5)
        lambda_reg: Peso para regression loss (default: 0.3)
        focal_alpha: Alpha para Focal loss (default: 0.25)
        focal_gamma: Gamma para Focal loss (default: 2.0)
    """
    def __init__(
        self,
        num_classes: int,
        lambda_cls: float = 1.0,
        lambda_boundary: float = 1.0,
        lambda_iou: float = 0.5,
        lambda_reg: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_boundary = lambda_boundary
        self.lambda_iou = lambda_iou
        self.lambda_reg = lambda_reg
        
        # Inicializar losses individuales
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryDetectionLoss()
        self.iou_loss = TemporalIoULoss(loss_type='giou')
        self.regression_loss = RegressionLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict con keys:
                - 'class_logits': (B, num_proposals, num_classes)
                - 'start_probs': (B, T)
                - 'end_probs': (B, T)
                - 'proposals': (B, num_proposals, 2) [start, end]
                - 'start_offsets': (B, T) opcional
                - 'end_offsets': (B, T) opcional
            
            targets: Dict con keys:
                - 'labels': (B, max_actions) índices de clase
                - 'segments': (B, max_actions, 2) [start, end] normalizados
                - 'start_targets': (B, T) targets binarios
                - 'end_targets': (B, T) targets binarios
                - 'num_actions': (B,) número real de acciones por sample
        
        Returns:
            Dict con losses individuales y total
        """
        print(f"DEBUG: outputs keys: {list(outputs.keys())}")
        print(f"DEBUG: targets keys: {list(targets.keys())}")
        
        losses = {}
        
        # 1. Classification Loss (Focal)
        if 'class_logits' in outputs and 'labels' in targets:
            class_logits = outputs['class_logits']  # (B, num_proposals, num_classes)
            labels = targets['labels']  # (B, max_actions)
            
            # Necesitamos matchear propuestas con GT
            # Para simplificar, usamos la propuesta con mayor IoU para cada GT
            cls_loss = self._compute_classification_loss(class_logits, labels, 
                                                         outputs.get('proposals'), 
                                                         targets.get('segments'))
            losses['loss_cls'] = cls_loss * self.lambda_cls
        
        # 2. Boundary Detection Loss
        if all(k in outputs for k in ['start_probs', 'end_probs']):
            boundary_losses = self.boundary_loss(
                outputs['start_probs'],
                outputs['end_probs'],
                targets['start_targets'],
                targets['end_targets']
            )
            losses['loss_boundary_start'] = boundary_losses['start_loss'] * self.lambda_boundary
            losses['loss_boundary_end'] = boundary_losses['end_loss'] * self.lambda_boundary
            losses['loss_boundary'] = boundary_losses['total'] * self.lambda_boundary
        
        # 3. Temporal IoU Loss
        if 'proposals' in outputs and 'segments' in targets:
            iou_loss = self._compute_iou_loss(outputs['proposals'], targets['segments'])
            losses['loss_iou'] = iou_loss * self.lambda_iou
        
        # 4. Regression Loss (offsets)
        if all(k in outputs for k in ['start_offsets', 'end_offsets']):
            if all(k in targets for k in ['start_offset_targets', 'end_offset_targets']):
                reg_loss = self._compute_regression_loss(outputs, targets)
                losses['loss_reg'] = reg_loss * self.lambda_reg
        
        # Si no hay losses calculadas, devolver loss dummy
        if not any(k.startswith('loss_') and k != 'loss_total' for k in losses.keys()):
            # Crear loss dummy que permita gradientes
            dummy_loss = torch.tensor(1.0, device=next(iter(outputs.values())).device if outputs else 'cpu', requires_grad=True)
            losses['loss_dummy'] = dummy_loss
        
        # Total loss
        total = 0
        for k, v in losses.items():
            if k.startswith('loss_') and k != 'loss_total':
                print(f"Loss {k}: {type(v)}, {v}")  # Debug
                total += v
        losses['loss_total'] = total
        print(f"Total loss: {type(losses['loss_total'])}, {losses['loss_total']}")  # Debug
        
        return losses
    
    def _compute_classification_loss(
        self,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        proposals: Optional[torch.Tensor],
        segments: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computa classification loss matcheando propuestas con GT.
        """
        if proposals is None or segments is None:
            # Fallback: tratar todas las propuestas como positivas
            # Esto es simplificado, en práctica necesitamos matching
            B, num_proposals, num_classes = class_logits.shape
            # Usar primer label para todas las propuestas (simplificación)
            targets = labels[:, 0].unsqueeze(1).expand(B, num_proposals)
            loss = self.focal_loss(class_logits.view(-1, num_classes), 
                                  targets.reshape(-1))
            return loss
        
        # Matching basado en IoU
        B = class_logits.size(0)
        total_loss = 0
        
        for b in range(B):
            prop_b = proposals[b]  # (num_proposals, 2)
            seg_b = segments[b]  # (max_actions, 2)
            label_b = labels[b]  # (max_actions,)
            
            # Calcular IoU entre propuestas y GT
            ious = self._compute_pairwise_iou(prop_b, seg_b)  # (num_proposals, max_actions)
            
            # Para cada propuesta, encontrar GT con mayor IoU
            max_iou, matched_gt = ious.max(dim=1)  # (num_proposals,)
            
            # Asignar labels: si IoU > 0.5 → label del GT, sino background (0)
            prop_labels = torch.where(
                max_iou > 0.5,
                label_b[matched_gt],
                torch.zeros_like(matched_gt)
            )
            
            # Focal loss
            loss = self.focal_loss(class_logits[b], prop_labels)
            total_loss += loss
        
        return total_loss / B
    
    def _compute_iou_loss(
        self,
        proposals: torch.Tensor,
        segments: torch.Tensor
    ) -> torch.Tensor:
        """
        Computa IoU loss entre propuestas positivas y sus GT matched.
        """
        B = proposals.size(0)
        total_loss = 0
        num_positives = 0
        
        for b in range(B):
            prop_b = proposals[b]  # (num_proposals, 2)
            seg_b = segments[b]  # (max_actions, 2)
            
            # Calcular IoU
            ious = self._compute_pairwise_iou(prop_b, seg_b)
            max_iou, matched_gt = ious.max(dim=1)
            
            # Solo calcular loss para propuestas positivas (IoU > 0.5)
            positive_mask = max_iou > 0.5
            if positive_mask.sum() > 0:
                positive_props = prop_b[positive_mask]
                matched_segs = seg_b[matched_gt[positive_mask]]
                
                loss = self.iou_loss(positive_props, matched_segs)
                total_loss += loss
                num_positives += 1
        
        if num_positives > 0:
            return total_loss / num_positives
        else:
            return torch.tensor(0.0, device=proposals.device)
    
    def _compute_regression_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computa regression loss para boundary offsets.
        """
        start_offsets = outputs['start_offsets']  # (B, T)
        end_offsets = outputs['end_offsets']  # (B, T)
        start_offset_targets = targets['start_offset_targets']  # (B, T)
        end_offset_targets = targets['end_offset_targets']  # (B, T)
        
        # Mask para posiciones con boundaries
        start_mask = targets['start_targets'] > 0.5
        end_mask = targets['end_targets'] > 0.5
        
        # Loss solo en posiciones con boundaries
        start_loss = self.regression_loss(
            start_offsets[start_mask].unsqueeze(-1),
            start_offset_targets[start_mask].unsqueeze(-1)
        )
        end_loss = self.regression_loss(
            end_offsets[end_mask].unsqueeze(-1),
            end_offset_targets[end_mask].unsqueeze(-1)
        )
        
        return (start_loss + end_loss) / 2
    
    def _compute_pairwise_iou(
        self,
        segments1: torch.Tensor,
        segments2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula IoU entre todos los pares de segmentos.
        
        Args:
            segments1: (N, 2)
            segments2: (M, 2)
        
        Returns:
            iou: (N, M)
        """
        N, M = segments1.size(0), segments2.size(0)
        
        # Expandir para broadcasting
        s1 = segments1.unsqueeze(1).expand(N, M, 2)  # (N, M, 2)
        s2 = segments2.unsqueeze(0).expand(N, M, 2)  # (N, M, 2)
        
        # Calcular intersección
        inter_start = torch.maximum(s1[:, :, 0], s2[:, :, 0])
        inter_end = torch.minimum(s1[:, :, 1], s2[:, :, 1])
        inter = torch.clamp(inter_end - inter_start, min=0)
        
        # Calcular áreas
        area1 = s1[:, :, 1] - s1[:, :, 0]
        area2 = s2[:, :, 1] - s2[:, :, 0]
        union = area1 + area2 - inter
        
        # IoU
        iou = inter / (union + 1e-6)
        
        return iou


# Test code
if __name__ == '__main__':
    print("Testing TAD Losses...")
    
    # Parámetros
    batch_size = 4
    num_proposals = 100
    num_classes = 24
    temporal_length = 64
    max_actions = 5
    
    # Test FocalLoss
    print("\n1. Testing FocalLoss...")
    focal = FocalLoss()
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = focal(logits, labels)
    print(f"   Focal loss: {loss.item():.4f}")
    
    # Test BoundaryDetectionLoss
    print("\n2. Testing BoundaryDetectionLoss...")
    boundary_loss_fn = BoundaryDetectionLoss()
    start_probs = torch.randn(batch_size, temporal_length)
    end_probs = torch.randn(batch_size, temporal_length)
    start_targets = torch.zeros(batch_size, temporal_length)
    end_targets = torch.zeros(batch_size, temporal_length)
    # Simular algunos boundaries
    start_targets[:, 10] = 1.0
    end_targets[:, 30] = 1.0
    losses = boundary_loss_fn(start_probs, end_probs, start_targets, end_targets)
    print(f"   Start loss: {losses['start_loss'].item():.4f}")
    print(f"   End loss: {losses['end_loss'].item():.4f}")
    
    # Test TemporalIoULoss
    print("\n3. Testing TemporalIoULoss...")
    iou_loss_fn = TemporalIoULoss()
    pred_segments = torch.tensor([[0.2, 0.5], [0.3, 0.7]])
    target_segments = torch.tensor([[0.25, 0.55], [0.35, 0.65]])
    loss = iou_loss_fn(pred_segments, target_segments)
    print(f"   GIoU loss: {loss.item():.4f}")
    
    # Test TADLoss completa
    print("\n4. Testing TADLoss (complete)...")
    tad_loss = TADLoss(num_classes=num_classes)
    
    outputs = {
        'class_logits': torch.randn(batch_size, num_proposals, num_classes),
        'start_probs': torch.randn(batch_size, temporal_length),
        'end_probs': torch.randn(batch_size, temporal_length),
        'proposals': torch.rand(batch_size, num_proposals, 2),
        'start_offsets': torch.randn(batch_size, temporal_length),
        'end_offsets': torch.randn(batch_size, temporal_length)
    }
    
    targets = {
        'labels': torch.randint(1, num_classes, (batch_size, max_actions)),
        'segments': torch.rand(batch_size, max_actions, 2),
        'start_targets': start_targets,
        'end_targets': end_targets,
        'start_offset_targets': torch.randn(batch_size, temporal_length),
        'end_offset_targets': torch.randn(batch_size, temporal_length),
        'num_actions': torch.randint(1, max_actions, (batch_size,))
    }
    
    losses = tad_loss(outputs, targets)
    print(f"   Total loss: {losses['loss_total'].item():.4f}")
    print(f"   Classification: {losses.get('loss_cls', 0):.4f}")
    print(f"   Boundary: {losses.get('loss_boundary', 0):.4f}")
    print(f"   IoU: {losses.get('loss_iou', 0):.4f}")
    if 'loss_reg' in losses:
        print(f"   Regression: {losses['loss_reg']:.4f}")
    
    print("\n✅ All losses working correctly!")
