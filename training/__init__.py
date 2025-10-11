"""
Training module para Temporal Action Detection
"""

from .losses import (
    FocalLoss,
    BoundaryDetectionLoss,
    TemporalIoULoss,
    RegressionLoss,
    TADLoss
)

from .metrics import (
    compute_temporal_iou,
    compute_map,
    compute_precision_recall,
    DetectionMetrics
)

from .trainer import TADTrainer

__all__ = [
    'FocalLoss',
    'BoundaryDetectionLoss',
    'TemporalIoULoss',
    'RegressionLoss',
    'TADLoss',
    'compute_temporal_iou',
    'compute_map',
    'compute_precision_recall',
    'DetectionMetrics',
    'TADTrainer'
]
