"""
Models module para Temporal Action Detection
"""

from .complete_model import TemporalActionDetector
from .backbones import get_backbone
from .detection import TemporalProposalNetwork, BoundaryDetector, ActionClassifier

__all__ = [
    'TemporalActionDetector',
    'get_backbone',
    'TemporalProposalNetwork',
    'BoundaryDetector',
    'ActionClassifier'
]
