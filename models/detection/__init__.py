"""
Detection components para Temporal Action Detection
"""

from .proposal_network import TemporalProposalNetwork
from .boundary_detector import BoundaryDetector
from .action_classifier import ActionClassifier

__all__ = [
    'TemporalProposalNetwork',
    'BoundaryDetector',
    'ActionClassifier'
]
