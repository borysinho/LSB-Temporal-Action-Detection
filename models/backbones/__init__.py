"""
Backbone models para feature extraction
"""

from .video_swin import VideoSwinTransformer
from .timesformer import TimeSformer
from .slowfast import SlowFast

def get_backbone(name, **kwargs):
    """
    Factory function para obtener backbone
    
    Args:
        name: Nombre del backbone ('video_swin', 'timesformer', 'slowfast')
        **kwargs: Argumentos adicionales
    
    Returns:
        Backbone model
    """
    backbones = {
        'video_swin': VideoSwinTransformer,
        'timesformer': TimeSformer,
        'slowfast': SlowFast
    }
    
    if name not in backbones:
        raise ValueError(f"Backbone '{name}' no reconocido. Opciones: {list(backbones.keys())}")
    
    return backbones[name](**kwargs)

__all__ = [
    'VideoSwinTransformer',
    'TimeSformer',
    'SlowFast',
    'get_backbone'
]
