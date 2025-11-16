"""
Backbone models para feature extraction
"""

# Lazy imports - solo importar cuando se necesite
def get_backbone(name, **kwargs):
    """
    Factory function para obtener backbone

    Args:
        name: Nombre del backbone ('video_swin', 'timesformer', 'slowfast')
        **kwargs: Argumentos adicionales

    Returns:
        Backbone model
    """
    if name == 'video_swin':
        from .video_swin import VideoSwinTransformer
        return VideoSwinTransformer(**kwargs)
    elif name == 'timesformer':
        from .timesformer import TimeSformer
        return TimeSformer(**kwargs)
    elif name == 'slowfast':
        from .slowfast import SlowFast
        return SlowFast(**kwargs)
    else:
        raise ValueError(f"Backbone '{name}' no reconocido. Opciones: ['video_swin', 'timesformer', 'slowfast']")

__all__ = [
    'get_backbone'
]
