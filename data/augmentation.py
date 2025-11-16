"""
Augmentations para videos de lenguaje de señas

Incluye augmentations espaciales y temporales
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, Tuple
import cv2


class SpatialAugmentation:
    """Augmentations espaciales para videos"""
    
    def __init__(
        self,
        crop_size: Tuple[int, int] = (224, 224),
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        rotation_degrees: int = 10
    ):
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.rotation_degrees = rotation_degrees
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Aplica augmentations espaciales
        
        Args:
            frames: (T, H, W, C)
        
        Returns:
            frames_aug: (T, H, W, C)
        """
        T, H, W, C = frames.shape
        
        # Random resized crop
        frames = self._random_resized_crop(frames)
        
        # Horizontal flip
        if random.random() < self.hflip_prob:
            frames = np.flip(frames, axis=2).copy()  # flip width
        
        # Color jitter
        if self.color_jitter:
            frames = self._color_jitter(frames)
        
        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            frames = self._rotate_frames(frames, angle)
        
        return frames
    
    def _random_resized_crop(self, frames: np.ndarray) -> np.ndarray:
        """Random resized crop como PyTorch"""
        T, H, W, C = frames.shape
        
        # Calcular tamaño de crop
        area = H * W
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        
        # Ajustar si se sale de los límites
        if w > W or h > H:
            w, h = W, H
        
        # Random position
        i = random.randint(0, H - h)
        j = random.randint(0, W - w)
        
        # Crop
        frames = frames[:, i:i+h, j:j+w, :]
        
        # Resize a crop_size
        frames_resized = []
        for t in range(T):
            frame_resized = cv2.resize(
                frames[t],
                self.crop_size,
                interpolation=cv2.INTER_LINEAR
            )
            frames_resized.append(frame_resized)
        
        return np.stack(frames_resized, axis=0)
    
    def _color_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Aplica color jitter"""
        # Brightness
        brightness_factor = random.uniform(0.8, 1.2)
        frames = np.clip(frames * brightness_factor, 0, 255)
        
        # Contrast
        contrast_factor = random.uniform(0.8, 1.2)
        mean = frames.mean(axis=(1, 2), keepdims=True)
        frames = np.clip((frames - mean) * contrast_factor + mean, 0, 255)
        
        # Saturation (en HSV)
        if random.random() < 0.5:
            sat_factor = random.uniform(0.8, 1.2)
            frames_hsv = []
            for t in range(len(frames)):
                frame_hsv = cv2.cvtColor(frames[t].astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                frame_hsv[:, :, 1] = np.clip(frame_hsv[:, :, 1] * sat_factor, 0, 255)
                frame_rgb = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                frames_hsv.append(frame_rgb)
            frames = np.stack(frames_hsv, axis=0).astype(np.float32)
        
        return frames
    
    def _rotate_frames(self, frames: np.ndarray, angle: float) -> np.ndarray:
        """Rota todos los frames"""
        T, H, W, C = frames.shape
        center = (W // 2, H // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        frames_rotated = []
        for t in range(T):
            frame_rot = cv2.warpAffine(
                frames[t],
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            frames_rotated.append(frame_rot)
        
        return np.stack(frames_rotated, axis=0)


class TemporalAugmentation:
    """Augmentations temporales"""
    
    def __init__(
        self,
        temporal_jitter: float = 0.1,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        frame_dropout_prob: float = 0.0
    ):
        self.temporal_jitter = temporal_jitter
        self.speed_range = speed_range
        self.frame_dropout_prob = frame_dropout_prob
    
    def __call__(self, frames: np.ndarray, annotations: Optional[dict] = None):
        """
        Aplica augmentations temporales
        
        Args:
            frames: (T, H, W, C)
            annotations: Dict con boundaries (opcional)
        
        Returns:
            frames_aug, annotations_aug
        """
        # Speed perturbation
        if random.random() < 0.5:
            speed_factor = random.uniform(*self.speed_range)
            frames, annotations = self._speed_perturbation(frames, annotations, speed_factor)
        
        # Frame dropout
        if self.frame_dropout_prob > 0:
            frames = self._frame_dropout(frames)
        
        # Temporal jitter (solo afecta annotations)
        if annotations is not None and self.temporal_jitter > 0:
            annotations = self._temporal_jitter(annotations, len(frames))
        
        return frames, annotations
    
    def _speed_perturbation(self, frames: np.ndarray, annotations: Optional[dict], factor: float):
        """Cambia la velocidad del video"""
        T, H, W, C = frames.shape
        new_T = int(T * factor)
        
        # Resample temporal
        indices = np.linspace(0, T - 1, new_T)
        frames_resampled = []
        
        for idx in indices:
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), T - 1)
            
            if idx_low == idx_high:
                frame = frames[idx_low]
            else:
                # Linear interpolation
                alpha = idx - idx_low
                frame = (1 - alpha) * frames[idx_low] + alpha * frames[idx_high]
            
            frames_resampled.append(frame)
        
        frames_new = np.stack(frames_resampled, axis=0)
        
        # Ajustar annotations
        if annotations is not None:
            annotations_new = annotations.copy()
            if 'boundaries' in annotations_new:
                annotations_new['boundaries'] = annotations_new['boundaries'] * factor
        else:
            annotations_new = None
        
        return frames_new, annotations_new
    
    def _frame_dropout(self, frames: np.ndarray) -> np.ndarray:
        """Elimina frames aleatoriamente"""
        T = len(frames)
        keep_mask = np.random.rand(T) > self.frame_dropout_prob
        
        # Asegurar que al menos la mitad de frames se mantienen
        if keep_mask.sum() < T // 2:
            return frames
        
        frames_kept = frames[keep_mask]
        
        return frames_kept
    
    def _temporal_jitter(self, annotations: dict, total_frames: int):
        """Añade jitter a las boundaries"""
        annotations_new = annotations.copy()
        
        if 'boundaries' in annotations_new:
            boundaries = annotations_new['boundaries']
            jitter_amount = int(total_frames * self.temporal_jitter)
            
            # Añadir jitter
            jitter = np.random.randint(-jitter_amount, jitter_amount + 1, boundaries.shape)
            boundaries_jittered = boundaries + jitter
            
            # Clip a rango válido
            boundaries_jittered = np.clip(boundaries_jittered, 0, total_frames - 1)
            
            annotations_new['boundaries'] = boundaries_jittered
        
        return annotations_new


class ComposeTransforms:
    """Compone múltiples transformaciones"""
    
    def __init__(self, spatial=None, temporal=None):
        self.spatial = spatial
        self.temporal = temporal
    
    def __call__(self, frames: np.ndarray, annotations=None):
        # Spatial augmentations
        if self.spatial is not None:
            frames = self.spatial(frames)
        
        # Temporal augmentations
        if self.temporal is not None:
            frames, annotations = self.temporal(frames, annotations)
        
        return frames, annotations


def get_train_transforms(config) -> ComposeTransforms:
    """Crea transformaciones de entrenamiento"""
    spatial = SpatialAugmentation(
        crop_size=(config['data']['input_size'], config['data']['input_size']),
        scale=config['augmentation']['spatial']['random_crop']['scale'],
        ratio=config['augmentation']['spatial']['random_crop']['ratio'],
        hflip_prob=config['augmentation']['spatial']['horizontal_flip']['prob'],
        color_jitter=config['augmentation']['spatial']['color_jitter']['enabled'],
        rotation_degrees=config['augmentation']['spatial']['rotation']['degrees']
    )
    
    temporal = TemporalAugmentation(
        temporal_jitter=config['augmentation']['temporal']['temporal_jitter']['max_shift_ratio'],
        speed_range=(
            1.0 - config['augmentation']['temporal']['speed_perturbation']['range'],
            1.0 + config['augmentation']['temporal']['speed_perturbation']['range']
        ),
        frame_dropout_prob=config['augmentation']['temporal']['frame_dropout']['prob']
    )
    
    return ComposeTransforms(spatial=spatial, temporal=temporal)


class CenterCrop:
    """Center crop para validación/test"""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        T, H, W, C = frames.shape
        # Resize manteniendo aspect ratio
        scale = self.size / min(H, W)
        new_H, new_W = int(H * scale), int(W * scale)

        frames_resized = []
        for t in range(T):
            frame_resized = cv2.resize(frames[t], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            frames_resized.append(frame_resized)
        frames = np.stack(frames_resized, axis=0)

        # Center crop
        T, H, W, C = frames.shape
        top = (H - self.size) // 2
        left = (W - self.size) // 2
        frames = frames[:, top:top+self.size, left:left+self.size, :]

        return frames


def get_train_transforms(target_size: Tuple[int, int], augmentation_config: dict):
    """Obtiene las transformaciones de entrenamiento"""
    
    # Spatial augmentations
    spatial = SpatialAugmentation(
        crop_size=target_size,
        scale=augmentation_config.get('spatial', {}).get('random_scale', [0.8, 1.0]),
        hflip_prob=augmentation_config.get('spatial', {}).get('random_flip', 0.5),
        color_jitter=augmentation_config.get('spatial', {}).get('color_jitter', {}),
        rotation_degrees=augmentation_config.get('spatial', {}).get('random_rotation', 10)
    )
    
    # Temporal augmentations
    temporal = TemporalAugmentation(
        temporal_jitter=augmentation_config.get('temporal', {}).get('temporal_jitter', 0.1),
        speed_range=tuple(augmentation_config.get('temporal', {}).get('speed_perturbation', [0.9, 1.1])),
        frame_dropout_prob=augmentation_config.get('temporal', {}).get('frame_dropout', 0.0)
    )
    
    return ComposeTransforms(spatial=spatial, temporal=temporal)


def get_val_transforms(target_size: Tuple[int, int]):
    """Obtiene las transformaciones de validación"""
    
    # Solo resize y center crop para validación
    spatial = CenterCrop(target_size[0])  # Usar el primer valor (altura)
    
    return ComposeTransforms(spatial=spatial, temporal=None)
if __name__ == '__main__':
    # Test
    print("Testing Augmentations...")
    
    # Crear frames dummy
    frames = np.random.randint(0, 255, (64, 256, 256, 3), dtype=np.uint8).astype(np.float32)
    
    # Test spatial
    spatial_aug = SpatialAugmentation()
    frames_aug = spatial_aug(frames)
    print(f"Spatial augmentation: {frames.shape} -> {frames_aug.shape}")
    
    # Test temporal
    annotations = {'boundaries': np.array([[10, 40], [50, 60]])}
    temporal_aug = TemporalAugmentation()
    frames_aug, ann_aug = temporal_aug(frames, annotations)
    print(f"Temporal augmentation: {frames.shape} -> {frames_aug.shape}")
    print(f"Boundaries: {annotations['boundaries']} -> {ann_aug['boundaries']}")
    
    print("✓ Test completado")
