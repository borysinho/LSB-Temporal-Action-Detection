"""
Dataset para Temporal Action Detection de Lenguaje de Señas

Este dataset maneja videos con múltiples señas y sus anotaciones temporales.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path


class TemporalAnnotation:
    """Clase para manejar una anotación temporal individual"""
    
    def __init__(self, class_name: str, start_frame: int, end_frame: int, 
                 start_time: float, end_time: float, class_id: int):
        self.class_name = class_name
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_time = start_time
        self.end_time = end_time
        self.class_id = class_id
        self.duration_frames = end_frame - start_frame
        self.duration_time = end_time - start_time
    
    def temporal_iou(self, other: 'TemporalAnnotation') -> float:
        """Calcula IoU temporal con otra anotación"""
        intersection_start = max(self.start_frame, other.start_frame)
        intersection_end = min(self.end_frame, other.end_frame)
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(self.start_frame, other.start_frame)
        union_end = max(self.end_frame, other.end_frame)
        union = union_end - union_start
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            'class': self.class_name,
            'class_id': self.class_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration_time
        }


class TemporalActionDataset(Dataset):
    """
    Dataset para Temporal Action Detection
    
    Formato de anotaciones JSON:
    {
        "video_id": "FAMILIA_001.mp4",
        "video_path": "/path/to/video.mp4",
        "duration": 5.2,
        "fps": 30,
        "total_frames": 156,
        "annotations": [
            {
                "class": "FAMILIA",
                "class_id": 5,
                "start_time": 1.2,
                "end_time": 3.5,
                "start_frame": 36,
                "end_frame": 105
            },
            ...
        ]
    }
    """
    
    def __init__(
        self,
        annotations_file: str,
        videos_root: str,
        clip_length: int = 64,
        sampling_rate: int = 2,
        mode: str = 'train',
        transform = None,
        cache_features: bool = False
    ):
        """
        Args:
            annotations_file: Ruta al archivo JSON con anotaciones
            videos_root: Directorio raíz de videos
            clip_length: Número de frames por clip
            sampling_rate: Tasa de muestreo temporal (1 = todos los frames)
            mode: 'train', 'val' o 'test'
            transform: Transformaciones a aplicar
            cache_features: Si cachear features extraídas
        """
        self.annotations_file = annotations_file
        self.videos_root = videos_root
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.transform = transform
        self.cache_features = cache_features
        
        # Cargar anotaciones
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Construir índice de clips
        self.clips = self._build_clip_index()
        
        print(f"Dataset cargado: {len(self.data)} videos, {len(self.clips)} clips")
    
    def _build_clip_index(self) -> List[Dict]:
        """
        Construye índice de clips de entrenamiento
        
        Para cada video:
        - Si es training: genera múltiples clips con overlap
        - Si es val/test: genera clips sin overlap cubriendo todo el video
        """
        clips = []
        
        for video_idx, video_data in enumerate(self.data):
            total_frames = video_data['total_frames']
            fps = video_data['fps']
            
            # Calcular stride
            if self.mode == 'train':
                stride = self.clip_length // 2  # 50% overlap para training
            else:
                stride = self.clip_length  # No overlap para val/test
            
            # Generar clips
            for start_frame in range(0, total_frames, stride):
                end_frame = start_frame + self.clip_length
                
                if end_frame > total_frames:
                    # Último clip: ajustar para que termine en el final
                    start_frame = max(0, total_frames - self.clip_length)
                    end_frame = total_frames
                
                # Encontrar anotaciones que intersectan con este clip
                clip_annotations = []
                for ann in video_data['annotations']:
                    ann_start = ann['start_frame']
                    ann_end = ann['end_frame']
                    
                    # Calcular intersección
                    intersection_start = max(start_frame, ann_start)
                    intersection_end = min(end_frame, ann_end)
                    
                    if intersection_end > intersection_start:
                        # Hay intersección
                        # Convertir a coordenadas relativas del clip
                        relative_start = max(0, ann_start - start_frame)
                        relative_end = min(self.clip_length, ann_end - start_frame)
                        
                        clip_annotations.append({
                            'class': ann['class'],
                            'class_id': ann['class_id'],
                            'start': relative_start,
                            'end': relative_end,
                            'original_start': ann_start,
                            'original_end': ann_end
                        })
                
                clips.append({
                    'video_idx': video_idx,
                    'video_id': video_data['video_id'],
                    'video_path': video_data['video_path'],
                    'fps': fps,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'annotations': clip_annotations
                })
                
                # En val/test, evitar generar clips más allá del final
                if self.mode != 'train' and end_frame >= total_frames:
                    break
        
        return clips
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Retorna un clip y sus anotaciones
        
        Returns:
            clip: Tensor (C, T, H, W) con frames del clip
            targets: Dict con anotaciones y metadata
        """
        clip_data = self.clips[idx]
        
        # Cargar video
        frames = self._load_video_clip(
            clip_data['video_path'],
            clip_data['start_frame'],
            clip_data['end_frame']
        )
        
        # Aplicar transformaciones
        if self.transform is not None:
            if hasattr(self.transform, '__call__'):
                result = self.transform(frames)
                if isinstance(result, tuple):
                    frames, _ = result  # Desempaquetar si devuelve (frames, annotations)
                else:
                    frames = result
        
        # Convertir a tensor (T, H, W, C) -> (C, T, H, W)
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames).float()
        
        if frames.ndim == 4:  # (T, H, W, C)
            frames = frames.permute(3, 0, 1, 2)
        
        # Normalizar a [0, 1] si no está normalizado
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        # Preparar targets
        targets = {
            'video_id': clip_data['video_id'],
            'clip_start': clip_data['start_frame'],
            'clip_end': clip_data['end_frame'],
            'fps': clip_data['fps'],
            'annotations': clip_data['annotations'],
            'num_annotations': len(clip_data['annotations'])
        }
        
        # Convertir anotaciones a tensors para el modelo
        if len(clip_data['annotations']) > 0:
            targets['labels'] = torch.tensor(
                [ann['class_id'] for ann in clip_data['annotations']],
                dtype=torch.long
            )
            targets['boundaries'] = torch.tensor(
                [[ann['start'], ann['end']] for ann in clip_data['annotations']],
                dtype=torch.float32
            )
        else:
            # Clip sin anotaciones (background)
            targets['labels'] = torch.empty(0, dtype=torch.long)
            targets['boundaries'] = torch.empty((0, 2), dtype=torch.float32)
        
        return frames, targets
    
    def _load_video_clip(
        self, 
        video_path: str, 
        start_frame: int, 
        end_frame: int
    ) -> np.ndarray:
        """
        Carga frames de un video
        
        Args:
            video_path: Ruta al video
            start_frame: Frame inicial
            end_frame: Frame final
        
        Returns:
            frames: np.ndarray (T, H, W, C)
        """
        # Construir ruta completa
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.videos_root, video_path)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Posicionar en start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_count = 0
        target_frames = end_frame - start_frame
        
        while frame_count < target_frames:
            ret, frame = cap.read()
            
            if not ret:
                # Si no hay más frames, repetir el último
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    # Video vacío o corrupto
                    raise ValueError(f"No se pudieron leer frames del video: {video_path}")
            else:
                # Convertir BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        # Aplicar sampling rate si es necesario
        if self.sampling_rate > 1:
            frames = frames[::self.sampling_rate]
        
        # Asegurar que tengamos exactamente clip_length frames
        if len(frames) < self.clip_length:
            # Padding
            while len(frames) < self.clip_length:
                frames.append(frames[-1].copy())
        else:
            # Cropping
            frames = frames[:self.clip_length]
        
        # Redimensionar frames a tamaño consistente
        resized_frames = []
        target_height, target_width = 224, 224  # Usar tamaño del config
        
        for frame in frames:
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame)
        
        frames = np.stack(resized_frames, axis=0)  # (T, H, W, C)
        
        return frames
    
    def get_class_names(self) -> List[str]:
        """Retorna lista de nombres de clases"""
        class_names = set()
        for video_data in self.data:
            for ann in video_data['annotations']:
                class_names.add(ann['class'])
        return sorted(list(class_names))
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Retorna distribución de clases en el dataset"""
        distribution = {}
        for clip in self.clips:
            for ann in clip['annotations']:
                class_name = ann['class']
                distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


def collate_fn(batch):
    """
    Collate function personalizada para batch de videos con diferentes números de anotaciones
    
    Args:
        batch: Lista de (frames, targets)
    
    Returns:
        batched_frames: Tensor (B, C, T, H, W)
        batched_targets: Dict con targets batcheados y padding
    """
    frames_list = []
    targets_list = []
    
    for frames, targets in batch:
        frames_list.append(frames)
        targets_list.append(targets)
    
    # Encontrar la longitud máxima temporal
    max_temporal_len = max(f.shape[1] for f in frames_list)  # frames ya están en (C, T, H, W)
    
    # Padding de frames para que todos tengan la misma longitud
    padded_frames = []
    for frames in frames_list:
        c, t, h, w = frames.shape
        if t < max_temporal_len:
            # Padding con el último frame
            padding = frames[:, -1:, :, :].repeat(1, max_temporal_len - t, 1, 1)
            frames = torch.cat([frames, padding], dim=1)
        elif t > max_temporal_len:
            # Cropping
            frames = frames[:, :max_temporal_len, :, :]
        padded_frames.append(frames)
    
    # Stack frames
    batched_frames = torch.stack(padded_frames, dim=0)
    
    # Encontrar máximo número de anotaciones
    max_annotations = max(len(targets.get('labels', [])) for targets in targets_list)
    
    # Crear targets batcheados con padding
    batched_targets = {
        'labels': [],
        'boundaries': [],
        'num_annotations': []
    }
    
    for targets in targets_list:
        labels = targets.get('labels', torch.empty(0, dtype=torch.long))
        boundaries = targets.get('boundaries', torch.empty((0, 2), dtype=torch.float32))
        num_ann = len(labels)
        
        # Padding
        if len(labels) < max_annotations:
            pad_labels = torch.full((max_annotations - len(labels),), -1, dtype=torch.long)
            labels = torch.cat([labels, pad_labels])
            
            pad_boundaries = torch.full((max_annotations - len(boundaries), 2), -1, dtype=torch.float32)
            boundaries = torch.cat([boundaries, pad_boundaries])
        
        batched_targets['labels'].append(labels)
        batched_targets['boundaries'].append(boundaries)
        batched_targets['num_annotations'].append(num_ann)
    
    # Stack
    batched_targets['labels'] = torch.stack(batched_targets['labels'])
    batched_targets['boundaries'] = torch.stack(batched_targets['boundaries'])
    batched_targets['num_annotations'] = torch.tensor(batched_targets['num_annotations'], dtype=torch.long)
    
    # Crear start_targets y end_targets para boundary detection
    start_targets_list = []
    end_targets_list = []
    
    for boundaries in batched_targets['boundaries']:
        start_targets = torch.zeros(max_temporal_len, dtype=torch.float32)
        end_targets = torch.zeros(max_temporal_len, dtype=torch.float32)
        
        for boundary in boundaries:
            start_pos, end_pos = boundary
            if start_pos >= 0 and start_pos < max_temporal_len:
                start_targets[int(start_pos)] = 1.0
            if end_pos >= 0 and end_pos < max_temporal_len:
                end_targets[int(end_pos)] = 1.0
        
        start_targets_list.append(start_targets)
        end_targets_list.append(end_targets)
    
    batched_targets['start_targets'] = torch.stack(start_targets_list)
    batched_targets['end_targets'] = torch.stack(end_targets_list)
    
    return {
        'frames': batched_frames,
        'targets': batched_targets
    }


if __name__ == '__main__':
    # Test del dataset
    print("Testing TemporalActionDataset...")
    
    # Crear anotación de ejemplo
    sample_annotations = [
        {
            "video_id": "test_001.mp4",
            "video_path": "test_001.mp4",
            "duration": 5.0,
            "fps": 30,
            "total_frames": 150,
            "annotations": [
                {
                    "class": "FAMILIA",
                    "class_id": 0,
                    "start_time": 1.0,
                    "end_time": 2.5,
                    "start_frame": 30,
                    "end_frame": 75
                },
                {
                    "class": "SALUDO",
                    "class_id": 1,
                    "start_time": 3.0,
                    "end_time": 4.2,
                    "start_frame": 90,
                    "end_frame": 126
                }
            ]
        }
    ]
    
    # Guardar JSON temporal
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_annotations, f)
        temp_file = f.name
    
    try:
        # Crear dataset (fallará porque no hay videos reales, pero mostrará la estructura)
        dataset = TemporalActionDataset(
            annotations_file=temp_file,
            videos_root='.',
            clip_length=64,
            sampling_rate=2,
            mode='train'
        )
        
        print(f"Dataset creado: {len(dataset)} clips")
        print(f"Clases: {dataset.get_class_names()}")
        print(f"Distribución: {dataset.get_class_distribution()}")
        
    except Exception as e:
        print(f"Error esperado (no hay videos): {e}")
    finally:
        os.unlink(temp_file)
    
    print("✓ Dataset test completado")
