"""
Utilidades para procesar anotaciones temporales
"""

import json
import os
from typing import List, Dict
from pathlib import Path


class AnnotationProcessor:
    """Procesa y valida anotaciones temporales"""
    
    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: Lista de nombres de clases
        """
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
    
    def validate_annotation(self, annotation: Dict) -> bool:
        """
        Valida una anotación temporal
        
        Returns:
            True si es válida, False si no
        """
        required_fields = ['class', 'start_frame', 'end_frame', 'start_time', 'end_time']
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in annotation:
                print(f"Advertencia: Falta campo '{field}' en anotación")
                return False
        
        # Verificar clase válida
        if annotation['class'] not in self.class_names:
            print(f"Advertencia: Clase '{annotation['class']}' no reconocida")
            return False
        
        # Verificar boundaries válidos
        if annotation['start_frame'] >= annotation['end_frame']:
            print(f"Advertencia: start_frame >= end_frame")
            return False
        
        if annotation['start_time'] >= annotation['end_time']:
            print(f"Advertencia: start_time >= end_time")
            return False
        
        return True
    
    def add_class_ids(self, annotation: Dict) -> Dict:
        """Añade class_id a una anotación"""
        annotation = annotation.copy()
        annotation['class_id'] = self.class_to_id[annotation['class']]
        return annotation
    
    def process_video_annotations(self, video_data: Dict) -> Dict:
        """
        Procesa todas las anotaciones de un video
        
        Args:
            video_data: Dict con metadata y anotaciones del video
        
        Returns:
            video_data procesado
        """
        video_data = video_data.copy()
        
        # Validar y procesar cada anotación
        valid_annotations = []
        for ann in video_data.get('annotations', []):
            if self.validate_annotation(ann):
                ann = self.add_class_ids(ann)
                valid_annotations.append(ann)
        
        video_data['annotations'] = valid_annotations
        
        return video_data
    
    def process_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Procesa dataset completo"""
        processed = []
        for video_data in dataset:
            processed.append(self.process_video_annotations(video_data))
        return processed
    
    def save_annotations(self, dataset: List[Dict], output_path: str):
        """Guarda anotaciones procesadas"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Anotaciones guardadas en: {output_path}")


def temporal_iou(seg1: List[int], seg2: List[int]) -> float:
    """
    Calcula Temporal IoU entre dos segmentos
    
    Args:
        seg1: [start, end]
        seg2: [start, end]
    
    Returns:
        IoU temporal (0-1)
    """
    intersection_start = max(seg1[0], seg2[0])
    intersection_end = min(seg1[1], seg2[1])
    intersection = max(0, intersection_end - intersection_start)
    
    union_start = min(seg1[0], seg2[0])
    union_end = max(seg1[1], seg2[1])
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0.0


if __name__ == '__main__':
    # Test
    print("Testing AnnotationProcessor...")
    
    classes = ['FAMILIA', 'SALUDO', 'GRACIAS']
    processor = AnnotationProcessor(classes)
    
    # Test annotation
    ann = {
        'class': 'FAMILIA',
        'start_frame': 30,
        'end_frame': 75,
        'start_time': 1.0,
        'end_time': 2.5
    }
    
    is_valid = processor.validate_annotation(ann)
    print(f"Anotación válida: {is_valid}")
    
    ann_with_id = processor.add_class_ids(ann)
    print(f"Anotación con ID: {ann_with_id}")
    
    # Test temporal IoU
    seg1 = [30, 75]
    seg2 = [50, 100]
    iou = temporal_iou(seg1, seg2)
    print(f"Temporal IoU: {iou:.3f}")
    
    print("✓ Test completado")
