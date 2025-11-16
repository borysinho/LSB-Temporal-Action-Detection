"""
Script para preparar anotaciones desde videos segmentados.

Convierte la estructura actual de videos segmentados a formato JSON
requerido por TemporalActionDataset:

Input (estructura actual):
    videos/
        CATEGORIA1.mp4 (video completo)
        CATEGORIA1/
            seña1.mp4
            seña2.mp4
            ...

Output (JSON):
    {
        "video_path": "videos/CATEGORIA1.mp4",
        "duration": 120.5,
        "fps": 30.0,
        "total_frames": 3615,
        "annotations": [
            {
                "label": "seña1",
                "class_id": 1,
                "start_frame": 0,
                "end_frame": 45,
                "start_time": 0.0,
                "end_time": 1.5
            },
            ...
        ]
    }

Usage:
    python scripts/prepare_annotations.py \
        --videos_dir "1 videos originales/1" \
        --segments_dir "2 segmentacion/1_procesados" \
        --output_dir "data/annotations" \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15
"""

import os
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Dict:
    """
    Extrae información de un video.
    
    Args:
        video_path: Ruta al video
    
    Returns:
        Dict con duration, fps, total_frames, width, height
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'width': width,
        'height': height
    }


def find_segment_in_video(
    segment_path: str,
    full_video_path: str,
    method: str = 'optical_flow'
) -> Tuple[int, int]:
    """
    Encuentra la posición temporal de un segmento dentro del video completo.
    
    Args:
        segment_path: Ruta al video segmentado
        full_video_path: Ruta al video completo
        method: 'optical_flow' o 'frame_matching'
    
    Returns:
        (start_frame, end_frame)
    """
    # Abrir ambos videos
    seg_cap = cv2.VideoCapture(segment_path)
    full_cap = cv2.VideoCapture(full_video_path)
    
    if not seg_cap.isOpened() or not full_cap.isOpened():
        raise ValueError("No se pudieron abrir los videos")
    
    # Obtener info
    seg_frames = int(seg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_frames = int(full_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Leer primer frame del segmento
    ret, seg_first = seg_cap.read()
    if not ret:
        seg_cap.release()
        full_cap.release()
        raise ValueError("No se pudo leer el segmento")
    
    # Convertir a escala de grises y redimensionar para matching
    seg_first_gray = cv2.cvtColor(seg_first, cv2.COLOR_BGR2GRAY)
    seg_first_gray = cv2.resize(seg_first_gray, (320, 240))
    
    # Buscar en video completo
    best_match_score = -1
    best_start_frame = 0
    
    logger.info(f"Buscando segmento en video completo...")
    logger.info(f"  Segmento: {seg_frames} frames")
    logger.info(f"  Video: {full_frames} frames")
    
    # Buscar cada 10 frames para eficiencia
    search_step = 10
    
    for frame_idx in range(0, full_frames - seg_frames, search_step):
        full_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, full_frame = full_cap.read()
        
        if not ret:
            break
        
        # Convertir y redimensionar
        full_gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        full_gray = cv2.resize(full_gray, (320, 240))
        
        # Template matching
        result = cv2.matchTemplate(full_gray, seg_first_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_match_score:
            best_match_score = max_val
            best_start_frame = frame_idx
        
        # Si encontramos un match muy bueno, salir
        if max_val > 0.95:
            break
    
    # Refinar búsqueda alrededor del mejor match
    refine_range = search_step * 2
    start = max(0, best_start_frame - refine_range)
    end = min(full_frames - seg_frames, best_start_frame + refine_range)
    
    for frame_idx in range(start, end):
        full_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, full_frame = full_cap.read()
        
        if not ret:
            break
        
        full_gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        full_gray = cv2.resize(full_gray, (320, 240))
        
        result = cv2.matchTemplate(full_gray, seg_first_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_match_score:
            best_match_score = max_val
            best_start_frame = frame_idx
    
    seg_cap.release()
    full_cap.release()
    
    start_frame = best_start_frame
    end_frame = best_start_frame + seg_frames
    
    logger.info(f"  Match encontrado: frames {start_frame}-{end_frame} (score: {best_match_score:.3f})")
    
    return start_frame, end_frame


def process_video_category(
    full_video_path: str,
    segments_dir: str,
    class_mapping: Dict[str, int]
) -> Dict:
    """
    Procesa una categoría de video completa.
    
    Args:
        full_video_path: Ruta al video completo
        segments_dir: Directorio con segmentos
        class_mapping: Dict {nombre_clase: class_id}
    
    Returns:
        Dict con anotaciones para este video
    """
    logger.info(f"\nProcesando: {full_video_path}")
    
    # Obtener info del video completo
    video_info = get_video_info(full_video_path)
    
    # Encontrar segmentos
    segments_path = Path(segments_dir)
    segment_files = sorted(segments_path.glob("*.mp4"))
    
    if len(segment_files) == 0:
        logger.warning(f"No se encontraron segmentos en {segments_dir}")
        return None
    
    logger.info(f"Encontrados {len(segment_files)} segmentos")
    
    # Procesar cada segmento
    annotations = []
    
    for seg_file in segment_files:
        # Extraer nombre de clase del archivo
        class_name = seg_file.stem  # Nombre sin extensión
        
        # Obtener o crear class_id
        if class_name not in class_mapping:
            class_mapping[class_name] = len(class_mapping) + 1  # +1 porque 0 es background
        
        class_id = class_mapping[class_name]
        
        # Encontrar posición en video completo
        try:
            start_frame, end_frame = find_segment_in_video(
                str(seg_file),
                full_video_path
            )
            
            # Calcular tiempos
            fps = video_info['fps']
            start_time = start_frame / fps if fps > 0 else 0
            end_time = end_frame / fps if fps > 0 else 0
            
            annotation = {
                'label': class_name,
                'class_id': class_id,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'start_time': float(start_time),
                'end_time': float(end_time)
            }
            
            annotations.append(annotation)
            logger.info(f"  ✓ {class_name}: frames {start_frame}-{end_frame}")
            
        except Exception as e:
            logger.error(f"  ✗ Error procesando {seg_file.name}: {e}")
            continue
    
    # Ordenar anotaciones por start_frame
    annotations.sort(key=lambda x: x['start_frame'])
    
    # Crear anotación completa
    video_annotation = {
        'video_path': full_video_path,
        'duration': video_info['duration'],
        'fps': video_info['fps'],
        'total_frames': video_info['total_frames'],
        'width': video_info['width'],
        'height': video_info['height'],
        'annotations': annotations
    }
    
    return video_annotation


def split_annotations(
    all_annotations: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Divide anotaciones en train/val/test.
    
    Args:
        all_annotations: Lista de anotaciones
        train_ratio: Proporción para train
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        seed: Semilla random
    
    Returns:
        (train_annotations, val_annotations, test_annotations)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Shuffle con seed fija
    np.random.seed(seed)
    indices = np.random.permutation(len(all_annotations))
    
    # Calcular splits
    n_train = int(len(all_annotations) * train_ratio)
    n_val = int(len(all_annotations) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_annotations = [all_annotations[i] for i in train_indices]
    val_annotations = [all_annotations[i] for i in val_indices]
    test_annotations = [all_annotations[i] for i in test_indices]
    
    return train_annotations, val_annotations, test_annotations


def main(args):
    """Función principal."""
    logger.info("=" * 60)
    logger.info("Preparación de Anotaciones para TAD")
    logger.info("=" * 60)
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapeo de clases
    class_mapping = {}
    
    # Procesar todos los videos
    all_annotations = []
    
    videos_dir = Path(args.videos_dir)
    segments_base = Path(args.segments_dir)
    
    # Buscar videos completos (.mp4 en directorio raíz)
    video_files = sorted(videos_dir.glob("*.mp4"))
    
    logger.info(f"Encontrados {len(video_files)} videos completos")
    
    for video_file in video_files:
        # Buscar directorio de segmentos correspondiente
        # Ej: COLORES.mp4 → COLORES_senas/
        video_name = video_file.stem
        segments_dir = segments_base / f"{video_name}_senas"

        if not segments_dir.exists():
            logger.warning(f"No se encontró directorio de segmentos: {segments_dir}")
            continue
        
        # Procesar
        annotation = process_video_category(
            str(video_file),
            str(segments_dir),
            class_mapping
        )
        
        if annotation:
            all_annotations.append(annotation)
    
    logger.info(f"\n✓ Procesados {len(all_annotations)} videos")
    logger.info(f"✓ Encontradas {len(class_mapping)} clases únicas")
    
    # Guardar mapeo de clases
    class_mapping_file = output_dir / 'class_mapping.json'
    with open(class_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Mapeo de clases guardado: {class_mapping_file}")
    
    # Split train/val/test
    train_annots, val_annots, test_annots = split_annotations(
        all_annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    logger.info(f"\nSplits:")
    logger.info(f"  Train: {len(train_annots)} videos")
    logger.info(f"  Val:   {len(val_annots)} videos")
    logger.info(f"  Test:  {len(test_annots)} videos")
    
    # Guardar anotaciones
    for split_name, split_data in [('train', train_annots), 
                                     ('val', val_annots), 
                                     ('test', test_annots)]:
        output_file = output_dir / f'annotations_{split_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Guardado: {output_file}")
    
    # Estadísticas
    logger.info("\n" + "=" * 60)
    logger.info("Estadísticas del Dataset")
    logger.info("=" * 60)
    
    total_signs = sum(len(a['annotations']) for a in all_annotations)
    total_duration = sum(a['duration'] for a in all_annotations)
    
    logger.info(f"Total videos: {len(all_annotations)}")
    logger.info(f"Total señas: {total_signs}")
    logger.info(f"Duración total: {total_duration / 60:.1f} minutos")
    logger.info(f"Clases únicas: {len(class_mapping)}")
    if len(all_annotations) > 0:
        logger.info(f"Promedio señas/video: {total_signs / len(all_annotations):.1f}")
    else:
        logger.info("Promedio señas/video: N/A (no hay videos procesados)")
    
    # Distribución por clase
    class_counts = defaultdict(int)
    for annot in all_annotations:
        for sign in annot['annotations']:
            class_counts[sign['label']] += 1
    
    logger.info("\nTop 10 clases más frecuentes:")
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {label}: {count}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Preparación completada!")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparar anotaciones para TAD')
    
    parser.add_argument(
        '--videos_dir',
        type=str,
        required=True,
        help='Directorio con videos completos'
    )
    parser.add_argument(
        '--segments_dir',
        type=str,
        required=True,
        help='Directorio base con segmentos procesados'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/annotations',
        help='Directorio de salida para anotaciones'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Proporción para entrenamiento'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Proporción para validación'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Proporción para test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla random para splits'
    )
    
    args = parser.parse_args()
    main(args)
