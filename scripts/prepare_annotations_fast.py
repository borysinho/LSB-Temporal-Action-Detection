"""
Script rápido para preparar anotaciones usando información de nombres de archivos.

Este script asume que los nombres de archivos de segmentos contienen información temporal
en el formato: seña_XX_frames_START-END.mp4
"""

import os
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_filename(filename: str) -> Tuple[int, int]:
    """
    Extrae start_frame y end_frame del nombre del archivo.

    Args:
        filename: Nombre del archivo (ej: "seña_01_frames_0000-0360.mp4")

    Returns:
        start_frame, end_frame
    """
    # Buscar patrón frames_START-END
    match = re.search(r'frames_(\d+)-(\d+)', filename)
    if match:
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        return start_frame, end_frame

    # Fallback: extraer número de seña y asumir duración estándar
    match = re.search(r'seña_(\d+)', filename)
    if match:
        sign_num = int(match.group(1))
        # Asumir duración promedio de 100 frames por seña
        start_frame = (sign_num - 1) * 120  # 120 frames de separación aproximada
        end_frame = start_frame + 100
        return start_frame, end_frame

    # Último fallback
    return 0, 100


def get_video_info(video_path: str) -> Dict:
    """Extrae información básica del video."""
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


def process_single_segment(video_path: str, video_info: Dict, video_name: str, class_mapping: Dict) -> Dict:
    """
    Procesa un video individual como segmento completo.
    """
    logger.info(f"Procesando segmento individual: {video_name}")

    # Extraer información del nombre del archivo
    start_frame, end_frame = extract_frames_from_filename(video_name)

    # Si no se pudo extraer información de frames, asumir segmento completo
    if start_frame == 0 and end_frame == 100:  # Valores por defecto
        start_frame = 0
        end_frame = video_info['total_frames']

    # Validar frames
    if end_frame > video_info['total_frames']:
        end_frame = video_info['total_frames']
    if start_frame >= end_frame:
        start_frame = max(0, end_frame - 50)  # Mínimo 50 frames

    # Calcular tiempos
    start_time = start_frame / video_info['fps'] if video_info['fps'] > 0 else 0
    end_time = end_frame / video_info['fps'] if video_info['fps'] > 0 else 0

    # Extraer nombre de clase del nombre del archivo
    # Buscar patrones comunes en nombres de segmentos
    class_name = extract_class_from_filename(video_name)

    # Asignar ID de clase
    if class_name not in class_mapping:
        class_mapping[class_name] = len(class_mapping)

    class_id = class_mapping[class_name]

    annotation = {
        'label': class_name,
        'class': class_name,
        'class_id': class_id,
        'start_frame': int(start_frame),
        'end_frame': int(end_frame),
        'start_time': float(start_time),
        'end_time': float(end_time)
    }

    logger.info(f"  ✓ Segmento: frames {start_frame}-{end_frame} (clase: {class_name})")

    # Crear anotación completa
    video_annotation = {
        'video_id': video_name,
        'video_path': str(video_path),
        'duration': video_info['duration'],
        'fps': video_info['fps'],
        'total_frames': video_info['total_frames'],
        'width': video_info['width'],
        'height': video_info['height'],
        'annotations': [annotation]
    }

    return video_annotation


def extract_class_from_filename(filename: str) -> str:
    """
    Extrae el nombre de la clase del nombre del archivo.
    """
    # Remover extensión
    name = filename.replace('.mp4', '')

    # Buscar patrones comunes
    # Ejemplo: "TOMATE_frames_0000-0360" -> "TOMATE"
    if '_frames_' in name:
        return name.split('_frames_')[0]

    # Ejemplo: "seña_01_TOMATE" -> "TOMATE"
    if 'seña_' in name:
        parts = name.split('_')
        # Buscar la parte que no es numérica después de "seña_"
        for part in parts[1:]:
            if not part.isdigit():
                return part

    # Fallback: usar el nombre completo
    return name.lower().replace(' ', '_')


def process_video_fast(
    video_path: str,
    segments_dir: str,
    class_mapping: Dict[str, int]
) -> Dict:
    """
    Procesa un video usando información de nombres de archivos (rápido).
    """
    logger.info(f"Procesando: {video_path}")

    # Información del video
    video_info = get_video_info(video_path)
    video_name = Path(video_path).stem

    # Buscar directorio de segmentos (buscar recursivamente)
    video_parent = Path(video_path).parent
    segments_candidates = []

    # Buscar carpetas con patrón {video_name}_senas
    segments_candidates.extend(list(video_parent.rglob(f"{video_name}_senas")))

    # Buscar cualquier carpeta que contenga "segmentos" en el nombre
    segments_candidates.extend(list(video_parent.rglob("*segmentos*")))

    # Buscar carpetas que contengan archivos .mp4 (posibles segmentos)
    for subdir in video_parent.rglob("*"):
        if subdir.is_dir() and not str(subdir).endswith('_senas') and 'segmentos' not in str(subdir).lower():
            mp4_files = list(subdir.glob("*.mp4"))
            if mp4_files:
                segments_candidates.append(subdir)

    # Si no se encontraron carpetas de segmentos, tratar el video como segmento individual
    if not segments_candidates:
        logger.info(f"No se encontraron carpetas de segmentos. Tratando '{video_name}' como segmento individual.")
        return process_single_segment(video_path, video_info, video_name, class_mapping)

    segments_path = segments_candidates[0]  # Tomar el primero encontrado
    logger.info(f"Encontrado directorio de segmentos: {segments_path}")

    # Procesar todos los segmentos
    annotations = []
    segment_files = sorted(segments_path.glob("*.mp4"))

    logger.info(f"Encontrados {len(segment_files)} segmentos en carpeta")

    # Si la carpeta de segmentos está vacía, buscar videos pre-segmentados con múltiples patrones
    if len(segment_files) == 0:
        logger.info(f"Carpeta de segmentos vacía. Buscando videos pre-segmentados...")
        parent_dir = Path(video_path).parent

        # Patrón 1: nombre_XXX.mp4 (ej: TIEMPO_001.mp4, TIEMPO_002.mp4)
        segmented_pattern1 = re.compile(rf'^{re.escape(video_name)}_(\d+)\.mp4$')
        segment_files_p1 = []
        for file_path in parent_dir.glob(f"{video_name}_*.mp4"):
            match = segmented_pattern1.match(file_path.name)
            if match:
                segment_files_p1.append((int(match.group(1)), file_path))

        # Patrón 2: Buscar cualquier .mp4 en subcarpetas dentro de la carpeta *_senas
        segment_files_p2 = list(segments_path.rglob("*.mp4"))

        # Patrón 3: Buscar en el directorio padre con patrones más flexibles
        segment_files_p3 = []
        for file_path in parent_dir.glob("*.mp4"):
            if file_path != Path(video_path):  # Excluir el video principal
                # Verificar si el nombre contiene el nombre base
                if video_name.lower() in file_path.stem.lower():
                    segment_files_p3.append(file_path)

        # Usar el patrón que encontró más archivos
        all_patterns = [
            (segment_files_p1, "patrón nombre_XXX.mp4"),
            (segment_files_p2, "subcarpetas en *_senas"),
            (segment_files_p3, "archivos relacionados")
        ]

        best_pattern = max(all_patterns, key=lambda x: len(x[0]))

        if best_pattern[0]:
            if best_pattern[1] == "patrón nombre_XXX.mp4":
                segment_files = [fp for _, fp in sorted(best_pattern[0])]
            else:
                segment_files = best_pattern[0]

            logger.info(f"Encontrados {len(segment_files)} videos pre-segmentados usando {best_pattern[1]}")
            for sf in segment_files[:3]:  # Mostrar primeros 3
                logger.info(f"  - {sf.name}")
        else:
            logger.warning(f"No se encontraron videos pre-segmentados para {video_name}")
            # Fallback: tratar el video principal como segmento único
            return process_single_segment(video_path, video_info, video_name, class_mapping)

    for seg_file in segment_files:
        try:
            # Verificar si estamos procesando videos pre-segmentados (fuera de carpeta *_senas)
            is_pre_segmented = len(list(segments_path.glob("*.mp4"))) == 0 and seg_file.parent == Path(video_path).parent

            if is_pre_segmented:
                # Procesar como segmento individual completo
                logger.info(f"Procesando segmento pre-segmentado: {seg_file.name}")
                segment_info = get_video_info(str(seg_file))

                # Para videos pre-segmentados, usar frames completos del segmento
                start_frame = 0
                end_frame = segment_info['total_frames']

                # Calcular tiempos
                start_time = 0
                end_time = end_frame / segment_info['fps'] if segment_info['fps'] > 0 else 0

                # Nombre de clase (usar nombre base del video)
                class_name = video_name.lower().replace(' ', '_')

            else:
                # Procesar segmento tradicional dentro de carpeta *_senas
                # Extraer información del nombre del archivo
                start_frame, end_frame = extract_frames_from_filename(seg_file.name)

                # Validar frames
                if end_frame > video_info['total_frames']:
                    end_frame = video_info['total_frames']
                if start_frame >= end_frame:
                    start_frame = max(0, end_frame - 50)  # Mínimo 50 frames

                # Calcular tiempos
                start_time = start_frame / video_info['fps'] if video_info['fps'] > 0 else 0
                end_time = end_frame / video_info['fps'] if video_info['fps'] > 0 else 0

                # Nombre de clase (usar nombre del video como clase base)
                class_name = video_name.lower().replace(' ', '_')

            # Asignar ID de clase
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            class_id = class_mapping[class_name]

            annotation = {
                'label': class_name,
                'class': class_name,  # Agregar clave 'class' para compatibilidad con dataset
                'class_id': class_id,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'start_time': float(start_time),
                'end_time': float(end_time)
            }

            annotations.append(annotation)
            logger.info(f"  ✓ {seg_file.name}: frames {start_frame}-{end_frame}")

        except Exception as e:
            logger.error(f"  ✗ Error procesando {seg_file.name}: {e}")
            continue

    # Crear anotación completa
    video_annotation = {
        'video_id': video_name,  # Agregar ID único del video
        'video_path': str(video_path),
        'duration': video_info['duration'],
        'fps': video_info['fps'],
        'total_frames': video_info['total_frames'],
        'width': video_info['width'],
        'height': video_info['height'],
        'annotations': annotations
    }

    return video_annotation


def main(args):
    logger.info("=" * 60)
    logger.info("Preparación Rápida de Anotaciones para TAD")
    logger.info("=" * 60)

    videos_dir = Path(args.videos_dir)
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar videos recursivamente en subcarpetas
    video_files = list(videos_dir.rglob("*.mp4"))
    logger.info(f"Encontrados {len(video_files)} archivos .mp4")

    # Detectar si todos los videos son segmentos individuales
    videos_with_segments = 0
    videos_without_segments = 0
    total_segment_dirs = 0
    empty_segment_dirs = 0

    for video_file in video_files:
        video_name = video_file.stem
        video_parent = video_file.parent

        # Verificar si existe carpeta de segmentos
        segment_dir = video_parent / f"{video_name}_senas"
        if segment_dir.exists():
            total_segment_dirs += 1
            mp4_files = list(segment_dir.glob("*.mp4"))
            if mp4_files:
                videos_with_segments += 1
            else:
                empty_segment_dirs += 1
                videos_without_segments += 1
        else:
            videos_without_segments += 1

    logger.info(f"Videos con segmentos tradicionales: {videos_with_segments}")
    logger.info(f"Videos/carpetas sin segmentos: {videos_without_segments}")
    logger.info(f"Total carpetas *_senas: {total_segment_dirs}, vacías: {empty_segment_dirs}")

    if videos_with_segments == 0 and videos_without_segments > 0:
        if total_segment_dirs == 0:
            logger.info("🎯 DETECTADO: Todos los archivos .mp4 son segmentos individuales (sin carpetas *_senas).")
        elif empty_segment_dirs == total_segment_dirs:
            logger.info("🎯 DETECTADO: Todos los archivos .mp4 son segmentos individuales (carpetas *_senas vacías).")
        else:
            logger.info("🎯 DETECTADO: Mezcla de videos con y sin segmentos.")

        logger.info("   Procesando como segmentos pre-segmentados...")

        # Procesar todos los .mp4 como segmentos individuales
        all_annotations = []
        class_mapping = {}

        for video_file in video_files:
            annotation = process_single_segment(
                str(video_file),
                get_video_info(str(video_file)),
                video_file.stem,
                class_mapping
            )
            if annotation:
                all_annotations.append(annotation)

    else:
        logger.info(f"Encontrados {videos_with_segments} videos con segmentos, {videos_without_segments} sin segmentos")
        logger.info("   Usando estructura tradicional de carpetas...")

        # Procesar videos tradicionales
        all_annotations = []
        class_mapping = {}

        for video_file in video_files:
            annotation = process_video_fast(
                str(video_file),
                str(segments_dir),
                class_mapping
            )

            if annotation and annotation['annotations']:
                all_annotations.append(annotation)

    logger.info(f"\n✓ Procesados {len(all_annotations)} videos")
    logger.info(f"✓ Encontradas {len(class_mapping)} clases únicas")

    # Guardar mapeo de clases
    class_mapping_file = output_dir / 'class_mapping.json'
    with open(class_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Mapeo de clases guardado: {class_mapping_file}")

    # Split train/val/test (simple por ahora)
    np.random.seed(args.seed)
    indices = np.random.permutation(len(all_annotations))

    train_split = int(len(all_annotations) * args.train_ratio)
    val_split = int(len(all_annotations) * (args.train_ratio + args.val_ratio))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_annots = [all_annotations[i] for i in train_indices]
    val_annots = [all_annotations[i] for i in val_indices]
    test_annots = [all_annotations[i] for i in test_indices]

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
        logger.info("Promedio señas/video: N/A")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Preparación completada!")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparar anotaciones para TAD (versión rápida)')

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
        help='Proporción para training'
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