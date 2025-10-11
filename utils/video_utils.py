"""
Utilidades para manipulación de videos

Implementa:
1. load_video: Carga eficiente con decord/av
2. extract_frames: Extrae frames específicos
3. save_video_with_detections: Guarda video con overlays
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path


def load_video(
    video_path: str,
    backend: str = 'cv2',
    target_fps: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Carga video completo en memoria.
    
    Args:
        video_path: Ruta al video
        backend: 'cv2', 'decord' o 'av'
        target_fps: FPS objetivo (resample si difiere)
    
    Returns:
        (frames, info) donde:
        - frames: (T, H, W, C) array
        - info: dict con fps, duration, etc.
    """
    if backend == 'cv2':
        return _load_video_cv2(video_path, target_fps)
    elif backend == 'decord':
        try:
            return _load_video_decord(video_path, target_fps)
        except ImportError:
            print("Warning: decord no disponible, usando cv2")
            return _load_video_cv2(video_path, target_fps)
    elif backend == 'av':
        try:
            return _load_video_av(video_path, target_fps)
        except ImportError:
            print("Warning: av no disponible, usando cv2")
            return _load_video_cv2(video_path, target_fps)
    else:
        raise ValueError(f"Backend no soportado: {backend}")


def _load_video_cv2(
    video_path: str,
    target_fps: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """Carga video usando OpenCV."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir: {video_path}")
    
    # Info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Leer frames
    frames = []
    
    # Calcular stride si target_fps diferente
    if target_fps and target_fps != fps:
        stride = int(fps / target_fps)
    else:
        stride = 1
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % stride == 0:
            # CV2 usa BGR, convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    frames = np.array(frames)  # (T, H, W, C)
    
    info = {
        'fps': target_fps if target_fps else fps,
        'original_fps': fps,
        'total_frames': len(frames),
        'width': width,
        'height': height,
        'duration': duration
    }
    
    return frames, info


def _load_video_decord(
    video_path: str,
    target_fps: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """Carga video usando decord (más rápido)."""
    from decord import VideoReader, cpu
    
    vr = VideoReader(video_path, ctx=cpu(0))
    
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    width, height = vr[0].shape[1], vr[0].shape[0]
    duration = total_frames / fps
    
    # Calcular índices de frames
    if target_fps and target_fps != fps:
        stride = int(fps / target_fps)
        indices = list(range(0, total_frames, stride))
    else:
        indices = list(range(total_frames))
    
    # Leer frames
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) ya en RGB
    
    info = {
        'fps': target_fps if target_fps else fps,
        'original_fps': fps,
        'total_frames': len(frames),
        'width': width,
        'height': height,
        'duration': duration
    }
    
    return frames, info


def _load_video_av(
    video_path: str,
    target_fps: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """Carga video usando PyAV."""
    import av
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    fps = float(stream.average_rate)
    total_frames = stream.frames
    width = stream.width
    height = stream.height
    duration = float(stream.duration * stream.time_base) if stream.duration else 0
    
    # Calcular stride
    if target_fps and target_fps != fps:
        stride = int(fps / target_fps)
    else:
        stride = 1
    
    # Leer frames
    frames = []
    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx % stride == 0:
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
    
    container.close()
    
    frames = np.array(frames)
    
    info = {
        'fps': target_fps if target_fps else fps,
        'original_fps': fps,
        'total_frames': len(frames),
        'width': width,
        'height': height,
        'duration': duration
    }
    
    return frames, info


def extract_frames(
    video_path: str,
    frame_indices: List[int],
    backend: str = 'cv2'
) -> np.ndarray:
    """
    Extrae frames específicos de un video.
    
    Args:
        video_path: Ruta al video
        frame_indices: Lista de índices de frames a extraer
        backend: 'cv2' o 'decord'
    
    Returns:
        frames: (N, H, W, C) array
    """
    if backend == 'cv2':
        return _extract_frames_cv2(video_path, frame_indices)
    elif backend == 'decord':
        try:
            return _extract_frames_decord(video_path, frame_indices)
        except ImportError:
            return _extract_frames_cv2(video_path, frame_indices)
    else:
        raise ValueError(f"Backend no soportado: {backend}")


def _extract_frames_cv2(
    video_path: str,
    frame_indices: List[int]
) -> np.ndarray:
    """Extrae frames usando OpenCV."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir: {video_path}")
    
    frames = []
    frame_indices_set = set(frame_indices)
    max_idx = max(frame_indices)
    
    frame_idx = 0
    while frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in frame_indices_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    # Ordenar según frame_indices original
    frame_dict = {idx: frame for idx, frame in zip(sorted(frame_indices_set), frames)}
    frames = [frame_dict[idx] for idx in frame_indices if idx in frame_dict]
    
    return np.array(frames)


def _extract_frames_decord(
    video_path: str,
    frame_indices: List[int]
) -> np.ndarray:
    """Extrae frames usando decord (más rápido para acceso random)."""
    from decord import VideoReader, cpu
    
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices).asnumpy()
    
    return frames


def save_video_with_detections(
    video_path: str,
    output_path: str,
    detections: List[dict],
    fps: Optional[int] = None,
    codec: str = 'mp4v'
):
    """
    Guarda video con detecciones dibujadas.
    
    Args:
        video_path: Video original
        output_path: Ruta de salida
        detections: Lista de detecciones
        fps: FPS (usa original si None)
        codec: Codec de video ('mp4v', 'avc1', etc.)
    """
    # Cargar video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir: {video_path}")
    
    # Props
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps or original_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Crear mapa de detecciones por frame
    frame_detections = {}
    for det in detections:
        start_frame = int(det['segment'][0] * original_fps)
        end_frame = int(det['segment'][1] * original_fps)
        
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx not in frame_detections:
                frame_detections[frame_idx] = []
            frame_detections[frame_idx].append(det)
    
    # Procesar frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar detecciones
        if frame_idx in frame_detections:
            for det in frame_detections[frame_idx]:
                # Barra superior con label
                label = f"Class {det['class_id']}: {det['score']:.2f}"
                cv2.rectangle(frame, (0, 0), (width, 50), (0, 255, 0), -1)
                cv2.putText(frame, label, (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"Video guardado: {output_path}")


# Test code
if __name__ == '__main__':
    print("Testing Video Utils...")
    
    print("\nFunciones disponibles:")
    print("  - load_video(path, backend='cv2')")
    print("  - extract_frames(path, indices)")
    print("  - save_video_with_detections(path, output, detections)")
    
    print("\nBackends soportados:")
    print("  - cv2 (siempre disponible)")
    
    try:
        import decord
        print("  - decord ✓")
    except ImportError:
        print("  - decord ✗ (pip install decord)")
    
    try:
        import av
        print("  - av ✓")
    except ImportError:
        print("  - av ✗ (pip install av)")
    
    print("\n✅ Video utilities ready!")
