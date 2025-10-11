"""
Utilidades de visualización para TAD

Implementa:
1. plot_detections: Timeline de detecciones vs ground truth
2. plot_training_history: Curvas de loss/métricas
3. create_demo_video: Video con detecciones overlayed
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path


def plot_detections(
    detections: List[Dict],
    ground_truths: List[Dict],
    duration: float,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 6)
):
    """
    Visualiza detecciones en timeline comparado con ground truth.
    
    Args:
        detections: Lista de detecciones predichas
            - 'class_id', 'segment' [start, end], 'score'
        ground_truths: Lista de ground truths
            - 'class_id', 'segment' [start, end]
        duration: Duración total del video en segundos
        class_names: Dict {class_id: nombre} opcional
        save_path: Ruta para guardar figura
        figsize: Tamaño de figura
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Colores por clase
    colors = plt.cm.get_cmap('tab20')
    
    def get_color(class_id):
        return colors(class_id % 20)
    
    def get_label(class_id):
        if class_names and class_id in class_names:
            return class_names[class_id]
        return f"Class {class_id}"
    
    # Plot Ground Truth
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('GT', fontsize=12)
    
    for gt in ground_truths:
        start, end = gt['segment']
        class_id = gt['class_id']
        
        rect = patches.Rectangle(
            (start, 0.2), end - start, 0.6,
            linewidth=2,
            edgecolor=get_color(class_id),
            facecolor=get_color(class_id),
            alpha=0.7,
            label=get_label(class_id)
        )
        ax1.add_patch(rect)
        
        # Texto con clase
        ax1.text(
            (start + end) / 2, 0.5,
            get_label(class_id),
            ha='center', va='center',
            fontsize=9,
            fontweight='bold',
            color='white'
        )
    
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)
    
    # Plot Predictions
    ax2.set_title('Predictions', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Pred', fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    
    for det in detections:
        start, end = det['segment']
        class_id = det['class_id']
        score = det['score']
        
        rect = patches.Rectangle(
            (start, 0.2), end - start, 0.6,
            linewidth=2,
            edgecolor=get_color(class_id),
            facecolor=get_color(class_id),
            alpha=0.7 * score  # Alpha según confidence
        )
        ax2.add_patch(rect)
        
        # Texto con clase y score
        label_text = f"{get_label(class_id)}\n{score:.2f}"
        ax2.text(
            (start + end) / 2, 0.5,
            label_text,
            ha='center', va='center',
            fontsize=8,
            fontweight='bold',
            color='white'
        )
    
    ax2.set_xlim(0, duration)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Timeline guardado en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history_file: str,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Visualiza historial de entrenamiento.
    
    Args:
        history_file: Ruta al JSON con historial
        save_path: Ruta para guardar figura
        figsize: Tamaño de figura
    """
    # Cargar historial
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mAP
    ax = axes[1]
    ax.plot(epochs, history['val_map'], 'g-', label='Val mAP@0.5', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Validation mAP', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[2]
    ax.plot(epochs, history['lr'], 'orange', label='Learning Rate', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('LR', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Historial guardado en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_demo_video(
    video_path: str,
    detections: List[Dict],
    output_path: str,
    class_names: Optional[Dict[int, str]] = None,
    fps: int = 30,
    show_confidence: bool = True
):
    """
    Crea video demo con detecciones overlayed.
    
    Args:
        video_path: Ruta al video original
        detections: Lista de detecciones
        output_path: Ruta para video de salida
        class_names: Dict {class_id: nombre}
        fps: FPS del video
        show_confidence: Mostrar confidence score
    """
    import cv2
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir video: {video_path}")
    
    # Propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    
    print(f"Procesando video: {total_frames} frames...")
    
    # Colores por clase
    np.random.seed(42)
    colors = {
        class_id: tuple(int(c) for c in np.random.randint(0, 255, 3))
        for class_id in set(d['class_id'] for d in detections)
    }
    
    # Procesar frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar detecciones activas
        if frame_idx in frame_detections:
            for det in frame_detections[frame_idx]:
                class_id = det['class_id']
                score = det['score']
                
                # Nombre de clase
                if class_names and class_id in class_names:
                    label = class_names[class_id]
                else:
                    label = f"Class {class_id}"
                
                # Agregar confidence si se solicita
                if show_confidence:
                    label += f" {score:.2f}"
                
                # Color
                color = colors.get(class_id, (0, 255, 0))
                
                # Barra superior con label
                bar_height = 40
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, bar_height),
                    color,
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame,
                    label,
                    (10, 28),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )
        
        # Escribir frame
        out.write(frame)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Procesado {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"✓ Video demo guardado: {output_path}")


# Test/Example code
if __name__ == '__main__':
    print("Testing Visualization Utils...")
    
    # Test data
    detections = [
        {'class_id': 1, 'segment': np.array([2.0, 5.0]), 'score': 0.95},
        {'class_id': 2, 'segment': np.array([6.0, 9.5]), 'score': 0.87},
        {'class_id': 1, 'segment': np.array([11.0, 14.5]), 'score': 0.92},
    ]
    
    ground_truths = [
        {'class_id': 1, 'segment': np.array([2.2, 5.3])},
        {'class_id': 2, 'segment': np.array([6.1, 9.2])},
        {'class_id': 1, 'segment': np.array([11.5, 14.8])},
    ]
    
    class_names = {
        1: 'HOLA',
        2: 'GRACIAS'
    }
    
    # Test plot_detections
    print("\n1. Testing plot_detections...")
    plot_detections(
        detections,
        ground_truths,
        duration=16.0,
        class_names=class_names,
        save_path='test_timeline.png'
    )
    
    # Test plot_training_history (requiere history.json)
    print("\n2. Testing plot_training_history...")
    print("   (Requiere archivo history.json del entrenamiento)")
    
    # Test create_demo_video (requiere video real)
    print("\n3. Testing create_demo_video...")
    print("   (Requiere video de entrada)")
    
    print("\n✅ Visualization utilities working!")
