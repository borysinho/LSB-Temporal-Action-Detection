# Temporal Action Detection - LSB Sign Language
# Sistema TAD para detección de Lenguaje de Señas Boliviano
# GPU: T4/P100 | Tiempo: ~6-7 horas | Esperado: 75-82% mAP

# ==============================================================================
# CELDA 1: VERIFICAR HARDWARE Y DEPENDENCIAS
# ==============================================================================

import os
import sys
import subprocess

print("🔍 Verificando entorno Kaggle...")

# Verificar GPU
import torch
print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ GPU disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ WARNING: GPU no detectada! Activar GPU en Settings")

# Verificar espacio en disco
import shutil
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"\n💾 Espacio disponible: {free / 1e9:.1f} GB")

print("\n" + "="*60)

# ==============================================================================
# CELDA 2: INSTALAR DEPENDENCIAS
# ==============================================================================

print("📦 Instalando dependencias...")

# Instalar paquetes
!pip install -q einops timm transformers pyyaml opencv-python-headless

# Opcional: Decord para carga rápida de videos
!pip install -q decord

print("✅ Dependencias instaladas correctamente")

# ==============================================================================
# CELDA 3: CREAR ESTRUCTURA DE DIRECTORIOS
# ==============================================================================

from pathlib import Path

print("📁 Creando estructura de directorios...")

# Directorios del proyecto
dirs = [
    'data',
    'models/backbones',
    'models/detection',
    'training',
    'utils',
    'scripts',
    '/kaggle/working/checkpoints',
    '/kaggle/working/logs',
    '/kaggle/working/results'
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

# Crear __init__.py vacíos
init_files = [
    'data/__init__.py',
    'models/__init__.py',
    'models/backbones/__init__.py',
    'models/detection/__init__.py',
    'training/__init__.py',
    'utils/__init__.py'
]

for f in init_files:
    Path(f).touch()

print("✅ Estructura creada")

# ==============================================================================
# CELDA 4: CONFIGURACIÓN DEL PROYECTO
# ==============================================================================

# IMPORTANTE: Ajustar esta ruta según tu Kaggle Dataset
DATASET_PATH = "/kaggle/input/lsb-sign-language-dataset"  # ⬅️ CAMBIAR ESTO

# Verificar que el dataset existe
if not Path(DATASET_PATH).exists():
    print(f"⚠️ ERROR: Dataset no encontrado en {DATASET_PATH}")
    print("   Por favor, adjunta tu dataset en 'Add Data' →")
else:
    print(f"✅ Dataset encontrado: {DATASET_PATH}")

# Crear config.yaml
config_yaml = f"""
model:
  num_classes: 24
  backbone: video_swin
  variant: tiny
  feature_dim: 768
  hidden_dim: 512
  num_proposal_levels: 4

data:
  video_dir: "{DATASET_PATH}/videos"
  train_annotations: "{DATASET_PATH}/annotations/annotations_train.json"
  val_annotations: "{DATASET_PATH}/annotations/annotations_val.json"
  test_annotations: "{DATASET_PATH}/annotations/annotations_test.json"
  class_mapping: "{DATASET_PATH}/annotations/class_mapping.json"
  
  clip_length: 64
  target_size: [224, 224]
  sampling_rate: 1
  batch_size: 4
  num_workers: 2
  train_overlap: 0.5

training:
  num_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  min_lr: 1.0e-6
  
  batch_size: 4
  gradient_accumulation_steps: 2
  use_amp: true
  early_stopping_patience: 10

detection:
  proposal_lengths: [8, 16, 32, 64]
  max_proposals: 100
  nms_threshold: 0.5
  score_threshold: 0.3

loss_weights:
  classification: 1.0
  boundary: 1.0
  iou: 0.5
  regression: 0.3
  focal_alpha: 0.25
  focal_gamma: 2.0

augmentation:
  spatial:
    random_crop: true
    random_flip: true
    color_jitter: 0.2
  temporal:
    speed_perturbation: 0.1
    temporal_jitter: 0.05
"""

with open('config.yaml', 'w') as f:
    f.write(config_yaml)

print("✅ config.yaml creado")

# ==============================================================================
# CELDA 5: COPIAR CÓDIGO DEL PROYECTO
# ==============================================================================

print("📝 Método de importación del código:")
print("   Opción 1: Git clone desde GitHub")
print("   Opción 2: Dataset de Kaggle con el código")
print("   Opción 3: %%writefile manual (recomendado para testing)")
print("\nElige un método y ejecuta las celdas correspondientes abajo:")

# ==============================================================================
# CELDA 6A: OPCIÓN 1 - CLONAR DESDE GITHUB
# ==============================================================================

# Descomentar si tienes el código en GitHub:
# !git clone https://github.com/TU-USUARIO/tad-lsb.git
# 
# # Copiar archivos al working directory
# !cp -r tad-lsb/data .
# !cp -r tad-lsb/models .
# !cp -r tad-lsb/training .
# !cp -r tad-lsb/utils .
# !cp -r tad-lsb/scripts .
# 
# print("✅ Código clonado desde GitHub")

# ==============================================================================
# CELDA 6B: OPCIÓN 2 - DATASET DE KAGGLE CON CÓDIGO
# ==============================================================================

# Si subiste el código como Kaggle Dataset:
# CODE_DATASET = "/kaggle/input/tad-code"  # Tu dataset de código
# 
# !cp -r {CODE_DATASET}/data .
# !cp -r {CODE_DATASET}/models .
# !cp -r {CODE_DATASET}/training .
# !cp -r {CODE_DATASET}/utils .
# !cp -r {CODE_DATASET}/scripts .
# 
# print("✅ Código copiado desde Kaggle Dataset")

# ==============================================================================
# CELDA 6C: OPCIÓN 3 - INSTRUCCIONES PARA %%writefile
# ==============================================================================

print("""
⚠️ Si usas %%writefile, necesitarás crear celdas para cada archivo:

Celdas necesarias (copiar contenido completo de cada archivo):

1. data/dataset.py (~450 líneas)
2. data/temporal_annotations.py (~150 líneas)
3. data/augmentation.py (~400 líneas)
4. models/backbones/video_swin.py (~750 líneas)
5. models/detection/proposal_network.py (~550 líneas)
6. models/detection/boundary_detector.py (~500 líneas)
7. models/detection/action_classifier.py (~500 líneas)
8. models/complete_model.py (~450 líneas)
9. training/losses.py (~600 líneas)
10. training/metrics.py (~600 líneas)
11. training/trainer.py (~800 líneas)
12. utils/temporal_nms.py (~350 líneas)
13. utils/visualization.py (~350 líneas)
14. utils/video_utils.py (~200 líneas)
15. scripts/train.py (~350 líneas)
16. scripts/evaluate.py (~450 líneas)

TOTAL: ~7,500 líneas

Ejemplo de celda:
%%writefile data/dataset.py
[PEGAR CONTENIDO COMPLETO DE data/dataset.py]

Repetir para cada archivo.
""")

# ==============================================================================
# CELDA 7: VERIFICAR IMPORTACIONES
# ==============================================================================

print("🔍 Verificando que el código se importó correctamente...")

try:
    from data.dataset import TemporalActionDataset
    print("✅ data.dataset")
except Exception as e:
    print(f"❌ data.dataset: {e}")

try:
    from models.complete_model import build_model
    print("✅ models.complete_model")
except Exception as e:
    print(f"❌ models.complete_model: {e}")

try:
    from training.losses import TADLoss
    print("✅ training.losses")
except Exception as e:
    print(f"❌ training.losses: {e}")

try:
    from training.metrics import DetectionMetrics
    print("✅ training.metrics")
except Exception as e:
    print(f"❌ training.metrics: {e}")

try:
    from training.trainer import TADTrainer
    print("✅ training.trainer")
except Exception as e:
    print(f"❌ training.trainer: {e}")

print("\nSi ves errores, asegúrate de haber copiado todos los archivos")

# ==============================================================================
# CELDA 8: VERIFICAR DATASET
# ==============================================================================

import json

print("📊 Verificando dataset...")

# Verificar estructura
required_files = [
    f"{DATASET_PATH}/annotations/annotations_train.json",
    f"{DATASET_PATH}/annotations/annotations_val.json",
    f"{DATASET_PATH}/annotations/class_mapping.json"
]

all_exist = True
for file in required_files:
    if Path(file).exists():
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - NO ENCONTRADO")
        all_exist = False

if all_exist:
    # Cargar y mostrar estadísticas
    with open(f"{DATASET_PATH}/annotations/annotations_train.json", 'r') as f:
        train_data = json.load(f)
    
    with open(f"{DATASET_PATH}/annotations/annotations_val.json", 'r') as f:
        val_data = json.load(f)
    
    with open(f"{DATASET_PATH}/annotations/class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    print(f"\n📈 Estadísticas del Dataset:")
    print(f"   Videos entrenamiento: {len(train_data)}")
    print(f"   Videos validación: {len(val_data)}")
    print(f"   Clases totales: {len(class_mapping)}")
    
    # Contar señas
    total_signs_train = sum(len(v['annotations']) for v in train_data)
    total_signs_val = sum(len(v['annotations']) for v in val_data)
    
    print(f"   Señas entrenamiento: {total_signs_train}")
    print(f"   Señas validación: {total_signs_val}")
    
    # Duración total
    total_duration = sum(v['duration'] for v in train_data + val_data)
    print(f"   Duración total: {total_duration / 60:.1f} minutos")
    
    print("\n✅ Dataset verificado correctamente")
else:
    print("\n⚠️ Dataset incompleto. Verifica la estructura.")

# ==============================================================================
# CELDA 9: ENTRENAMIENTO
# ==============================================================================

print("🏋️ Iniciando entrenamiento...")
print("="*60)

import yaml
import torch
from torch.utils.data import DataLoader

# Cargar config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Crear modelo
from models.complete_model import build_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(config)
model = model.to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Modelo creado:")
print(f"   Total parámetros: {total_params / 1e6:.2f}M")
print(f"   Parámetros entrenables: {trainable_params / 1e6:.2f}M")

# Crear datasets
from data.dataset import TemporalActionDataset, collate_fn
from data.augmentation import get_train_transforms, get_val_transforms

print(f"\n📦 Cargando datasets...")

train_transforms = get_train_transforms(
    target_size=tuple(config['data']['target_size']),
    augmentation_config=config.get('augmentation', {})
)

val_transforms = get_val_transforms(
    target_size=tuple(config['data']['target_size'])
)

train_dataset = TemporalActionDataset(
    video_dir=config['data']['video_dir'],
    annotation_file=config['data']['train_annotations'],
    clip_length=config['data']['clip_length'],
    split='train',
    transform=train_transforms,
    overlap=config['data']['train_overlap']
)

val_dataset = TemporalActionDataset(
    video_dir=config['data']['video_dir'],
    annotation_file=config['data']['val_annotations'],
    clip_length=config['data']['clip_length'],
    split='val',
    transform=val_transforms,
    overlap=0.0
)

print(f"   Train clips: {len(train_dataset)}")
print(f"   Val clips: {len(val_dataset)}")

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=config['data']['num_workers'],
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=False,
    num_workers=config['data']['num_workers'],
    pin_memory=True,
    collate_fn=collate_fn
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['training']['num_epochs'],
    eta_min=config['training']['min_lr']
)

# Loss function
from training.losses import TADLoss

loss_fn = TADLoss(
    num_classes=config['model']['num_classes'],
    lambda_cls=config['loss_weights']['classification'],
    lambda_boundary=config['loss_weights']['boundary'],
    lambda_iou=config['loss_weights']['iou'],
    lambda_reg=config['loss_weights']['regression']
)

# Trainer
from training.trainer import TADTrainer

trainer = TADTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    num_classes=config['model']['num_classes'],
    config=config,
    device=device,
    checkpoint_dir='/kaggle/working/checkpoints',
    log_dir='/kaggle/working/logs',
    use_amp=config['training']['use_amp'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

print(f"\n🚀 Comenzando entrenamiento...")
print(f"   Épocas: {config['training']['num_epochs']}")
print(f"   Device: {device}")
print(f"   Mixed Precision: {config['training']['use_amp']}")
print("="*60)

# ENTRENAR
trainer.train(
    num_epochs=config['training']['num_epochs'],
    early_stopping_patience=config['training']['early_stopping_patience'],
    save_every=5,
    eval_every=1
)

print("\n✅ Entrenamiento completado!")

# ==============================================================================
# CELDA 10: BACKUP DE CHECKPOINTS
# ==============================================================================

print("💾 Creando backup de checkpoints...")

# Comprimir checkpoints
!cd /kaggle/working && zip -r checkpoints_backup.zip checkpoints/

# Verificar tamaño
import os
size = os.path.getsize('/kaggle/working/checkpoints_backup.zip') / 1e6
print(f"✅ Backup creado: checkpoints_backup.zip ({size:.1f} MB)")
print(f"\n📥 IMPORTANTE: Descarga este archivo desde el panel derecho")
print(f"   antes de que expire la sesión de Kaggle!")

# ==============================================================================
# CELDA 11: EVALUACIÓN EN TEST SET
# ==============================================================================

print("📊 Evaluando modelo en test set...")

from training.metrics import DetectionMetrics

model.eval()
metrics = DetectionMetrics(num_classes=config['model']['num_classes'])

# Cargar test dataset si existe
test_annotation_file = f"{DATASET_PATH}/annotations/annotations_test.json"

if Path(test_annotation_file).exists():
    test_dataset = TemporalActionDataset(
        video_dir=config['data']['video_dir'],
        annotation_file=test_annotation_file,
        clip_length=config['data']['clip_length'],
        split='test',
        transform=val_transforms,
        overlap=0.0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'] * 2,  # Batch más grande para eval
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    print(f"   Test clips: {len(test_dataset)}")
    
    # Evaluar
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            frames = batch['frames'].to(device)
            targets = batch['targets']
            
            outputs = model(frames)
            
            # Detecciones
            detections = outputs.get('detections', [])
            
            # Ground truths
            gts = []
            for b in range(frames.size(0)):
                labels = targets['labels'][b]
                segments = targets['segments'][b]
                num_actions = targets.get('num_actions', [len(labels)])[b]
                
                video_gts = []
                for i in range(num_actions):
                    if labels[i] > 0:
                        video_gts.append({
                            'class_id': labels[i].item(),
                            'segment': segments[i].cpu().numpy()
                        })
                gts.append(video_gts)
            
            all_predictions.extend(detections)
            all_ground_truths.extend(gts)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Procesado {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calcular métricas
    metrics.add_batch(all_predictions, all_ground_truths)
    results = metrics.compute_metrics()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS DE EVALUACIÓN - TEST SET")
    print("="*60)
    print(f"\nmAP:")
    print(f"  mAP@0.5:0.95: {results.get('mAP@0.5:0.95', 0):.4f}")
    print(f"  mAP@0.50:     {results.get('mAP@0.50', 0):.4f}")
    print(f"  mAP@0.75:     {results.get('mAP@0.75', 0):.4f}")
    
    print(f"\nMétricas de Clasificación:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    
    print("="*60)
    
    # Guardar resultados
    import json
    with open('/kaggle/working/results/test_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Resultados guardados en: /kaggle/working/results/test_metrics.json")
    
else:
    print("⚠️ Test annotations no encontrado, saltando evaluación de test")

# ==============================================================================
# CELDA 12: VISUALIZACIÓN
# ==============================================================================

print("📈 Generando visualizaciones...")

from utils.visualization import plot_training_history
import matplotlib.pyplot as plt

# Plot curvas de entrenamiento
plot_training_history(
    '/kaggle/working/logs/history.json',
    save_path='/kaggle/working/results/training_curves.png'
)

# Mostrar imagen
from IPython.display import Image, display
display(Image('/kaggle/working/results/training_curves.png'))

print("✅ Visualización completada")

# ==============================================================================
# CELDA 13: RESUMEN Y DESCARGA
# ==============================================================================

print("\n" + "="*60)
print("🎉 PROCESO COMPLETADO")
print("="*60)

print("\n📥 Archivos listos para descargar:")
print("   (Panel derecho → Output)")
print()
print("   1. /kaggle/working/checkpoints/best.pth")
print("      → Mejor modelo (mayor mAP)")
print()
print("   2. /kaggle/working/checkpoints_backup.zip")
print("      → Backup de todos los checkpoints")
print()
print("   3. /kaggle/working/logs/history.json")
print("      → Historial de entrenamiento")
print()
print("   4. /kaggle/working/results/test_metrics.json")
print("      → Métricas de evaluación")
print()
print("   5. /kaggle/working/results/training_curves.png")
print("      → Gráficas de entrenamiento")

print("\n⚠️ IMPORTANTE: Descarga estos archivos AHORA")
print("   Las sesiones de Kaggle expiran y los datos se pierden!")

print("\n📊 Resultados Esperados:")
print("   mAP@0.5: 75-82% (excelente si >80%)")
print("   Precision: 78-85%")
print("   Recall: 75-82%")

print("\n💡 Próximos pasos:")
print("   1. Descargar checkpoints")
print("   2. Usar modelo localmente para inferencia")
print("   3. Si resultados < esperados, ajustar hyperparámetros y re-entrenar")

print("\n" + "="*60)
print("✅ Notebook completado exitosamente!")
print("="*60)

# ==============================================================================
# FIN DEL NOTEBOOK
# ==============================================================================
