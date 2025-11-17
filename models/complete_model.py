"""
Modelo Completo de Temporal Action Detection

Integra todos los componentes en un pipeline end-to-end:
- Backbone (Video Swin Transformer)
- Temporal Proposal Network
- Boundary Detector
- Action Classifier
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import yaml

from .backbones import get_backbone
from .detection.proposal_network import TemporalProposalNetwork
from .detection.boundary_detector import BoundaryDetector, EnhancedBoundaryDetector
from .detection.action_classifier import ActionClassifier


class TemporalActionDetector(nn.Module):
    """
    Modelo completo de Temporal Action Detection
    
    Pipeline:
    1. Backbone: Extrae features espacio-temporales
    2. Proposal Network: Genera candidatos temporales
    3. Boundary Detector: Refina boundaries
    4. Action Classifier: Clasifica cada proposal
    5. Post-processing: NMS, filtering, etc.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dict con configuración del modelo
        """
        super().__init__()
        
        self.config = config
        self.num_classes = config['model']['num_classes']
        
        # 1. Backbone
        self.backbone = self._build_backbone(config)
        
        # 2. Temporal Proposal Network
        self.proposal_net = TemporalProposalNetwork(
            in_channels=config['model']['backbone_config']['feature_dim'],
            hidden_channels=config['model']['detection']['hidden_dim'],
            num_pyramid_levels=config['model']['detection'].get('num_pyramid_levels', 3),
            proposal_lengths=config['model']['detection']['proposal_lengths'],
            min_score=config['model']['detection'].get('min_proposal_score', 0.1),
            max_proposals=config['model']['detection']['num_proposals']
        )
        
        # 3. Boundary Detector
        use_enhanced = config['model']['detection'].get('use_enhanced_boundary', True)
        if use_enhanced:
            self.boundary_detector = EnhancedBoundaryDetector(
                in_channels=config['model']['backbone_config']['feature_dim'],
                hidden_channels=config['model']['detection']['hidden_dim'],
                num_conv_layers=config['model']['boundary']['num_layers'],
                kernel_sizes=[config['model']['boundary']['kernel_size']]
            )
        else:
            self.boundary_detector = BoundaryDetector(
                in_channels=config['model']['backbone_config']['feature_dim'],
                hidden_channels=config['model']['detection']['hidden_dim']
            )
        
        # 4. Classification Head
        self.classifier = ActionClassifier(
            in_channels=config['model']['backbone_config']['feature_dim'],
            hidden_channels=config['model']['classifier']['hidden_dim'],
            num_classes=self.num_classes,
            use_attention=config['model']['detection'].get('use_attention', True),
            dropout=config['model']['classifier']['dropout']
        )
        
        # Post-processing config
        self.nms_threshold = config['model']['detection'].get('nms_iou_threshold', 0.5)
        self.confidence_threshold = config['model']['detection'].get('confidence_threshold', 0.5)
    
    def _build_backbone(self, config: Dict) -> nn.Module:
        """Construye el backbone según configuración"""
        backbone_name = config['model']['backbone']
        
        if backbone_name == 'video_swin':
            from .backbones.video_swin import video_swin_tiny, video_swin_small, video_swin_base
            
            variant = config['model'].get('backbone_variant', 'tiny')
            if variant == 'tiny':
                backbone = video_swin_tiny(num_classes=0)
            elif variant == 'small':
                backbone = video_swin_small(num_classes=0)
            elif variant == 'base':
                backbone = video_swin_base(num_classes=0)
            else:
                raise ValueError(f"Unknown Video Swin variant: {variant}")
        
        elif backbone_name == 'timesformer':
            # TODO: Implementar cuando tengamos TimeSformer
            raise NotImplementedError("TimeSformer no implementado aún")
        
        elif backbone_name == 'slowfast':
            # TODO: Implementar cuando tengamos SlowFast
            raise NotImplementedError("SlowFast no implementado aún")
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        return backbone
    
    def forward(
        self,
        videos: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            videos: (B, C, T, H, W) - Batch de videos
            targets: Lista de targets (para training)
        
        Returns:
            Dict con outputs del modelo
        """
        B, C, T, H, W = videos.shape
        
        # 1. Extract features con backbone
        features = self.backbone.get_feature_maps(videos)  # (B, T', H', W', D)
        
        # Reshape para processing temporal: (B, T', H', W', D) -> (B, T', D)
        # Aplicar spatial pooling
        B, Tp, Hp, Wp, D = features.shape
        features_spatial_pooled = features.mean(dim=(2, 3))  # (B, T', D)
        
        # 2. Generate proposals
        proposal_output = self.proposal_net(
            features_spatial_pooled,
            return_intermediate=True
        )
        proposals = proposal_output['proposals']
        start_probs = proposal_output['start_probs']
        end_probs = proposal_output['end_probs']
        
        # 3. Refine boundaries
        start_probs_refined, end_probs_refined, start_offsets, end_offsets = \
            self.boundary_detector(features_spatial_pooled)
        
        # Refinar proposals usando offsets
        refined_proposals = []
        for batch_proposals in proposals:
            refined = self.boundary_detector.refine_boundaries(
                batch_proposals,
                start_probs_refined,
                end_probs_refined,
                start_offsets,
                end_offsets
            )
            refined_proposals.append(refined)
        
        # 4. Classify proposals
        classifications = self.classifier.classify_proposals(
            features_spatial_pooled,
            refined_proposals
        )
        
        # Stack classifications into batched tensor for loss computation
        # classifications is List[Tensor(N_i, num_classes)] -> (B, max_N, num_classes)
        max_proposals = max(len(cls) for cls in classifications)
        batch_size = len(classifications)
        num_classes = classifications[0].shape[1] if classifications else self.num_classes
        
        # Pad shorter sequences with zeros
        padded_classifications = []
        for cls in classifications:
            if len(cls) < max_proposals:
                padding = torch.zeros(max_proposals - len(cls), num_classes, device=cls.device)
                cls = torch.cat([cls, padding], dim=0)
            padded_classifications.append(cls)
        
        # Stack into batch tensor
        class_logits_batched = torch.stack(padded_classifications, dim=0)  # (B, max_N, num_classes)
        
        # Preparar output
        if self.training and targets is not None:
            # Modo training: retornar raw outputs para que loss_fn externa los procese
            return {
                'class_logits': class_logits_batched,  # (B, max_N, num_classes) tensor batched
                'start_probs': start_probs_refined,  # (B, T)
                'end_probs': end_probs_refined,  # (B, T)
                'proposals': refined_proposals,  # List[Tensor] - proposals por batch
                'start_offsets': start_offsets,  # (B, T)
                'end_offsets': end_offsets  # (B, T)
            }
        else:
            # Modo inference: retornar detecciones
            detections = self.post_process(
                proposals=refined_proposals,
                classifications=classifications
            )
            
            return {
                'detections': detections,
                'proposals': refined_proposals,
                'classifications': classifications,
                'start_probs': start_probs_refined,
                'end_probs': end_probs_refined
            }
    
    def compute_losses(
        self,
        proposals: List[torch.Tensor],
        classifications: List[torch.Tensor],
        start_probs: torch.Tensor,
        end_probs: torch.Tensor,
        start_offsets: torch.Tensor,
        end_offsets: torch.Tensor,
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula losses para training
        
        Esta es una implementación simplificada.
        La versión completa estará en training/losses.py
        """
        # TODO: Implementar losses completas
        # Por ahora, placeholder
        
        total_loss = torch.tensor(0.0, device=proposals[0].device, requires_grad=True)
        
        losses = {
            'loss_classification': total_loss * 0.0,
            'loss_boundary': total_loss * 0.0,
            'loss_regression': total_loss * 0.0,
            'loss_total': total_loss
        }
        
        return losses
    
    def post_process(
        self,
        proposals: List[torch.Tensor],
        classifications: List[torch.Tensor]
    ) -> List[List[Dict]]:
        """
        Post-procesa proposals y clasificaciones
        
        1. Combina proposal coordinates + classification
        2. Aplica NMS temporal
        3. Filtra por confidence threshold
        
        Args:
            proposals: List[Tensor(N, 3)] - [start, end, score]
            classifications: List[Tensor(N, num_classes)]
        
        Returns:
            detections: List[List[Dict]] - Detecciones por batch
        """
        all_detections = []
        
        for batch_proposals, batch_classifications in zip(proposals, classifications):
            detections = []
            
            if len(batch_proposals) == 0:
                all_detections.append(detections)
                continue
            
            # Obtener clase y confidence
            class_probs = torch.softmax(batch_classifications, dim=1)
            class_confidences, class_ids = class_probs.max(dim=1)
            
            # Combinar con proposals
            for i in range(len(batch_proposals)):
                prop = batch_proposals[i]
                class_id = class_ids[i].item()
                confidence = class_confidences[i].item()
                
                # Filtrar por confidence
                if confidence < self.confidence_threshold:
                    continue
                
                detection = {
                    'start': prop[0].item(),
                    'end': prop[1].item(),
                    'class_id': class_id,
                    'confidence': confidence,
                    'proposal_score': prop[2].item()
                }
                
                detections.append(detection)
            
            # Aplicar NMS temporal
            detections = self._temporal_nms(detections, self.nms_threshold)
            
            all_detections.append(detections)
        
        return all_detections
    
    def _temporal_nms(
        self,
        detections: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """
        Non-Maximum Suppression temporal
        
        Args:
            detections: Lista de detecciones
            threshold: IoU threshold
        
        Returns:
            filtered_detections: Detecciones después de NMS
        """
        if len(detections) <= 1:
            return detections
        
        # Ordenar por confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            # Tomar mejor detección
            best = detections.pop(0)
            keep.append(best)
            
            # Filtrar overlapping
            filtered = []
            for det in detections:
                iou = self._compute_iou(best, det)
                if iou < threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep
    
    def _compute_iou(self, det1: Dict, det2: Dict) -> float:
        """Calcula IoU temporal entre dos detecciones"""
        # Intersection
        inter_start = max(det1['start'], det2['start'])
        inter_end = min(det1['end'], det2['end'])
        inter = max(0, inter_end - inter_start)
        
        # Union
        union_start = min(det1['start'], det2['start'])
        union_end = max(det1['end'], det2['end'])
        union = union_end - union_start
        
        return inter / union if union > 0 else 0.0
    
    def detect_video(
        self,
        video: torch.Tensor,
        return_features: bool = False
    ) -> Dict:
        """
        Detecta señas en un video completo
        
        Args:
            video: (1, C, T, H, W) - Video individual
            return_features: Si retornar features intermedias
        
        Returns:
            Dict con detecciones y opcionalmente features
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(video, targets=None)
        
        detections = outputs['detections'][0]  # Primer (y único) batch
        
        result = {
            'detections': detections,
            'num_detections': len(detections)
        }
        
        if return_features:
            result['proposals'] = outputs['proposals'][0]
            result['start_probs'] = outputs['start_probs']
            result['end_probs'] = outputs['end_probs']
        
        return result


def build_model(config) -> TemporalActionDetector:
    """
    Construye modelo desde configuración
    
    Args:
        config: Dict de configuración o ruta a config.yaml
    
    Returns:
        model: TemporalActionDetector
    """
    # Si config es un string, cargarlo desde archivo
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    model = TemporalActionDetector(config)
    
    return model


def load_pretrained(
    checkpoint_path: str,
    config_path: str,
    device: str = 'cuda'
) -> TemporalActionDetector:
    """
    Carga modelo pre-entrenado
    
    Args:
        checkpoint_path: Ruta al checkpoint
        config_path: Ruta a config.yaml
        device: 'cuda' o 'cpu'
    
    Returns:
        model: Modelo cargado
    """
    # Construir modelo
    model = build_model(config_path)
    
    # Cargar pesos
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == '__main__':
    # Test del modelo completo
    print("Testing Complete Temporal Action Detector...")
    
    # Crear config dummy
    config = {
        'model': {
            'backbone': 'video_swin',
            'backbone_variant': 'tiny',
            'num_classes': 24,
            'feature_dim': 768,
            'detection': {
                'proposal_hidden_dim': 256,
                'boundary_hidden_dim': 256,
                'classifier_hidden_dim': 512,
                'num_pyramid_levels': 4,
                'use_enhanced_boundary': True,
                'use_attention': True
            }
        },
        'detection': {
            'proposal_lengths': [8, 16, 32, 64],
            'min_proposal_score': 0.1,
            'max_proposals_per_video': 100,
            'nms_iou_threshold': 0.5,
            'confidence_threshold': 0.3
        },
        'training': {
            'dropout': 0.5
        }
    }
    
    # Crear modelo
    model = TemporalActionDetector(config)
    model.eval()
    
    # Input dummy
    batch_size = 2
    video = torch.randn(batch_size, 3, 64, 224, 224)
    
    print(f"Input video shape: {video.shape}")
    
    # Forward pass (inference)
    with torch.no_grad():
        outputs = model(video)
    
    print(f"\nOutputs:")
    print(f"  Number of detections:")
    for i, dets in enumerate(outputs['detections']):
        print(f"    Batch {i}: {len(dets)} detecciones")
        
        if len(dets) > 0:
            print(f"    Primera detección:")
            det = dets[0]
            print(f"      Start: {det['start']:.1f}")
            print(f"      End: {det['end']:.1f}")
            print(f"      Class ID: {det['class_id']}")
            print(f"      Confidence: {det['confidence']:.3f}")
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNúmero total de parámetros: {num_params:,}")
    
    # Desglose por componente
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    proposal_params = sum(p.numel() for p in model.proposal_net.parameters())
    boundary_params = sum(p.numel() for p in model.boundary_detector.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"\nDesglose de parámetros:")
    print(f"  Backbone: {backbone_params:,} ({backbone_params/num_params*100:.1f}%)")
    print(f"  Proposal Network: {proposal_params:,} ({proposal_params/num_params*100:.1f}%)")
    print(f"  Boundary Detector: {boundary_params:,} ({boundary_params/num_params*100:.1f}%)")
    print(f"  Classifier: {classifier_params:,} ({classifier_params/num_params*100:.1f}%)")
    
    print("\n✓ Complete model test completado")
