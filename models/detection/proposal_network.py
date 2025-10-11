"""
Temporal Proposal Network

Genera propuestas temporales (candidatos de señas) a partir de features del backbone.
Inspirado en Boundary Matching Network (BMN) y ActionFormer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class FeaturePyramid(nn.Module):
    """
    Feature Pyramid Network para detección multi-escala
    
    Genera features a múltiples escalas temporales para detectar
    señas de diferentes duraciones.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 256,
        num_levels: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # Lateral connections (1x1 conv para reducir canales)
        self.lateral_convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            for _ in range(num_levels)
        ])
        
        # Output convs (3x1 conv para refinar)
        self.output_convs = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])
        
        # Downsampling convs para crear pirámide
        self.downsample_convs = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) - Features del backbone
        
        Returns:
            pyramid: List[(B, C_out, T_i)] - Features multi-escala
        """
        # Primera escala (más fina)
        x = self.lateral_convs[0](x)
        pyramid = [self.output_convs[0](x)]
        
        # Escalas subsiguientes (downsampling)
        current = x
        for i in range(1, self.num_levels):
            current = self.downsample_convs[i - 1](current)
            lateral = self.lateral_convs[i](current)
            output = self.output_convs[i](lateral)
            pyramid.append(output)
        
        return pyramid


class BoundaryMatchingModule(nn.Module):
    """
    Boundary Matching Module
    
    Predice probabilidades de inicio y fin en cada posición temporal.
    Luego combina estos para generar proposals.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256
    ):
        super().__init__()
        
        # Start boundary predictor
        self.start_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1)
        )
        
        # End boundary predictor
        self.end_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        
        Returns:
            start_probs: (B, 1, T) - Probabilidades de inicio
            end_probs: (B, 1, T) - Probabilidades de fin
        """
        start_logits = self.start_conv(x)
        end_logits = self.end_conv(x)
        
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        
        return start_probs, end_probs


class ProposalGenerator(nn.Module):
    """
    Genera proposals combinando start y end boundaries
    
    Combina las probabilidades de inicio/fin para generar
    proposals (segmentos temporales candidatos).
    """
    
    def __init__(
        self,
        proposal_lengths: List[int] = [8, 16, 32, 64],
        min_score: float = 0.1,
        max_proposals: int = 100
    ):
        super().__init__()
        self.proposal_lengths = proposal_lengths
        self.min_score = min_score
        self.max_proposals = max_proposals
    
    def forward(
        self,
        start_probs: torch.Tensor,
        end_probs: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Genera proposals a partir de start/end probabilities
        
        Args:
            start_probs: (B, 1, T)
            end_probs: (B, 1, T)
        
        Returns:
            proposals: List[Tensor(N_i, 3)] donde cada tensor es [start, end, score]
        """
        batch_size = start_probs.shape[0]
        T = start_probs.shape[2]
        
        batch_proposals = []
        
        for b in range(batch_size):
            start_p = start_probs[b, 0].cpu().numpy()
            end_p = end_probs[b, 0].cpu().numpy()
            
            proposals = []
            
            # Para cada longitud de proposal
            for length in self.proposal_lengths:
                # Generar proposals de esta longitud
                for start_idx in range(T - length + 1):
                    end_idx = start_idx + length
                    
                    # Score como producto de start y end probs
                    start_score = start_p[start_idx]
                    end_score = end_p[end_idx - 1]
                    score = np.sqrt(start_score * end_score)  # Media geométrica
                    
                    if score >= self.min_score:
                        proposals.append([start_idx, end_idx, score])
            
            # Ordenar por score
            proposals = sorted(proposals, key=lambda x: x[2], reverse=True)
            
            # Tomar top-k
            proposals = proposals[:self.max_proposals]
            
            # Convertir a tensor
            if len(proposals) > 0:
                proposals_tensor = torch.tensor(proposals, dtype=torch.float32)
            else:
                # Si no hay proposals, crear uno dummy
                proposals_tensor = torch.tensor([[0, T, 0.0]], dtype=torch.float32)
            
            batch_proposals.append(proposals_tensor)
        
        return batch_proposals


class TemporalProposalNetwork(nn.Module):
    """
    Temporal Proposal Network completa
    
    Combina Feature Pyramid, Boundary Matching y Proposal Generation
    para generar propuestas temporales de alta calidad.
    
    Pipeline:
    1. Feature Pyramid: Genera features multi-escala
    2. Boundary Matching: Predice start/end en cada escala
    3. Proposal Generation: Combina boundaries para generar proposals
    4. Scoring: Asigna confidence a cada proposal
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_pyramid_levels: int = 4,
        proposal_lengths: List[int] = [8, 16, 32, 64],
        min_score: float = 0.1,
        max_proposals: int = 100
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Feature Pyramid
        self.fpn = FeaturePyramid(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_levels=num_pyramid_levels
        )
        
        # Boundary Matching para cada nivel de la pirámide
        self.boundary_modules = nn.ModuleList([
            BoundaryMatchingModule(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels
            )
            for _ in range(num_pyramid_levels)
        ])
        
        # Proposal Generator
        self.proposal_generator = ProposalGenerator(
            proposal_lengths=proposal_lengths,
            min_score=min_score,
            max_proposals=max_proposals
        )
        
        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, D) - Features del backbone
            return_intermediate: Si retornar outputs intermedios
        
        Returns:
            Dict con:
                - proposals: List[Tensor(N, 3)] - [start, end, score]
                - start_probs: Tensor(B, 1, T) - Probabilidades de inicio
                - end_probs: Tensor(B, 1, T) - Probabilidades de fin
                - confidence_map: Tensor(B, 1, T) - Mapa de confianza
        """
        # Reshape para convolutions: (B, T, D) -> (B, D, T)
        B, T, D = features.shape
        features = features.permute(0, 2, 1)  # (B, D, T)
        
        # Feature Pyramid
        pyramid_features = self.fpn(features)
        
        # Boundary prediction en cada nivel
        all_start_probs = []
        all_end_probs = []
        
        for i, (feat, boundary_module) in enumerate(zip(pyramid_features, self.boundary_modules)):
            start_p, end_p = boundary_module(feat)
            
            # Upsample a resolución original si es necesario
            if feat.shape[2] != T:
                start_p = F.interpolate(start_p, size=T, mode='linear', align_corners=False)
                end_p = F.interpolate(end_p, size=T, mode='linear', align_corners=False)
            
            all_start_probs.append(start_p)
            all_end_probs.append(end_p)
        
        # Combinar predicciones de múltiples escalas (promedio)
        start_probs = torch.stack(all_start_probs, dim=0).mean(dim=0)
        end_probs = torch.stack(all_end_probs, dim=0).mean(dim=0)
        
        # Generar proposals
        proposals = self.proposal_generator(start_probs, end_probs)
        
        # Mapa de confianza
        confidence_map = self.confidence_head(pyramid_features[0])
        if confidence_map.shape[2] != T:
            confidence_map = F.interpolate(confidence_map, size=T, mode='linear', align_corners=False)
        
        output = {
            'proposals': proposals,
            'start_probs': start_probs,
            'end_probs': end_probs,
            'confidence_map': confidence_map
        }
        
        if return_intermediate:
            output['pyramid_features'] = pyramid_features
            output['all_start_probs'] = all_start_probs
            output['all_end_probs'] = all_end_probs
        
        return output
    
    def generate_proposals_with_nms(
        self,
        features: torch.Tensor,
        nms_threshold: float = 0.5
    ) -> List[torch.Tensor]:
        """
        Genera proposals y aplica NMS temporal
        
        Args:
            features: (B, T, D)
            nms_threshold: Threshold de IoU para NMS
        
        Returns:
            proposals: List[Tensor(N, 3)] después de NMS
        """
        output = self.forward(features)
        proposals = output['proposals']
        
        # Aplicar NMS a cada batch
        nms_proposals = []
        for props in proposals:
            if len(props) > 1:
                props = self._temporal_nms(props, nms_threshold)
            nms_proposals.append(props)
        
        return nms_proposals
    
    def _temporal_nms(
        self,
        proposals: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Non-Maximum Suppression temporal
        
        Args:
            proposals: (N, 3) [start, end, score]
            threshold: IoU threshold
        
        Returns:
            filtered_proposals: Proposals después de NMS
        """
        if len(proposals) == 0:
            return proposals
        
        # Ordenar por score
        scores = proposals[:, 2]
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            # Tomar el de mayor score
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Calcular IoU con el resto
            current_prop = proposals[current_idx]
            rest_props = proposals[sorted_indices[1:]]
            
            ious = self._compute_temporal_iou(
                current_prop[None, :2],
                rest_props[:, :2]
            )
            
            # Filtrar los que tienen IoU < threshold
            mask = ious < threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return proposals[keep]
    
    def _compute_temporal_iou(
        self,
        proposals1: torch.Tensor,
        proposals2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula IoU temporal entre proposals
        
        Args:
            proposals1: (N, 2) [start, end]
            proposals2: (M, 2) [start, end]
        
        Returns:
            iou: (N, M) o (M,) si N=1
        """
        # Intersection
        inter_start = torch.max(proposals1[:, 0:1], proposals2[:, 0])
        inter_end = torch.min(proposals1[:, 1:2], proposals2[:, 1])
        inter = torch.clamp(inter_end - inter_start, min=0)
        
        # Union
        union_start = torch.min(proposals1[:, 0:1], proposals2[:, 0])
        union_end = torch.max(proposals1[:, 1:2], proposals2[:, 1])
        union = union_end - union_start
        
        # IoU
        iou = inter / (union + 1e-8)
        
        if iou.shape[0] == 1:
            iou = iou.squeeze(0)
        
        return iou


if __name__ == '__main__':
    # Test
    print("Testing Temporal Proposal Network...")
    
    # Crear modelo
    model = TemporalProposalNetwork(
        in_channels=768,
        hidden_channels=256,
        num_pyramid_levels=4,
        proposal_lengths=[8, 16, 32, 64],
        max_proposals=100
    )
    
    # Input dummy (features del backbone)
    batch_size = 2
    T = 64
    D = 768
    features = torch.randn(batch_size, T, D)
    
    print(f"Input features shape: {features.shape}")
    
    # Forward pass
    output = model(features, return_intermediate=True)
    
    print(f"\nOutputs:")
    print(f"  Proposals batch 0: {output['proposals'][0].shape}")
    print(f"  Proposals batch 1: {output['proposals'][1].shape}")
    print(f"  Start probs shape: {output['start_probs'].shape}")
    print(f"  End probs shape: {output['end_probs'].shape}")
    print(f"  Confidence map shape: {output['confidence_map'].shape}")
    
    # Mostrar algunos proposals
    print(f"\nEjemplo de proposals (batch 0):")
    props = output['proposals'][0][:5]
    for i, prop in enumerate(props):
        print(f"  Proposal {i}: start={prop[0]:.1f}, end={prop[1]:.1f}, score={prop[2]:.3f}")
    
    # Test con NMS
    nms_proposals = model.generate_proposals_with_nms(features, nms_threshold=0.5)
    print(f"\nProposals después de NMS:")
    print(f"  Batch 0: {len(nms_proposals[0])} proposals")
    print(f"  Batch 1: {len(nms_proposals[1])} proposals")
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNúmero de parámetros: {num_params:,}")
    
    print("\n✓ Temporal Proposal Network test completado")
