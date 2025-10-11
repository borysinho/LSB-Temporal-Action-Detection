"""
Boundary Detector

Detecta con precisión los límites (inicio y fin) de las señas.
Utiliza regresión para refinamiento sub-frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BoundaryDetector(nn.Module):
    """
    Boundary Detector con refinamiento por regresión
    
    Detecta boundaries (inicio/fin) con dos componentes:
    1. Classification: Probabilidad de ser boundary
    2. Regression: Offset para refinamiento preciso
    
    Esto permite precisión sub-frame en la detección de boundaries.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_conv_layers: int = 3,
        kernel_sizes: list = [3, 5, 7]
    ):
        """
        Args:
            in_channels: Canales de entrada (del backbone)
            hidden_channels: Canales ocultos
            num_conv_layers: Número de capas convolucionales
            kernel_sizes: Tamaños de kernel para multi-scale
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        # Multi-scale temporal convolutions
        self.multi_scale_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layers = []
            for i in range(num_conv_layers):
                conv_layers.extend([
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(inplace=True)
                ])
            self.multi_scale_convs.append(nn.Sequential(*conv_layers))
        
        # Fusion layer
        fusion_in_channels = hidden_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Start boundary heads
        self.start_cls_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
        )
        
        self.start_reg_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
        )
        
        # End boundary heads
        self.end_cls_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
        )
        
        self.end_reg_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            features: (B, T, D) - Features del backbone
        
        Returns:
            start_probs: (B, 1, T) - Probabilidades de inicio
            end_probs: (B, 1, T) - Probabilidades de fin
            start_offsets: (B, 1, T) - Offsets de regresión para inicio
            end_offsets: (B, 1, T) - Offsets de regresión para fin
        """
        # Reshape: (B, T, D) -> (B, D, T)
        B, T, D = features.shape
        x = features.permute(0, 2, 1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Multi-scale processing
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # Fuse multi-scale features
        x = torch.cat(multi_scale_features, dim=1)
        x = self.fusion(x)
        
        # Start boundary prediction
        start_logits = self.start_cls_head(x)
        start_probs = torch.sigmoid(start_logits)
        start_offsets = self.start_reg_head(x)
        
        # End boundary prediction
        end_logits = self.end_cls_head(x)
        end_probs = torch.sigmoid(end_logits)
        end_offsets = self.end_reg_head(x)
        
        return start_probs, end_probs, start_offsets, end_offsets
    
    def refine_boundaries(
        self,
        proposals: torch.Tensor,
        start_probs: torch.Tensor,
        end_probs: torch.Tensor,
        start_offsets: torch.Tensor,
        end_offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Refina boundaries de proposals usando regresión
        
        Args:
            proposals: (N, 3) [start, end, score]
            start_probs: (B, 1, T)
            end_probs: (B, 1, T)
            start_offsets: (B, 1, T)
            end_offsets: (B, 1, T)
        
        Returns:
            refined_proposals: (N, 3) con boundaries refinados
        """
        refined = proposals.clone()
        
        for i in range(len(proposals)):
            start_idx = int(proposals[i, 0])
            end_idx = int(proposals[i, 1])
            
            # Aplicar offset de regresión
            if start_idx < start_offsets.shape[2]:
                start_offset = start_offsets[0, 0, start_idx].item()
                refined[i, 0] = max(0, proposals[i, 0] + start_offset)
            
            if end_idx < end_offsets.shape[2]:
                end_offset = end_offsets[0, 0, end_idx].item()
                refined[i, 1] = proposals[i, 1] + end_offset
        
        return refined
    
    def detect_boundaries(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[list, list]:
        """
        Detecta boundaries como peaks en las probabilidades
        
        Args:
            features: (B, T, D)
            threshold: Threshold mínimo de probabilidad
        
        Returns:
            start_positions: List de posiciones de inicio por batch
            end_positions: List de posiciones de fin por batch
        """
        start_probs, end_probs, start_offsets, end_offsets = self.forward(features)
        
        B = start_probs.shape[0]
        start_positions = []
        end_positions = []
        
        for b in range(B):
            start_p = start_probs[b, 0].cpu().numpy()
            end_p = end_probs[b, 0].cpu().numpy()
            
            # Encontrar peaks
            starts = self._find_peaks(start_p, threshold)
            ends = self._find_peaks(end_p, threshold)
            
            start_positions.append(starts)
            end_positions.append(ends)
        
        return start_positions, end_positions
    
    def _find_peaks(self, signal, threshold):
        """Encuentra peaks en señal 1D"""
        peaks = []
        
        for i in range(1, len(signal) - 1):
            # Es un peak si es mayor que sus vecinos y supera threshold
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        
        return peaks


class TemporalContextModule(nn.Module):
    """
    Módulo de contexto temporal
    
    Captura contexto largo para mejorar detección de boundaries.
    Usa dilated convolutions para capturar patrones temporales largos.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        dilation_rates: list = [1, 2, 4, 8]
    ):
        super().__init__()
        
        self.dilated_convs = nn.ModuleList()
        for dilation in dilation_rates:
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * len(dilation_rates), out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        
        Returns:
            out: (B, C, T) con contexto temporal
        """
        outputs = []
        for conv in self.dilated_convs:
            outputs.append(conv(x))
        
        out = torch.cat(outputs, dim=1)
        out = self.fusion(out)
        
        return out


class EnhancedBoundaryDetector(BoundaryDetector):
    """
    Versión mejorada del Boundary Detector con contexto temporal
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_conv_layers: int = 3,
        kernel_sizes: list = [3, 5, 7],
        use_temporal_context: bool = True
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_conv_layers=num_conv_layers,
            kernel_sizes=kernel_sizes
        )
        
        # Temporal context module
        if use_temporal_context:
            self.temporal_context = TemporalContextModule(
                in_channels=hidden_channels,
                out_channels=hidden_channels
            )
        else:
            self.temporal_context = None
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            features: (B, T, D)
        
        Returns:
            start_probs, end_probs, start_offsets, end_offsets
        """
        # Reshape: (B, T, D) -> (B, D, T)
        B, T, D = features.shape
        x = features.permute(0, 2, 1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Temporal context
        if self.temporal_context is not None:
            x = x + self.temporal_context(x)  # Residual connection
        
        # Multi-scale processing
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # Fuse
        x = torch.cat(multi_scale_features, dim=1)
        x = self.fusion(x)
        
        # Predictions
        start_logits = self.start_cls_head(x)
        start_probs = torch.sigmoid(start_logits)
        start_offsets = self.start_reg_head(x)
        
        end_logits = self.end_cls_head(x)
        end_probs = torch.sigmoid(end_logits)
        end_offsets = self.end_reg_head(x)
        
        return start_probs, end_probs, start_offsets, end_offsets


if __name__ == '__main__':
    # Test
    print("Testing Boundary Detector...")
    
    # Crear modelo básico
    model_basic = BoundaryDetector(
        in_channels=768,
        hidden_channels=256,
        kernel_sizes=[3, 5, 7]
    )
    
    # Input dummy
    batch_size = 2
    T = 64
    D = 768
    features = torch.randn(batch_size, T, D)
    
    print(f"Input features shape: {features.shape}")
    
    # Forward pass
    start_probs, end_probs, start_offsets, end_offsets = model_basic(features)
    
    print(f"\nOutputs (Basic):")
    print(f"  Start probs shape: {start_probs.shape}")
    print(f"  End probs shape: {end_probs.shape}")
    print(f"  Start offsets shape: {start_offsets.shape}")
    print(f"  End offsets shape: {end_offsets.shape}")
    
    # Test detección de boundaries
    start_pos, end_pos = model_basic.detect_boundaries(features, threshold=0.5)
    print(f"\nBoundaries detectados (batch 0):")
    print(f"  Start positions: {start_pos[0][:5] if len(start_pos[0]) > 0 else []}")
    print(f"  End positions: {end_pos[0][:5] if len(end_pos[0]) > 0 else []}")
    
    # Test modelo mejorado
    model_enhanced = EnhancedBoundaryDetector(
        in_channels=768,
        hidden_channels=256,
        use_temporal_context=True
    )
    
    start_probs2, end_probs2, start_offsets2, end_offsets2 = model_enhanced(features)
    
    print(f"\nOutputs (Enhanced):")
    print(f"  Start probs shape: {start_probs2.shape}")
    
    # Contar parámetros
    num_params_basic = sum(p.numel() for p in model_basic.parameters())
    num_params_enhanced = sum(p.numel() for p in model_enhanced.parameters())
    
    print(f"\nNúmero de parámetros:")
    print(f"  Basic: {num_params_basic:,}")
    print(f"  Enhanced: {num_params_enhanced:,}")
    
    print("\n✓ Boundary Detector test completado")
