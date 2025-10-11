"""
Action Classifier

Clasifica cada proposal temporal en una de las clases de señas.
Usa temporal pooling y attention para clasificación robusta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalPooling(nn.Module):
    """
    Temporal Pooling Module
    
    Combina multiple pooling strategies para capturar
    información completa de la seña.
    """
    
    def __init__(self, pool_types: list = ['avg', 'max']):
        super().__init__()
        self.pool_types = pool_types
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        
        Returns:
            pooled: (B, C * num_pool_types)
        """
        pooled_features = []
        
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pooled = x.mean(dim=1)
            elif pool_type == 'max':
                pooled = x.max(dim=1)[0]
            elif pool_type == 'first':
                pooled = x[:, 0, :]
            elif pool_type == 'last':
                pooled = x[:, -1, :]
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")
            
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=1)


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention Module
    
    Permite al modelo enfocarse en frames importantes de la seña.
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        
        Returns:
            out: (B, T, C)
        """
        B, T, C = x.shape
        
        # Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out


class ActionClassifier(nn.Module):
    """
    Action Classifier para proposals temporales
    
    Clasifica cada proposal en una de las clases de señas.
    Usa attention temporal y multiple pooling para robustez.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 512,
        num_classes: int = 24,
        use_attention: bool = True,
        pool_types: list = ['avg', 'max'],
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: Dimensión de features de entrada
            hidden_channels: Dimensión oculta
            num_classes: Número de clases de señas
            use_attention: Si usar temporal attention
            pool_types: Tipos de pooling a usar
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temporal attention
        if use_attention:
            self.attention = TemporalAttention(hidden_channels, num_heads=8)
            self.attn_norm = nn.LayerNorm(hidden_channels)
        
        # Temporal pooling
        self.pooling = TemporalPooling(pool_types=pool_types)
        pooled_dim = hidden_channels * len(pool_types)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) - Features de un proposal
        
        Returns:
            logits: (B, num_classes)
        """
        # Input projection
        x = self.input_proj(features)
        
        # Temporal attention
        if self.use_attention:
            attn_out = self.attention(x)
            x = x + attn_out  # Residual
            x = self.attn_norm(x)
        
        # Temporal pooling
        x = self.pooling(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def extract_proposal_features(
        self,
        full_features: torch.Tensor,
        proposals: torch.Tensor
    ) -> torch.Tensor:
        """
        Extrae features para un conjunto de proposals
        
        Args:
            full_features: (B, T_full, D) - Features completas del video
            proposals: (N, 3) - [start, end, score]
        
        Returns:
            proposal_features: (N, T_prop, D)
        """
        proposal_feats = []
        
        for prop in proposals:
            start = int(prop[0])
            end = int(prop[1])
            
            # Extraer features del proposal
            feat = full_features[0, start:end, :]  # (T_prop, D)
            
            # Si el proposal está vacío, usar un frame dummy
            if feat.shape[0] == 0:
                feat = full_features[0, 0:1, :]
            
            proposal_feats.append(feat.unsqueeze(0))  # (1, T_prop, D)
        
        # No podemos hacer batch directamente porque proposals tienen diferentes longitudes
        # Retornamos lista
        return proposal_feats
    
    def classify_proposals(
        self,
        full_features: torch.Tensor,
        proposals: list
    ) -> list:
        """
        Clasifica múltiples proposals
        
        Args:
            full_features: (B, T, D)
            proposals: List[Tensor(N_i, 3)] - Proposals por batch
        
        Returns:
            classifications: List[Tensor(N_i, num_classes)]
        """
        all_classifications = []
        
        for b, batch_proposals in enumerate(proposals):
            if len(batch_proposals) == 0:
                all_classifications.append(torch.empty(0, self.num_classes))
                continue
            
            # Extraer features de cada proposal
            proposal_feats = self.extract_proposal_features(
                full_features[b:b+1],
                batch_proposals
            )
            
            # Clasificar cada proposal
            batch_logits = []
            for prop_feat in proposal_feats:
                logits = self.forward(prop_feat)  # (1, num_classes)
                batch_logits.append(logits)
            
            batch_logits = torch.cat(batch_logits, dim=0)  # (N, num_classes)
            all_classifications.append(batch_logits)
        
        return all_classifications


class LightweightActionClassifier(nn.Module):
    """
    Versión ligera del Action Classifier
    
    Usa solo pooling sin attention para mayor velocidad.
    Útil para streaming en tiempo real.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_classes: int = 24,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Simple pooling + classification
        self.pooling = TemporalPooling(pool_types=['avg', 'max'])
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D)
        
        Returns:
            logits: (B, num_classes)
        """
        x = self.pooling(features)
        logits = self.classifier(x)
        return logits


class MultiScaleActionClassifier(ActionClassifier):
    """
    Action Classifier con procesamiento multi-escala
    
    Captura patrones a diferentes escalas temporales
    antes de la clasificación.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 512,
        num_classes: int = 24,
        temporal_scales: list = [1, 2, 4],
        dropout: float = 0.5
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            use_attention=True,
            dropout=dropout
        )
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList()
        for scale in temporal_scales:
            self.temporal_convs.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=scale,
                    dilation=scale
                )
            )
        
        # Fusion después de multi-scale
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_channels * len(temporal_scales), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D)
        
        Returns:
            logits: (B, num_classes)
        """
        B, T, D = features.shape
        
        # Input projection
        x = self.input_proj(features)
        
        # Attention
        if self.use_attention:
            attn_out = self.attention(x)
            x = x + attn_out
            x = self.attn_norm(x)
        
        # Multi-scale processing
        x_transpose = x.permute(0, 2, 1)  # (B, C, T)
        multi_scale_feats = []
        
        for conv in self.temporal_convs:
            scale_feat = conv(x_transpose)  # (B, C, T)
            scale_feat = scale_feat.permute(0, 2, 1)  # (B, T, C)
            multi_scale_feats.append(scale_feat.mean(dim=1))  # (B, C)
        
        # Fuse scales
        multi_scale_feat = torch.cat(multi_scale_feats, dim=1)  # (B, C*scales)
        fused = self.scale_fusion(multi_scale_feat)  # (B, C)
        
        # También usar pooling normal
        pooled = self.pooling(x)  # (B, C*pool_types)
        
        # Combinar
        combined = fused + pooled.mean(dim=1, keepdim=True).expand_as(fused)
        
        # Classification (modificar para usar combined)
        logits = self.classifier[0](combined.unsqueeze(1))
        logits = logits.squeeze(1)
        
        # Resto de layers
        for layer in self.classifier[1:]:
            logits = layer(logits)
        
        return logits


if __name__ == '__main__':
    # Test
    print("Testing Action Classifier...")
    
    # Crear modelo
    model = ActionClassifier(
        in_channels=768,
        hidden_channels=512,
        num_classes=24,
        use_attention=True
    )
    
    # Input dummy (features de un proposal)
    batch_size = 4
    T_prop = 32  # Longitud del proposal
    D = 768
    features = torch.randn(batch_size, T_prop, D)
    
    print(f"Input features shape: {features.shape}")
    
    # Forward pass
    logits = model(features)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
    
    # Test con proposals
    full_features = torch.randn(2, 64, 768)  # Video completo
    proposals = [
        torch.tensor([[10, 30, 0.9], [40, 55, 0.8]]),  # Batch 0: 2 proposals
        torch.tensor([[5, 25, 0.85]])  # Batch 1: 1 proposal
    ]
    
    classifications = model.classify_proposals(full_features, proposals)
    print(f"\nClassifications:")
    print(f"  Batch 0: {classifications[0].shape}")
    print(f"  Batch 1: {classifications[1].shape}")
    
    # Test lightweight
    model_light = LightweightActionClassifier(
        in_channels=768,
        hidden_channels=256,
        num_classes=24
    )
    
    logits_light = model_light(features)
    print(f"\nLightweight output shape: {logits_light.shape}")
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters())
    num_params_light = sum(p.numel() for p in model_light.parameters())
    
    print(f"\nNúmero de parámetros:")
    print(f"  Standard: {num_params:,}")
    print(f"  Lightweight: {num_params_light:,}")
    
    print("\n✓ Action Classifier test completado")
