import torch
from torch import nn

from .input_feature import InputFeature


class ThreatsV1(InputFeature):
    """Model-side counterpart of the self-owned threat encoder (threat_encoder/,
    ENCODER_VERSION "ThreatsV1"). The C++ dataloader emits these features via the
    CstThreatsExtractor ("ThreatsV1"); this owns the matching first-layer weights.

    Like Full_Threats it is a flat sparse feature with no virtual/factorized
    weights, so coalesce() is a no-op and export weights are identity.

    IMPORTANT — must stay in lockstep with the C++ encoder:
      * NUM_INPUTS          == threats::ThreatTables::NUM_INPUTS (selftest prints it)
      * MAX_ACTIVE_FEATURES == CstThreatsExtractor MAX_ACTIVE
    Any change to the C++ threat_config knobs changes NUM_INPUTS -> bump the
    version name on BOTH sides and retrain; old nets are incompatible.
    """

    HASH = 0x54563031  # "TV01"; distinct per feature, tags the exported net
    FEATURE_NAME = "ThreatsV1"
    INPUT_FEATURE_NAME = "ThreatsV1"  # <- the exact string handed to the C++ loader
    MAX_ACTIVE_FEATURES = 256

    NUM_INPUTS = 75320
    NUM_REAL_FEATURES = 75320
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Threats carry no piece values, so PSQT columns are zero."""
        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    def clip_weights(self, quantization) -> None:
        """Clamp threat weights to the quantization-safe range."""
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
