import torch
from torch import nn

from .input_feature import InputFeature


class AttacksV3KNP(InputFeature):
    """Model-side counterpart of the self-owned encoder (threat_encoder/,
    ENCODER_VERSION "AttacksV3KNP"). The C++ dataloader emits these via the
    CstThreatsExtractor ("AttacksV3KNP"); this owns the matching first-layer weights.

    Two-block feature space (see threat_encoder/ATTACKS_V3_KNP_SCOPE.md):
      * CONTROL (K/N/P): dense control maps, indices [0, 384). At EXPORT these fold
        into the HalfKA placement rows (free at inference) -- but during TRAINING they
        are ordinary sparse features here, learned like any other.
      * THREAT  (B/R/Q): sparse on-occupied threats, indices [384, NUM_INPUTS).

    Flat sparse feature, no virtual/factorized weights (coalesce is a no-op, export
    weights identity) -- the control->placement fold is done in the trainer's export.py,
    not here.

    IMPORTANT -- must stay in lockstep with the C++ encoder:
      * NUM_INPUTS          == threats::ThreatTables::NUM_INPUTS (build_tables)
      * MAX_ACTIVE_FEATURES == CstThreatsExtractor MAX_ACTIVE
    Any change to the C++ threat_config knobs changes NUM_INPUTS -> bump the version
    name on BOTH sides and retrain; nets are permanently tied to an encoder version.
    """

    HASH = 0x41563450  # "AV4P"; AttacksV4 (KNP control folded + QRB DENSE control). AV3P superseded.
    FEATURE_NAME = "AttacksV3KNP"        # loader/registry key kept as-is (opaque); content is now V4
    INPUT_FEATURE_NAME = "AttacksV3KNP"  # <- the exact string handed to the C++ loader
    MAX_ACTIVE_FEATURES = 256            # matches CstThreatsExtractor MAX_ACTIVE (KNP + QRB dense control)

    NUM_INPUTS = 640           # CONTROL 384 (KNP, folds) + SLIDER_CONTROL 256 (QRB dense, live)
    NUM_REAL_FEATURES = 640
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
        """Control/threat features carry no piece values, so PSQT columns are zero."""
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
        """Clamp weights to the quantization-safe range."""
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
