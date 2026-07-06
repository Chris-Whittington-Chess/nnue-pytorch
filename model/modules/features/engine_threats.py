import torch
from torch import nn

from .input_feature import InputFeature


class EngineThreats(InputFeature):
    """Model-side counterpart of the self-owned EngineThreats encoder (threat_encoder/,
    encode_engine_threat + threat_buildpin.h). The C++ dataloader emits these via the
    EngineThreatsExtractor ("EngineThreats"); this owns the matching first-layer weights.

    A compact, victim-keyed threat block read from the engine's BuildPinData bitboards
    (threats[] / lesserthreats[]) -- the value-based + count-based tactical signal the engine
    already computes during search, for free. Recovers the K/N/P attacker x victim conjunction
    signal that AttacksV3KNP's fold drops, without giving up the +9% NPS.

      index = (block*2 + victim_side) * (6*64) + victim_type*64 + orient(square)
        block 0 = value-threat (piece attacked by a strictly cheaper enemy)
        block 1 = loose        (own piece attacked but under-defended)
      NUM_INPUTS = 2 blocks * 2 sides * 6 types * 64 squares = 1536

    Flat sparse feature, no virtual/factorized weights, no fold (it is a genuine LIVE feature).

    IMPORTANT -- must stay in lockstep with the C++ encoder:
      * NUM_INPUTS          == threats::ENGINE_THREAT_INPUTS
      * MAX_ACTIVE_FEATURES == EngineThreatsExtractor MAX_ACTIVE_ET
    """

    HASH = 0x45543031  # "ET01"; tags the exported net, distinct from AttacksV3KNP (0x41563350)
    FEATURE_NAME = "EngineThreats"
    INPUT_FEATURE_NAME = "EngineThreats"  # <- the exact string handed to the C++ loader
    MAX_ACTIVE_FEATURES = 16              # matches EngineThreatsExtractor MAX_ACTIVE_ET (measured max 9)

    NUM_INPUTS = 1536
    NUM_REAL_FEATURES = 1536
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
        """Threat features carry no piece values, so PSQT columns are zero."""
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
