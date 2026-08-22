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
        block 2 = equalexchange(even trade available)
        block 3 = pinned       (piece absolutely pinned to its king)
        block 4 = pinner       (slider doing an absolute pin)
        block 5 = safe check   (a check is available; keyed by CHECKER type, LANDING square)
        block 6 = severe check (the subset leaving the king <= 1 flight square)
      NUM_INPUTS = 7 blocks * 2 sides * 6 types * 64 squares = 5376
                 + king-ring pressure 162 states x 2 sides                 =  324
                 + x-ray sliders on ring 3 states x 2 sides                =    6
                                                                            -----
                                                                             5706

    Blocks 5/6 differ in kind from 0-4: they are keyed by the checker, and the landing square is
    usually EMPTY, so the piece type cannot be read off the board -- it travels with the datum.
    They come from the engine's FindSafeChecks (threat_encoder/threat_safecheck.h, proven
    bit-exact against engine-generated golden vectors), NOT from BuildPinData's victim bitboards.

    Flat sparse feature, no virtual/factorized weights, no fold (it is a genuine LIVE feature).

    IMPORTANT -- must stay in lockstep with the C++ encoder, and NOTHING ENFORCES IT:
      * NUM_INPUTS          == threats::ENGINE_THREAT_INPUTS
      * MAX_ACTIVE_FEATURES == EngineThreatsExtractor MAX_ACTIVE_ET
    The values below are the 7-block ones, so the dataloader MUST be compiled with
    -DSAFECHECK_FEATURES=1. Build it without and the loader emits a 3840-wide space into a
    5376-wide layer: no error, just 1536 permanently dead columns and blocks 5/6 never learned.
    Check `selftest.exe`, which prints ENCODER_VERSION -- ThreatsV1nkET7fp is the 7-block build,
    ThreatsV1nkET5fp the 5-block one.
    """

    HASH = 0x45544B50  # "ETKP"; 7 blocks + the king-ring pressure section. ET07/ET05/ET02 superseded.
    FEATURE_NAME = "EngineThreats"
    INPUT_FEATURE_NAME = "EngineThreats"  # <- the exact string handed to the C++ loader
    # MAX_ACTIVE_ET measured over 800,000 perspectives (threat_encoder/et_active_stats.cpp):
    # mean 2.148, max 24. 48 is ~2x the maximum, the margin the 5-block 32 carried over its 13.
    # +4: the pressure section contributes exactly 4 features per perspective (a state and an
    # x-ray count for each colour) and is ALWAYS active, unlike blocks 5/6 which are live in
    # only 25-43% of perspectives. Measured max for the victim-keyed part was 24 over 800,000
    # perspectives, so 48 + 4 keeps the same 2x margin.
    MAX_ACTIVE_FEATURES = 52

    NUM_INPUTS = 5706
    NUM_REAL_FEATURES = 5706
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
