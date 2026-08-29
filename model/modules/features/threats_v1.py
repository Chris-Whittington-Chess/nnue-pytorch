import os

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

    # HASH separates incompatible feature spaces. "TV0B" = fan-pruned 61096 (distinct from
    # the 66920 and 75320 nets); "TV0C" = 59920, additionally pawn-reciprocity-pruned (B4).
    HASH = 0x54563043 if os.environ.get("CST_PAWN_PRUNE", "0") == "1" else 0x54563042
    FEATURE_NAME = "ThreatsV1"
    INPUT_FEATURE_NAME = "ThreatsV1"  # <- the exact string handed to the C++ loader
    MAX_ACTIVE_FEATURES = 96  # cap; fewer now (king attacker dropped)

    # Fan-pruned (ENCODER_VERSION "ThreatsV1nkET5fp", 2026-08-04): B->Q and R->Q
    # dropped -- a bishop/rook attacking a queen is the same physical relation as
    # the queen attacking back, so it is kept from the queen's side only.
    # T[BISHOP] and T[ROOK] go 5 -> 4, removing 4*(F[B]+F[R]) = 4*(560+896) = 5824.
    # Verified against build_tables(): 66920 -> 61096.
    #
    # B4 PAWN RECIPROCITY PRUNE (2026-08-29), env-driven: CST_PAWN_PRUNE=1.
    # Restricts PAWN attackers to {N,R}, the non-reciprocal targets -- a pawn attacking
    # P/B/Q is a mutual relation already emitted from the target's side. T[PAWN] 5 -> 2
    # removes 2 sides * 2 target-sides * 3 types * F[P]=98 = 1176.
    # Verified against build_tables(): 61096 -> 59920, ENCODER_VERSION "ThreatsV1nkET5fpPR".
    #
    # THIS MUST MATCH the -DPAWN_RECIPROCITY_PRUNE the data loader was compiled with. It is
    # not checked automatically at this layer: get it wrong one way and 1176 columns train
    # dead, the other way and indices overflow. The exported net carries the C++ encoder's
    # variant id, so the ENGINE catches a mismatch at load -- but only after a training run.
    # Hence the banner below: a training log should show which space it used.
    _PAWN_PRUNE = os.environ.get("CST_PAWN_PRUNE", "0") == "1"

    NUM_INPUTS = 59920 if _PAWN_PRUNE else 61096
    NUM_REAL_FEATURES = NUM_INPUTS
    EXPORT_WEIGHT_DTYPE = torch.int8

    # flush=True is load-bearing, not tidiness. A training run's stdout is redirected to a
    # FILE, so it is block-buffered; this banner prints during import, thousands of lines
    # before the buffer fills, and a run launched on 2026-08-29 showed no banner at all for
    # its first four minutes. The banner's entire purpose is that a LOG proves which feature
    # space was trained -- unflushed, it proves nothing exactly when you need it.
    print(
        "[ThreatsV1] CST_PAWN_PRUNE=1 -> B4 pawn reciprocity prune ACTIVE: NUM_INPUTS=59920,"
        " data loader MUST be built with -DPAWN_RECIPROCITY_PRUNE=1 (ThreatsV1nkET5fpPR)"
        if _PAWN_PRUNE else
        "[ThreatsV1] NUM_INPUTS=61096 (baseline, no pawn prune)",
        flush=True,
    )

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
