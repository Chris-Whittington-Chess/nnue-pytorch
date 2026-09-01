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

    # Both build flags are read FIRST, because HASH and NUM_INPUTS below depend on the pair.
    _PAWN_PRUNE = os.environ.get("CST_PAWN_PRUNE", "0") == "1"
    _PAWN_PAIRS = os.environ.get("CST_PAWN_PAIRS", "0") == "1"

    # HASH separates incompatible feature spaces. "TV0B" = fan-pruned 61096 (distinct from
    # the 66920 and 75320 nets); "TV0C" = 59920, additionally pawn-reciprocity-pruned (B4);
    # "TV0D" = 64480, B4 prune + B5 pawn pairs (the SF-faithful arm); "TV0E" = 65656, pairs
    # without the prune. Four spaces, four ids -- a shared id across incompatible widths is
    # the failure this field exists to prevent.
    if _PAWN_PAIRS:
        HASH = 0x54563044 if _PAWN_PRUNE else 0x54563045
    else:
        HASH = 0x54563043 if _PAWN_PRUNE else 0x54563042
    FEATURE_NAME = "ThreatsV1"
    INPUT_FEATURE_NAME = "ThreatsV1"  # <- the exact string handed to the C++ loader
    # B5 changes the ACTIVE count, not just the width: C(16,2) = 120 pawn pairs on top of the
    # threats stream. Measured over 327,680 perspectives: mean 62.73, max 168 (threats alone
    # were mean 14.6, max 49). 96 would truncate a normal opening, so this must track the data
    # loader's MAX_ACTIVE in cst_threats_extractor.cpp -- 256 when pairs are on.
    MAX_ACTIVE_FEATURES = 256 if _PAWN_PAIRS else 96  # cap; fewer now (king attacker dropped)

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
    # B5 PAWN-PAIR SET (2026-08-31), env-driven: CST_PAWN_PAIRS=1. Appends an unordered
    # pawn-pair section AFTER every attacker block -- C(96,2) = 4,560 indices, keyed by which
    # two pawns exist rather than by any attack relation. Built as the SF-faithful hedge for
    # the fork linkage B4 measured at -9.04 +/- 6.99 Elo/node.
    # Verified against build_tables(): 59920 -> 64480, ENCODER_VERSION "ThreatsV1nkET5fpPRPP".
    # Same warning as the prune above: this MUST match the -DPAWNPAIR_FEATURES the data loader
    # was compiled with, and nothing at this layer checks it.
    _PP_INPUTS = 4560

    NUM_INPUTS = (59920 if _PAWN_PRUNE else 61096) + (_PP_INPUTS if _PAWN_PAIRS else 0)
    NUM_REAL_FEATURES = NUM_INPUTS
    EXPORT_WEIGHT_DTYPE = torch.int8

    # flush=True is load-bearing, not tidiness. A training run's stdout is redirected to a
    # FILE, so it is block-buffered; this banner prints during import, thousands of lines
    # before the buffer fills, and a run launched on 2026-08-29 showed no banner at all for
    # its first four minutes. The banner's entire purpose is that a LOG proves which feature
    # space was trained -- unflushed, it proves nothing exactly when you need it.
    # Derived from the flags rather than written out per case: with two independent knobs there
    # are four spaces, and a hardcoded banner that disagrees with NUM_INPUTS would defeat the
    # whole point of printing it.
    _ENC = "ThreatsV1nkET5fp" + ("PR" if _PAWN_PRUNE else "") + ("PP" if _PAWN_PAIRS else "")
    print(
        f"[ThreatsV1] CST_PAWN_PRUNE={int(_PAWN_PRUNE)} CST_PAWN_PAIRS={int(_PAWN_PAIRS)}"
        f" -> NUM_INPUTS={NUM_INPUTS} ({_ENC}), max_active={MAX_ACTIVE_FEATURES}."
        " The data loader MUST be built with the MATCHING"
        f" -DPAWN_RECIPROCITY_PRUNE={int(_PAWN_PRUNE)} -DPAWNPAIR_FEATURES={int(_PAWN_PAIRS)}",
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
