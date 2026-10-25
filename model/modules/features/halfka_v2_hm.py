import os

import chess
import torch
from torch import nn

from .input_feature import InputFeature

# Q[2]K[32] QUEEN PHASE (Chris, 2026-10-25). The king bucket is prefixed with a
# queen-presence flag, so the space doubles: bucket = Q*32 + king_bucket, with Q = 1 if
# EITHER side has at least one queen.
#
# MUST match the data loader, which reads the SAME env var at runtime
# (training_data_loader.cpp, queen_phase_on()). One variable for both sides, deliberately:
# ThreatsV1's equivalent is a -D define in C++ and an env var in Python, and its own comment
# warns nothing checks they agree. Default OFF, so every existing net stays reproducible.
_QUEEN_PHASE = os.environ.get("CST_QUEEN_PHASE", "0") == "1"


# fmt: off
KingBuckets = [
  -1, -1, -1, -1, 31, 30, 29, 28,
  -1, -1, -1, -1, 27, 26, 25, 24,
  -1, -1, -1, -1, 23, 22, 21, 20,
  -1, -1, -1, -1, 19, 18, 17, 16,
  -1, -1, -1, -1, 15, 14, 13, 12,
  -1, -1, -1, -1, 11, 10, 9, 8,
  -1, -1, -1, -1, 7, 6, 5, 4,
  -1, -1, -1, -1, 3, 2, 1, 0
]
# fmt: on

# Inverse mapping: bucket -> oriented king square
InverseKingBuckets = [0] * 32
for _sq, _bucket in enumerate(KingBuckets):
    if _bucket >= 0:
        InverseKingBuckets[_bucket] = _sq


def _orient(is_white_pov: bool, sq: int, ksq: int) -> int:
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    """Feature index using 12 piece types (no king merging)."""
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768


class HalfKav2Hm(InputFeature):
    # HASH separates incompatible feature spaces, as ThreatsV1's does. XOR with ord('Q')
    # marks the queen-phase space so a 32-bucket net and a 64-bucket net can never be
    # mistaken for each other.
    HASH = 0x7F234CB8 ^ (0x51 if _QUEEN_PHASE else 0x00)
    FEATURE_NAME = "HalfKAv2_hm^"
    INPUT_FEATURE_NAME = "HalfKAv2_hm"
    MAX_ACTIVE_FEATURES = 32

    NUM_SQ = 64
    NUM_PT = 12
    NUM_PLANES = NUM_SQ * NUM_PT  # 768
    # 32, or 64 under the queen phase: Q*32 + king_bucket
    NUM_BUCKETS = (NUM_SQ // 2) * (2 if _QUEEN_PHASE else 1)
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS  # 24,576, or 49,152
    NUM_INPUTS_VIRTUAL = NUM_PLANES  # 768

    # Export size uses 11 piece types (704 per bucket)
    NUM_REAL_FEATURES = 704 * NUM_BUCKETS  # 22,528, or 45,056

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.virtual_weight.repeat(self.NUM_BUCKETS, 1)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.add_(self.virtual_weight.repeat(self.NUM_BUCKETS, 1))
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Initialize virtual weights to zero and set PSQT columns."""
        self.virtual_weight.zero_()

        # PSQT ablated (CST_NO_PSQT -> PSQT_BUCKETS 0): there are no PSQT columns to seed.
        # NOTE what goes with them: the seeding below is a HAND-SET MATERIAL PRIOR
        # (halfka_psqts: P 126, N 781, B 825, R 1276, Q 2538). Ablating PSQT therefore does
        # not merely drop an output term, it removes the material head start, and the body
        # must learn material from nothing. Expect slower early convergence.
        if num_psqt_buckets == 0:
            return

        scale = 1.0 / nnue2score
        L1 = self.num_outputs - num_psqt_buckets

        initial_values = self.halfka_psqts()
        # halfka_psqts() emits one value per 32-bucket slot. Under the queen phase the space
        # is doubled, and a piece's material value does not depend on Q, so the same block is
        # tiled across both halves.
        if len(initial_values) != self.NUM_INPUTS:
            reps, rem = divmod(self.NUM_INPUTS, len(initial_values))
            assert rem == 0, "halfka_psqts() does not tile into NUM_INPUTS"
            initial_values = initial_values * reps
        assert len(initial_values) == self.NUM_INPUTS

        new_weights = (
            torch.tensor(
                initial_values,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            * scale
        )

        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = new_weights

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        """Return coalesced weight remapped from 12->11 piece types for export.

        Returns a float tensor with NUM_REAL_FEATURES rows.
        """
        # Coalesce virtual weights
        coalesced = self.weight.data + self.virtual_weight.data.repeat(
            self.NUM_BUCKETS, 1
        )

        # Remap 12 piece types -> 11 piece types
        export = coalesced.new_zeros(self.NUM_REAL_FEATURES, coalesced.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * self.NUM_PLANES  # 768 features per bucket
            dst_offset = b * 704  # 704 features per bucket in export

            # Copy first 10 piece types (p_idx 0..9) -- 640 features
            export[dst_offset : dst_offset + 640] = coalesced[
                src_offset : src_offset + 640
            ]

            # Merge own king (p_idx=10) and opponent king (p_idx=11) into single block
            own_king_src = src_offset + 10 * 64
            opp_king_src = src_offset + 11 * 64
            dst_king = dst_offset + 10 * 64
            # b % 32: under the queen phase buckets 32..63 are the SAME king buckets with
            # Q=1, so the king square for bucket b is the one for b mod 32. Without the
            # modulo this indexes off the end of a 32-entry table.
            ksq = InverseKingBuckets[b % 32]

            export[dst_king : dst_king + 64] = coalesced[
                opp_king_src : opp_king_src + 64
            ]
            export[dst_king + ksq] = coalesced[own_king_src + ksq]

        return export

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        """Load export-format weights (11 piece types) and expand to 12.

        Takes a float tensor of shape (NUM_REAL_FEATURES, num_outputs).
        Expands 11->12 piece types and assigns to self.weight.
        Zeros self.virtual_weight.
        """
        expanded = export_weight.new_zeros(self.NUM_INPUTS, export_weight.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * 704
            dst_offset = b * self.NUM_PLANES

            # Copy first 10 piece types
            expanded[dst_offset : dst_offset + 640] = export_weight[
                src_offset : src_offset + 640
            ]

            # Split merged king block back into p_idx 10 and 11
            src_king = src_offset + 10 * 64
            ksq = InverseKingBuckets[b % 32]   # see get_export_weights

            # Own king: only weight at ksq matters (rest stays zero)
            expanded[dst_offset + 10 * 64 + ksq] = export_weight[src_king + ksq]

            # Opponent king: all squares from merged, except ksq -> 0
            expanded[dst_offset + 11 * 64 : dst_offset + 12 * 64] = export_weight[
                src_king : src_king + 64
            ]
            expanded[dst_offset + 11 * 64 + ksq] = 0

        self.weight.data.copy_(expanded)
        self.virtual_weight.zero_()

    @staticmethod
    def halfka_psqts() -> list[int]:
        """PSQT initial values using 12 piece types (24,576 values)."""
        piece_values = {
            chess.PAWN: 126,
            chess.KNIGHT: 781,
            chess.BISHOP: 825,
            chess.ROOK: 1276,
            chess.QUEEN: 2538,
        }

        num_inputs = 768 * 32  # 24,576
        values = [0] * num_inputs

        for ksq in range(64):
            for s in range(64):
                for pt, val in piece_values.items():
                    idxw = _halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
                    idxb = _halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
                    values[idxw] = val
                    values[idxb] = -val

        return values
