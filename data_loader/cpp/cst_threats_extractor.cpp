// Bucket 2 — trainer-side glue for the self-owned threat encoder.
//
// Enumerates threats DIRECTLY off the dataloader's Position: occupancy-aware
// magic attacks (pos.attacks) intersected with the occupied non-king pieces,
// each hit fed to the self-owned encode(). No RefPosition rebuild, no ray-walk,
// no std::vector/sort -- the encoder module (threat_encoder/) supplies only the
// index math; the fast board + attacks come from the loader.
#include "cst_threats_extractor.h"

#include <atomic>
#include <cstdio>

using namespace binpack;
using namespace chess;

namespace {

// Built once, shared read-only across worker threads (thread-safe local static).
const threats::ThreatTables& tables() {
    static const threats::ThreatTables T = threats::build_tables();
    return T;
}

// Cap on active threats per position per perspective. Measured max over 3.3M
// real positions is 70 (mean 23.5, p99.9 60); 96 clears that with margin. The GPU
// FT kernel processes this padded width, so keeping it tight matters for speed.
// It is a CAP, not a hard bound: a rarer threat "explosion" is safely truncated
// (low-signal tail; the net loses ~nothing), and a one-shot warning notes it.
constexpr int MAX_ACTIVE = 96;

}  // namespace

int CstThreatsExtractor::inputs() const { return tables().NUM_INPUTS; }
int CstThreatsExtractor::max_active_features() const { return MAX_ACTIVE; }

std::pair<int, int> CstThreatsExtractor::fill_features_sparse(
    const TrainingDataEntry& e, int* features, float* values, Color color) const {

    const threats::ThreatTables& T = tables();
    const Position& pos = e.pos;

    // Kings still BLOCK slider rays (pos.attacks sees the full board occupancy)
    // but are never threat TARGETS, so drop them from the target mask only.
    const Bitboard occupied = pos.piecesBB();
    const Bitboard kings    = Bitboard::square(pos.kingSquare(Color::White))
                            | Bitboard::square(pos.kingSquare(Color::Black));
    const Bitboard targets  = occupied & ~kings;

    const int persp = static_cast<int>(color);
    const int ksq   = static_cast<int>(pos.kingSquare(color));

    int k = 0;
    for (Square from : occupied) {
        const Piece p     = pos.pieceAt(from);
        const int   acol  = static_cast<int>(p.color());
        const int   atype = static_cast<int>(p.type());

        // Bitboard of the pieces this attacker hits: magic attacks (pawns =
        // diagonal captures only, matching the V1 encoder) & the non-king pieces.
        Bitboard hits = (p.type() == PieceType::Pawn)
                      ? bb::pawnAttacks(Bitboard::square(from), p.color())
                      : pos.attacks(from);
        hits &= targets;

        for (Square to : hits) {
            const Piece q   = pos.pieceAt(to);
            const int   idx = threats::encode(T, persp, acol, atype, static_cast<int>(from),
                                              static_cast<int>(q.color()), static_cast<int>(q.type()),
                                              static_cast<int>(to), ksq);
            if (idx < 0) continue;  // defensive; king targets already masked out
            if (k >= MAX_ACTIVE) {
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true))
                    std::fprintf(stderr,
                                 "[CstThreatsExtractor] active features exceeded MAX_ACTIVE %d; "
                                 "truncating -- raise MAX_ACTIVE.\n", MAX_ACTIVE);
                return {k, T.NUM_INPUTS};
            }
            features[k] = idx;
            values[k]   = 1.0f;
            ++k;
        }
    }
    return {k, T.NUM_INPUTS};
}
