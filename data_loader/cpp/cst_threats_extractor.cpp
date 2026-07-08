// AttacksV3KNP — trainer-side glue for the self-owned attack/threat encoder.
//
// Enumerates features DIRECTLY off the dataloader's Position (fast board + magic
// attacks), fed to the self-owned index math in threat_encoder/. Two paths:
//   CONTROL (K/N/P): dense control map -- EVERY attacked square (occupancy-independent),
//     occupied or not, via encode_control(). Count-based: emitted per attacker, so two
//     same-type attackers of a square duplicate its index (the FT sums them). Must match
//     the oracle full_encode() exactly, including duplicates -> NO dedup here.
//   THREAT  (B/R/Q): attacks landing on occupied non-king targets, via encode() (V1 scheme).
#include "cst_threats_extractor.h"
#include "threat_buildpin.h"   // shared BuildPinData port + emit_engine_threats

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

// Cap on active features per position per perspective. V3 adds dense K/N/P control (~3x the
// V1 active count), so this is well above V1's measured max of 70. It is a CAP, not a hard
// bound: a rarer overflow is safely truncated (low-signal tail) with a one-shot warning.
// The GPU FT kernel pads to this width, so it trades training speed for headroom.
constexpr int MAX_ACTIVE = 192;

void warn_truncate() {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
        std::fprintf(stderr, "[CstThreatsExtractor] active features exceeded MAX_ACTIVE %d; "
                             "truncating -- raise MAX_ACTIVE.\n", MAX_ACTIVE);
}

}  // namespace

int CstThreatsExtractor::inputs() const { return tables().NUM_INPUTS; }
int CstThreatsExtractor::max_active_features() const { return MAX_ACTIVE; }

std::pair<int, int> CstThreatsExtractor::fill_features_sparse(
    const TrainingDataEntry& e, int* features, float* values, Color color) const {

    const threats::ThreatTables& T = tables();
    const Position& pos = e.pos;

    // Kings still BLOCK slider rays (pos.attacks sees the full occupancy) but are never
    // threat TARGETS, so drop them from the target mask only.
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

        if (threats::is_control_attacker(atype)) {
            // CONTROL (K/N/P): every attacked square (occupancy-independent for these pieces),
            // regardless of occupant. Pawns = diagonal attacks only (match the encoder).
            Bitboard atk = (p.type() == PieceType::Pawn)
                         ? bb::pawnAttacks(Bitboard::square(from), p.color())
                         : pos.attacks(from);
            for (Square to : atk) {
                const int idx = threats::encode_control(persp, acol, atype,
                                                        static_cast<int>(to), ksq);
                if (idx < 0) continue;
                if (k >= MAX_ACTIVE) { warn_truncate(); return {k, T.NUM_INPUTS}; }
                features[k] = idx;
                values[k]   = 1.0f;
                ++k;
            }
        } else {
            // THREAT (B/R/Q): occupancy-aware attacks intersected with occupied non-king pieces.
            Bitboard hits = pos.attacks(from) & targets;
            for (Square to : hits) {
                const Piece q   = pos.pieceAt(to);
                const int   idx = threats::encode(T, persp, acol, atype, static_cast<int>(from),
                                                  static_cast<int>(q.color()), static_cast<int>(q.type()),
                                                  static_cast<int>(to), ksq);
                if (idx < 0) continue;
                if (k >= MAX_ACTIVE) { warn_truncate(); return {k, T.NUM_INPUTS}; }
                features[k] = idx;
                values[k]   = 1.0f;
                ++k;
            }
        }
    }
    return {k, T.NUM_INPUTS};
}

// ---- EngineThreats ---------------------------------------------------------------------------

namespace {
constexpr int MAX_ACTIVE_ET = 24;   // 3-block (threats/lesser|pawn/equalexch); measured max ~12 both sides + margin
void warn_truncate_et() {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
        std::fprintf(stderr, "[EngineThreatsExtractor] active features exceeded MAX_ACTIVE %d; "
                             "truncating -- raise MAX_ACTIVE_ET.\n", MAX_ACTIVE_ET);
}
}  // namespace

int EngineThreatsExtractor::inputs() const { return threats::ENGINE_THREAT_INPUTS; }
int EngineThreatsExtractor::max_active_features() const { return MAX_ACTIVE_ET; }

std::pair<int, int> EngineThreatsExtractor::fill_features_sparse(
    const TrainingDataEntry& e, int* features, float* values, Color color) const {

    const Position& pos = e.pos;

    // Build per-color, per-type bitboards for the shared port (threats:: type order == loader's).
    uint64_t bb[2][6] = {};
    for (Square s : pos.piecesBB()) {
        const Piece p = pos.pieceAt(s);
        bb[static_cast<int>(p.color())][static_cast<int>(p.type())] |= (uint64_t(1) << static_cast<int>(s));
    }
    const int king_sq[2] = { static_cast<int>(pos.kingSquare(Color::White)),
                             static_cast<int>(pos.kingSquare(Color::Black)) };
    const int persp = static_cast<int>(color);

    int k = 0;
    threats::emit_engine_threats(bb, king_sq, persp, [&](int idx) {
        if (k >= MAX_ACTIVE_ET) { warn_truncate_et(); return; }
        features[k] = idx;
        values[k]   = 1.0f;
        ++k;
    });
    return { k, threats::ENGINE_THREAT_INPUTS };
}
