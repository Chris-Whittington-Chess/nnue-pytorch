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
#include "threat_buildpin.h"    // shared BuildPinData port + emit_engine_threats
#include "threat_safecheck.h"   // shared FindSafeChecks port + emit_engine_threats_all

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
    Bitboard targets = occupied & ~kings;

    // C3 filter: emit only where the TARGET is attacked by an enemy of its owner.
    // Built from the SHARED threats::threatened_mask() rather than pos.attacks(), so
    // this cannot drift from the oracle or the engine -- three private answers to
    // "is this square attacked" would diverge on exactly the cases the golden
    // vectors exist to catch.
    if (threats::cfg::SKIP_UNATTACKED_TARGETS) {
        uint64_t bb[2][threats::NUM_PT] = {};
        for (Square s : occupied) {
            const Piece q = pos.pieceAt(s);
            bb[static_cast<int>(q.color())][static_cast<int>(q.type())]
                |= (1ull << static_cast<int>(s));
        }
        const uint64_t keep = threats::threatened_mask(bb, occupied.bits());
        targets &= Bitboard::fromBits(keep);
    }

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
            // With cfg::XRAY_THREATS the slider also reaches the man BEHIND its first
            // blocker, emitting the SAME index a direct attack would. Must match the
            // oracle's threat_attacks_geo() and the engine's piece_attacks() exactly --
            // three independent implementations, and parity is what proves they agree.
            const Bitboard direct = pos.attacks(from);
            Bitboard hits = direct & targets;
            if (threats::cfg::XRAY_THREATS &&
                (atype == threats::BISHOP || atype == threats::ROOK || atype == threats::QUEEN)) {
                const Bitboard blockers = direct & occupied;
                if (blockers.any()) {
                    const Bitboard through = bb::attacks(p.type(), from, occupied ^ blockers);
                    hits |= (through & ~direct) & targets;
                }
            }
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
// MEASURED, not guessed -- the extractor TRUNCATES above this and silently corrupts every
// position that exceeds it, so the number has to come from a count. threat_encoder/
// et_active_stats.cpp over 400,000 unique non-check positions from match/*.pgn (800,000
// perspectives, 7 blocks):
//
//   mean active  2.148        max active  24   (et5 half alone: 13; blocks 5/6 alone: 24)
//   busiest      8/8/8/2K5/8/8/1Q4Q1/3k4 w   -- two queens hunting a bare king
//
// 48 is ~2x the observed maximum, the same margin the 5-block 32 carried over its measured 13.
// Note the earlier plan's "raise to ~64" was a guess and the measurement says otherwise: the
// safe-check blocks are SPARSE (live in 25.66% of perspectives, mean 0.656), not bulky.
constexpr int MAX_ACTIVE_ET = 48;
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
    // _all: blocks 0-4 from BuildPinData plus, when SAFECHECK_FEATURES is on, blocks 5/6 from
    // the safe-check port. One ThreatBB serves both. With the macro off this is exactly the
    // old et5 behaviour.
    threats::emit_engine_threats_all(bb, king_sq, persp, [&](int idx) {
        if (k >= MAX_ACTIVE_ET) { warn_truncate_et(); return; }
        features[k] = idx;
        values[k]   = 1.0f;
        ++k;
    });
    return { k, threats::ENGINE_THREAT_INPUTS };
}
