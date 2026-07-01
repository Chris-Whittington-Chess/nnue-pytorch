// Bucket 2 — trainer-side glue for the self-owned threat encoder.
#include "cst_threats_extractor.h"

#include <atomic>
#include <cstdio>
#include <vector>

using namespace binpack;
using namespace chess;

namespace {

// Built once, shared read-only across worker threads (thread-safe local static).
const threats::ThreatTables& tables() {
    static const threats::ThreatTables T = threats::build_tables();
    return T;
}

// Safe upper bound on active threats per position per perspective. Well above
// the observed worst case; a one-shot warning fires if a position ever exceeds
// it (rather than silently overflowing the batch arrays).
// TODO: profile over the real binpack mix and tighten/confirm this.
constexpr int MAX_ACTIVE = 256;

// Read the GPL Position into the encoder's minimal board. Piece/color/square
// ordinals match the encoder's enums exactly (Pawn=0..King=5, White=0/Black=1,
// square = rank*8+file), so this is a straight copy.
void position_to_ref(const Position& pos, threats::RefPosition& rp) {
    for (int i = 0; i < threats::NSQ; ++i) rp.pc[i] = -1;
    for (Square sq : pos.piecesBB()) {
        Piece p = pos.pieceAt(sq);
        int c = static_cast<int>(p.color());   // White=0, Black=1
        int t = static_cast<int>(p.type());    // Pawn=0 .. King=5
        rp.pc[static_cast<int>(sq)] = static_cast<int8_t>(c * threats::NUM_PT + t);
    }
    rp.king_sq[threats::WHITE] = static_cast<int>(pos.kingSquare(Color::White));
    rp.king_sq[threats::BLACK] = static_cast<int>(pos.kingSquare(Color::Black));
}

} // namespace

int CstThreatsExtractor::inputs() const { return tables().NUM_INPUTS; }
int CstThreatsExtractor::max_active_features() const { return MAX_ACTIVE; }

std::pair<int, int> CstThreatsExtractor::fill_features_sparse(
    const TrainingDataEntry& e, int* features, float* values, Color color) const {

    const threats::ThreatTables& T = tables();
    threats::RefPosition rp;
    position_to_ref(e.pos, rp);

    // Reuse the ORACLE so the trainer emits exactly the golden-vector features.
    // (A direct-to-array enumeration is the later optimization, once parity is
    // locked in.) color's ordinal (White=0/Black=1) is the perspective.
    std::vector<int> v = threats::full_encode(T, rp, static_cast<int>(color));

    int n = static_cast<int>(v.size());
    if (n > MAX_ACTIVE) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr,
                         "[CstThreatsExtractor] active features %d exceeded MAX_ACTIVE %d; "
                         "truncating — raise MAX_ACTIVE.\n", n, MAX_ACTIVE);
        n = MAX_ACTIVE;
    }
    for (int i = 0; i < n; ++i) { features[i] = v[i]; values[i] = 1.0f; }
    return {n, T.NUM_INPUTS};
}
