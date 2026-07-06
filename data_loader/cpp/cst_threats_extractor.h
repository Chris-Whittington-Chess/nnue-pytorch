// Bucket 2 — trainer-side glue for the self-owned threat encoder.
//
// This file lives inside the (GPL) dataloader and is trainer-only. It reads the
// GPL Position and calls into the self-owned threat_encoder module. The encoder
// itself (threat_encoder/) is separate, independently authored, and is the copy
// that also compiles into CST.
#pragma once
#include "training_data_loader_internal.h"
#include "threat_encoder.h"   // encode + tables, from threat_encoder/ (self-owned)

struct CstThreatsExtractor : IFeatureExtractor {
    int inputs() const override;
    int max_active_features() const override;
    std::pair<int, int> fill_features_sparse(const struct binpack::TrainingDataEntry& e,
                                             int*        features,
                                             float*      values,
                                             chess::Color color) const override;
};

// EngineThreats: victim-keyed features from the engine's BuildPinData threats[]/lesserthreats[]
// bitboards, computed by the shared port (threat_buildpin.h) off the dataloader Position.
struct EngineThreatsExtractor : IFeatureExtractor {
    int inputs() const override;
    int max_active_features() const override;
    std::pair<int, int> fill_features_sparse(const struct binpack::TrainingDataEntry& e,
                                             int*        features,
                                             float*      values,
                                             chess::Color color) const override;
};
