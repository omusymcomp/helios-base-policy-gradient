#ifndef SAMPLE_FIELD_EVALUATOR_H
#define SAMPLE_FIELD_EVALUATOR_H

#include "field_evaluator.h"
#include "predict_state.h"

#include <vector>
#include <memory>
#include <string>
#include <torch/script.h> // PyTorch Script モデル

namespace rcsc {

class SampleFieldEvaluator : public FieldEvaluator {
private:
    bool use_nn_;                              // NNを使うかどうか
    std::string model_load_path_;              // モデル読み込みパス
    std::shared_ptr<torch::jit::script::Module> nn_model_;  // NN モデル本体

public:
    SampleFieldEvaluator();
    virtual ~SampleFieldEvaluator();

    // 評価関数のオーバーロード
    virtual double operator()(const PredictState & state,
                              const std::vector<ActionStatePair> & path) const override;
    
    // ヒューリスティックを計算する関数
    std::vector<double> calculateHeuristics(const PredictState & state) const;

    static double legacyFieldEvaluationRaw(const PredictState & state);
    static double legacyFieldEvaluationNormalized(const PredictState & state);
};

} // namespace rcsc

#endif // SAMPLE_FIELD_EVALUATOR_H
