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

    // 評価値を計算する関数
    double calculateFieldEvaluation(const std::vector<double> &heuristics,
                                    const std::vector<double> &weights) const;

public:
    SampleFieldEvaluator();
    virtual ~SampleFieldEvaluator();

    // 評価関数のオーバーロード
    virtual double operator()(const PredictState & state,
                              const std::vector<ActionStatePair> & path) const override;
    
    // ヒューリスティックを計算する関数
    std::vector<double> calculateHeuristics(const PredictState & state) const;
};

} // namespace rcsc

#endif // SAMPLE_FIELD_EVALUATOR_H