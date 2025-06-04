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
    bool save_model_;                          // モデルを保存するかどうか
    std::string model_load_path_;              // モデル読み込みパス
    std::string model_save_path_;              // モデル保存パス
    std::shared_ptr<torch::jit::script::Module> nn_model_;  // NN モデル本体

    // 最適化対象パラメータ
    double goal_reward_;
    double self_bonus_;
    double enemy_goal_bonus_;
    double our_goal_penalty_;
    double progress_coeff_;
    double progress_base_;

    // 外部ファイルからパラメータを読み込む関数
    void loadParametersFromFile(const std::string &file_path);

    // 評価関数本体
    double evaluate_state(const PredictState & state,
                          double goal_reward,
                          double self_bonus,
                          double enemy_goal_bonus,
                          double our_goal_penalty,
                          double progress_coeff,
                          double progress_base_) const;

public:
    SampleFieldEvaluator();
    virtual ~SampleFieldEvaluator();

    virtual double operator()(const PredictState & state,
                              const std::vector<ActionStatePair> & path) const override;
};

} // namespace rcsc

#endif // SAMPLE_FIELD_EVALUATOR_H