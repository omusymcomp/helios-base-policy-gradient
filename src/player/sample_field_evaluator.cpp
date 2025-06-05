#include "sample_field_evaluator.h"

#include "field_analyzer.h"
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>

#include <torch/script.h>
#include <iostream>
#include <cmath>
#include <cfloat>

using namespace rcsc;

SampleFieldEvaluator::SampleFieldEvaluator()
    : use_nn_(true),
      model_load_path_("/home/okayama/rcss/policy-gradient/model_with_attention.pt"),
      nn_model_(nullptr)
{
    if (use_nn_) {
        try {
            nn_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_load_path_));
            nn_model_->eval();
            //std::cerr << "[INFO] NN model loaded from: " << model_load_path_ << std::endl;
        } catch (const c10::Error &e) {
            std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        }
    }
}

SampleFieldEvaluator::~SampleFieldEvaluator() {}

double SampleFieldEvaluator::operator()(const PredictState & state,
                                        const std::vector<ActionStatePair> & /*path*/) const
{
    if (!use_nn_ || !nn_model_) {
        std::cerr << "[ERROR] NN model is not loaded." << std::endl;
        return -DBL_MAX;
    }

    // ヒューリスティックを計算
    std::vector<double> heuristics = calculateHeuristics(state);

    // 状態とヒューリスティックをテンソルに変換
    torch::Tensor state_tensor = torch::tensor({
        state.ball().pos().x / 50.0,
        state.ball().pos().y / 34.0,
        state.self().pos().x / 50.0,
        state.self().pos().y / 34.0,
        state.self().vel().x / 5.0,
        state.self().vel().y / 5.0
    }, torch::kFloat).unsqueeze(0); // shape: [1, feature_dim]

    torch::Tensor heuristics_tensor = torch::tensor(heuristics, torch::kFloat).unsqueeze(0); // shape: [1, num_heuristics]

    try {
        auto outputs = nn_model_->forward({state_tensor, heuristics_tensor}).toTuple();
        // 正しい順序で取得
        torch::Tensor logits_tensor = outputs->elements()[0].toTensor();   // optional
        torch::Tensor weights_tensor = outputs->elements()[1].toTensor();  // ← 正しくここから取得！

        std::vector<double> weights(weights_tensor.data_ptr<float>(),
                                    weights_tensor.data_ptr<float>() + weights_tensor.numel());

        return calculateFieldEvaluation(heuristics, weights);
    } catch (const c10::Error &e) {
        std::cerr << "[ERROR] NN forward failed: " << e.what() << std::endl;
        return -DBL_MAX;
    }

}

std::vector<double> SampleFieldEvaluator::calculateHeuristics(const PredictState & state) const {
    std::vector<double> heuristics;
    const ServerParam & SP = ServerParam::i();

    const AbstractPlayerObject * holder = state.ballHolder();

    // h₁: ボールの x 位置
    heuristics.push_back(state.ball().pos().x);

    // h₂: ボールの y 位置（絶対値）
    heuristics.push_back(std::abs(state.ball().pos().y));

    // h₃: ゴールとの距離（expでスケーリング）
    double dist_to_goal = SP.theirTeamGoalPos().dist(state.ball().pos());
    heuristics.push_back(std::exp(-dist_to_goal / 10.0)); // 距離に応じた正規化

    // h₄: 自分がボールキック可能か（距離ベースで近似）
    const PlayerType * self_type = state.self().playerTypePtr();
    double kickable_area = self_type->kickableArea();
    bool is_kickable = state.self().pos().dist(state.ball().pos()) <= kickable_area;
    heuristics.push_back(is_kickable ? 1.0 : 0.0);

    // h₅: 自分とボールの速度差（正規化）
    double rel_vel = (state.self().vel() - state.ball().vel()).r();
    heuristics.push_back(std::tanh(rel_vel)); // -1〜1に圧縮

    // h₆: ボールの速度
    heuristics.push_back(std::tanh(state.ball().vel().r())); // -1〜1に圧縮

    // h₇: ボールと自分の距離（近いほど有利）
    double dist_to_ball = state.self().pos().dist(state.ball().pos());
    heuristics.push_back(std::exp(-dist_to_ball));

    // h₈: ボールの位置がゴールラインに近いか（±xの端に寄っているか）
    double goal_line_proximity = std::max(0.0, std::abs(state.ball().pos().x) - (SP.pitchHalfLength() - 5.0)) / 5.0;
    heuristics.push_back(goal_line_proximity);

    // h₉: 自分の速度（ダッシュ中か）
    heuristics.push_back(std::tanh(state.self().vel().r()));

    // h₁₀: 自分の角度（正面を向いているか）→ 簡易として body angle の cos を使う
    heuristics.push_back(std::cos(state.self().body().radian()));

    return heuristics;
}

double SampleFieldEvaluator::calculateFieldEvaluation(const std::vector<double> &heuristics,
                                                      const std::vector<double> &weights) const {
    if (heuristics.size() != weights.size()) {
        std::cerr << "[ERROR] Heuristics and weights size mismatch!" << std::endl;
        return -DBL_MAX;
    }

    double evaluation = 0.0;
    for (size_t i = 0; i < heuristics.size(); ++i) {
        evaluation += weights[i] * heuristics[i];
    }
    return evaluation;
}