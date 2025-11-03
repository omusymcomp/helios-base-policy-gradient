#include "sample_field_evaluator.h"

#include "field_analyzer.h"
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>

#include <torch/script.h>
#include <array>
#include <iostream>
#include <cmath>
#include <cfloat>

using namespace rcsc;

SampleFieldEvaluator::SampleFieldEvaluator()
    : use_nn_(true),
      model_load_path_("/home/okayama/rcss/policy-gradient/model.pt"),
      nn_model_(nullptr)
{
    if (use_nn_)
    {
        try
        {
            nn_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_load_path_));
            nn_model_->eval();
            // std::cerr << "[INFO] NN model loaded from: " << model_load_path_ << std::endl;
        }
        catch (const c10::Error &e)
        {
            std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        }
    }
}

SampleFieldEvaluator::~SampleFieldEvaluator() {}

double SampleFieldEvaluator::operator()(const PredictState &state,
                                        const std::vector<ActionStatePair> & /*path*/) const
{
    if (!use_nn_ || !nn_model_)
    {
        std::cerr << "[ERROR] NN model is not loaded." << std::endl;
        return -DBL_MAX;
    }

    // ヒューリスティックを計算
    std::vector<double> heuristics = calculateHeuristics(state);

    // 状態特徴をテンソルに変換（学習時と同じスケールを維持）
    std::array<float, 6> state_values = {
        static_cast<float>(state.ball().pos().x),
        static_cast<float>(state.ball().pos().y),
        static_cast<float>(state.self().pos().x),
        static_cast<float>(state.self().pos().y),
        static_cast<float>(state.self().vel().x),
        static_cast<float>(state.self().vel().y)};

    torch::Tensor state_tensor = torch::from_blob(state_values.data(), {(long)state_values.size()}, torch::kFloat32).clone().unsqueeze(0);

    std::vector<float> heuristics_values;
    heuristics_values.reserve(heuristics.size());
    for (double value : heuristics)
    {
        heuristics_values.push_back(static_cast<float>(value));
    }
    torch::Tensor heuristics_tensor = torch::from_blob(heuristics_values.data(), {(long)heuristics_values.size()}, torch::kFloat32).clone().unsqueeze(0);

    try
    {
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(state_tensor);
        inputs.emplace_back(heuristics_tensor);

        const auto outputs_tuple = nn_model_->forward(inputs).toTuple();
        if (!outputs_tuple || outputs_tuple->elements().size() < 2)
        {
            std::cerr << "[ERROR] Unexpected NN output format." << std::endl;
            return -DBL_MAX;
        }

        torch::Tensor value_tensor = outputs_tuple->elements()[1].toTensor();
        return value_tensor.squeeze().item<double>();
    }
    catch (const c10::Error &e)
    {
        std::cerr << "[ERROR] NN forward failed: " << e.what() << std::endl;
        return -DBL_MAX;
    }
}

std::vector<double> SampleFieldEvaluator::calculateHeuristics(const PredictState &state) const
{
    std::vector<double> heuristics;
    const ServerParam &SP = ServerParam::i();

    const AbstractPlayerObject *holder = state.ballHolder();

    // h₁: ボールの x 位置
    heuristics.push_back(state.ball().pos().x);

    // h₂: ボールの y 位置（絶対値）
    heuristics.push_back(std::abs(state.ball().pos().y));

    // h₃: ゴールとの距離（expでスケーリング）
    double dist_to_goal = SP.theirTeamGoalPos().dist(state.ball().pos());
    heuristics.push_back(std::exp(-dist_to_goal / 10.0)); // 距離に応じた正規化

    // h₄: 自分がボールキック可能か（距離ベースで近似）
    const PlayerType *self_type = state.self().playerTypePtr();
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
