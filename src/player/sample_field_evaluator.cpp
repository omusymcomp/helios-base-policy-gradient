// -*-c++-*-

#include "sample_field_evaluator.h"

#include "field_analyzer.h"
#include "simple_pass_checker.h"

#include <rcsc/player/player_evaluator.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>
#include <rcsc/math_util.h>

#include <torch/script.h>
#include <torch/serialize.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cfloat>

using namespace rcsc;

static const int VALID_PLAYER_THRESHOLD = 8;

static double evaluate_state(const PredictState & state,
                              double goal_reward,
                              double self_bonus,
                              double enemy_goal_bonus,
                              double our_goal_penalty,
                              double progress_coeff);

SampleFieldEvaluator::SampleFieldEvaluator()
    : use_nn_(false),
      save_model_(false),
      model_load_path_("model.pt"),
      model_save_path_("output_model.pt"),
      nn_model_(nullptr),
      goal_reward_(1.0e+6),
      self_bonus_(5.0e+5),
      enemy_goal_bonus_(1.0e+7),
      our_goal_penalty_(-1.0e+7),
      progress_coeff_(1.0)
      {
        if (use_nn_) {
            try {
                nn_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_load_path_));
                nn_model_->eval();
                std::cout << "[INFO] NN model loaded from: " << model_load_path_ << std::endl;
            } catch (const c10::Error& e) {
                std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
            }
        }
    }
    
    SampleFieldEvaluator::~SampleFieldEvaluator()
    {
        if (use_nn_ && save_model_ && nn_model_) {
            try {
                nn_model_->save(model_save_path_);
                std::cout << "[INFO] NN model saved to: " << model_save_path_ << std::endl;
            } catch (const c10::Error& e) {
                std::cerr << "[ERROR] Failed to save model: " << e.what() << std::endl;
            }
        }
    }

double SampleFieldEvaluator::operator()(const PredictState & state,
                                        const std::vector<ActionStatePair> & /*path*/) const
{
    if (use_nn_ && nn_model_) {
        std::vector<double> features;  // 将来的に使う可能性があれば保持
        // features = extractFeatures(state);

        torch::Tensor input = torch::tensor(features).unsqueeze(0);
        torch::Tensor output = nn_model_->forward({input}).toTensor();

        return output.item<double>();
    }

    return evaluate_state(state, goal_reward_, self_bonus_, enemy_goal_bonus_, our_goal_penalty_, progress_coeff_);
}

static double evaluate_state(const PredictState & state,
                             double goal_reward,
                             double self_bonus,
                             double enemy_goal_bonus,
                             double our_goal_penalty,
                             double progress_coeff)
{
    const ServerParam & SP = ServerParam::i();
    const AbstractPlayerObject * holder = state.ballHolder();

    if (!holder)
        return -DBL_MAX / 2.0;

    const int holder_unum = holder->unum();

    if (state.ball().pos().x > + (SP.pitchHalfLength() - 0.1)
        && state.ball().pos().absY() < SP.goalHalfWidth() + 2.0)
        return enemy_goal_bonus;

    if (state.ball().pos().x < - (SP.pitchHalfLength() - 0.1)
        && state.ball().pos().absY() < SP.goalHalfWidth())
        return our_goal_penalty;

    if (state.ball().pos().absX() > SP.pitchHalfLength()
        || state.ball().pos().absY() > SP.pitchHalfWidth())
        return -DBL_MAX / 2.0;

    double point = state.ball().pos().x;
    point += std::max(0.0, progress_coeff * (40.0 - SP.theirTeamGoalPos().dist(state.ball().pos())));

    if (FieldAnalyzer::can_shoot_from(holder->unum() == state.self().unum(),
                                      holder->pos(),
                                      state.getPlayers(new OpponentOrUnknownPlayerPredicate(state.ourSide())),
                                      VALID_PLAYER_THRESHOLD)) {
        point += goal_reward;
        if (holder_unum == state.self().unum()) {
            point += self_bonus;
        }
    }

    return point;
}

// setter を外部から呼べるようにしても良い（例：config 読み込みやコマンドラインから）
