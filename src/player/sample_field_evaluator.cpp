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
#include <cstdlib>  // for getenv

using namespace rcsc;

static const int VALID_PLAYER_THRESHOLD = 8;

void SampleFieldEvaluator::loadParametersFromFile(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open parameter file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            double value = std::stod(line.substr(pos + 1));

            if (key == "goal_reward") {
                goal_reward_ = value;
            } else if (key == "self_bonus") {
                self_bonus_ = value;
            } else if (key == "enemy_goal_bonus") {
                enemy_goal_bonus_ = value;
            } else if (key == "our_goal_penalty") {
                our_goal_penalty_ = value;
            } else if (key == "progress_coeff") {
                progress_coeff_ = value;
            } else if (key == "progress_base") {
                progress_base_ = value;
            }
        }
    }

    std::cout << "[INFO] Parameters loaded from: " << file_path << std::endl;
}

SampleFieldEvaluator::SampleFieldEvaluator()
    : use_nn_(false),
      save_model_(false),
      model_load_path_("/home/okayama/rcss/policy-gradient/model.pt"),
      model_save_path_("/home/okayama/rcss/policy-gradient/output_model.pt"),
      nn_model_(nullptr),
      goal_reward_(1.0e+6),
      self_bonus_(5.0e+5),
      enemy_goal_bonus_(1.0e+7),
      our_goal_penalty_(-1.0e+7),
      progress_coeff_(1.0),
      progress_base_(0.1) // デフォルト値
{
    loadParametersFromFile("/home/okayama/rcss/policy-gradient/config/parameters.conf");

    if (use_nn_) {
        try {
            nn_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_load_path_));
            nn_model_->eval();
            std::cout << "[INFO] NN model loaded from: " << model_load_path_ << std::endl;
        } catch (const c10::Error &e) {
            std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        }
    }
}

SampleFieldEvaluator::~SampleFieldEvaluator() {
    if (use_nn_ && save_model_ && nn_model_) {
        try {
            nn_model_->save(model_save_path_);
            std::cout << "[INFO] NN model saved to: " << model_save_path_ << std::endl;
        } catch (const c10::Error &e) {
            std::cerr << "[ERROR] Failed to save model: " << e.what() << std::endl;
        }
    }
}

double SampleFieldEvaluator::operator()(const PredictState & state,
                                        const std::vector<ActionStatePair> & /*path*/) const
{
    return evaluate_state(state, goal_reward_, self_bonus_, enemy_goal_bonus_,
                          our_goal_penalty_, progress_coeff_, progress_base_);
}

double SampleFieldEvaluator::evaluate_state(const PredictState & state,
                                            double goal_reward,
                                            double self_bonus,
                                            double enemy_goal_bonus,
                                            double our_goal_penalty,
                                            double progress_coeff,
                                            double progress_base_) const
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

    double dist = SP.theirTeamGoalPos().dist(state.ball().pos());
    point += std::exp(progress_base_ * (40.0 - dist));

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