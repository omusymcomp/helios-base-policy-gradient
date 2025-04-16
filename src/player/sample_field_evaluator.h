// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hiroki SHIMORA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifndef SAMPLE_FIELD_EVALUATOR_H
#define SAMPLE_FIELD_EVALUATOR_H

#include "field_evaluator.h"
#include "predict_state.h"

#include <vector>
#include <memory>
#include <string>
#include <torch/script.h>              // PyTorch Script モデル

namespace rcsc {
class AbstractPlayerObject;
class Vector2D;
}

class ActionStatePair;

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

public:
    SampleFieldEvaluator();
    virtual ~SampleFieldEvaluator();

    virtual double operator()( const PredictState & state,
                               const std::vector<ActionStatePair> & path ) const;

    // 設定関連
    void setUseNN(bool flag) { use_nn_ = flag; }
    void setSaveModel(bool flag) { save_model_ = flag; }
    void setModelLoadPath(const std::string & path) { model_load_path_ = path; }
    void setModelSavePath(const std::string & path) { model_save_path_ = path; }

    bool isUsingNN() const { return use_nn_; }

    // パラメータの setter
    void setGoalReward(double val) { goal_reward_ = val; }
    void setSelfBonus(double val) { self_bonus_ = val; }
    void setEnemyGoalBonus(double val) { enemy_goal_bonus_ = val; }
    void setOurGoalPenalty(double val) { our_goal_penalty_ = val; }
    void setProgressCoeff(double val) { progress_coeff_ = val; }

    // 将来の利用向け特徴抽出（今は未使用）
    static std::vector<double> extractFeatures(const PredictState & state);
};

#endif
