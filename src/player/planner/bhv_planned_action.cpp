// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hiroki SHIMORA, Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_planned_action.h"

#include "action_chain_holder.h"
#include "action_chain_graph.h"
#include "action_state_pair.h"
#include "field_analyzer.h"

#include "bhv_pass_kick_find_receiver.h"
#include "bhv_normal_dribble.h"
#include "body_force_shoot.h"

#include "neck_turn_to_receiver.h"

#include "basic_actions/bhv_scan_field.h"
#include "basic_actions/body_clear_ball.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_hold_ball.h"
#include "basic_actions/body_turn_to_point.h"
#include "basic_actions/neck_scan_field.h"
#include "basic_actions/neck_turn_to_goalie_or_scan.h"

#include "utils/step_data.h"
#include "utils/episode_logger.h"

#include "basic_actions/kick_table.h"
#include <rcsc/player/intercept_table.h>
#include <rcsc/player/soccer_intention.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/audio_memory.h>

#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <random>

#include <fstream> // CSV出力用
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <array> // 使わなければ不要
#include <cmath> // pow を使うので
#include <cstdlib>
#include <iostream>

using namespace rcsc;

namespace
{
    // ==== ε-greedy 用ユーティリティ ====

    inline double readEnvDouble(const char *name, double default_value)
    {
        if (const char *val = std::getenv(name))
        {
            char *end_ptr = nullptr;
            const double parsed = std::strtod(val, &end_ptr);
            if (end_ptr != val)
            {
                return parsed;
            }
        }
        return default_value;
    }

    inline long readEnvLong(const char *name, long default_value)
    {
        if (const char *val = std::getenv(name))
        {
            char *end_ptr = nullptr;
            const long parsed = std::strtol(val, &end_ptr, 10);
            if (end_ptr != val)
            {
                return parsed;
            }
        }
        return default_value;
    }

    inline bool isTrainingMode()
    {
        const char *mode = std::getenv("RL_TRAIN_MODE");
        return !(mode && mode[0] == '0');
    }

    struct EpsilonSchedule
    {
        double epsilon;
        double epsilon_min;
        double epsilon_decay;
    };

    inline EpsilonSchedule initEpsilonSchedule()
    {
        EpsilonSchedule schedule{0.20, 0.01, 0.999};

        const double eps_start = readEnvDouble("RL_EPS_START", 0.20);
        const double eps_end = readEnvDouble("RL_EPS_END", 0.05);
        const double eps_min_env = readEnvDouble("RL_EPS_MIN", eps_end);
        const double eps_decay = readEnvDouble("RL_EPS_DECAY", 0.999);
        const long total_matches = readEnvLong("RL_TOTAL_MATCHES", 0);
        const long match_index = readEnvLong("RL_MATCH_INDEX", 0);

        schedule.epsilon_decay = eps_decay;

        if (!isTrainingMode())
        {
            schedule.epsilon = 0.0;
            schedule.epsilon_min = 0.0;
            return schedule;
        }

        const double low = std::min(eps_start, eps_end);
        const double high = std::max(eps_start, eps_end);

        double progress = 0.0;
        if (total_matches > 1 && match_index > 0)
        {
            progress = static_cast<double>(match_index - 1) / static_cast<double>(total_matches - 1);
            progress = std::clamp(progress, 0.0, 1.0);
        }

        double scheduled = eps_start + (eps_end - eps_start) * progress;
        scheduled = std::clamp(scheduled, low, high);

        double eps_min = std::clamp(eps_min_env, low, high);
        if (eps_start >= eps_end)
        {
            eps_min = std::max(eps_min, eps_end);
        }
        else
        {
            eps_min = std::min(eps_min, eps_end);
        }

        schedule.epsilon = scheduled;
        schedule.epsilon_min = eps_min;

        return schedule;
    }

    const EpsilonSchedule S_EPS_SCHEDULE = initEpsilonSchedule();
    const double S_POLICY_TEMPERATURE = std::max(1e-6, readEnvDouble("RL_POLICY_TEMPERATURE", 1.0));
    static double s_epsilon = S_EPS_SCHEDULE.epsilon;
    static double s_eps_min = S_EPS_SCHEDULE.epsilon_min;
    static double s_eps_decay = S_EPS_SCHEDULE.epsilon_decay; // 1サイクルごとに減衰(例)

    // 簡易 0..1 乱数（サイクルと背番号で決定論シードにして再現性確保）
    inline double rand01_det(int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> d(0.0, 1.0);
        return d(gen);
    }

    // 探索用に整数を1つ選ぶ
    inline int randint_det(int seed, int lo, int hi)
    { // [lo, hi]
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> d(lo, hi);
        return d(gen);
    }

    // 必要であれば「実行不能」行動を弾く
    inline bool is_feasible(const CooperativeAction &a, const WorldModel &wm)
    {
        const auto gm = wm.gameMode().type();
        if ((a.category() == CooperativeAction::Dribble ||
             a.category() == CooperativeAction::Hold) &&
            gm != GameMode::PlayOn && !wm.gameMode().isPenaltyKickMode())
        {
            return false;
        }
        return true;
    }

    inline int policyIndexFromCategory(CooperativeAction::ActionCategory category)
    {
        switch (category)
        {
        case CooperativeAction::Hold:
            return 0;
        case CooperativeAction::Dribble:
            return 1;
        case CooperativeAction::Pass:
            return 2;
        case CooperativeAction::Shoot:
            return 3;
        default:
            return -1;
        }
    }

    const char *categoryName(CooperativeAction::ActionCategory category)
    {
        switch (category)
        {
        case CooperativeAction::Hold:
            return "Hold";
        case CooperativeAction::Dribble:
            return "Dribble";
        case CooperativeAction::Pass:
            return "Pass";
        case CooperativeAction::Shoot:
            return "Shoot";
        case CooperativeAction::Clear:
            return "Clear";
        case CooperativeAction::Move:
            return "Move";
        case CooperativeAction::NoAction:
            return "NoAction";
        default:
            return "Unknown";
        }
    }

    std::vector<double> computeHeuristics(const WorldModel &wm)
    {
        std::vector<double> heuristics;
        heuristics.reserve(10);

        const ServerParam &SP = ServerParam::i();

        heuristics.push_back(wm.ball().pos().x);
        heuristics.push_back(std::abs(wm.ball().pos().y));

        double dist_to_goal = SP.theirTeamGoalPos().dist(wm.ball().pos());
        heuristics.push_back(std::exp(-dist_to_goal / 10.0));

        const PlayerType *self_type = wm.self().playerTypePtr();
        double kickable_area = self_type ? self_type->kickableArea() : 0.0;
        bool is_kickable = wm.self().pos().dist(wm.ball().pos()) <= kickable_area;
        heuristics.push_back(is_kickable ? 1.0 : 0.0);

        double rel_vel = (wm.self().vel() - wm.ball().vel()).r();
        heuristics.push_back(std::tanh(rel_vel));

        heuristics.push_back(std::tanh(wm.ball().vel().r()));

        double dist_to_ball = wm.self().pos().dist(wm.ball().pos());
        heuristics.push_back(std::exp(-dist_to_ball));

        double goal_line_proximity = std::max(0.0, std::abs(wm.ball().pos().x) - (SP.pitchHalfLength() - 5.0)) / 5.0;
        heuristics.push_back(goal_line_proximity);

        heuristics.push_back(std::tanh(wm.self().vel().r()));

        heuristics.push_back(std::cos(wm.self().body().radian()));

        return heuristics;
    }

    class IntentionTurnTo
        : public SoccerIntention
    {
    private:
        int M_step;
        Vector2D M_target_point;

    public:
        IntentionTurnTo(const Vector2D &target_point)
            : M_step(0),
              M_target_point(target_point)
        {
        }

        bool finished(const PlayerAgent *agent);

        bool execute(PlayerAgent *agent);

    private:
    };

    // // 構造体を修正
    // struct StepData
    // {
    //     std::vector<double> features;
    //     int cycle; // サイクル数
    //     int action_index;
    //     double reward;
    //     int player_num;
    // };
    // static std::vector<StepData> episode_buffer;
    // static bool prev_our_ball = false;

    /*-------------------------------------------------------------------*/
    /*!

     */
    bool
    IntentionTurnTo::finished(const PlayerAgent *agent)
    {
        ++M_step;

        dlog.addText(Logger::TEAM,
                     __FILE__ ": (finished) step=%d",
                     M_step);

        if (M_step >= 2)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": (finished) time over");
            return true;
        }

        const WorldModel &wm = agent->world();

        //
        // check kickable
        //

        if (!wm.self().isKickable())
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": (finished) no kickable");
            return true;
        }

        //
        // check opponent
        //

        if (wm.kickableOpponent())
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": (finished) exist kickable opponent");
            return true;
        }

        if (wm.interceptTable().opponentStep() <= 1)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": (finished) opponent may be kickable");
            return true;
        }

        //
        // check next kickable
        //

        double kickable2 = std::pow(wm.self().playerType().kickableArea() - wm.self().vel().r() * ServerParam::i().playerRand() - wm.ball().vel().r() * ServerParam::i().ballRand() - 0.15,
                                    2);
        Vector2D self_next = wm.self().pos() + wm.self().vel();
        Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

        if (self_next.dist2(ball_next) > kickable2)
        {
            // unkickable if turn is performed.
            dlog.addText(Logger::TEAM,
                         __FILE__ ": (finished) unkickable at next cycle");
            return true;
        }

        return false;
    }

    /*-------------------------------------------------------------------*/
    /*!

     */
    bool
    IntentionTurnTo::execute(PlayerAgent *agent)
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (intention) facePoint=(%.1f %.1f)",
                     M_target_point.x, M_target_point.y);
        agent->debugClient().addMessage("IntentionTurnToForward");

        Body_TurnToPoint(M_target_point).execute(agent);
        agent->setNeckAction(new Neck_ScanField());

        return true;
    }

}

/*-------------------------------------------------------------------*/
/*!

 */
std::shared_ptr<torch::jit::script::Module> Bhv_PlannedAction::nn_model_ = nullptr;

Bhv_PlannedAction::Bhv_PlannedAction()
    : M_chain_graph(ActionChainHolder::i().graph())
{
    // モデルのロード（静的メンバー変数を使用）
    if (!nn_model_)
    {
        try
        {
            std::string model_path = "/home/okayama/rcss/policy-gradient/model.pt";

            // ファイルが存在するか確認
            if (std::filesystem::exists(model_path))
            {
                nn_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
                nn_model_->eval(); // 推論モードに設定
                std::cerr << "[INFO] Successfully loaded NN model from: " << model_path << std::endl;
            }
            else
            {
                std::cerr << "[WARN] Model file not found. Using default weights." << std::endl;
                // デフォルトの重みを設定（ランダムまたは固定値）
                nn_model_ = nullptr; // 必要に応じて、モデルなしで動作するロジックを追加
            }
        }
        catch (const c10::Error &e)
        {
            std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
            nn_model_ = nullptr; // モデルがロードできなかった場合は nullptr に設定
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool Bhv_PlannedAction::execute(PlayerAgent *agent)
{
    dlog.addText(Logger::TEAM, __FILE__ ": Bhv_PlannedAction");

    if (doTurnToForward(agent))
        return true;

    const ServerParam &SP = ServerParam::i();
    const WorldModel &wm = agent->world();

    std::vector<double> features = {
        wm.ball().pos().x, wm.ball().pos().y,
        wm.self().pos().x, wm.self().pos().y,
        wm.self().vel().x, wm.self().vel().y};

    std::vector<double> heuristics = computeHeuristics(wm);

    std::vector<double> policy_probs;
    bool policy_available = false;

    if (nn_model_)
    {
        std::vector<float> state_input;
        state_input.reserve(features.size());
        for (double value : features)
        {
            state_input.push_back(static_cast<float>(value));
        }

        std::vector<float> heuristic_input;
        heuristic_input.reserve(heuristics.size());
        for (double value : heuristics)
        {
            heuristic_input.push_back(static_cast<float>(value));
        }

        torch::Tensor state_tensor = torch::from_blob(state_input.data(), {(long)state_input.size()}, torch::kFloat32).clone().unsqueeze(0);
        torch::Tensor heuristic_tensor = torch::from_blob(heuristic_input.data(), {(long)heuristic_input.size()}, torch::kFloat32).clone().unsqueeze(0);

        try
        {
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(state_tensor);
            inputs.emplace_back(heuristic_tensor);

            const auto outputs_tuple = nn_model_->forward(inputs).toTuple();
            if (outputs_tuple && outputs_tuple->elements().size() >= 2)
            {
                torch::Tensor logits_tensor = outputs_tuple->elements()[0].toTensor().squeeze(0);
                if (S_POLICY_TEMPERATURE > 1e-6 && std::fabs(S_POLICY_TEMPERATURE - 1.0) > 1e-6)
                {
                    logits_tensor = logits_tensor / S_POLICY_TEMPERATURE;
                }
                torch::Tensor probs_tensor = torch::softmax(logits_tensor, 0);
                torch::Tensor probs_cpu = probs_tensor.detach().cpu();

                policy_probs.resize(probs_cpu.size(0));
                for (int i = 0; i < probs_cpu.size(0); ++i)
                {
                    policy_probs[i] = probs_cpu[i].item<double>();
                }
                policy_available = true;

                std::cout << "[NN-ACT] cycle=" << wm.time().cycle()
                          << " unum=" << wm.self().unum()
                          << " probs=";
                for (int i = 0; i < static_cast<int>(policy_probs.size()); ++i)
                {
                    std::cout << std::fixed << std::setprecision(6)
                              << policy_probs[i];
                    if (i + 1 < static_cast<int>(policy_probs.size()))
                    {
                        std::cout << ",";
                    }
                }
                std::cout << " eps=" << std::fixed << std::setprecision(3) << s_epsilon << std::endl;
            }
            else
            {
                std::cerr << "[ERROR] Unexpected output format from NN policy." << std::endl;
            }
        }
        catch (const c10::Error &e)
        {
            std::cerr << "[ERROR] NN forward failed in Bhv_PlannedAction: " << e.what() << std::endl;
        }
    }

    const Vector2D goal_pos = SP.theirTeamGoalPos();
    agent->setNeckAction(new Neck_TurnToReceiver(M_chain_graph));

    // 候補取得＆デバッグ表示
    const CooperativeAction &first_action = M_chain_graph.getFirstAction();
    const auto all_chain = M_chain_graph.getAllChain();
    ActionChainGraph::debug_send_chain(agent, all_chain);

    // ε 減衰
    s_epsilon = std::max(s_epsilon * s_eps_decay, s_eps_min);

    // ε-greedy 選択
    const CooperativeAction *chosen_ptr = &first_action; // フォールバック(=活用)
    int chosen_idx = 0;

    if (!all_chain.empty())
    {
        std::vector<int> feasible_idx;
        feasible_idx.reserve(all_chain.size());
        for (int i = 0; i < (int)all_chain.size(); ++i)
        {
            if (is_feasible(all_chain[i].action(), wm))
                feasible_idx.push_back(i);
        }

        const int seed_base = wm.time().cycle() * 131 + wm.self().unum();
        if (feasible_idx.empty())
        {
            chosen_idx = 0;
            chosen_ptr = &all_chain[chosen_idx].action();
            agent->debugClient().addMessage("EXPLOIT_NOFEAS");
            dlog.addText(Logger::TEAM, __FILE__ " no feasible action, fallback to first cat=%d",
                         (int)chosen_ptr->category());
        }
        else
        {
            const bool explore = (rand01_det(seed_base) < s_epsilon);

            if (explore)
            {
                const int pick = randint_det(seed_base + 17, 0, (int)feasible_idx.size() - 1);
                chosen_idx = feasible_idx[pick];
                chosen_ptr = &all_chain[chosen_idx].action();
                agent->debugClient().addMessage("EXPLORE_eps");
                dlog.addText(Logger::TEAM, __FILE__ " ε-greedy: EXPLORE eps=%.3f cat=%d",
                             s_epsilon, (int)chosen_ptr->category());
            }
            else
            {
                if (policy_available)
                {
                    int best_index = -1;
                    double best_prob = -1.0;
                    for (int idx : feasible_idx)
                    {
                        int policy_index = policyIndexFromCategory(all_chain[idx].action().category());
                        if (policy_index < 0 || policy_index >= static_cast<int>(policy_probs.size()))
                        {
                            continue;
                        }

                        double prob = policy_probs[policy_index];
                        if (prob > best_prob)
                        {
                            best_prob = prob;
                            best_index = idx;
                        }
                    }

                    if (best_index != -1)
                    {
                        chosen_idx = best_index;
                        chosen_ptr = &all_chain[chosen_idx].action();
                        agent->debugClient().addMessage("EXPLOIT_POLICY");
                        dlog.addText(Logger::TEAM, __FILE__ " policy exploit eps=%.3f cat=%d prob=%.3f",
                                     s_epsilon, (int)chosen_ptr->category(), best_prob);
                        std::cout << "[NN-ACT] select=" << categoryName(chosen_ptr->category())
                                  << " prob=" << std::fixed << std::setprecision(6) << best_prob
                                  << std::endl;
                    }
                    else
                    {
                        chosen_idx = feasible_idx.front();
                        chosen_ptr = &all_chain[chosen_idx].action();
                        agent->debugClient().addMessage("EXPLOIT_FALLBACK");
                        dlog.addText(Logger::TEAM, __FILE__ " policy fallback eps=%.3f cat=%d",
                                     s_epsilon, (int)chosen_ptr->category());
                    }
                }
                else
                {
                    chosen_idx = feasible_idx.front();
                    chosen_ptr = &all_chain[chosen_idx].action();
                    agent->debugClient().addMessage("EXPLOIT_DEFAULT");
                    dlog.addText(Logger::TEAM, __FILE__ " default exploit eps=%.3f cat=%d",
                                 s_epsilon, (int)chosen_ptr->category());
                }
            }
        }
    }

    const CooperativeAction &chosen_action = *chosen_ptr;

    /******************* 報酬計算 *******************/
    double reward = 0.0;

    static bool s_progress_initialized = false;
    static double s_prev_ball_x = 0.0;
    static double s_prev_goal_dist = 0.0;

    const double current_ball_x = wm.ball().pos().x;
    const double current_goal_dist = goal_pos.dist(wm.ball().pos());

    if (!s_progress_initialized)
    {
        s_prev_ball_x = current_ball_x;
        s_prev_goal_dist = current_goal_dist;
        s_progress_initialized = true;
    }

    const double delta_ball_x = current_ball_x - s_prev_ball_x;
    const double delta_goal_dist = s_prev_goal_dist - current_goal_dist;

    // 進行/接近量を±1でクリップし、1.0刻みの報酬に統一
    const double clamped_dx = std::clamp(delta_ball_x, -1.0, 1.0);
    const double clamped_goal = std::clamp(delta_goal_dist, -1.0, 1.0);
    reward += clamped_dx;
    reward += clamped_goal;

    s_prev_ball_x = current_ball_x;
    s_prev_goal_dist = current_goal_dist;

    /******************* 特徴量 & バッファ *******************/
    StepData step;
    step.features = features;
    step.cycle = wm.time().cycle();
    step.action_index = static_cast<int>(chosen_action.category()); // ← 修正：typo & chosen_action
    step.reward = reward;
    step.player_num = wm.self().unum();
    step.heuristics = heuristics;
    episode_buffer.push_back(step);

    /******************* 行動実行 *******************/
    switch (chosen_action.category())
    {
    case CooperativeAction::Shoot:
    {
        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) shoot");
        if (Body_ForceShoot().execute(agent))
        {
            agent->setNeckAction(new Neck_TurnToGoalieOrScan(2));
            return true;
        }
        break;
    }

    case CooperativeAction::Dribble:
    {
        if (wm.gameMode().type() != GameMode::PlayOn && !wm.gameMode().isPenaltyKickMode())
        {
            agent->debugClient().addMessage("CancelChainDribble");
            dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) cancel dribble");
            return false;
        }

        const Vector2D &dribble_target = chosen_action.targetPoint(); // ← first→chosen

        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) dribble target=(%.1f %.1f)",
                     dribble_target.x, dribble_target.y);

        NeckAction::Ptr neck;
        double goal_dist = goal_pos.dist(dribble_target);
        int count_thr = (goal_dist < 18.0 ? (goal_dist < 13.0 ? -1 : 0) : 0);
        if (goal_dist < 18.0)
        {
            agent->debugClient().addMessage("ChainDribble:LookGoalie");
            neck = NeckAction::Ptr(new Neck_TurnToGoalieOrScan(count_thr));
        }

        if (Bhv_NormalDribble(chosen_action, neck).execute(agent))
        { // ← first→chosen
            return true;
        }
        break;
    }

    case CooperativeAction::Hold:
    {
        if (wm.gameMode().type() != GameMode::PlayOn)
        {
            agent->debugClient().addMessage("CancelChainHold");
            dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) cancel hold");
            return false;
        }

        if (wm.ball().pos().x < -SP.pitchHalfLength() + 8.0 && wm.ball().pos().absY() < SP.goalHalfWidth() + 1.0)
        {
            agent->debugClient().addMessage("ChainHold:Clear");
            dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) cancel hold. clear ball");
            Body_ClearBall().execute(agent);
            agent->setNeckAction(new Neck_ScanField());
            return true;
        }

        agent->debugClient().addMessage("hold");
        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) hold");
        Body_HoldBall().execute(agent);
        agent->setNeckAction(new Neck_ScanField());
        return true;
    }

    case CooperativeAction::Pass:
    {
        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) pass");
        Bhv_PassKickFindReceiver(M_chain_graph).execute(agent);
        return true;
    }

    case CooperativeAction::Move:
    {
        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) move");
        if (Body_GoToPoint(chosen_action.targetPoint(), 1.0, SP.maxDashPower()).execute(agent))
        { // ← first→chosen
            agent->setNeckAction(new Neck_ScanField());
            return true;
        }
        break;
    }

    case CooperativeAction::NoAction:
    {
        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) no action");
        return true;
    }

    default:
        dlog.addText(Logger::TEAM, __FILE__ " (Bhv_PlannedAction) invalid category");
        break;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool Bhv_PlannedAction::doTurnToForward(PlayerAgent *agent)
{
    const WorldModel &wm = agent->world();

    if (wm.gameMode().type() != GameMode::PlayOn)
    {
        return false;
    }

    Vector2D face_point(42.0, 0.0);

    const double body_angle_diff = ((face_point - wm.self().pos()).th() - wm.self().body()).abs();
    if (body_angle_diff < 110.0)
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ " (doTurnToForward) already facing the forward direction. angle_diff=%.1f",
                     body_angle_diff);
        return false;
    }

    dlog.addText(Logger::TEAM,
                 __FILE__ " (doTurnToForward) angle_diff=%.1f. try turn",
                 body_angle_diff);

    // const double opponent_dist_thr = ( wm.self().pos().x > ServerParam::i().theirPenaltyAreaLineX() - 2.0
    //                                    && wm.self().pos().absY() > ServerParam::i().goalHalfWidth()
    //                                    ? 2.7
    //                                    : 4.0 );
    const double opponent_dist_thr = 4.0;

    for (PlayerObject::Cont::const_iterator o = wm.opponentsFromSelf().begin(),
                                            end = wm.opponentsFromSelf().end();
         o != end;
         ++o)
    {
        double dist = (*o)->distFromSelf();
        dist -= bound(0, (*o)->posCount(), 3) * (*o)->playerTypePtr()->realSpeedMax();

        if (dist < opponent_dist_thr)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ " (doTurnToForward) exist opponent");
            return false;
        }

        if (dist > 10.0)
        {
            break;
        }
    }

    // TODO: find the best scan target angle
    face_point.y = wm.self().pos().y * 0.5;

    double kickable2 = std::pow(wm.self().playerType().kickableArea() - wm.self().vel().r() * ServerParam::i().playerRand() - wm.ball().vel().r() * ServerParam::i().ballRand() - 0.2,
                                2);
    Vector2D self_next = wm.self().pos() + wm.self().vel();
    Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

    if (self_next.dist2(ball_next) < kickable2)
    {
        Body_TurnToPoint(face_point).execute(agent);
        agent->setNeckAction(new Neck_ScanField());
        return true;
    }

    Vector2D ball_vel = getKeepBallVel(agent->world());

    if (!ball_vel.isValid())
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (doKeepBall) no candidate.");

        return false;
    }

    //
    // perform first kick
    //

    Vector2D kick_accel = ball_vel - wm.ball().vel();
    double kick_power = kick_accel.r() / wm.self().kickRate();
    AngleDeg kick_angle = kick_accel.th() - wm.self().body();

    dlog.addText(Logger::TEAM,
                 __FILE__ ": (doTurnToForward) "
                          " ballVel=(%.2f %.2f)"
                          " kickPower=%.1f kickAngle=%.1f",
                 ball_vel.x, ball_vel.y,
                 kick_power,
                 kick_angle.degree());

    if (kick_power > ServerParam::i().maxPower())
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (doTurnToForward) over kick power");
        Body_HoldBall(true,
                      face_point)
            .execute(agent);
        agent->setNeckAction(new Neck_ScanField());
    }
    else
    {
        agent->doKick(kick_power, kick_angle);
        agent->setNeckAction(new Neck_ScanField());
    }

    agent->debugClient().addMessage("Chain:Turn:Keep");
    agent->debugClient().setTarget(face_point);

    //
    // set turn intention
    //

    dlog.addText(Logger::TEAM,
                 __FILE__ ": (doTurnToFoward) register intention");
    agent->setIntention(new IntentionTurnTo(face_point));

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
Vector2D
Bhv_PlannedAction::getKeepBallVel(const WorldModel &wm)
{
    static GameTime s_update_time(0, 0);
    static Vector2D s_best_ball_vel(0.0, 0.0);

    if (s_update_time == wm.time())
    {
        return s_best_ball_vel;
    }
    s_update_time = wm.time();

    //
    //
    //

    const int ANGLE_DIVS = 12;

    const ServerParam &SP = ServerParam::i();
    const PlayerType &ptype = wm.self().playerType();
    const double collide_dist2 = std::pow(ptype.playerSize() + SP.ballSize(),
                                          2);
    const double keep_dist = ptype.playerSize() + ptype.kickableMargin() * 0.5 + ServerParam::i().ballSize();

    const Vector2D next_self_pos = wm.self().pos() + wm.self().vel();
    const Vector2D next2_self_pos = next_self_pos + wm.self().vel() * ptype.playerDecay();

    //
    // create keep target point
    //

    Vector2D best_ball_vel = Vector2D::INVALIDATED;
    int best_opponent_step = 0;
    double best_ball_speed = 1000.0;

    for (int a = 0; a < ANGLE_DIVS; ++a)
    {
        Vector2D keep_pos = next2_self_pos + Vector2D::from_polar(keep_dist,
                                                                  360.0 / ANGLE_DIVS * a);
        if (keep_pos.absX() > SP.pitchHalfLength() - 0.2 || keep_pos.absY() > SP.pitchHalfWidth() - 0.2)
        {
            continue;
        }

        Vector2D ball_move = keep_pos - wm.ball().pos();
        double ball_speed = ball_move.r() / (1.0 + SP.ballDecay());

        Vector2D max_vel = KickTable::calc_max_velocity(ball_move.th(),
                                                        wm.self().kickRate(),
                                                        wm.ball().vel());
        if (max_vel.r2() < std::pow(ball_speed, 2))
        {
            continue;
        }

        Vector2D ball_next_next = keep_pos;

        Vector2D ball_vel = ball_move.setLengthVector(ball_speed);
        Vector2D ball_next = wm.ball().pos() + ball_vel;

        if (next_self_pos.dist2(ball_next) < collide_dist2)
        {
            ball_next_next = ball_next;
            ball_next_next += ball_vel * (SP.ballDecay() * -0.1);
        }

#ifdef DEBUG_PRINT
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (getKeepBallVel) %d: ball_move th=%.1f speed=%.2f max=%.2f",
                     a,
                     ball_move.th().degree(),
                     ball_speed,
                     max_vel.r());
        dlog.addText(Logger::TEAM,
                     __FILE__ ": __ ball_next=(%.2f %.2f) ball_next2=(%.2f %.2f)",
                     ball_next.x, ball_next.y,
                     ball_next_next.x, ball_next_next.y);
#endif

        //
        // check opponent
        //

        int min_step = 1000;
        for (PlayerObject::Cont::const_iterator o = wm.opponentsFromSelf().begin(),
                                                end = wm.opponentsFromSelf().end();
             o != end;
             ++o)
        {
            if ((*o)->distFromSelf() > 10.0)
            {
                break;
            }

            int o_step = FieldAnalyzer::predict_player_reach_cycle(*o,
                                                                   ball_next_next,
                                                                   (*o)->playerTypePtr()->kickableArea(),
                                                                   0.0, // penalty distance
                                                                   1,   // body count thr
                                                                   1,   // default turn step
                                                                   0,   // wait cycle
                                                                   true);

            if (o_step <= 0)
            {
                break;
            }

            if (o_step < min_step)
            {
                min_step = o_step;
            }
        }
#ifdef DEBUG_PRINT
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (getKeepBallVel) %d: keepPos=(%.2f %.2f)"
                              " ballNext2=(%.2f %.2f) ballVel=(%.2f %.2f) speed=%.2f o_step=%d",
                     a,
                     keep_pos.x, keep_pos.y,
                     ball_next_next.x, ball_next_next.y,
                     ball_vel.x, ball_vel.y,
                     ball_speed,
                     min_step);
#endif
        if (min_step > best_opponent_step)
        {
            best_ball_vel = ball_vel;
            best_opponent_step = min_step;
            best_ball_speed = ball_speed;
        }
        else if (min_step == best_opponent_step)
        {
            if (best_ball_speed > ball_speed)
            {
                best_ball_vel = ball_vel;
                best_opponent_step = min_step;
                best_ball_speed = ball_speed;
            }
        }
    }

    s_best_ball_vel = best_ball_vel;
    return s_best_ball_vel;
}
