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

#include "basic_actions/kick_table.h"
#include "sample_field_evaluator.h" // ヒューリスティック計算用

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

using namespace rcsc;

namespace {

class IntentionTurnTo
    : public SoccerIntention {
private:
    int M_step;
    Vector2D M_target_point;

public:

    IntentionTurnTo( const Vector2D & target_point )
        : M_step( 0 ),
          M_target_point( target_point )
      { }

    bool finished( const PlayerAgent * agent );

    bool execute( PlayerAgent * agent );

private:

};

// 構造体を修正
struct StepData {
    std::vector<double> features;
    int cycle; // サイクル数
    int action_index;
    double reward;
};
static std::vector<StepData> episode_buffer;
static bool prev_our_ball = false;


/*-------------------------------------------------------------------*/
/*!

 */
bool
IntentionTurnTo::finished( const PlayerAgent * agent )
{
    ++M_step;

    dlog.addText( Logger::TEAM,
                  __FILE__": (finished) step=%d",
                  M_step );

    if ( M_step >= 2 )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (finished) time over" );
        return true;
    }

    const WorldModel & wm = agent->world();

    //
    // check kickable
    //

    if ( ! wm.self().isKickable() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (finished) no kickable" );
        return true;
    }

    //
    // check opponent
    //

    if ( wm.kickableOpponent() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (finished) exist kickable opponent" );
        return true;
    }

    if ( wm.interceptTable().opponentStep() <= 1 )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (finished) opponent may be kickable" );
        return true;
    }

    //
    // check next kickable
    //

    double kickable2 = std::pow( wm.self().playerType().kickableArea()
                                 - wm.self().vel().r() * ServerParam::i().playerRand()
                                 - wm.ball().vel().r() * ServerParam::i().ballRand()
                                 - 0.15,
                                 2 );
    Vector2D self_next = wm.self().pos() + wm.self().vel();
    Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

    if ( self_next.dist2( ball_next ) > kickable2 )
    {
        // unkickable if turn is performed.
        dlog.addText( Logger::TEAM,
                      __FILE__": (finished) unkickable at next cycle" );
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
IntentionTurnTo::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::TEAM,
                  __FILE__": (intention) facePoint=(%.1f %.1f)",
                  M_target_point.x, M_target_point.y );
    agent->debugClient().addMessage( "IntentionTurnToForward" );

    Body_TurnToPoint( M_target_point ).execute( agent );
    agent->setNeckAction( new Neck_ScanField() );

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
    if (!nn_model_) {
        try {
            std::string model_path = "/home/okayama/rcss/policy-gradient/model_with_attention.pt";

            // ファイルが存在するか確認
            if (std::filesystem::exists(model_path)) {
                nn_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
                nn_model_->eval(); // 推論モードに設定
                std::cerr << "[INFO] Successfully loaded NN model from: " << model_path << std::endl;
            } else {
                std::cerr << "[WARN] Model file not found. Using default weights." << std::endl;
                // デフォルトの重みを設定（ランダムまたは固定値）
                nn_model_ = nullptr; // 必要に応じて、モデルなしで動作するロジックを追加
            }
        } catch (const c10::Error &e) {
            std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
            nn_model_ = nullptr; // モデルがロードできなかった場合は nullptr に設定
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool Bhv_PlannedAction::execute(PlayerAgent * agent)
{
    dlog.addText(Logger::TEAM, __FILE__": Bhv_PlannedAction");
    std::cerr << "[DEBUG] Entering Bhv_PlannedAction::execute for player: " 
              << agent->world().self().unum() << std::endl;

    if (doTurnToForward(agent)) {
        return true;
    }

    const ServerParam & SP = ServerParam::i();
    const WorldModel & wm = agent->world();

    // 候補アクションの取得
    std::vector<ActionStatePair> candidates;

    // PredictState を生成
    PredictState predict_state(wm);

    // アクション候補を生成
    ActionChainHolder::instance().actionGenerator()->generate(&candidates, predict_state, wm, {});

    std::cerr << "[DEBUG] Number of action candidates: " << candidates.size() << std::endl;
    if (candidates.empty()) {
        std::cerr << "[ERROR] No action candidates generated." << std::endl;
        return false;
    }

    // ニューラルネットワークがロードされていない場合の処理
    if (!nn_model_) {
        std::cerr << "[WARN] NN model is not loaded. Using heuristic-based action selection." << std::endl;

        // ランダムに行動を選択
        static std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
        int selected_idx = dist(gen);

        const CooperativeAction & selected_action = candidates[selected_idx].action();
        std::cerr << "[INFO] Selected action (heuristic): " << selected_action.category() << std::endl;
    }

    // 状態ベクトル作成
    torch::Tensor input = torch::tensor({
        wm.ball().pos().x / 50.0,
        wm.ball().pos().y / 34.0,
        wm.self().pos().x / 50.0,
        wm.self().pos().y / 34.0,
        wm.self().vel().x / 5.0,
        wm.self().vel().y / 5.0
    }, torch::kFloat).unsqueeze(0);  // shape: [1, 6]

    // ヒューリスティックベクトル作成（例として10個のダミー値）
    std::vector<float> dummy_heuristics_vec = {1.0, 0.5, 0.3, 0.7, 0.2, 0.9, 0.1, 0.6, 0.4, 0.8};
    torch::Tensor heuristics = torch::from_blob(dummy_heuristics_vec.data(), {1, 10, 1}, torch::kFloat32).clone();  // shape: [1, 10, 1]

    std::cerr << "[DEBUG] Input tensor shape: " << input.sizes() << std::endl;
    std::cerr << "[DEBUG] Heuristics tensor shape: " << heuristics.sizes() << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    inputs.push_back(heuristics);

    torch::Tensor logits, weights;
    try {
        auto outputs = nn_model_->forward(inputs).toTuple();
        logits = outputs->elements()[0].toTensor().squeeze();   // ✅ ここで次元圧縮
        weights = outputs->elements()[1].toTensor().squeeze();  // ✅ ここも同様に
        std::cerr << "[DEBUG] Model logits: " << logits << std::endl;
        std::cerr << "[DEBUG] Model weights: " << weights << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "[ERROR] NN forward failed: " << e.what() << std::endl;
        return false;
    }

    // Softmax による確率計算
    torch::Tensor probabilities = torch::softmax(logits, 0); // logits is 1D tensor: [num_actions]
    if (probabilities.dim() != 1 || probabilities.size(0) <= 1) {
        std::cerr << "[ERROR] Softmax applied to invalid or trivial output." << std::endl;
        return false;
    }
    std::cerr << "[DEBUG] Probabilities: " << probabilities << std::endl;

    // 確率に基づいて行動をサンプリング
    std::vector<float> probs(
        probabilities.data_ptr<float>(),
        probabilities.data_ptr<float>() + probabilities.size(0)
    );

    static std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int selected_idx = dist(gen);

    if (selected_idx < 0 || selected_idx >= static_cast<int>(candidates.size())) {
        dlog.addText(Logger::TEAM, __FILE__": Invalid action index sampled: %d", selected_idx);
        return false;
    }

    const CooperativeAction & selected_action = candidates[selected_idx].action();
    dlog.addText(Logger::TEAM, __FILE__": Selected action category: %d", selected_action.category());

    /********************************************************************
    * 報酬の計算ブロック
    *******************************************************************/
    double reward = 0.0;

    // 1. 前進距離（正：敵ゴール方向，負：自陣方向）
    const double ball_vel_x = wm.ball().vel().x;
    const double VEL_COEFF = 1.0;

    if (std::abs(ball_vel_x) > 0.1) { // ある程度動いていれば
        if (ball_vel_x > 0.0) {
            reward += VEL_COEFF;
            std::cerr << "[DEBUG] BALL MOVING FORWARD reward: +" << VEL_COEFF << std::endl;
        } else {
            reward -= VEL_COEFF;
            std::cerr << "[DEBUG] BALL MOVING BACKWARD penalty: -" << VEL_COEFF << std::endl;
        }
    }

    // 2. ゴール・失点（AfterGoal_ 時に判定）
    if (wm.gameMode().type() == GameMode::AfterGoal_) {
        if (wm.lastKickerSide() == wm.ourSide()) {
            reward += 100.0;
            std::cerr << "[DEBUG] GOAL reward: +100.0" << std::endl;
        } else {
            reward -= 100.0;
            std::cerr << "[DEBUG] LOST GOAL penalty: -100.0" << std::endl;
        }
    }

    // 3. 枠内シュート可能か
    rcsc::AbstractPlayerObject::Cont opponents;
    for (const auto &opponent : wm.opponentsFromSelf()) {
        opponents.push_back(opponent);
    }
    if (FieldAnalyzer::can_shoot_from(true, wm.self().pos(), opponents, 8)) {
        reward += 5.0;
        std::cerr << "[DEBUG] SHOOTABLE reward: +5.0" << std::endl;
    }

    // 4. パス成功
    const rcsc::AbstractPlayerObject * holder = predict_state.ballHolder();
    if (holder != nullptr && holder->side() == wm.ourSide()) {
        reward += 3.0;
        std::cerr << "[DEBUG] PASS SUCCESS reward: +3.0" << std::endl;
    }

    // 5. ボールロスト
    static const rcsc::AbstractPlayerObject * prev_holder = nullptr;
    const rcsc::AbstractPlayerObject * current_holder = predict_state.ballHolder();
    if (current_holder && current_holder->side() != wm.ourSide()) {
        reward -= 1.0;
        std::cerr << "[DEBUG] BALL CONTROL BY OPPONENT: -1.0" << std::endl;
    }
    prev_holder = current_holder;

    // 6. 奪われそうな位置に相手がいる
    if (wm.kickableOpponent() != nullptr) {
        reward -= 1.0;
        std::cerr << "[DEBUG] KICKABLE OPPONENT penalty: -1.0" << std::endl;
    }

    std::cerr << "[DEBUG] TOTAL REWARD: " << reward << std::endl;


    // SampleFieldEvaluator のインスタンスを作成
    SampleFieldEvaluator evaluator;

    // heuristics_tensor ではなく、最初から vector で受け取る！
    std::vector<double> heuristics_vec = evaluator.calculateHeuristics(predict_state);


    // 状態特徴量ベクトルを作成
    std::vector<double> features = {
        wm.ball().pos().x, wm.ball().pos().y,
        wm.self().pos().x, wm.self().pos().y,
        wm.self().vel().x, wm.self().vel().y
    };

    // バッファに追加
    StepData step;
    step.features = features;
    step.cycle = wm.time().cycle();
    step.action_index = static_cast<int>(selected_idx);
    step.reward = reward;
    episode_buffer.push_back(step);

    // エピソード終了判定
    bool our_ball = (predict_state.ballHolder()->side() == wm.ourSide());
    bool episode_end = false;
    if (prev_our_ball && (!our_ball || wm.gameMode().type() == GameMode::AfterGoal_ || wm.gameMode().type() != GameMode::PlayOn)) {
        episode_end = true;
    }
    prev_our_ball = our_ball;

    // エピソード終了時のCSV出力
    if (episode_end && !episode_buffer.empty()) {
        double gamma = 0.99;
        double G = 0.0;
        std::vector<double> returns(episode_buffer.size());
        for (int t = episode_buffer.size() - 1; t >= 0; --t) {
            G = episode_buffer[t].reward + gamma * G;
            returns[t] = G;
        }
        static std::ofstream csv_file("/home/okayama/rcss/policy-gradient/logs/data.csv", std::ios::out | std::ios::app);
        if (csv_file.tellp() == 0) {
            csv_file << "ball_x,ball_y,player_x,player_y,player_vel_x,player_vel_y,cycle,action_index,discounted_reward" << std::endl;
        }
        for (size_t t = 0; t < episode_buffer.size(); ++t) {
            for (auto v : episode_buffer[t].features) csv_file << v << ",";
            csv_file << episode_buffer[t].cycle << ",";
            csv_file << episode_buffer[t].action_index << "," << returns[t] << std::endl;
        }
        episode_buffer.clear();
}

    // 選択したアクションを実行
    switch (selected_action.category()) {
    case CooperativeAction::Shoot:
        {
            dlog.addText( Logger::TEAM,
                          __FILE__" (Bhv_PlannedAction) shoot" );
            if ( Body_ForceShoot().execute( agent ) )
            {
                agent->setNeckAction( new Neck_TurnToGoalieOrScan( 2 ) );
                return true;
            }

            break;
        }

    case CooperativeAction::Dribble:
        {
            if ( wm.gameMode().type() != GameMode::PlayOn
                 && ! wm.gameMode().isPenaltyKickMode() )
            {
                agent->debugClient().addMessage( "CancelChainDribble" );
                dlog.addText( Logger::TEAM,
                              __FILE__" (Bhv_PlannedAction) cancel dribble" );
                return false;
            }

            const Vector2D & dribble_target = selected_action.targetPoint();

            dlog.addText( Logger::TEAM,
                          __FILE__" (Bhv_PlannedAction) dribble target=(%.1f %.1f)",
                          dribble_target.x, dribble_target.y );

            NeckAction::Ptr neck;
            const Vector2D goal_pos = SP.theirTeamGoalPos();
            double goal_dist = goal_pos.dist( dribble_target );
            if ( goal_dist < 18.0 )
            {
                int count_thr = 0;
                if ( goal_dist < 13.0 )
                {
                    count_thr = -1;
                }
                agent->debugClient().addMessage( "ChainDribble:LookGoalie" );
                neck = NeckAction::Ptr( new Neck_TurnToGoalieOrScan( count_thr ) );
            }

            if ( Bhv_NormalDribble( selected_action, neck ).execute( agent ) )
            {
                return true;
            }
            break;
        }

    case CooperativeAction::Hold:
        {
            if ( wm.gameMode().type() != GameMode::PlayOn )
            {
                agent->debugClient().addMessage( "CancelChainHold" );
                dlog.addText( Logger::TEAM,
                              __FILE__" (Bhv_PlannedAction) cancel hold" );
                return false;
            }

            if ( wm.ball().pos().x < -SP.pitchHalfLength() + 8.0
                 && wm.ball().pos().absY() < SP.goalHalfWidth() + 1.0 )
            {
                agent->debugClient().addMessage( "ChainHold:Clear" );
                dlog.addText( Logger::TEAM,
                              __FILE__" (Bhv_PlannedAction) cancel hold. clear ball" );
                Body_ClearBall().execute( agent );
                agent->setNeckAction( new Neck_ScanField() );
                return true;
            }

            agent->debugClient().addMessage( "hold" );
            dlog.addText( Logger::TEAM,
                          __FILE__" (Bhv_PlannedAction) hold" );

            Body_HoldBall().execute( agent );
            agent->setNeckAction( new Neck_ScanField() );
            return true;
            break;
        }

    case CooperativeAction::Pass:
        {
            dlog.addText( Logger::TEAM,
                          __FILE__" (Bhv_PlannedAction) pass" );
            Bhv_PassKickFindReceiver( M_chain_graph ).execute( agent );
            return true;
            break;
        }

    case CooperativeAction::Move:
        {
            dlog.addText( Logger::TEAM,
                          __FILE__" (Bhv_PlannedAction) move" );

            if ( Body_GoToPoint( selected_action.targetPoint(),
                                 1.0,
                                 SP.maxDashPower() ).execute( agent ) )
            {
                agent->setNeckAction( new Neck_ScanField() );
                return true;
            }

            break;
        }

    case CooperativeAction::NoAction:
        {
            dlog.addText( Logger::TEAM,
                          __FILE__" (Bhv_PlannedAction) no action" );

            return true;
            break;
        }

    default:
        dlog.addText( Logger::TEAM,
                      __FILE__" (Bhv_PlannedAction) invalid category" );
        break;
    }

    return false;
}
/*-------------------------------------------------------------------*/
/*!

 */
bool
Bhv_PlannedAction::doTurnToForward( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    if ( wm.gameMode().type() != GameMode::PlayOn )
    {
        return false;
    }

    Vector2D face_point( 42.0, 0.0 );

    const double body_angle_diff = ( ( face_point - wm.self().pos() ).th() - wm.self().body() ).abs();
    if ( body_angle_diff < 110.0 )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__" (doTurnToForward) already facing the forward direction. angle_diff=%.1f",
                      body_angle_diff );
        return false;
    }

    dlog.addText( Logger::TEAM,
                  __FILE__" (doTurnToForward) angle_diff=%.1f. try turn",
                  body_angle_diff );

    // const double opponent_dist_thr = ( wm.self().pos().x > ServerParam::i().theirPenaltyAreaLineX() - 2.0
    //                                    && wm.self().pos().absY() > ServerParam::i().goalHalfWidth()
    //                                    ? 2.7
    //                                    : 4.0 );
    const double opponent_dist_thr = 4.0;

    for ( PlayerObject::Cont::const_iterator o = wm.opponentsFromSelf().begin(),
              end = wm.opponentsFromSelf().end();
          o != end;
          ++o )
    {
        double dist = (*o)->distFromSelf();
        dist -= bound( 0, (*o)->posCount(), 3 ) * (*o)->playerTypePtr()->realSpeedMax();

        if ( dist < opponent_dist_thr )
        {
            dlog.addText( Logger::TEAM,
                      __FILE__" (doTurnToForward) exist opponent" );
            return false;
        }

        if ( dist > 10.0 )
        {
            break;
        }
    }

    // TODO: find the best scan target angle
    face_point.y = wm.self().pos().y * 0.5;


    double kickable2 = std::pow( wm.self().playerType().kickableArea()
                                 - wm.self().vel().r() * ServerParam::i().playerRand()
                                 - wm.ball().vel().r() * ServerParam::i().ballRand()
                                 - 0.2,
                                 2 );
    Vector2D self_next = wm.self().pos() + wm.self().vel();
    Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

    if ( self_next.dist2( ball_next ) < kickable2 )
    {
        Body_TurnToPoint( face_point ).execute( agent );
        agent->setNeckAction( new Neck_ScanField() );
        return true;
    }


    Vector2D ball_vel = getKeepBallVel( agent->world() );

    if ( ! ball_vel.isValid() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (doKeepBall) no candidate." );

        return false;
    }

    //
    // perform first kick
    //

    Vector2D kick_accel = ball_vel - wm.ball().vel();
    double kick_power = kick_accel.r() / wm.self().kickRate();
    AngleDeg kick_angle = kick_accel.th() - wm.self().body();

    dlog.addText( Logger::TEAM,
                  __FILE__": (doTurnToForward) "
                  " ballVel=(%.2f %.2f)"
                  " kickPower=%.1f kickAngle=%.1f",
                  ball_vel.x, ball_vel.y,
                  kick_power,
                  kick_angle.degree() );

    if ( kick_power > ServerParam::i().maxPower() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (doTurnToForward) over kick power" );
        Body_HoldBall( true,
                       face_point ).execute( agent );
        agent->setNeckAction( new Neck_ScanField() );
    }
    else
    {
        agent->doKick( kick_power, kick_angle );
        agent->setNeckAction( new Neck_ScanField() );
    }

    agent->debugClient().addMessage( "Chain:Turn:Keep" );
    agent->debugClient().setTarget( face_point );

    //
    // set turn intention
    //

    dlog.addText( Logger::TEAM,
                  __FILE__": (doTurnToFoward) register intention" );
    agent->setIntention( new IntentionTurnTo( face_point ) );

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
Vector2D
Bhv_PlannedAction::getKeepBallVel( const WorldModel & wm )
{
    static GameTime s_update_time( 0, 0 );
    static Vector2D s_best_ball_vel( 0.0, 0.0 );

    if ( s_update_time == wm.time() )
    {
        return s_best_ball_vel;
    }
    s_update_time = wm.time();

    //
    //
    //

    const int ANGLE_DIVS = 12;

    const ServerParam & SP = ServerParam::i();
    const PlayerType & ptype = wm.self().playerType();
    const double collide_dist2 = std::pow( ptype.playerSize()
                                           + SP.ballSize(),
                                           2 );
    const double keep_dist = ptype.playerSize()
        + ptype.kickableMargin() * 0.5
        + ServerParam::i().ballSize();

    const Vector2D next_self_pos
        = wm.self().pos() + wm.self().vel();
    const Vector2D next2_self_pos
        = next_self_pos
        + wm.self().vel() * ptype.playerDecay();

    //
    // create keep target point
    //

    Vector2D best_ball_vel = Vector2D::INVALIDATED;
    int best_opponent_step = 0;
    double best_ball_speed = 1000.0;


    for ( int a = 0; a < ANGLE_DIVS; ++a )
    {
        Vector2D keep_pos
            = next2_self_pos
            + Vector2D::from_polar( keep_dist,
                                    360.0/ANGLE_DIVS * a );
        if ( keep_pos.absX() > SP.pitchHalfLength() - 0.2
             || keep_pos.absY() > SP.pitchHalfWidth() - 0.2 )
        {
            continue;
        }

        Vector2D ball_move = keep_pos - wm.ball().pos();
        double ball_speed = ball_move.r() / ( 1.0 + SP.ballDecay() );

        Vector2D max_vel
            = KickTable::calc_max_velocity( ball_move.th(),
                                            wm.self().kickRate(),
                                            wm.ball().vel() );
        if ( max_vel.r2() < std::pow( ball_speed, 2 ) )
        {
            continue;
        }

        Vector2D ball_next_next = keep_pos;

        Vector2D ball_vel = ball_move.setLengthVector( ball_speed );
        Vector2D ball_next = wm.ball().pos() + ball_vel;

        if ( next_self_pos.dist2( ball_next ) < collide_dist2 )
        {
            ball_next_next = ball_next;
            ball_next_next += ball_vel * ( SP.ballDecay() * -0.1 );
        }

#ifdef DEBUG_PRINT
        dlog.addText( Logger::TEAM,
                      __FILE__": (getKeepBallVel) %d: ball_move th=%.1f speed=%.2f max=%.2f",
                      a,
                      ball_move.th().degree(),
                      ball_speed,
                      max_vel.r() );
        dlog.addText( Logger::TEAM,
                      __FILE__": __ ball_next=(%.2f %.2f) ball_next2=(%.2f %.2f)",
                      ball_next.x, ball_next.y,
                      ball_next_next.x, ball_next_next.y );
#endif

        //
        // check opponent
        //

        int min_step = 1000;
        for ( PlayerObject::Cont::const_iterator o = wm.opponentsFromSelf().begin(),
                  end = wm.opponentsFromSelf().end();
              o != end;
              ++o )
        {
            if ( (*o)->distFromSelf() > 10.0 )
            {
                break;
            }

            int o_step = FieldAnalyzer::predict_player_reach_cycle( *o,
                                                                    ball_next_next,
                                                                    (*o)->playerTypePtr()->kickableArea(),
                                                                    0.0, // penalty distance
                                                                    1, // body count thr
                                                                    1, // default turn step
                                                                    0, // wait cycle
                                                                    true );

            if ( o_step <= 0 )
            {
                break;
            }

            if ( o_step < min_step )
            {
                min_step = o_step;
            }
        }
#ifdef DEBUG_PRINT
        dlog.addText( Logger::TEAM,
                      __FILE__": (getKeepBallVel) %d: keepPos=(%.2f %.2f)"
                      " ballNext2=(%.2f %.2f) ballVel=(%.2f %.2f) speed=%.2f o_step=%d",
                      a,
                      keep_pos.x, keep_pos.y,
                      ball_next_next.x, ball_next_next.y,
                      ball_vel.x, ball_vel.y,
                      ball_speed,
                      min_step );
#endif
        if ( min_step > best_opponent_step )
        {
            best_ball_vel = ball_vel;
            best_opponent_step = min_step;
            best_ball_speed = ball_speed;
        }
        else if ( min_step == best_opponent_step )
        {
            if ( best_ball_speed > ball_speed )
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
