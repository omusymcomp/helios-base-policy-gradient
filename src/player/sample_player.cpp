// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "sample_player.h"

#include "strategy.h"
#include "field_analyzer.h"

#include "action_chain_holder.h"
#include "action_state_pair.h"
#include "cooperative_action.h"
#include "sample_field_evaluator.h"

#include "soccer_role.h"

#include "sample_communication.h"
#include "keepaway_communication.h"
#include "sample_freeform_message_parser.h"

#include "bhv_penalty_kick.h"
#include "bhv_set_play.h"
#include "bhv_set_play_kick_in.h"
#include "bhv_set_play_indirect_free_kick.h"

#include "bhv_custom_before_kick_off.h"
#include "bhv_strict_check_shoot.h"

#include "view_tactical.h"

#include "intention_receive.h"

#include "basic_actions/basic_actions.h"
#include "basic_actions/bhv_emergency.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_intercept.h"
#include "basic_actions/body_kick_one_step.h"
#include "basic_actions/neck_scan_field.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include "basic_actions/view_synch.h"
#include "basic_actions/kick_table.h"

#include "basic_actions/body_pass.h"
#include "basic_actions/body_dribble.h"
#include "basic_actions/body_hold_ball.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_clear_ball.h"
#include "basic_actions/body_smart_kick.h"

#include <rcsc/formation/formation.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/player/say_message_builder.h>
#include <rcsc/player/audio_sensor.h>

#include <rcsc/common/abstract_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/player_param.h>
#include <rcsc/common/audio_memory.h>
#include <rcsc/common/say_message_parser.h>

#include <rcsc/param/param_map.h>
#include <rcsc/param/cmd_line_parser.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <torch/script.h>
#include <torch/torch.h>
#include <random>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

 */
SamplePlayer::SamplePlayer()
    : PlayerAgent(),
      M_communication()
{
    M_field_evaluator = createFieldEvaluator();
    M_action_generator = createActionGenerator();

    std::shared_ptr< AudioMemory > audio_memory( new AudioMemory );

    M_worldmodel.setAudioMemory( audio_memory );

    //
    // set communication message parser
    //
    addSayMessageParser( new BallMessageParser( audio_memory ) );
    addSayMessageParser( new PassMessageParser( audio_memory ) );
    addSayMessageParser( new InterceptMessageParser( audio_memory ) );
    addSayMessageParser( new GoalieMessageParser( audio_memory ) );
    addSayMessageParser( new GoalieAndPlayerMessageParser( audio_memory ) );
    addSayMessageParser( new OffsideLineMessageParser( audio_memory ) );
    addSayMessageParser( new DefenseLineMessageParser( audio_memory ) );
    addSayMessageParser( new WaitRequestMessageParser( audio_memory ) );
    addSayMessageParser( new PassRequestMessageParser( audio_memory ) );
    addSayMessageParser( new DribbleMessageParser( audio_memory ) );
    addSayMessageParser( new BallGoalieMessageParser( audio_memory ) );
    addSayMessageParser( new OnePlayerMessageParser( audio_memory ) );
    addSayMessageParser( new TwoPlayerMessageParser( audio_memory ) );
    addSayMessageParser( new ThreePlayerMessageParser( audio_memory ) );
    addSayMessageParser( new SelfMessageParser( audio_memory ) );
    addSayMessageParser( new TeammateMessageParser( audio_memory ) );
    addSayMessageParser( new OpponentMessageParser( audio_memory ) );
    addSayMessageParser( new BallPlayerMessageParser( audio_memory ) );
    addSayMessageParser( new StaminaMessageParser( audio_memory ) );
    addSayMessageParser( new RecoveryMessageParser( audio_memory ) );

    // addSayMessageParser( new FreeMessageParser< 9 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 8 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 7 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 6 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 5 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 4 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 3 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 2 >( audio_memory ) );
    // addSayMessageParser( new FreeMessageParser< 1 >( audio_memory ) );

    //
    // set freeform message parser
    //
    addFreeformMessageParser( new OpponentPlayerTypeMessageParser( M_worldmodel ) );

    //
    // set communication planner
    //
    M_communication = Communication::Ptr( new SampleCommunication() );
}

/*-------------------------------------------------------------------*/
/*!

 */
SamplePlayer::~SamplePlayer()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
SamplePlayer::initImpl( CmdLineParser & cmd_parser )
{
    bool result = PlayerAgent::initImpl( cmd_parser );

    // read additional options
    result &= Strategy::instance().init( cmd_parser );

    rcsc::ParamMap my_params( "Additional options" );
#if 0
    std::string param_file_path = "params";
    param_map.add()
        ( "param-file", "", &param_file_path, "specified parameter file" );
#endif

    cmd_parser.parse( my_params );

    if ( cmd_parser.count( "help" ) > 0 )
    {
        my_params.printHelp( std::cout );
        return false;
    }

    if ( cmd_parser.failed() )
    {
        std::cerr << "player: ***WARNING*** detected unsuppprted options: ";
        cmd_parser.print( std::cerr );
        std::cerr << std::endl;
    }

    if ( ! result )
    {
        return false;
    }

    if ( ! Strategy::instance().read( config().configDir() ) )
    {
        std::cerr << "***ERROR*** Failed to read team strategy." << std::endl;
        return false;
    }

    if ( KickTable::instance().read( config().configDir() + "/kick-table" ) )
    {
        std::cerr << "Loaded the kick table: ["
                  << config().configDir() << "/kick-table]"
                  << std::endl;
    }

    //モデル読み込み処理を追加
    std::string model_path = "/home/okayama/rcss/policy-gradient/model_with_attention.pt"; // 修正
    if ( !loadModel(model_path) )
    {
        std::cerr << "***ERROR*** Failed to load NN model from "
                  << model_path << std::endl;
        return false;
    }

    return true;
}

/*-------------------------------------------------------------------*/
/*!
  main decision
  virtual method in super class
*/
void SamplePlayer::actionImpl()
{
    //std::cerr << "[DEBUG] Entering actionImpl. Time: " << world().time() << std::endl;
    if (this->audioSensor().trainerMessageTime() == world().time())
    {
        std::cerr << world().ourTeamName() << ' ' << world().self().unum()
                  << ' ' << world().time()
                  << " receive trainer message["
                  << this->audioSensor().trainerMessage() << ']'
                  << std::endl;
    }

    // 戦略とフィールド解析を更新
    Strategy::instance().update(world());
    FieldAnalyzer::instance().update(world());

    // アクションチェーンの準備
    M_field_evaluator = createFieldEvaluator();
    M_action_generator = createActionGenerator();

    ActionChainHolder::instance().setFieldEvaluator(M_field_evaluator);
    ActionChainHolder::instance().setActionGenerator(M_action_generator);

    // 特殊状況の処理
    if (doPreprocess())
    {   
        //std::cerr << "[DEBUG] doPreprocess returned true. Exiting actionImpl." << std::endl;
        dlog.addText(Logger::TEAM, __FILE__ ": preprocess done");
        return;
    }

     //
    // update action chain
    //
    //std::cerr << "[DEBUG] Before updating ActionChainHolder." << std::endl;
    ActionChainHolder::instance().update(world());
    //std::cerr << "[DEBUG] After updating ActionChainHolder." << std::endl;

    //
    // create current role
    //
    SoccerRole::Ptr role_ptr;
    {
        role_ptr = Strategy::i().createRole( world().self().unum(), world() );

        if (!role_ptr) {
            //std::cerr << "[ERROR] Failed to create role for player: " << world().self().unum() << std::endl;
            return;
        }
        //std::cerr << "[DEBUG] Assigned role: " << role_ptr << " for player: " << world().self().unum() << std::endl;
    }


    //
    // override execute if role accept
    //
    if (role_ptr->acceptExecution(world()))
    {
        //std::cerr << "[DEBUG] Role accepted execution: " << role_ptr << std::endl;
        role_ptr->execute(this);
        return;
    }
    else
    {
        //std::cerr << "[DEBUG] Role did not accept execution." << std::endl;
    }

    // PlayOn モードの場合
    //std::cerr << "[DEBUG] Current game mode: " << world().gameMode().type() << std::endl;

    if (world().gameMode().type() == GameMode::PlayOn)
    {
        std::cerr << "[DEBUG] Entering PlayOn mode." << std::endl;

        ActionChainHolder::instance().update(world());
        std::cerr << "[DEBUG] ActionChainHolder updated." << std::endl;

        // 候補アクションの取得
        std::vector<ActionStatePair> candidates = ActionChainHolder::instance().graph().getAllChain();
        std::cerr << "[DEBUG] Number of action candidates: " << candidates.size() << std::endl;
        for (const auto &candidate : candidates) {
            std::cerr << "[DEBUG] Candidate action category: " << candidate.action().category() << std::endl;
        }

        if (candidates.empty())
        {
            std::cerr << "[WARN] No action candidates available." << std::endl;
            return;
        }

        torch::Tensor input = this->extractFeatures(world());
        std::cerr << "[DEBUG] Extracted features: " << input << std::endl;

        // 入力次元のチェック
        if (input.size(1) != 6) {
            std::cerr << "[ERROR] Invalid input dimensions for the model: "
                    << input.sizes() << std::endl;
            return;
        }

        // NNモデルでスコアを予測
        torch::Tensor logits;
        try
        {
            logits = nn_model_->forward({input}).toTensor(); // shape: [1, num_actions]
            std::cerr << "[DEBUG] Model logits: " << logits << std::endl;
        }
        catch (const c10::Error &e)
        {
            std::cerr << "[ERROR] NN forward failed: " << e.what() << std::endl;
            return;
        }

        // Softmax による確率計算
        torch::Tensor probabilities = torch::softmax(logits, 1); // shape: [1, num_actions]
        std::cerr << "[DEBUG] Probabilities: " << probabilities << std::endl;

        // 確率ベクトルを取得
        if (probabilities.dim() != 2 || probabilities.size(0) != 1) {
            std::cerr << "[ERROR] Invalid shape for probabilities tensor: "
                    << probabilities.sizes() << std::endl;
            return;
        }

        std::vector<float> probs(probabilities.data_ptr<float>(), probabilities.data_ptr<float>() + probabilities.size(1));

        // 確率の正規化を確認
        float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (std::abs(sum_probs - 1.0f) > 1e-5) {
            std::cerr << "[ERROR] Probabilities do not sum to 1: " << sum_probs << std::endl;
            return;
        }

        // 負の確率値がないか確認
        for (float p : probs) {
            if (p < 0.0f) {
                std::cerr << "[ERROR] Negative probability value: " << p << std::endl;
                return;
            }
        }

        // 確率に基づいて行動をサンプリング
        static std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int selected_idx = dist(gen);
        std::cerr << "[DEBUG] Selected action index: " << selected_idx << std::endl;

        if (selected_idx < 0 || selected_idx >= static_cast<int>(probs.size())) {
            std::cerr << "[ERROR] Invalid action index sampled: " << selected_idx << std::endl;
            return;
        }

        // 選択したアクションを取得
        const CooperativeAction &selected_action = candidates[selected_idx].action();
        std::cerr << "[DEBUG] Selected action category: " << selected_action.category() << std::endl;

        // 選択したアクションを実行
        doAction(selected_action);
        std::cerr << "[DEBUG] Action executed." << std::endl;
        return;
    }

    // ペナルティキックモードの場合
    if (world().gameMode().isPenaltyKickMode())
    {
        dlog.addText(Logger::TEAM, __FILE__ ": penalty kick");
        Bhv_PenaltyKick().execute(this);
        return;
    }

    // その他のセットプレイモードの場合
    Bhv_SetPlay().execute(this);
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SamplePlayer::handleActionStart()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
void
SamplePlayer::handleActionEnd()
{
    if ( world().self().posValid() )
    {
#if 0
        const ServerParam & SP = ServerParam::i();
        //
        // inside of pitch
        //

        // top,lower
        debugClient().addLine( Vector2D( world().ourOffenseLineX(),
                                         -SP.pitchHalfWidth() ),
                               Vector2D( world().ourOffenseLineX(),
                                         -SP.pitchHalfWidth() + 3.0 ) );
        // top,lower
        debugClient().addLine( Vector2D( world().ourDefenseLineX(),
                                         -SP.pitchHalfWidth() ),
                               Vector2D( world().ourDefenseLineX(),
                                         -SP.pitchHalfWidth() + 3.0 ) );

        // bottom,upper
        debugClient().addLine( Vector2D( world().theirOffenseLineX(),
                                         +SP.pitchHalfWidth() - 3.0 ),
                               Vector2D( world().theirOffenseLineX(),
                                         +SP.pitchHalfWidth() ) );
        //
        debugClient().addLine( Vector2D( world().offsideLineX(),
                                         world().self().pos().y - 15.0 ),
                               Vector2D( world().offsideLineX(),
                                         world().self().pos().y + 15.0 ) );

        // outside of pitch

        // top,upper
        debugClient().addLine( Vector2D( world().ourOffensePlayerLineX(),
                                         -SP.pitchHalfWidth() - 3.0 ),
                               Vector2D( world().ourOffensePlayerLineX(),
                                         -SP.pitchHalfWidth() ) );
        // top,upper
        debugClient().addLine( Vector2D( world().ourDefensePlayerLineX(),
                                         -SP.pitchHalfWidth() - 3.0 ),
                               Vector2D( world().ourDefensePlayerLineX(),
                                         -SP.pitchHalfWidth() ) );
        // bottom,lower
        debugClient().addLine( Vector2D( world().theirOffensePlayerLineX(),
                                         +SP.pitchHalfWidth() ),
                               Vector2D( world().theirOffensePlayerLineX(),
                                         +SP.pitchHalfWidth() + 3.0 ) );
        // bottom,lower
        debugClient().addLine( Vector2D( world().theirDefensePlayerLineX(),
                                         +SP.pitchHalfWidth() ),
                               Vector2D( world().theirDefensePlayerLineX(),
                                         +SP.pitchHalfWidth() + 3.0 ) );
#else
        // top,lower
        debugClient().addLine( Vector2D( world().ourDefenseLineX(),
                                         world().self().pos().y - 2.0 ),
                               Vector2D( world().ourDefenseLineX(),
                                         world().self().pos().y + 2.0 ) );

        //
        debugClient().addLine( Vector2D( world().offsideLineX(),
                                         world().self().pos().y - 15.0 ),
                               Vector2D( world().offsideLineX(),
                                         world().self().pos().y + 15.0 ) );
#endif
    }

    //
    // ball position & velocity
    //
    dlog.addText( Logger::WORLD,
                  "WM: BALL pos=(%lf, %lf), vel=(%lf, %lf, r=%lf, ang=%lf)",
                  world().ball().pos().x,
                  world().ball().pos().y,
                  world().ball().vel().x,
                  world().ball().vel().y,
                  world().ball().vel().r(),
                  world().ball().vel().th().degree() );


    dlog.addText( Logger::WORLD,
                  "WM: SELF move=(%lf, %lf, r=%lf, th=%lf)",
                  world().self().lastMove().x,
                  world().self().lastMove().y,
                  world().self().lastMove().r(),
                  world().self().lastMove().th().degree() );

    if ( world().prevBall().rpos().isValid() )
    {
        Vector2D diff = world().ball().rpos() - world().prevBall().rpos();
        dlog.addText( Logger::WORLD,
                      "WM: BALL rpos=(%lf %lf) prev_rpos=(%lf %lf) diff=(%lf %lf)",
                  world().ball().rpos().x,
                      world().ball().rpos().y,
                      world().prevBall().rpos().x,
                      world().prevBall().rpos().y,
                      diff.x,
                      diff.y );

        Vector2D ball_move = diff + world().self().lastMove();
        Vector2D diff_vel = ball_move * ServerParam::i().ballDecay();
        dlog.addText( Logger::WORLD,
                      "---> ball_move=(%lf %lf) vel=(%lf, %lf, r=%lf, th=%lf)",
                      ball_move.x,
                      ball_move.y,
                      diff_vel.x,
                      diff_vel.y,
                      diff_vel.r(),
                      diff_vel.th().degree() );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SamplePlayer::handleInitMessage()
{
    {
        // Initializing the order of penalty kickers
        std::vector< int > unum_order_pk_kickers = { 10, 9, 2, 11, 3, 4, 1, 5, 6, 7, 8 };
        M_worldmodel.setPenaltyKickTakerOrder( unum_order_pk_kickers );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SamplePlayer::handleServerParam()
{
    if ( ServerParam::i().keepawayMode() )
    {
        std::cerr << "set Keepaway mode communication." << std::endl;
        M_communication = Communication::Ptr( new KeepawayCommunication() );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SamplePlayer::handlePlayerParam()
{
    if ( KickTable::instance().createTables() )
    {
        std::cerr << world().teamName() << ' '
                  << world().self().unum() << ": "
                  << " KickTable created."
                  << std::endl;
    }
    else
    {
        std::cerr << world().teamName() << ' '
                  << world().self().unum() << ": "
                  << " KickTable failed..."
                  << std::endl;
        M_client->setServerAlive( false );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SamplePlayer::handlePlayerType()
{

}

/*-------------------------------------------------------------------*/
/*!
  communication decision.
  virtual method in super class
*/
void
SamplePlayer::communicationImpl()
{
    if ( M_communication )
    {
        M_communication->execute( this );
    }
}

/*-------------------------------------------------------------------*/
/*!
*/
bool
SamplePlayer::doPreprocess()
{
    // check tackle expires
    // check self position accuracy
    // ball search
    // check queued intention
    // check simultaneous kick

    const WorldModel & wm = this->world();

    dlog.addText( Logger::TEAM,
                  __FILE__": (doPreProcess)" );

    //
    // freezed by tackle effect
    //
    if ( wm.self().isFrozen() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": tackle wait. expires= %d",
                      wm.self().tackleExpires() );
        // face neck to ball
        this->setViewAction( new View_Tactical() );
        this->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );
        return true;
    }

    //
    // BeforeKickOff or AfterGoal. jump to the initial position
    //
    if ( wm.gameMode().type() == GameMode::BeforeKickOff
         || wm.gameMode().type() == GameMode::AfterGoal_ )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": before_kick_off" );
        Vector2D move_point =  Strategy::i().getPosition( wm.self().unum() );
        Bhv_CustomBeforeKickOff( move_point ).execute( this );
        this->setViewAction( new View_Tactical() );
        return true;
    }

    //
    // self localization error
    //
    if ( ! wm.self().posValid() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": invalid my pos" );
        Bhv_Emergency().execute( this ); // includes change view
        return true;
    }

    //
    // ball localization error
    //
    const int count_thr = ( wm.self().goalie()
                            ? 10
                            : 5 );
    if ( wm.ball().posCount() > count_thr
         || ( wm.gameMode().type() != GameMode::PlayOn
              && wm.ball().seenPosCount() > count_thr + 10 ) )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": search ball" );
        this->setViewAction( new View_Tactical() );
        Bhv_NeckBodyToBall().execute( this );
        return true;
    }

    //
    // set default change view
    //

    this->setViewAction( new View_Tactical() );

    //
    // check shoot chance
    //
    if ( doShoot() )
    {
        return true;
    }

    //
    // check queued action
    //
    if ( this->doIntention() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": do queued intention" );
        return true;
    }

    //
    // check simultaneous kick
    //
    if ( doForceKick() )
    {
        return true;
    }

    //
    // check pass message
    //
    if ( doHeardPassReceive() )
    {
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
SamplePlayer::doShoot()
{
    const WorldModel & wm = this->world();

    if ( wm.gameMode().type() != GameMode::IndFreeKick_
         && wm.time().stopped() == 0
         && wm.self().isKickable()
         && Bhv_StrictCheckShoot().execute( this ) )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": shooted" );

        // reset intention
        this->setIntention( static_cast< SoccerIntention * >( 0 ) );
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
SamplePlayer::doForceKick()
{
    const WorldModel & wm = this->world();

    if ( wm.gameMode().type() == GameMode::PlayOn
         && ! wm.self().goalie()
         && wm.self().isKickable()
         && wm.kickableOpponent() )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": simultaneous kick" );
        this->debugClient().addMessage( "SimultaneousKick" );
        Vector2D goal_pos( ServerParam::i().pitchHalfLength(), 0.0 );

        if ( wm.self().pos().x > 36.0
             && wm.self().pos().absY() > 10.0 )
        {
            goal_pos.x = 45.0;
            dlog.addText( Logger::TEAM,
                          __FILE__": simultaneous kick cross type" );
        }
        Body_KickOneStep( goal_pos,
                          ServerParam::i().ballSpeedMax()
                          ).execute( this );
        this->setNeckAction( new Neck_ScanField() );
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
SamplePlayer::doHeardPassReceive()
{
    const WorldModel & wm = this->world();

    if ( wm.audioMemory().passTime() != wm.time()
         || wm.audioMemory().pass().empty()
         || wm.audioMemory().pass().front().receiver_ != wm.self().unum() )
    {

        return false;
    }

    int self_min = wm.interceptTable().selfStep();
    Vector2D intercept_pos = wm.ball().inertiaPoint( self_min );
    Vector2D heard_pos = wm.audioMemory().pass().front().receive_pos_;

    dlog.addText( Logger::TEAM,
                  __FILE__":  (doHeardPassReceive) heard_pos(%.2f %.2f) intercept_pos(%.2f %.2f)",
                  heard_pos.x, heard_pos.y,
                  intercept_pos.x, intercept_pos.y );

    if ( ! wm.kickableTeammate()
         && wm.ball().posCount() <= 1
         && wm.ball().velCount() <= 1
         && self_min < 20
         //&& intercept_pos.dist( heard_pos ) < 3.0 ) //5.0 )
         )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (doHeardPassReceive) intercept cycle=%d. intercept",
                      self_min );
        this->debugClient().addMessage( "Comm:Receive:Intercept" );
        Body_Intercept().execute( this );
        this->setNeckAction( new Neck_TurnToBall() );
    }
    else
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (doHeardPassReceive) intercept cycle=%d. go to receive point",
                      self_min );
        this->debugClient().setTarget( heard_pos );
        this->debugClient().addMessage( "Comm:Receive:GoTo" );
        Body_GoToPoint( heard_pos,
                    0.5,
                        ServerParam::i().maxDashPower()
                        ).execute( this );
        this->setNeckAction( new Neck_TurnToBall() );
    }

    this->setIntention( new IntentionReceive( heard_pos,
                                              ServerParam::i().maxDashPower(),
                                              0.9,
                                              5,
                                              wm.time() ) );

    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
FieldEvaluator::ConstPtr
SamplePlayer::getFieldEvaluator() const
{
    return M_field_evaluator;
}

/*-------------------------------------------------------------------*/
/*!

*/
FieldEvaluator::ConstPtr
SamplePlayer::createFieldEvaluator() const
{
    return FieldEvaluator::ConstPtr( new SampleFieldEvaluator );
}
/*-------------------------------------------------------------------*/
/*!

*/
torch::Tensor SamplePlayer::extractFeatures(const rcsc::WorldModel & wm) 
{
    std::vector<double> features = {
        wm.ball().pos().x / 50.0,  // 正規化例
        wm.ball().pos().y / 34.0,
        wm.self().pos().x / 50.0,
        wm.self().pos().y / 34.0,
        wm.self().vel().x / 5.0,
        wm.self().vel().y / 5.0
    }; // shape: [1, feature_dim]
}
/*-------------------------------------------------------------------*/
/*!

*/ 
// int sample_from_probs(torch::Tensor probs) {
//     float r = static_cast<float>(rand()) / RAND_MAX;
//     float cum = 0.0;
//     for (int i = 0; i < probs.size(0); ++i) {
//         cum += probs[i].item<float>();
//         if (r < cum) return i;
//     }
//     return probs.size(0) - 1;
// }
/*-------------------------------------------------------------------*/
/*!

*/
void SamplePlayer::doAction(const CooperativeAction & action)
{
    std::cerr << "[DEBUG] Executing action of category: " << action.category() << std::endl;

    switch (action.category()) {
        case CooperativeAction::Hold:
            if (!Body_HoldBall().execute(this)) {
                std::cerr << "[ERROR] Failed to execute Hold action. Trying Move action." << std::endl;
                Body_GoToPoint(world().ball().pos(), 0.5, ServerParam::i().maxDashPower()).execute(this);
            } else {
                std::cerr << "[DEBUG] Successfully executed Hold action." << std::endl;
            }
            break;

        case CooperativeAction::Dribble:
            if (!Body_Dribble(action.targetPoint(), 0.5, action.firstDashPower(), 3).execute(this)) {
                std::cerr << "[ERROR] Failed to execute Dribble action. Trying Move action." << std::endl;
                Body_GoToPoint(world().ball().pos(), 0.5, ServerParam::i().maxDashPower()).execute(this);
            } else {
                std::cerr << "[DEBUG] Successfully executed Dribble action." << std::endl;
            }
            break;

        case CooperativeAction::Pass:
            if (!Body_Pass().execute(this)) {
                std::cerr << "[ERROR] Failed to execute Pass action." << std::endl;
            } else {
                std::cerr << "[DEBUG] Successfully executed Pass action." << std::endl;
            }
            break;

        case CooperativeAction::Shoot:
            if (!Body_SmartKick(action.targetPoint(),
                                ServerParam::i().ballSpeedMax(),
                                ServerParam::i().ballSpeedMax() * 0.96,
                                3).execute(this)) {
                std::cerr << "[ERROR] Failed to execute Shoot action." << std::endl;
            } else {
                std::cerr << "[DEBUG] Successfully executed Shoot action." << std::endl;
            }
            break;

        case CooperativeAction::Clear:
            if (!Body_ClearBall().execute(this)) {
                std::cerr << "[ERROR] Failed to execute Clear action." << std::endl;
            } else {
                std::cerr << "[DEBUG] Successfully executed Clear action." << std::endl;
            }
            break;

        case CooperativeAction::Move:
            if (!Body_GoToPoint(action.targetPoint(), 0.5, action.firstDashPower()).execute(this)) {
                std::cerr << "[ERROR] Failed to execute Move action." << std::endl;
            } else {
                std::cerr << "[DEBUG] Successfully executed Move action." << std::endl;
            }
            break;

        case CooperativeAction::NoAction:
        default:
            std::cerr << "[WARN] Unknown or NoAction category: "
                      << static_cast<int>(action.category()) << std::endl;
            break;
    }

    std::cerr << "[DEBUG] Finished executing action of category: " << action.category() << std::endl;
}
/*-------------------------------------------------------------------*/
/*!

*/
bool SamplePlayer::loadModel(const std::string & model_path)
{
    try {
        nn_model_ = std::make_shared<torch::jit::script::Module>(
            torch::jit::load("/home/okayama/rcss/policy-gradient/model_with_attention.pt") // 修正
        );
        std::cerr << "[INFO] Loaded NN model from: " << model_path << std::endl;
        return true;
    } catch (const c10::Error &e) {
        std::cerr << "[ERROR] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}
/*-------------------------------------------------------------------*/

/*!

*/
#include "actgen_cross.h"
#include "actgen_direct_pass.h"
#include "actgen_self_pass.h"
#include "actgen_strict_check_pass.h"
#include "actgen_short_dribble.h"
#include "actgen_simple_dribble.h"
#include "actgen_shoot.h"
#include "actgen_action_chain_length_filter.h"

ActionGenerator::ConstPtr
SamplePlayer::createActionGenerator() const
{
    CompositeActionGenerator * g = new CompositeActionGenerator();

    //
    // shoot
    //
    g->addGenerator( new ActGen_RangeActionChainLengthFilter
                     ( new ActGen_Shoot(),
                       2, ActGen_RangeActionChainLengthFilter::MAX ) );

    //
    // strict check pass
    //
    g->addGenerator( new ActGen_MaxActionChainLengthFilter
                     ( new ActGen_StrictCheckPass(), 1 ) );

    //
    // cross
    //
    g->addGenerator( new ActGen_MaxActionChainLengthFilter
                     ( new ActGen_Cross(), 1 ) );

    //
    // direct pass
    //
    // g->addGenerator( new ActGen_RangeActionChainLengthFilter
    //                  ( new ActGen_DirectPass(),
    //                    2, ActGen_RangeActionChainLengthFilter::MAX ) );

    //
    // short dribble
    //
    g->addGenerator( new ActGen_MaxActionChainLengthFilter
                     ( new ActGen_ShortDribble(), 1 ) );

    //
    // self pass (long dribble)
    //
    g->addGenerator( new ActGen_MaxActionChainLengthFilter
                     ( new ActGen_SelfPass(), 1 ) );

    //
    // simple dribble
    //
    // g->addGenerator( new ActGen_RangeActionChainLengthFilter
    //                  ( new ActGen_SimpleDribble(),
    //                    2, ActGen_RangeActionChainLengthFilter::MAX ) );

    return ActionGenerator::ConstPtr( g );
}
