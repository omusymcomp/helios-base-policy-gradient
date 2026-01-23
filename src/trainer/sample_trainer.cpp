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
#include "config.h"
#endif

#include "sample_trainer.h"

#include <rcsc/trainer/trainer_command.h>
#include <rcsc/trainer/trainer_config.h>
#include <rcsc/coach/coach_world_model.h>
#include <rcsc/coach/coach_player_object.h>
#include <rcsc/common/abstract_client.h>
#include <rcsc/player/abstract_player_object.h>
#include <rcsc/common/player_param.h>
#include <rcsc/common/player_type.h>
#include <rcsc/common/server_param.h>
#include <rcsc/formation/formation_parser.h>
#include <rcsc/formation/formation.h>
#include <rcsc/formation/formation_data.h>
#include <rcsc/param/param_map.h>
#include <rcsc/param/cmd_line_parser.h>
#include <rcsc/random.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

 */
SampleTrainer::SampleTrainer()
    : TrainerAgent()
    , M_state( EpisodeState::Idle )
    , M_resume_cycle( 0 )
    , M_episode_start_cycle( 0 )
    , M_episode_time_limit( 200 )
{
    M_attack_ours.valid = false;
    M_attack_opponents.valid = false;
    M_current_ball_start.assign( 0.0, 0.0 );
    M_team_left = "HELIOS_base";
    M_team_right = "Opponent";
}

/*-------------------------------------------------------------------*/
/*!

 */
SampleTrainer::~SampleTrainer()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
SampleTrainer::initImpl( CmdLineParser & cmd_parser )
{
    bool result = TrainerAgent::initImpl( cmd_parser );

#if 0
    ParamMap my_params;

    std::string formation_conf;
    my_map.add()
        ( &conf_path, "fconf" )
        ;

    cmd_parser.parse( my_params );
#endif

    if ( cmd_parser.failed() )
    {
        std::cerr << "coach: ***WARNING*** detected unsupported options: ";
        cmd_parser.print( std::cerr );
        std::cerr << std::endl;
    }

    if ( ! result )
    {
        return false;
    }

    loadOffenseFormation();
    M_team_left = config().teamName();

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::actionImpl()
{
    const GameTime current_time = world().time();
    if ( world().teamNameLeft().empty() )
    {
        std::cout << "[trainer] waiting team name at cycle "
                  << current_time << std::endl;
        doTeamNames();
    }
    else if ( M_team_left != world().teamNameLeft() )
    {
        M_team_left = world().teamNameLeft();
        std::cout << "[trainer] detected left team: "
                  << M_team_left << " at cycle "
                  << current_time << std::endl;
    }

    if ( ! world().teamNameRight().empty()
         && M_team_right != world().teamNameRight() )
    {
        M_team_right = world().teamNameRight();
        std::cout << "[trainer] detected right team: "
                  << M_team_right << std::endl;
    }

    if ( M_state == EpisodeState::Idle )
    {
        std::cout << "[trainer] actionImpl: state=Idle, startEpisode() at cycle "
                  << current_time << std::endl;
        startEpisode();
        return;
    }

    if ( M_state == EpisodeState::WaitingSync )
    {
        if ( world().time().cycle() >= M_resume_cycle
             && world().gameMode().type() == GameMode::PlayOn )
        {
            M_state = EpisodeState::Running;
            M_episode_start_cycle = world().time().cycle();
            std::cout << "[trainer] actionImpl: enter Running at cycle "
                      << M_episode_start_cycle << std::endl;
        }
        return;
    }

    if ( M_state == EpisodeState::Running )
    {
        const GameMode::Type gtype = world().gameMode().type();
        if ( gtype != GameMode::PlayOn )
        {
            if ( gtype == GameMode::AfterGoal_
                 || gtype == GameMode::KickIn_
                 || gtype == GameMode::CornerKick_
                 || gtype == GameMode::GoalKick_
                 || gtype == GameMode::FoulCharge_
                 || gtype == GameMode::OffSide_ )
            {
                std::cout << "[trainer] reset reason=playmode("
                          << static_cast<int>( gtype ) << ") at cycle "
                          << world().time().cycle() << std::endl;
                startEpisode();
                return;
            }
        }

        if ( isBallOut()
             || isBallLost()
             || world().ball().pos().x < M_current_ball_start.x - 25.0 )
        {
            std::cout << "[trainer] reset reason="
                      << ( isBallOut() ? "ball_out"
                           : ( isBallLost() ? "ball_lost" : "ball_back" ) )
                      << " at cycle " << world().time().cycle() << std::endl;
            startEpisode();
            return;
        }

        if ( M_episode_time_limit > 0
             && world().time().cycle() - M_episode_start_cycle > M_episode_time_limit )
        {
            std::cout << "[trainer] reset reason=time_limit at cycle "
                      << world().time().cycle() << std::endl;
            startEpisode();
            return;
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

*/
void
SampleTrainer::handleInitMessage()
{

}

/*-------------------------------------------------------------------*/
/*!

*/
void
SampleTrainer::handleServerParam()
{

}

/*-------------------------------------------------------------------*/
/*!

*/
void
SampleTrainer::handlePlayerParam()
{

}

/*-------------------------------------------------------------------*/
/*!

*/
void
SampleTrainer::handlePlayerType()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::startEpisode()
{
    const ServerParam & SP = ServerParam::i();
    Vector2D ball_pos = M_attack_ours.valid
        ? M_attack_ours.ball
        : Vector2D( SP.pitchHalfLength() - 20.0, 0.0 );

    M_current_ball_start = ball_pos;

    doRecover();
    placeOpponents( ball_pos );
    placeOurPlayers();
    doMoveBall( ball_pos, Vector2D( 0.0, 0.0 ) );
    doChangeMode( PM_PlayOn );

    M_state = EpisodeState::WaitingSync;
    M_resume_cycle = world().time().cycle() + 3;
    M_episode_start_cycle = world().time().cycle();

    std::cout << "trainer: reset attack scenario at cycle "
              << world().time().cycle()
              << std::endl;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::placeOurPlayers()
{
    const std::string team = M_team_left.empty()
        ? config().teamName()
        : M_team_left;

    if ( M_attack_ours.valid )
    {
        for ( int unum = 1; unum <= 11; ++unum )
        {
            doMovePlayer( team, unum, M_attack_ours.players[unum - 1], 0.0 );
        }
        return;
    }

    const Vector2D & ball_pos = M_current_ball_start;
    std::array< Vector2D, 11 > fallback = {
        Vector2D( -50.0, 0.0 ),                    // 1 GK
        ball_pos + Vector2D( -20.0, -6.0 ),        // 2
        ball_pos + Vector2D( -20.0,  6.0 ),        // 3
        ball_pos + Vector2D( -14.0, -12.0 ),       // 4
        ball_pos + Vector2D( -14.0,  12.0 ),       // 5
        ball_pos + Vector2D( -10.0,  0.0 ),        // 6
        ball_pos + Vector2D(  -6.0,  6.0 ),        // 7
        ball_pos + Vector2D(  -6.0, -6.0 ),        // 8
        ball_pos + Vector2D(  -2.0,  8.0 ),        // 9
        ball_pos + Vector2D(  -2.0, -8.0 ),        // 10
        ball_pos + Vector2D(   0.5,  0.0 )         // 11 striker
    };

    for ( int unum = 1; unum <= 11; ++unum )
    {
        doMovePlayer( team, unum, fallback[unum - 1], 0.0 );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::placeOpponents( const Vector2D & ball_pos )
{
    const std::string team = M_team_right.empty()
        ? "Opponent"
        : M_team_right;

    if ( M_attack_opponents.valid )
    {
        for ( int unum = 1; unum <= 11; ++unum )
        {
            doMovePlayer( team, unum, M_attack_opponents.players[unum - 1], 180.0 );
        }
        return;
    }

    const ServerParam & SP = ServerParam::i();
    const double base_x = std::min( SP.pitchHalfLength() - 2.0, ball_pos.x + 4.0 );
    std::array< Vector2D, 11 > fallback = {
        Vector2D( SP.pitchHalfLength() - 1.0, 0.0 ),          // GK
        Vector2D( base_x,  0.0 ),
        Vector2D( base_x,  8.0 ),
        Vector2D( base_x, -8.0 ),
        Vector2D( base_x + 2.0, 12.0 ),
        Vector2D( base_x + 2.0, -12.0 ),
        Vector2D( base_x - 4.0,  5.0 ),
        Vector2D( base_x - 4.0, -5.0 ),
        Vector2D( base_x - 8.0, 10.0 ),
        Vector2D( base_x - 8.0, -10.0 ),
        Vector2D( base_x - 6.0,  0.0 )
    };

    for ( int unum = 1; unum <= 11; ++unum )
    {
        doMovePlayer( team, unum, fallback[unum - 1], 180.0 );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
SampleTrainer::isBallOut() const
{
    const ServerParam & SP = ServerParam::i();
    const Vector2D & pos = world().ball().pos();
    return ( std::fabs( pos.x ) > SP.pitchHalfLength() - 0.5
             || std::fabs( pos.y ) > SP.pitchHalfWidth() - 0.5 );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
SampleTrainer::isBallLost() const
{
    const auto & ball = world().ball();

    auto is_kickable = [&]( const CoachPlayerObject * player ) -> bool
    {
        int type = world().playerTypeId( player->side(), player->unum() );
        const PlayerType * param = PlayerTypeSet::i().get( type );
        double kickable_area = ( param
                                 ? param->kickableArea()
                                 : ServerParam::i().defaultKickableArea() );
        return player->pos().dist2( ball.pos() ) < std::pow( kickable_area, 2 );
    };

    // check if any of our players keeps possession
    for ( const CoachPlayerObject * p : world().playersLeft() )
    {
        if ( is_kickable( p ) )
        {
            return false;
        }
    }

    // if opponent is kickable, we lost the ball
    for ( const CoachPlayerObject * p : world().playersRight() )
    {
        if ( is_kickable( p ) )
        {
            return true;
        }
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::sampleAction()
{
    // sample training to test a ball interception.

    static int s_state = 0;
    static int s_wait_counter = 0;

    static Vector2D s_last_player_move_pos;

    if ( world().existKickablePlayer() )
    {
        s_state = 1;
    }

    switch ( s_state ) {
    case 0:
        // nothing to do
        break;
    case 1:
        // exist kickable left player

        // recover stamina
        doRecover();
        // move ball to center
        doMoveBall( Vector2D( 0.0, 0.0 ),
                    Vector2D( 0.0, 0.0 ) );
        // change playmode to play_on
        doChangeMode( PM_PlayOn );
        {
            // move player to random point
            UniformReal uni01( 0.0, 1.0 );
            Vector2D move_pos
                = Vector2D::polar2vector( 15.0, //20.0,
                                          AngleDeg( 360.0 * uni01() ) );
            s_last_player_move_pos = move_pos;

            doMovePlayer( config().teamName(),
                          1, // uniform number
                          move_pos,
                          move_pos.th() - 180.0 );
        }
        // change player type
        {
            static int type = 0;
            doChangePlayerType( world().teamNameLeft(), 1, type );
            type = ( type + 1 ) % PlayerParam::i().playerTypes();
        }

        doSay( "move player" );
        s_state = 2;
        std::cout << "trainer: actionImpl init episode." << std::endl;
        break;
    case 2:
        ++s_wait_counter;
        if ( s_wait_counter > 3
             && ! world().playersLeft().empty() )
        {
            // add velocity to the ball
            //UniformReal uni_spd( 2.7, 3.0 );
            //UniformReal uni_spd( 2.5, 3.0 );
            UniformReal uni_spd( 2.3, 2.7 );
            //UniformReal uni_ang( -50.0, 50.0 );
            UniformReal uni_ang( -10.0, 10.0 );
            Vector2D velocity
                = Vector2D::polar2vector( uni_spd(),
                                          s_last_player_move_pos.th()
                                          + uni_ang() );
            doMoveBall( Vector2D( 0.0, 0.0 ),
                        velocity );
            s_state = 0;
            s_wait_counter = 0;
            std::cout << "trainer: actionImpl start ball" << std::endl;
        }
        break;

    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::recoverForever()
{
    if ( world().playersLeft().empty() )
    {
        return;
    }

    if ( world().time().stopped() == 0
         && world().time().cycle() % 50 == 0 )
    {
        // recover stamina
        doRecover();
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::doSubstitute()
{
    static bool s_substitute = false;
    if ( ! s_substitute
         && world().time().cycle() == 0
         && world().time().stopped() >= 10 )
    {
        std::cerr << "trainer " << world().time() << " team name = "
                  << world().teamNameLeft()
                  << std::endl;

        if ( ! world().teamNameLeft().empty() )
        {
            UniformInt uni( 0, PlayerParam::i().ptMax() );
            doChangePlayerType( world().teamNameLeft(),
                                1,
                                uni() );

            s_substitute = true;
        }
    }

    if ( world().time().stopped() == 0
         && world().time().cycle() % 100 == 1
         && ! world().teamNameLeft().empty() )
    {
        static int type = 0;
        doChangePlayerType( world().teamNameLeft(), 1, type );
        type = ( type + 1 ) % PlayerParam::i().playerTypes();
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::doKeepaway()
{
    if ( world().trainingTime() == world().time() )
    {
        std::cerr << "trainer: "
                  << world().time()
                  << " keepaway training time." << std::endl;
    }

}
void
SampleTrainer::loadOffenseFormation()
{
    const std::string base_dir =
        "/home/okayama/rcss/teams/policy_gradient/helios-base-policy-gradient/src/formations-dt/";
    const std::string offense_path = base_dir + "offense-formation.conf";
    const std::string opponent_path = base_dir + "attack-situation.conf";

    auto apply_entry_to_snapshot =
        []( FormationSnapshot & snapshot,
            const FormationData::Data & entry )
    {
        if ( entry.players_.size() < 11 )
        {
            return false;
        }
        snapshot.ball = entry.ball_;
        for ( int i = 0; i < 11; ++i )
        {
            snapshot.players[i] = entry.players_[i];
        }
        snapshot.valid = true;
        return true;
    };

    constexpr int desired_attack_index = 43;
    auto offense = FormationParser::parse( offense_path );
    if ( ! offense )
    {
        std::cerr << "[trainer] Failed to parse offense formation: "
                  << offense_path << std::endl;
    }
    else
    {
        FormationData::Ptr data = offense->toData();
        const FormationData::Data * best = nullptr;
        const FormationData::Data * fallback = nullptr;
        double best_x = -1e9;
        double best_y_abs = 1e9;
        if ( data )
        {
            for ( const auto & entry : data->dataCont() )
            {
                if ( entry.players_.size() < 11 )
                {
                    continue;
                }
                if ( entry.index_ == desired_attack_index )
                {
                    best = &entry;
                    break;
                }
                const double bx = entry.ball_.x;
                const double by_abs = std::fabs( entry.ball_.y );
                if ( bx > best_x + 1e-6
                     || ( std::fabs( bx - best_x ) < 1e-6 && by_abs < best_y_abs ) )
                {
                    fallback = &entry;
                    best_x = bx;
                    best_y_abs = by_abs;
                }
            }
            if ( ! best )
            {
                best = fallback;
            }
        }

        if ( best )
        {
            apply_entry_to_snapshot( M_attack_ours, *best );
            std::cout << "[trainer] Use offense formation: ball=("
                      << M_attack_ours.ball.x << ", "
                      << M_attack_ours.ball.y << ") index="
                      << best->index_ << std::endl;
        }
        else
        {
            std::cerr << "[trainer] Could not find a valid offense entry." << std::endl;
        }
    }

    auto defense = FormationParser::parse( opponent_path );
    if ( ! defense )
    {
        std::cerr << "[trainer] Failed to parse opponent formation: "
                  << opponent_path << std::endl;
    }
    else
    {
        FormationData::Ptr data = defense->toData();
        if ( data && ! data->dataCont().empty() )
        {
            apply_entry_to_snapshot( M_attack_opponents, data->dataCont().front() );
            std::cout << "[trainer] Use opponent static formation." << std::endl;
        }
        else
        {
            std::cerr << "[trainer] Opponent formation has no data: "
                      << opponent_path << std::endl;
        }
    }

    if ( ! M_attack_ours.valid )
    {
        std::cerr << "[trainer] Offense formation fallback will be used." << std::endl;
    }
    else
    {
        const ServerParam & SP = ServerParam::i();
        const double dist_to_goal = std::max( 1.0, SP.pitchHalfLength() - M_attack_ours.ball.x );
        M_episode_time_limit = static_cast< int >( std::clamp( dist_to_goal * 6.0, 120.0, 400.0 ) );
    }
    if ( ! M_attack_opponents.valid )
    {
        std::cerr << "[trainer] Opponent formation fallback will be used." << std::endl;
    }
}
