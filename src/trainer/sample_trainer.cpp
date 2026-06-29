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
#include <rcsc/common/abstract_client.h>
#include <rcsc/common/player_param.h>
#include <rcsc/common/player_type.h>
#include <rcsc/common/server_param.h>
#include <rcsc/formation/formation.h>
#include <rcsc/formation/formation_parser.h>
#include <rcsc/param/param_map.h>
#include <rcsc/param/cmd_line_parser.h>
#include <rcsc/random.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace rcsc;

namespace {

struct HeatmapConfig {
    std::string formation_dir;
    std::string our_formation_file;
    std::string opp_formation_file;
    std::string done_file;
    std::string placements_file;
    std::string points;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double grid_step;
    double ball_vx;
    double ball_vy;
    double player_vx;
    double player_vy;
    double holder_offset_x;
    double holder_offset_y;
    double fixed_body_deg;
    int holder_unum;
    int prepare_cycles;
    int settle_cycles;
    int eval_cycles;
    bool mirror_opponent;
    bool force_holder_near_ball;
    bool refix_state_every_cycle;
    std::string body_mode;
};

static std::string env_string( const char * name, const std::string & default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        if ( *value ) return std::string( value );
    }
    return default_value;
}

static double env_double( const char * name, const double default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        if ( *value ) return std::atof( value );
    }
    return default_value;
}

static int env_int( const char * name, const int default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        if ( *value ) return std::atoi( value );
    }
    return default_value;
}

static bool env_bool( const char * name, const bool default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        const std::string s( value );
        return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
    }
    return default_value;
}

static HeatmapConfig load_heatmap_config()
{
    HeatmapConfig c;
    c.formation_dir = env_string( "RCSS_HEATMAP_FORMATION_DIR",
                                  "/home/okayama/rcss/teams/base_team/helios-base/src/formations-dt" );
    c.our_formation_file = env_string( "RCSS_HEATMAP_OUR_FORMATION_FILE",
                                       "offense-formation.conf" );
    c.opp_formation_file = env_string( "RCSS_HEATMAP_OPP_FORMATION_FILE",
                                       "defense-formation.conf" );
    c.done_file = env_string( "RCSS_HEATMAP_DONE_FILE", "" );
    c.placements_file = env_string( "RCSS_HEATMAP_PLACEMENTS_FILE", "" );
    c.points = env_string( "RCSS_HEATMAP_POINTS", "" );
    c.x_min = env_double( "RCSS_HEATMAP_X_MIN", -50.0 );
    c.x_max = env_double( "RCSS_HEATMAP_X_MAX", 50.0 );
    c.y_min = env_double( "RCSS_HEATMAP_Y_MIN", -30.0 );
    c.y_max = env_double( "RCSS_HEATMAP_Y_MAX", 30.0 );
    c.grid_step = std::max( 0.1, env_double( "RCSS_HEATMAP_GRID_STEP", 2.5 ) );
    c.ball_vx = env_double( "RCSS_HEATMAP_BALL_VX", 0.0 );
    c.ball_vy = env_double( "RCSS_HEATMAP_BALL_VY", 0.0 );
    c.player_vx = env_double( "RCSS_HEATMAP_PLAYER_VX", 0.0 );
    c.player_vy = env_double( "RCSS_HEATMAP_PLAYER_VY", 0.0 );
    c.holder_offset_x = env_double( "RCSS_HEATMAP_HOLDER_OFFSET_X", -1.0 );
    c.holder_offset_y = env_double( "RCSS_HEATMAP_HOLDER_OFFSET_Y", 0.0 );
    c.fixed_body_deg = env_double( "RCSS_HEATMAP_FIXED_BODY_DEG", 0.0 );
    c.holder_unum = env_int( "RCSS_HEATMAP_HOLDER_UNUM", 10 );
    c.prepare_cycles = std::max( 0, env_int( "RCSS_HEATMAP_PREPARE_CYCLES", 10 ) );
    c.settle_cycles = std::max( 0, env_int( "RCSS_HEATMAP_SETTLE_CYCLES", 5 ) );
    c.eval_cycles = std::max( 1, env_int( "RCSS_HEATMAP_EVAL_CYCLES", 3 ) );
    c.mirror_opponent = env_bool( "RCSS_HEATMAP_MIRROR_OPPONENT", true );
    c.force_holder_near_ball = env_bool( "RCSS_HEATMAP_FORCE_HOLDER_NEAR_BALL", true );
    c.refix_state_every_cycle = env_bool( "RCSS_HEATMAP_REFIX_STATE_EVERY_CYCLE", true );
    c.body_mode = env_string( "RCSS_HEATMAP_BODY_MODE", "goal" );
    return c;
}

static std::string join_path( const std::string & dir, const std::string & file )
{
    if ( dir.empty() || dir[dir.size() - 1] == '/' ) return dir + file;
    return dir + "/" + file;
}

static std::vector< Vector2D > make_grid_points( const HeatmapConfig & c )
{
    std::vector< Vector2D > points;
    if ( ! c.points.empty() )
    {
        std::string normalized = c.points;
        for ( char & ch : normalized )
        {
            if ( ch == ',' || ch == ';' ) ch = ' ';
        }
        std::istringstream in( normalized );
        double x = 0.0;
        double y = 0.0;
        while ( in >> x >> y )
        {
            points.push_back( Vector2D( x, y ) );
        }
        return points;
    }

    for ( double x = c.x_min; x <= c.x_max + 1.0e-6; x += c.grid_step )
    {
        for ( double y = c.y_min; y <= c.y_max + 1.0e-6; y += c.grid_step )
        {
            points.push_back( Vector2D( x, y ) );
        }
    }
    return points;
}

static Vector2D clamp_to_pitch( const Vector2D & p )
{
    const ServerParam & sp = ServerParam::i();
    return Vector2D( std::max( -sp.pitchHalfLength() + 0.5,
                               std::min( sp.pitchHalfLength() - 0.5, p.x ) ),
                     std::max( -sp.pitchHalfWidth() + 0.5,
                               std::min( sp.pitchHalfWidth() - 0.5, p.y ) ) );
}

static std::vector< Vector2D > formation_positions( const Formation::Ptr & formation,
                                                    const Vector2D & ball,
                                                    const bool mirror )
{
    std::vector< Vector2D > positions( 11 );
    const Vector2D focus = mirror ? Vector2D( -ball.x, -ball.y ) : ball;
    formation->getPositions( focus, positions );
    if ( mirror )
    {
        for ( Vector2D & p : positions )
        {
            p.x = -p.x;
            p.y = -p.y;
        }
    }
    for ( Vector2D & p : positions )
    {
        p = clamp_to_pitch( p );
    }
    return positions;
}

static AngleDeg body_angle( const HeatmapConfig & c, const bool left_team )
{
    if ( c.body_mode == "fixed" ) return AngleDeg( c.fixed_body_deg );
    return AngleDeg( left_team ? 0.0 : 180.0 );
}

} // namespace

/*-------------------------------------------------------------------*/
/*!

 */
SampleTrainer::SampleTrainer()
    : TrainerAgent()
{

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

    //////////////////////////////////////////////////////////////////
    // Add your code here.
    //////////////////////////////////////////////////////////////////

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
SampleTrainer::actionImpl()
{
    if ( world().teamNameLeft().empty() )
    {
        doTeamNames();
        return;
    }

    if ( heatmapModeEnabled() )
    {
        doHeatmapEvaluation();
        return;
    }

    //////////////////////////////////////////////////////////////////
    // Add your code here.

    //sampleAction();
    //recoverForever();
    //doSubstitute();
    doKeepaway();
}

/*-------------------------------------------------------------------*/
/*!
 */
bool
SampleTrainer::heatmapModeEnabled() const
{
    return env_bool( "RCSS_HEATMAP_MODE", false );
}

/*-------------------------------------------------------------------*/
/*!
 */
void
SampleTrainer::doHeatmapEvaluation()
{
    static bool s_initialized = false;
    static bool s_failed = false;
    static HeatmapConfig s_config;
    static Formation::Ptr s_our_formation;
    static Formation::Ptr s_opp_formation;
    static std::vector< Vector2D > s_points;
    static size_t s_index = 0;
    static int s_prepare_wait = 0;
    static int s_wait = 0;
    static GameTime s_last_wait_time( -1, -1 );
    static bool s_done = false;
    static bool s_wrote_placement_header = false;
    static bool s_episode_initialized = false;

    if ( s_done || s_failed )
    {
        return;
    }

    if ( world().teamNameLeft().empty() || world().teamNameRight().empty() )
    {
        doTeamNames();
        return;
    }

    if ( ! s_initialized )
    {
        s_config = load_heatmap_config();
        s_our_formation = FormationParser::parse( join_path( s_config.formation_dir,
                                                             s_config.our_formation_file ) );
        s_opp_formation = FormationParser::parse( join_path( s_config.formation_dir,
                                                             s_config.opp_formation_file ) );
        if ( ! s_our_formation || ! s_opp_formation )
        {
            std::cerr << "[HEATMAP] failed to load formations: "
                      << s_config.formation_dir << std::endl;
            s_failed = true;
            return;
        }

        s_points = make_grid_points( s_config );
        if ( ! s_config.placements_file.empty() )
        {
            std::ofstream ofs( s_config.placements_file.c_str(), std::ios::out );
            ofs << "placement_id,target_ball_x,target_ball_y,cycle_start,cycle_end,eval_start,eval_end\n";
            s_wrote_placement_header = true;
        }
        s_initialized = true;
        std::cerr << "[HEATMAP] trainer initialized: points=" << s_points.size()
                  << " our=" << s_config.our_formation_file
                  << " opp=" << s_config.opp_formation_file
                  << " holder=" << s_config.holder_unum
                  << " prepare=" << s_config.prepare_cycles
                  << std::endl;
    }

    if ( s_prepare_wait < s_config.prepare_cycles )
    {
        if ( s_prepare_wait == 0 )
        {
            doChangeMode( PM_BeforeKickOff );
            std::cerr << "[HEATMAP] prepare cycles=" << s_config.prepare_cycles << std::endl;
        }
        ++s_prepare_wait;
        return;
    }

    if ( s_index >= s_points.size() )
    {
        doChangeMode( PM_TimeOver );
        if ( ! s_config.done_file.empty() )
        {
            std::ofstream ofs( s_config.done_file.c_str() );
            ofs << "done\n";
        }
        std::cerr << "[HEATMAP] trainer completed all points." << std::endl;
        s_done = true;
        return;
    }

    const Vector2D ball = s_points[s_index];
    if ( ! s_episode_initialized )
    {
        if ( ! s_config.placements_file.empty() )
        {
            std::ofstream ofs( s_config.placements_file.c_str(), std::ios::out | std::ios::app );
            if ( ofs )
            {
                const int cycle_start = world().time().cycle();
                const int eval_start = cycle_start + s_config.settle_cycles + 1;
                const int eval_end = eval_start + s_config.eval_cycles - 1;
                const int cycle_end = eval_end;
                if ( ! s_wrote_placement_header )
                {
                    ofs << "placement_id,target_ball_x,target_ball_y,cycle_start,cycle_end,eval_start,eval_end\n";
                    s_wrote_placement_header = true;
                }
                ofs << s_index << ','
                    << ball.x << ','
                    << ball.y << ','
                    << cycle_start << ','
                    << cycle_end << ','
                    << eval_start << ','
                    << eval_end << '\n';
            }
        }
        std::cerr << "[HEATMAP] episode=" << s_index
                  << " ball=(" << ball.x << "," << ball.y << ")"
                  << std::endl;
        s_episode_initialized = true;
    }

    if ( s_wait == 0 || s_config.refix_state_every_cycle )
    {
        std::vector< Vector2D > our_positions = formation_positions( s_our_formation, ball, false );
        std::vector< Vector2D > opp_positions = formation_positions( s_opp_formation,
                                                                     ball,
                                                                     s_config.mirror_opponent );
        if ( s_config.force_holder_near_ball )
        {
            int holder_unum = s_config.holder_unum;
            if ( holder_unum <= 0 )
            {
                double best_dist = 1.0e10;
                for ( int unum = 1; unum <= 11; ++unum )
                {
                    const double dist = our_positions[static_cast<size_t>( unum - 1 )].dist( ball );
                    if ( dist < best_dist )
                    {
                        best_dist = dist;
                        holder_unum = unum;
                    }
                }
            }
            if ( 1 <= holder_unum && holder_unum <= 11 )
            {
                our_positions[static_cast<size_t>( holder_unum - 1 )]
                    = clamp_to_pitch( Vector2D( ball.x + s_config.holder_offset_x,
                                                ball.y + s_config.holder_offset_y ) );
            }
        }

        if ( s_wait == 0 || s_wait < s_config.settle_cycles )
        {
            doChangeMode( PM_BeforeKickOff );
        }
        doRecover();
        doMoveBall( ball, Vector2D( s_config.ball_vx, s_config.ball_vy ) );

        const AngleDeg left_body = body_angle( s_config, true );
        const AngleDeg right_body = body_angle( s_config, false );
        const Vector2D player_vel( s_config.player_vx, s_config.player_vy );
        for ( int unum = 1; unum <= 11; ++unum )
        {
            doMovePlayer( world().teamNameLeft(),
                          unum,
                          our_positions[static_cast<size_t>( unum - 1 )],
                          left_body,
                          player_vel );
            doMovePlayer( world().teamNameRight(),
                          unum,
                          opp_positions[static_cast<size_t>( unum - 1 )],
                          right_body,
                          player_vel );
        }
    }

    if ( s_wait == s_config.settle_cycles )
    {
        doChangeMode( PM_PlayOn );
    }

    const GameTime current_time = world().time();
    const bool in_eval_phase = s_wait >= s_config.settle_cycles;
    const bool should_count_wait =
        in_eval_phase
        ? ( world().gameMode().type() == GameMode::PlayOn
            && current_time.cycle() != s_last_wait_time.cycle() )
        : ( current_time != s_last_wait_time );
    if ( should_count_wait )
    {
        ++s_wait;
        s_last_wait_time = current_time;
    }
    if ( s_wait >= s_config.settle_cycles + s_config.eval_cycles )
    {
        s_wait = 0;
        s_last_wait_time = GameTime( -1, -1 );
        s_episode_initialized = false;
        ++s_index;
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
