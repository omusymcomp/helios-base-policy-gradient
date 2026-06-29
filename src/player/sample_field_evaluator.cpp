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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "sample_field_evaluator.h"

#include "field_analyzer.h"

#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace rcsc;

namespace {

static std::atomic<long long> g_inference_call_count{0};

struct HeatmapEvalLogger {
    bool initialized = false;
    bool enabled = false;
    bool pair_enabled = false;
    bool pair_return_nn = false;
    bool require_self_holder = true;
    bool current_only = false;
    bool finite_only = true;
    int self_unum = 10;
    int log_stride = 1;
    long long accepted = 0;
    SideID side = LEFT;
    std::string path;
    std::string pair_original_path;
    std::string pair_imitation_path;
    std::string run_type;
    std::ofstream out;
    std::mutex mutex;
};

static HeatmapEvalLogger g_heatmap_logger;

static
bool env_bool( const char * name,
               const bool default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        const std::string s( value );
        return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
    }
    return default_value;
}

static
int env_int( const char * name,
             const int default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        if ( *value ) return std::atoi( value );
    }
    return default_value;
}

static
std::string env_string( const char * name,
                        const std::string & default_value )
{
    if ( const char * value = std::getenv( name ) )
    {
        if ( *value ) return std::string( value );
    }
    return default_value;
}

static
SideID env_side( const char * name,
                 const SideID default_value )
{
    const std::string value = env_string( name, default_value == LEFT ? "l" : "r" );
    if ( value == "r" || value == "right" || value == "RIGHT" ) return RIGHT;
    return LEFT;
}

static
void init_heatmap_logger()
{
    if ( g_heatmap_logger.initialized )
    {
        return;
    }

    g_heatmap_logger.initialized = true;
    g_heatmap_logger.self_unum = env_int( "RCSS_HEATMAP_SELF_UNUM", 10 );
    g_heatmap_logger.log_stride = std::max( 1, env_int( "RCSS_HEATMAP_LOG_STRIDE", 1 ) );
    g_heatmap_logger.require_self_holder = env_bool( "RCSS_HEATMAP_REQUIRE_SELF_HOLDER", true );
    g_heatmap_logger.current_only = env_bool( "RCSS_HEATMAP_CURRENT_ONLY", false );
    g_heatmap_logger.finite_only = env_bool( "RCSS_HEATMAP_FINITE_ONLY", true );
    g_heatmap_logger.side = env_side( "RCSS_HEATMAP_SIDE", LEFT );
    g_heatmap_logger.run_type = env_string( "RCSS_HEATMAP_RUN_TYPE", "unknown" );
    g_heatmap_logger.path = env_string( "RCSS_HEATMAP_EVAL_LOG", "" );
    g_heatmap_logger.pair_original_path = env_string( "RCSS_HEATMAP_PAIR_ORIGINAL_LOG", "" );
    g_heatmap_logger.pair_imitation_path = env_string( "RCSS_HEATMAP_PAIR_IMITATION_LOG", "" );
    g_heatmap_logger.pair_enabled = ! g_heatmap_logger.pair_original_path.empty()
        && ! g_heatmap_logger.pair_imitation_path.empty();
    g_heatmap_logger.pair_return_nn = env_bool( "RCSS_HEATMAP_PAIR_RETURN_NN", false );
    g_heatmap_logger.enabled = ! g_heatmap_logger.path.empty() || g_heatmap_logger.pair_enabled;
    if ( ! g_heatmap_logger.enabled )
    {
        return;
    }

    auto can_open = []( const std::string & path ) -> bool {
        if ( path.empty() ) return true;
        std::ofstream test_out( path.c_str(), std::ios::out | std::ios::app );
        return static_cast<bool>( test_out );
    };
    {
        if ( ! can_open( g_heatmap_logger.path )
             || ! can_open( g_heatmap_logger.pair_original_path )
             || ! can_open( g_heatmap_logger.pair_imitation_path ) )
        {
            g_heatmap_logger.enabled = false;
            std::cerr << "[HEATMAP] failed to open eval log: "
                      << g_heatmap_logger.path << " "
                      << g_heatmap_logger.pair_original_path << " "
                      << g_heatmap_logger.pair_imitation_path << std::endl;
            return;
        }
    }

    std::cerr << "[HEATMAP] eval logger enabled: "
              << g_heatmap_logger.path << std::endl;
    if ( g_heatmap_logger.pair_enabled )
    {
        std::cerr << "[HEATMAP] paired eval logger enabled: original="
                  << g_heatmap_logger.pair_original_path
                  << " imitation=" << g_heatmap_logger.pair_imitation_path
                  << " return=" << ( g_heatmap_logger.pair_return_nn ? "nn" : "original" )
                  << std::endl;
    }
}

static
bool heatmap_log_accepts( const PredictState & state,
                          const double value )
{
    init_heatmap_logger();
    if ( ! g_heatmap_logger.enabled )
    {
        return false;
    }

    if ( state.ourSide() != g_heatmap_logger.side )
    {
        return false;
    }

    if ( g_heatmap_logger.current_only
         && state.spendTime() != 0 )
    {
        return false;
    }

    if ( g_heatmap_logger.finite_only
         && ! std::isfinite( value ) )
    {
        return false;
    }

    if ( g_heatmap_logger.self_unum > 0
         && state.self().unum() != g_heatmap_logger.self_unum )
    {
        return false;
    }

    const AbstractPlayerObject * holder = state.ballHolder();
    const int holder_unum = holder ? holder->unum() : Unum_Unknown;
    if ( g_heatmap_logger.require_self_holder
         && holder_unum != state.self().unum() )
    {
        return false;
    }

    return true;
}

static
const char * action_category_name( const CooperativeAction::ActionCategory category )
{
    switch ( category )
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
    default:
        return "NoAction";
    }
}

static
std::string action_signature( const CooperativeAction & action )
{
    const Vector2D & target = action.targetPoint();
    std::ostringstream os;
    os << action_category_name( action.category() )
       << ':' << action.index()
       << ':' << action.playerUnum()
       << ':' << action.targetPlayerUnum()
       << ':' << std::fixed << std::setprecision( 3 ) << target.x
       << ':' << std::fixed << std::setprecision( 3 ) << target.y
       << ':' << action.durationStep();
    return os.str();
}

static
std::string path_signature( const std::vector< ActionStatePair > & action_path,
                            const size_t upto )
{
    const size_t n = std::min( upto, action_path.size() );
    if ( n == 0 )
    {
        return "ROOT";
    }

    std::ostringstream os;
    for ( size_t i = 0; i < n; ++i )
    {
        if ( i > 0 )
        {
            os << '>';
        }
        os << action_signature( action_path[i].action() );
    }
    return os.str();
}

static
void append_action_columns( std::ostringstream & line,
                            const std::vector< ActionStatePair > & action_path )
{
    const int path_len = static_cast< int >( action_path.size() );
    const std::string parent_key = path_len > 0
        ? path_signature( action_path, action_path.size() - 1 )
        : std::string( "ROOT_PARENT" );
    const std::string path_key = path_signature( action_path, action_path.size() );

    auto append_action = [&]( const ActionStatePair * pair ) {
        if ( ! pair )
        {
            line << ",NA,-1,-1,-1,0,0,-1";
            return;
        }
        const CooperativeAction & action = pair->action();
        const Vector2D & target = action.targetPoint();
        line << ',' << action_category_name( action.category() )
             << ',' << action.index()
             << ',' << action.playerUnum()
             << ',' << action.targetPlayerUnum()
             << ',' << std::setprecision( 10 ) << target.x
             << ',' << std::setprecision( 10 ) << target.y
             << ',' << action.durationStep();
    };

    line << ',' << path_len
         << ',' << parent_key
         << ',' << path_signature( action_path, action_path.size() );
    append_action( action_path.empty() ? nullptr : &action_path.front() );
    append_action( action_path.empty() ? nullptr : &action_path.back() );
}

static
void write_heatmap_line( const std::string & path,
                         const std::string & run_type,
                         const PredictState & state,
                         const double value,
                         const char * source,
                         const std::string & pair_id,
                         const std::vector< ActionStatePair > & action_path )
{
    if ( path.empty() )
    {
        return;
    }

    const AbstractPlayerObject * holder = state.ballHolder();
    const int holder_unum = holder ? holder->unum() : Unum_Unknown;
    const GameTime & t = state.currentTime();
    std::ostringstream line;
    line << run_type << ','
         << t.cycle() << ','
         << t.stopped() << ','
         << state.spendTime() << ','
         << state.self().unum() << ','
         << holder_unum << ','
         << std::setprecision( 10 ) << state.ball().pos().x << ','
         << std::setprecision( 10 ) << state.ball().pos().y << ','
         << std::setprecision( 12 ) << value << ','
         << source << ','
         << pair_id;
    append_action_columns( line, action_path );
    line << '\n';

    const std::string text = line.str();
    const int fd = ::open( path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644 );
    if ( fd < 0 )
    {
        return;
    }
    const char * ptr = text.c_str();
    size_t remaining = text.size();
    while ( remaining > 0 )
    {
        const ssize_t written = ::write( fd, ptr, remaining );
        if ( written <= 0 ) break;
        ptr += written;
        remaining -= static_cast<size_t>( written );
    }
    ::close( fd );
}

static
void log_heatmap_value( const PredictState & state,
                        const double value,
                        const char * source,
                        const std::vector< ActionStatePair > & action_path )
{
    if ( g_heatmap_logger.initialized && g_heatmap_logger.path.empty() )
    {
        return;
    }
    if ( ! heatmap_log_accepts( state, value ) )
    {
        return;
    }
    if ( g_heatmap_logger.path.empty() )
    {
        return;
    }

    std::lock_guard< std::mutex > lock( g_heatmap_logger.mutex );
    ++g_heatmap_logger.accepted;
    if ( g_heatmap_logger.accepted % g_heatmap_logger.log_stride != 0 )
    {
        return;
    }

    write_heatmap_line( g_heatmap_logger.path,
                        g_heatmap_logger.run_type,
                        state,
                        value,
                        source,
                        "",
                        action_path );
}

static
void log_heatmap_pair( const PredictState & state,
                       const double original_value,
                       const double imitation_value,
                       const char * imitation_source,
                       const std::vector< ActionStatePair > & action_path )
{
    if ( ! heatmap_log_accepts( state, original_value )
         || ! heatmap_log_accepts( state, imitation_value ) )
    {
        return;
    }

    std::lock_guard< std::mutex > lock( g_heatmap_logger.mutex );
    ++g_heatmap_logger.accepted;
    if ( g_heatmap_logger.accepted % g_heatmap_logger.log_stride != 0 )
    {
        return;
    }

    std::ostringstream pair_id;
    pair_id << ::getpid() << ':' << g_heatmap_logger.accepted;

    write_heatmap_line( g_heatmap_logger.pair_original_path,
                        "original",
                        state,
                        original_value,
                        "paired_original",
                        pair_id.str(),
                        action_path );
    write_heatmap_line( g_heatmap_logger.pair_imitation_path,
                        "imitation",
                        state,
                        imitation_value,
                        imitation_source,
                        pair_id.str(),
                        action_path );
}

static
std::string resolve_model_path()
{
    if ( const char * p = std::getenv( "MODEL_PATH" ) )
    {
        if ( *p ) return std::string( p );
    }
    if ( const char * p = std::getenv( "RCSS_MODEL_PATH" ) )
    {
        if ( *p ) return std::string( p );
    }
    return "/home/okayama/rcss/policy-gradient/model.pt";
}

static
double evaluate_state_original( const PredictState & state )
{
    const ServerParam & sp = ServerParam::i();
    const AbstractPlayerObject * holder = state.ballHolder();

    if ( ! holder )
    {
        return -DBL_MAX / 2.0;
    }

    const int holder_unum = holder->unum();

    if ( state.ball().pos().x > + ( sp.pitchHalfLength() - 0.1 )
         && state.ball().pos().absY() < sp.goalHalfWidth() + 2.0 )
    {
        return +1.0e+7;
    }

    if ( state.ball().pos().x < - ( sp.pitchHalfLength() - 0.1 )
         && state.ball().pos().absY() < sp.goalHalfWidth() )
    {
        return -1.0e+7;
    }

    if ( state.ball().pos().absX() > sp.pitchHalfLength()
         || state.ball().pos().absY() > sp.pitchHalfWidth() )
    {
        return -DBL_MAX / 2.0;
    }

    double point = state.ball().pos().x;
    point += std::max( 0.0,
                       40.0 - sp.theirTeamGoalPos().dist( state.ball().pos() ) );

    if ( FieldAnalyzer::can_shoot_from( holder_unum == state.self().unum(),
                                        holder->pos(),
                                        state.getPlayers( new OpponentOrUnknownPlayerPredicate( state.ourSide() ) ),
                                        8 ) )
    {
        point += 1.0e+6;
        if ( holder_unum == state.self().unum() )
        {
            point += 5.0e+5;
        }
    }

    return point;
}

static
std::vector<float> build_model_input_features( const PredictState & state )
{
    std::vector<float> features;
    features.reserve( env_bool( "RCSS_APPEND_EVAL_MODE_ONE_HOT", false ) ? 55 : 52 );

    features.push_back( static_cast<float>( state.ball().pos().x ) );
    features.push_back( static_cast<float>( state.ball().pos().y ) );

    const AbstractPlayerObject * holder = state.ballHolder();
    if ( holder )
    {
        features.push_back( 1.0F );
        features.push_back( static_cast<float>( holder->unum() ) );
        features.push_back( static_cast<float>( holder->pos().x ) );
        features.push_back( static_cast<float>( holder->pos().y ) );
    }
    else
    {
        features.push_back( 0.0F );
        features.push_back( 0.0F );
        features.push_back( 0.0F );
        features.push_back( 0.0F );
    }

    features.push_back( static_cast<float>( state.self().unum() ) );
    features.push_back( state.ourSide() == LEFT ? -1.0F : ( state.ourSide() == RIGHT ? 1.0F : 0.0F ) );

    AbstractPlayerObject::Cont opponents
        = state.getPlayers( new OpponentOrUnknownPlayerPredicate( state.ourSide() ) );
    std::sort( opponents.begin(), opponents.end(),
               []( const AbstractPlayerObject * lhs, const AbstractPlayerObject * rhs ) {
                   const int lu = lhs ? lhs->unum() : Unum_Unknown;
                   const int ru = rhs ? rhs->unum() : Unum_Unknown;
                   const int lk = ( lu == Unum_Unknown ? 999 : lu );
                   const int rk = ( ru == Unum_Unknown ? 999 : ru );
                   if ( lk != rk ) return lk < rk;
                   const double lx = lhs ? lhs->pos().x : 0.0;
                   const double rx = rhs ? rhs->pos().x : 0.0;
                   if ( lx != rx ) return lx < rx;
                   const double ly = lhs ? lhs->pos().y : 0.0;
                   const double ry = rhs ? rhs->pos().y : 0.0;
                   return ly < ry;
               } );

    for ( int i = 0; i < 11; ++i )
    {
        if ( i < static_cast<int>( opponents.size() ) && opponents[static_cast<size_t>( i )] )
        {
            const AbstractPlayerObject * opp = opponents[static_cast<size_t>( i )];
            features.push_back( static_cast<float>( opp->unum() ) );
            features.push_back( static_cast<float>( opp->pos().x ) );
            features.push_back( static_cast<float>( opp->pos().y ) );
            features.push_back( 1.0F );
        }
        else
        {
            features.push_back( 0.0F );
            features.push_back( 0.0F );
            features.push_back( 0.0F );
            features.push_back( 0.0F );
        }
    }

    if ( env_bool( "RCSS_APPEND_EVAL_MODE_ONE_HOT", false ) )
    {
        const std::string mode = env_string( "RCSS_EVAL_MODE_NAME", "normal" );
        features.push_back( mode == "normal" ? 1.0F : 0.0F );
        features.push_back( mode == "advanced" ? 1.0F : 0.0F );
        features.push_back( mode == "penalty" ? 1.0F : 0.0F );
    }

    return features;
}

} // namespace

SampleFieldEvaluator::SampleFieldEvaluator()
    : use_nn_( true ),
      model_load_path_( resolve_model_path() ),
      nn_model_( nullptr )
{
    if ( const char * mode = std::getenv( "RCSS_MODE" ) )
    {
        if ( std::string( mode ) == "collect" )
        {
            use_nn_ = false;
            std::cerr << "[INFO] RCSS_MODE=collect, disable eval model inference." << std::endl;
        }
    }

    if ( ! use_nn_ )
    {
        return;
    }

    try
    {
        nn_model_ = std::make_shared< torch::jit::script::Module >( torch::jit::load( model_load_path_ ) );
        nn_model_->eval();
        std::cerr << "[INFO] loaded eval model: " << model_load_path_ << std::endl;
    }
    catch ( const c10::Error & e )
    {
        use_nn_ = false;
        nn_model_.reset();
        std::cerr << "[WARN] failed to load eval model (fallback to heuristic evaluator): "
                  << e.what() << std::endl;
    }
}

SampleFieldEvaluator::~SampleFieldEvaluator()
{
}

double
SampleFieldEvaluator::operator()( const PredictState & state,
                                  const std::vector< ActionStatePair > & path ) const
{
    const auto finish = [&]( const double value, const char * source ) -> double {
        log_heatmap_value( state, value, source, path );
        return value;
    };

    // PlayOn 以外（セットプレー等）は従来評価のみを使う
    if ( state.gameMode().type() != rcsc::GameMode::PlayOn )
    {
        return finish( evaluate_state_original( state ), "original_non_playon" );
    }

    // 推論は実質キッカー時（自己がボール保持者）に限定する
    const AbstractPlayerObject * holder = state.ballHolder();
    if ( ! holder || holder->unum() != state.self().unum() )
    {
        return finish( evaluate_state_original( state ), "original_not_self_holder" );
    }

    if ( ! use_nn_ || ! nn_model_ )
    {
        return finish( evaluate_state_original( state ), "original_evaluate_state" );
    }

    const std::vector< double > heuristics = calculateHeuristics( state );

    const std::vector<float> model_features = build_model_input_features( state );
    torch::Tensor state_tensor = torch::from_blob(
        const_cast<float *>( model_features.data() ),
        { 1, static_cast<long>( model_features.size() ) },
        torch::kFloat ).clone();
    const double original_value = evaluate_state_original( state );
    init_heatmap_logger();

    try
    {
        const long long call_index = ++g_inference_call_count;
        if ( const char * dbg = std::getenv( "RCSS_LOG_INFERENCE" ) )
        {
            if ( std::string( dbg ) == "1" && ( call_index <= 5 || ( call_index % 500 ) == 0 ) )
            {
                std::cerr << "[INFER] call=" << call_index
                          << " unum=" << state.self().unum()
                          << " mode=PlayOn" << std::endl;
            }
        }

        std::vector< torch::jit::IValue > inputs;
        inputs.push_back( state_tensor );

        torch::Tensor weights_tensor = nn_model_->forward( inputs ).toTensor().to( torch::kFloat );
        weights_tensor = weights_tensor.squeeze( 0 ).contiguous();

        const int64_t output_numel = weights_tensor.numel();
        if ( output_numel == 1 )
        {
            // Imitation-only mode: model directly predicts scalar field value.
            const float * ptr = weights_tensor.data_ptr< float >();
            const double nn_value = static_cast< double >( ptr[0] );
            if ( g_heatmap_logger.pair_enabled )
            {
                log_heatmap_pair( state, original_value, nn_value, "paired_nn_scalar", path );
                return g_heatmap_logger.pair_return_nn ? nn_value : original_value;
            }
            return finish( nn_value, "nn_scalar" );
        }

        if ( static_cast< size_t >( output_numel ) != heuristics.size() )
        {
            std::cerr << "[WARN] model output size mismatch: " << output_numel
                      << " vs " << heuristics.size() << std::endl;
            return finish( evaluate_state_original( state ), "original_output_mismatch" );
        }

        std::vector< double > weights;
        weights.reserve( heuristics.size() );
        const float * ptr = weights_tensor.data_ptr< float >();
        for ( size_t i = 0; i < heuristics.size(); ++i )
        {
            weights.push_back( static_cast< double >( ptr[i] ) );
        }

        const double nn_value = calculateFieldEvaluation( heuristics, weights );
        if ( g_heatmap_logger.pair_enabled )
        {
            log_heatmap_pair( state, original_value, nn_value, "paired_nn_weighted", path );
            return g_heatmap_logger.pair_return_nn ? nn_value : original_value;
        }
        return finish( nn_value, "nn_weighted" );
    }
    catch ( const c10::Error & e )
    {
        std::cerr << "[WARN] eval model forward failed (fallback): " << e.what() << std::endl;
        return finish( evaluate_state_original( state ), "original_forward_error" );
    }
}

std::vector< double >
SampleFieldEvaluator::calculateHeuristics( const PredictState & state ) const
{
    std::vector< double > heuristics;
    heuristics.reserve( 10 );

    const ServerParam & sp = ServerParam::i();

    const AbstractPlayerObject * holder = state.ballHolder();
    if ( ! holder )
    {
        heuristics.assign( 10, 0.0 );
        return heuristics;
    }

    heuristics.push_back( state.ball().pos().x );

    const double goal_progress =
        std::max( 0.0,
                  std::min( 1.0,
                            ( state.ball().pos().x + sp.pitchHalfLength() )
                            / ( 2.0 * sp.pitchHalfLength() ) ) );
    heuristics.push_back( goal_progress );

    const double dist_to_goal = sp.theirTeamGoalPos().dist( state.ball().pos() );
    heuristics.push_back( std::exp( -dist_to_goal / 10.0 ) );

    const PlayerType * self_type = state.self().playerTypePtr();
    const double kickable_area = self_type ? self_type->kickableArea() : 0.0;
    const bool is_kickable = state.self().pos().dist( state.ball().pos() ) <= kickable_area;
    heuristics.push_back( is_kickable ? 1.0 : 0.0 );

    const double rel_vel = ( state.self().vel() - state.ball().vel() ).r();
    heuristics.push_back( std::tanh( rel_vel ) );
    heuristics.push_back( std::tanh( state.ball().vel().r() ) );

    const double dist_to_ball = state.self().pos().dist( state.ball().pos() );
    heuristics.push_back( std::exp( -dist_to_ball ) );

    const double goal_line_proximity =
        std::max( 0.0, std::abs( state.ball().pos().x ) - ( sp.pitchHalfLength() - 5.0 ) ) / 5.0;
    heuristics.push_back( goal_line_proximity );
    heuristics.push_back( std::tanh( state.self().vel().r() ) );
    heuristics.push_back( std::cos( state.self().body().radian() ) );

    return heuristics;
}

double
SampleFieldEvaluator::calculateFieldEvaluation( const std::vector< double > & heuristics,
                                                 const std::vector< double > & weights ) const
{
    if ( heuristics.size() != weights.size() )
    {
        return -DBL_MAX;
    }

    double evaluation = 0.0;
    for ( size_t i = 0; i < heuristics.size(); ++i )
    {
        evaluation += weights[i] * heuristics[i];
    }
    return evaluation;
}
