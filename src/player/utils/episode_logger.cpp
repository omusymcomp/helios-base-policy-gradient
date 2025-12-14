#include "episode_logger.h"
#include "utils/step_data.h"

#include <fstream>
#include <iostream>
#include <rcsc/player/world_model.h>
#include "planner/predict_state.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>

extern std::string match_id;

// 試合IDを生成する関数
void initialize_match_id()
{
    std::ostringstream oss;
    std::time_t now = std::time(nullptr);
    oss << "match_" << std::put_time(std::localtime(&now), "%Y%m%d%H%M%S");
    match_id = oss.str();
    std::cerr << "[INFO] Match ID initialized: " << match_id << std::endl;
}


void flush_episode_if_needed(const rcsc::WorldModel &wm)
{
    bool our_ball = wm.lastKickerSide() == wm.ourSide();

    // // デバッグログを追加
    // std::cerr << "[DEBUG] Cycle: " << wm.time().cycle()
    //           << " | prev_our_ball: " << prev_our_ball
    //           << ", our_ball: " << our_ball << std::endl;

    const bool lost_possession = (prev_our_ball && !our_ball);
    bool episode_end = false;

    // エピソード終了条件：ボールロストまたはプレイオン以外の状態
    if (lost_possession || wm.gameMode().type() != rcsc::GameMode::PlayOn)
    {
        episode_end = true;
        // std::cerr << "[DEBUG] Cycle: " << wm.time().cycle()
        //           << " | Episode ended. Reason: Ball lost or GameMode not PlayOn" << std::endl;
    }
    prev_our_ball = our_ball;

    // if (episode_buffer.empty())
    // {
    // //     std::cerr << "[DEBUG] episode_buffer is empty, nothing to write to CSV" << std::endl;
    // // }
    // else
    // {
    //     std::cerr << "[DEBUG] episode_buffer size: " << episode_buffer.size() << std::endl;
    // }

    if (episode_end && !episode_buffer.empty())
    {
        double terminal_bonus = 0.0;
        if (lost_possession)
        {
            terminal_bonus -= 1.0;
        }
        if (wm.gameMode().type() == rcsc::GameMode::AfterGoal_)
        {
            terminal_bonus += (wm.lastKickerSide() == wm.ourSide() ? 3.0 : -3.0);
        }
        if (std::abs(terminal_bonus) > 1e-6)
        {
            episode_buffer.back().reward += terminal_bonus;
        }

        double gamma = 0.99;
        double G = 0.0;
        std::vector<double> returns(episode_buffer.size());
        for (int t = episode_buffer.size() - 1; t >= 0; --t)
        {
            G = episode_buffer[t].reward + gamma * G;
            returns[t] = G;
        }

        // ファイルを開いて即座に書き込む
        std::ofstream csv_file("/home/okayama/rcss/policy-gradient/logs/data.csv", std::ios::out | std::ios::app);
        if (!csv_file.is_open())
        {
            std::cerr << "[ERROR] Failed to open CSV file for writing" << std::endl;
            return;
        }

        // ファイルが空かどうかを確認してヘッダーを書き込む
        std::ifstream check_file("/home/okayama/rcss/policy-gradient/logs/data.csv", std::ios::ate | std::ios::binary);
        if (check_file.tellg() == 0) // ファイルサイズが0の場合
        {
            csv_file << "match_id,ball_x,ball_y,player_x,player_y,player_vel_x,player_vel_y,cycle,action_index,original_reward,discounted_reward,player_num";
            for (int i = 1; i <= 10; ++i)
            {
                csv_file << ",heuristic_" << i;
            }
            csv_file << "\n";
            // std::cerr << "[DEBUG] CSV header written." << std::endl;
        }

        for (size_t t = 0; t < episode_buffer.size(); ++t)
        {
            std::ostringstream line;
            line << match_id << ",";
            for (auto v : episode_buffer[t].features)
            {
                line << v << ",";
            }
            line << episode_buffer[t].cycle << ",";
            line << episode_buffer[t].action_index << ",";
            line << episode_buffer[t].reward << ",";
            line << returns[t] << ",";
            line << episode_buffer[t].player_num;

            // ヒューリスティック項を出力
            for (const auto &heuristic : episode_buffer[t].heuristics)
            {
                line << "," << heuristic;
            }

            line << "\n";
            csv_file << line.str();
        }

        csv_file.flush();

        // std::cerr << "[DEBUG] Cycle: " << wm.time().cycle()
        //           << " | Episode data written to CSV." << std::endl;

        episode_buffer.clear();
    }
}
