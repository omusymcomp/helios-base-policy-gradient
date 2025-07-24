#pragma once
#include <vector>
#include <string>

// 行動・状態・報酬など1ステップ分のログデータ構造
struct StepData {
    std::vector<double> features;
    std::vector<double> heuristics;
    int cycle;           // サイクル数
    int action_index;    // 選択された行動インデックス
    double reward;       // 単発報酬
    int player_num;      // 背番号
};

extern std::string match_id;