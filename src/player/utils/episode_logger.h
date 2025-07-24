#ifndef EPISODE_LOGGER_H
#define EPISODE_LOGGER_H

#include <vector>
#include <string>
#include "utils/step_data.h"
#include <rcsc/geom/vector_2d.h>
#include <rcsc/player/world_model.h>

extern std::vector<StepData> episode_buffer;
extern bool prev_our_ball;  // 型を bool に統一
extern std::string match_id;

// 試合IDを生成する関数の宣言
void initialize_match_id();
void flush_episode_if_needed(const rcsc::WorldModel &wm);

#endif // EPISODE_LOGGER_H