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

#ifndef SAMPLE_TRAINER_H
#define SAMPLE_TRAINER_H

#include <rcsc/trainer/trainer_agent.h>
#include <rcsc/geom/vector_2d.h>
#include <array>
#include <string>
#include <vector>

class SampleTrainer
    : public rcsc::TrainerAgent {
private:
    struct ScenarioSnapshot
    {
        rcsc::Vector2D ball{ 0.0, 0.0 };
        std::array< rcsc::Vector2D, 11 > players{};
        bool valid{ false };
    };

    ScenarioSnapshot M_our_snapshot;
    ScenarioSnapshot M_opp_snapshot;
    bool M_scenario_ready;
    bool M_need_reset;
    int M_last_reset_cycle;
    std::string M_team_left;
    std::string M_team_right;
    double M_episode_start_x;
    int M_episode_limit;

public:

    SampleTrainer();

    virtual
    ~SampleTrainer();

protected:

    /*!
      You can override this method.
      But you must call TrainerAgent::doInit() in this method.
    */
    virtual
    bool initImpl( rcsc::CmdLineParser & cmd_parser );

    //! main decision
    virtual
    void actionImpl();

    virtual
    void handleInitMessage();
    virtual
    void handleServerParam();
    virtual
    void handlePlayerParam();
    virtual
    void handlePlayerType();

private:

    bool loadScenario();
    void resetScenario();
    bool isBallOut() const;
    bool isBallLost() const;

};

#endif
