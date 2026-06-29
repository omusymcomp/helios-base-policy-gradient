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

/////////////////////////////////////////////////////////////////////

#ifndef SAMPLE_FIELD_EVALUATOR_H
#define SAMPLE_FIELD_EVALUATOR_H

#include "field_evaluator.h"
#include "predict_state.h"

#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>

class ActionStatePair;

class SampleFieldEvaluator
    : public FieldEvaluator {
private:
    bool use_nn_;
    std::string model_load_path_;
    std::shared_ptr< torch::jit::script::Module > nn_model_;

    double calculateFieldEvaluation( const std::vector< double > & heuristics,
                                     const std::vector< double > & weights ) const;

public:
    SampleFieldEvaluator();

    virtual
    ~SampleFieldEvaluator();

    virtual
    double operator()( const PredictState & state,
                       const std::vector< ActionStatePair > & path ) const;

    std::vector< double > calculateHeuristics( const PredictState & state ) const;
};

#endif
