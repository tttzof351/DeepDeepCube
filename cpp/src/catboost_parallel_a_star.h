#ifndef CATBOOST_PARALLEL_A_STAR_H
#define CATBOOST_PARALLEL_A_STAR_H

#include <omp.h>
#include <vector>

#include "utils.h"
#include "nodes.h"
#include "cube3_game.h"

// #include "../../assets/models/catboost_cube3.cpp"

class CatboostParallelAStar {
    public:
    
    CatboostParallelAStar(
        Cube3Game& game,
        int limit_size,
        double* init_state_raw,
        bool debug
    ) {        
    }

    ~CatboostParallelAStar() {

    }

    Node* search(Cube3Game& game) {
        return nullptr;
    }

    int close_size() {
        return 0;
    }
};


#endif