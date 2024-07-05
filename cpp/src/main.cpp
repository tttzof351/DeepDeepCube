#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <ctime>

#include <omp.h>
#include <unistd.h> // sleep() function

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utils.h"
#include "cube3_game.h"

#include "heuristic_a_star.h"
#include "catboost_a_star.h"

using namespace std;
namespace py = pybind11;

struct ResultSearch {
    vector<int> actions;
    vector<float> h_values;
    int visit_nodes = 0;
};

/* ============= Global variables ============= */

Cube3Game game = Cube3Game();

/* ============================================ */

void run_openmp_test() {
    auto start = std::chrono::high_resolution_clock::now();    
    
    {
        #pragma omp parallel for 
        for (int i = 0; i < 5; i++) {
            sleep(3);  
            std::cout << "Thread " << omp_get_thread_num() << " completed iteration " << i << std::endl;      
        }
    }
    
    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "Run openmp test (without 15000 ms): "
    << std::chrono::duration_cast<std::chrono::milliseconds>(done-start).count() << " ms"
    << std::endl << std::endl;
}

void init_envs(py::array_t<double> actions) {    
    init_wyhash();

    py::buffer_info action_info = actions.request();
    game.set_actions(
        action_info.shape[0], //18
        action_info.shape[1], //54
        (double*) action_info.ptr
    );    
}

ResultSearch heuristic_search_a(
    py::function heuristic,
    py::array_t<double> state, 
    int limit_size, 
    bool debug
) {
    py::buffer_info state_info = state.request();
    ResultSearch result;

    if (game.is_goal((double*) state_info.ptr)) {
        return result;
    }
    
    HeuristicAStar astar = HeuristicAStar(
        heuristic,
        game,
        limit_size,
        (double*) state_info.ptr,
        debug
    );

    Node* target = astar.search(game);
    result.visit_nodes = astar.close.size();

    if (target == nullptr) {
        return result;
    } else if (debug) {
        cout << "Found!" << endl;
    }

    vector<Node*> path = target->get_path();

    for (int i = 0; i < path.size(); i++) {
        Node* n = path[i];
        result.actions.push_back(n->action);
        result.h_values.push_back(n->h);
    }

    return result;   
}

ResultSearch catboost_search_a(
    py::array_t<double> state, 
    int limit_size, 
    bool debug
) {
    py::buffer_info state_info = state.request();
    ResultSearch result;

    if (game.is_goal((double*) state_info.ptr)) {
        return result;
    }
    
    CatboostAStar astar = CatboostAStar(        
        game,
        limit_size,
        (double*) state_info.ptr,
        debug
    );

    Node* target = astar.search(game);
    result.visit_nodes = astar.close.size();

    if (target == nullptr) {
        return result;
    } else if (debug) {
        cout << "Found!" << endl;
    }

    vector<Node*> path = target->get_path();

    for (int i = 0; i < path.size(); i++) {
        Node* n = path[i];
        result.actions.push_back(n->action);
        result.h_values.push_back(n->h);
    }

    return result;   
}


PYBIND11_MODULE(cpp_a_star, m) { 
    m.doc() = "cpp_a_star module"; 
    
    m.def("init_envs", &init_envs, "init_envs");
    m.def("run_openmp_test", &run_openmp_test, "run_openmp_test");

    py::class_<ResultSearch>(m, "ResultSearch")
            .def(py::init<>())
            .def_readwrite("actions", &ResultSearch::actions)
            .def_readwrite("h_values", &ResultSearch::h_values)
            .def_readwrite("visit_nodes", &ResultSearch::visit_nodes);

    m.def("heuristic_search_a", &heuristic_search_a, "heuristic_search_a"); 
    m.def("catboost_search_a", &catboost_search_a, "catboost_search_a"); 

    
    // m.def("test_allocation_dealocation", &test_allocation_dealocation, "test_allocation_dealocation");
}