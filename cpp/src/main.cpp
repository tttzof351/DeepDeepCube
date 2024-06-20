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

#include "../../assets/models/catboost_cube3.cpp"

#include "utils.h"
#include "cube3_game.h"
#include "a_star.h"

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

ResultSearch search_a(py::array_t<double> state, int limit_size, bool debug) {
    py::buffer_info state_info = state.request();
    ResultSearch result;

    if (game.is_goal((double*) state_info.ptr)) {
        return result;
    }
    
    AStar astar = AStar(
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

void test_allocation_dealocation() {
    cout << "Start alloc in C++" << endl;
    double mock_state[54];

    AStar astar = AStar(
        game,
        100'000'000,
        mock_state,
        true
    );
    for (int i = 0; i < 10'000'000; i++) {
        astar.open.insert(
            new Node(54)
        );
        astar.close.insert(
            new Node(54)
        );
        if (i % 10000 == 0) {
            cout << "i: " << i << endl;
        }
    }
    cout << "End alloc in C++" << endl;
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

    m.def("search_a", &search_a, "search_a"); 
    
    m.def("test_allocation_dealocation", &test_allocation_dealocation, "test_allocation_dealocation");
}