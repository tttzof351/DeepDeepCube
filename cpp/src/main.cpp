#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <ctime>


#include <omp.h>
#include <unistd.h> // sleep() function

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../assets/models/catboost_cube3.cpp"
#include "utils.h"
#include "wyhash.h"

using namespace std;

namespace py = pybind11;

/* ===== HASH ========= */
int seed = 42;
uint64_t _wyp_s[4];

uint64_t w_hash(const void *data, size_t len) {
    return wyhash(data, len, seed, _wyp_s);
}

class Cube3Game {
    public:
    
    int space_size = -1;
    int action_size = -1;
    int* actions = nullptr;
    bool debug = true;

    Cube3Game() {    
        ;
    }

    ~Cube3Game() {
        if (actions != nullptr) {
            delete[] actions;
            actions = nullptr;
        }
    }

    void set_actions(
        int action_size, // Must be 18
        int space_size, // Must be 54
        double* external_actions
    ) {
        if (debug) {
            cout << "set_actions; action_size: " << action_size << "; space_size: " << space_size << endl;
        }

        this->space_size = space_size;
        this->action_size = action_size;

        this->actions = new int[action_size * space_size];
        
        #pragma omp simd
        for (int i = 0; i < action_size * space_size; i++) {
            this->actions[i] = int(external_actions[i]);
        }

        // if (debug) {
        //     cout << "action[0] in c++: ";
        //     for (int i = 0; i < this->space_size; i++) {
        //         cout << this->actions[i] << " ";
        //     }
        //     cout << endl;
        // }
    }

    bool is_goal_by_state(vector<float>& state) {
        bool answer = true;
        #pragma omp simd        
        for (int i = 0; i < this->space_size; i++) {
            if (int(state[i]) != i) {
                answer = false;
            }            
        }
        return answer;
    }    

    void apply_action(
        vector<float>& in_state,
        vector<float>& out_state,
        int action
    ) {
        #pragma omp simd
        for (int i = 0; i < this->space_size; i++) {
            out_state[i] = int(in_state[
                int(this->actions[action * this->space_size + i])
            ]);
            // out_state[i] = this->space_size - i;
        }
    }
};


class Node {
    public:

    int space_size = -1;
    vector<float>* state = nullptr;
    uint64_t state_hash = -1;

    Node* parent = nullptr;

    int g = 0;
    int h = 0;
    int f = 0;

    Node(
        int space_size
    ) {
        this->space_size = space_size;
        this->state = new vector<float>(space_size);
    }

    ~Node() {
        if (this->state != nullptr) {
            delete this->state;
            this->state = nullptr;
        }
    }

    void reset_hash() {
        this->state_hash = w_hash(
            (void*)&this->state->at(0), 
            this->space_size * sizeof(float)
        );
    }

    void reset_f() {
        this->f = this->h + this->g;
    }
};

struct CompareNode {
    bool operator()(const Node* n1, const Node* n2) {
        return n1->f > n2->f;
    }
};

class NodeQueue {
    public:

    std::priority_queue <Node*, vector<Node*>, CompareNode> queue;
    std::map<uint64_t, Node*> hashes;

    NodeQueue() {
        ;
    }

    ~NodeQueue() {
        ;
    }

    void insert(Node* node) {
        this->queue.push(node);
        this->hashes[node->state_hash] = node;
    }

    Node* pop_min_element() {
        Node* node = this->queue.top();
        this->queue.pop();

        this->hashes.erase(node->state_hash);

        return node;
    }

    int size() {
        return this->queue.size();
    }

    bool is_contains(Node* node) {
        return this->hashes.find(node->state_hash) != this->hashes.end();
    }
};

class AStar {
    public:
    
    int limit_size = -1;
    bool debug = false;    
    NodeQueue open;
    NodeQueue close;

    AStar(
        Cube3Game& game,
        int limit_size,
        double* init_state_raw,
        bool debug
    ) {
        this->limit_size = limit_size;
        this->debug = debug;

        Node* root_node = new Node(game.space_size);
        
        //TODO: Memcopy
        for (int i = 0; i < game.space_size; i++) {
            root_node->state->at(i) = int(init_state_raw[i]);
        }
        
        root_node->h = ApplyCatboostModel(*root_node->state);
        root_node->reset_hash();
        root_node->reset_f();

        this->open.insert(root_node);
    }

    ~AStar() {        
        while (open.size() > 0) {            
            delete open.pop_min_element();
        }

        while (close.size() > 0) {
            delete close.pop_min_element();
        }
    }

    Node* search(Cube3Game& game) {
        if (this->debug) {
            cout << "search, size open: " << this->open.size() << endl;
        }
        
        Node* child_nodes[game.action_size];
        int global_i = 0;

        auto start = std::chrono::high_resolution_clock::now();
        while (this->open.size() > 0) {
            Node* best_node = this->open.pop_min_element();

            //Initialization childs
            #pragma omp parallel for
            for (int action = 0; action < game.action_size; action++) {
                Node* child = new Node(game.space_size);

                game.apply_action(
                    *(best_node->state), // in
                    *(child->state), // out
                    action
                );

                child->h = ApplyCatboostModel(*(child->state));
                child->g = best_node->g + 1;

                child->parent = best_node;                
                child->reset_hash();
                child->reset_f();

                child_nodes[action] = child;
            }

            for (int action = 0; action < game.action_size; action++) {
                Node* child = child_nodes[action];
                bool is_goal = game.is_goal_by_state(*(child->state));

                if (is_goal) {
                    //For prevent memory leak
                    for (int j = action; j < game.action_size; j++) {
                        delete child_nodes[j];
                    }

                    if (debug) {
                        cout << "Found! " << endl;
                    }
                    return child; 
                                
                } else if (this->close.is_contains(child)) {
                    delete child;
                    continue;
                } else if (this->open.is_contains(child)) {
                    //TODO: Need implementation
                    delete child;

                    continue;
                } else {
                    this->open.insert(child);
                }
            }

            this->close.insert(best_node);            
            
            global_i += 1;
            if (debug && global_i % 1000 == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                
                cout << "size open: " 
                << this->close.size() 
                << "; Duration: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()  
                << " ms"
                << endl; 

                start = end;
            }

            // if (global_i > 0) {
            //     break;s
            // }
        }

        return nullptr;
    }
};

/* ============= Global variables ============= */

Cube3Game game = Cube3Game();

void set_cube3_actions(py::array_t<double> actions) {
    cout << "c++ call set_cube3_actions" << endl;
    py::buffer_info action_info = actions.request();
    game.set_actions(
        action_info.shape[0], //18
        action_info.shape[1], //54
        (double*) action_info.ptr
    );
    cout << "c++ end set_cube3_actions" << endl;
}

void init_wyhash() {
    cout << "c++ call init_wyhash" << endl;
    make_secret(time(NULL), _wyp_s);
    cout << "c++ end init_wyhash" << endl;
}

int add_function(int i, int j) {
    return i + j;
}

void run_openmp_test() {
    auto start = std::chrono::high_resolution_clock::now();    
    
    // #pragma omp parallel 
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

void search_a(py::array_t<double> state, int limit_size, bool debug) {
    if (debug) {
        cout << "Start search!" << endl;
    }

    py::buffer_info state_info = state.request();
    AStar astar = AStar(
        game,
        limit_size,
        (double*) state_info.ptr,
        debug
    );

    astar.search(game);
}

PYBIND11_MODULE(a_star, m) { 
    m.doc() = "a_star module"; // optional module docstring

    m.def("add_function", &add_function, "add_function");
    
    m.def("set_cube3_actions", &set_cube3_actions, "set_cube3_actions");
    m.def("init_wyhash", &init_wyhash, "init_wyhash");
    m.def("search_a", &search_a, "search_a");
    m.def("run_openmp_test", &run_openmp_test, "run_openmp_test");
}


// int main() {
//     auto start = std::chrono::high_resolution_clock::now();
    
//     #pragma omp parallel for
//     for (int i = 0; i < 10'000; i++) {
//         std::vector<float> floatFeatures(54);
//         for (int j = 0; j < 54; j++) {
//             floatFeatures[j] = j;
//         }

//         float predict = ApplyCatboostModel(floatFeatures);
//     }
    
//     auto done = std::chrono::high_resolution_clock::now();
//     std::cout << "Catboost 10K: "
//     << std::chrono::duration_cast<std::chrono::milliseconds>(done-start).count() << " ms"
//     << std::endl << std::endl;

    
//     return 0;
// }