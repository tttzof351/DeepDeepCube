#ifndef A_STAR_H
#define A_STAR_H

#include "utils.h"
#include "cube3_game.h"

class Node {
    public:

    int space_size = -1;
    vector<float>* state = nullptr; //TODO: Move to stack
    uint64_t state_hash = -1;

    float g = 0;
    float h = 0;
    float f = 0;

    Node* parent = nullptr;
    int action = -1;

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

    vector<Node*> get_path() {
        Node* node = this;
        vector<Node*> path;

        while (node != nullptr) {
            path.push_back(node); // TODO: push_front ? 
            node = node->parent;
        }
        std::reverse(path.begin(), path.end());

        return move(path);
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
        for (auto it = open.hashes.begin(); it != open.hashes.end(); it++) {
            delete it->second;
        }

        for (auto it = close.hashes.begin(); it != close.hashes.end(); it++) {
            delete it->second;
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
                child->action = action;

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
                bool is_goal = game.is_goal(*(child->state));

                if (is_goal) {
                    //For prevent memory leak
                    for (int j = action + 1; j < game.action_size; j++) {
                        delete child_nodes[j];
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
                
                cout << "size close: " 
                << this->close.size() 
                << "; Duration: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()  
                << " ms"
                << endl; 

                start = end;
            }

            if (global_i > this->limit_size) {
                return nullptr;
            }
        }

        return nullptr;
    }
};


#endif
