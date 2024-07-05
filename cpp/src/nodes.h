#ifndef NODES_H
#define NODES_H

#include <vector>
#include <queue>
#include <map>

#include "utils.h"

using namespace std;

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

#endif