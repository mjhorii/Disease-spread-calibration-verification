#include "common.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

using namespace std;

std::random_device rd;
std::mt19937 gen;
std::uniform_real_distribution<double> rand_11(-1,1);   //uniform distribution between -1,1
std::uniform_real_distribution<double> rand_01(0,1);    //uniform distribution between -1,1
std::uniform_int_distribution<int> rand_int;            //uniform integer distribution
double d_IU_sq;                                         //radius of infection squared
std::vector<int> S_agents;                              //vector of ids of susceptible agents
std::vector<int> E_agents;                              //vector of ids of exposed agents
std::vector<int> I_agents;                              //vector of ids of infected agents
std::vector<int> DR_agents;                             //vector of ids of dead/recovered agents
vector<vector<double>> domain_limits_x;                 //geometric domain limits (x-dir)
vector<vector<double>> domain_limits_y;                 //geometric domain limits (y-dir)
std::vector<int> subPops;                               //vector of subpop ids (e.g., if there are 3 subpops, subPops = [0,1,2])
std::vector<int>::iterator iter; 
string savename2;                                       //filenames to save data
string savename3;
string savename4;
string savename5;
string savename6;


//Vectors of vectors to store simulation data:
//Contain number of agents of a specific status for each subpopulation
//for example, S[0][10] is number of susceptible agents in subpop 0 at timestep 10
std::vector<vector<int>> S; //susceptible
std::vector<vector<int>> E; //exposed
std::vector<vector<int>> I; //infected
std::vector<vector<int>> DR; //dead/recovered
std::vector<vector<int>> new_I; //newly infected

int count_in_subPop(agent* agents, std::vector<int> agent_IDs, int subPop_ID) {
    //count the number of agents (from agent_IDs) that are in a specific subpopulation (subPop_ID)
    int count = 0;
    for (int j : agent_IDs){
        if (agents[j].subPop == subPop_ID){
            ++count;
        }
    }
    return count;
}

int count_from_original_subPop(agent* agents, std::vector<int> agent_IDs, int subPop_ID) {
    //count the number of agents (from agent_IDs) that are from a specific original subpopulation (subPop_ID)
    int count = 0;
    for (int j : agent_IDs){
        if (agents[j].original_subPop == subPop_ID){
            ++count;
        }
    }
    return count;
}

void save_matrix_data(string save_name, std::vector<vector<int>> &matrix, int nsteps, int m){
    //save data from matrix to file with name save_name. nsteps and m (number of subpops) define dimensions of matrix
    std::ofstream fsave(save_name);
    for (int i = 0; i < nsteps+1; ++i) {
        for (int j = 0; j<m-1;++j){
            fsave << matrix[j][i] << " ";
        }
    fsave << matrix[m-1][i];
    fsave << "\n";
    }
}

// Move agents
void move(agent& agent, double jump_prob, double mobility, double del_t) {
    //sub-population jumping with probability jump_prob
    if (rand_01(gen)<jump_prob){
        std::vector<int> choices{subPops}; // make a copy of vec
        choices.erase(choices.begin() + agent.subPop);
        int ind = rand_int(gen);
        int new_subPop = choices[ind];
        agent.subPop = new_subPop;
    }

    // Move:
    agent.x = agent.x + mobility*del_t*rand_11(gen);
    agent.y = agent.y + mobility*del_t*rand_11(gen);

    // Bounce from walls
    double x_low = domain_limits_x[agent.subPop][0];
    double x_high = domain_limits_x[agent.subPop][1];
    double y_low = domain_limits_y[agent.subPop][0];
    double y_high = domain_limits_y[agent.subPop][1];

    while (agent.x < x_low || agent.x > x_high) {
        agent.x = agent.x < x_low ? 2 * x_low - agent.x : 2 * x_high - agent.x;
    }

    while (agent.y < y_low || agent.y > y_high) {
        agent.y = agent.y < y_low ? 2 * y_low - agent.y : 2 * y_high - agent.y;
    }
}

// Integrate the ODE
void check_infection(agent& infected, agent& neighbor, double d_IU) { //must pass in susceptible neighbors
    // // Calculate Distance
    double dx = neighbor.x - infected.x;
    double dy = neighbor.y - infected.y;
    double r2 = dx * dx + dy * dy;

    if (r2 < d_IU_sq){
        neighbor.status = 1; //if within infection radius, set status to exposed
        iter = std::find(S_agents.begin(), S_agents.end(), neighbor.id); 
        if (iter != S_agents.end()){ //make sure agent wasn't already exposed
            S_agents.erase(iter);
            E_agents.push_back(neighbor.id);
            neighbor.timer = 0;
        }
    }

}

void init_simulation(agent* agents, int total_pop, double d_IU, int m, vector<vector<double>> domain_limits_x_, vector<vector<double>> domain_limits_y_, int part_seed, char* savename){
// void init_simulation(agent* agents, int total_pop, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    //filenames:
    if (savename == nullptr){
        cout<<"Please provide output file name\n";
    } else{
        savename2 = "new_I_";
        savename2 = savename2 + savename;
        savename3 = "I_";
        savename3 = savename3 + savename;
        savename4 = "S_";
        savename4 = savename4 + savename;
        savename5 = "E_";
        savename5 = savename5 + savename;
        savename6 = "DR_";
        savename6 = savename6 + savename;
    }

    domain_limits_x = domain_limits_x_;
    domain_limits_y = domain_limits_y_;

    std::mt19937 gen(part_seed ? part_seed : rd());
    rand_int.param(std::uniform_int_distribution<int>::param_type(0, m-2));
    d_IU_sq = d_IU*d_IU;

    //Save agent IDs into vectors based on status
    for (int i = 0; i < total_pop; ++i){
        if(agents[i].status == 0){
            S_agents.push_back(agents[i].id);
        } else if(agents[i].status == 1){
            E_agents.push_back(agents[i].id);
        } else if(agents[i].status == 2){
            I_agents.push_back(agents[i].id);
        } else if(agents[i].status == 3){
            DR_agents.push_back(agents[i].id);
        }
    }

    for (int i = 0; i < m; i++) {
        subPops.push_back(i);
    }

    S.resize(m);
    E.resize(m);
    I.resize(m);
    DR.resize(m);
    new_I.resize(m);

    //Count and save how many agents have each status
    for (int i = 0; i < m; i++){
        S[i].push_back(count_in_subPop(agents, S_agents, i));
        E[i].push_back(count_in_subPop(agents, E_agents, i));
        I[i].push_back(count_in_subPop(agents, I_agents, i));
        DR[i].push_back(count_in_subPop(agents, DR_agents, i));
        // S[i].push_back(S_agents.size());
        // E[i].push_back(E_agents.size());
        // I[i].push_back(I_agents.size());
        // DR[i].push_back(DR_agents.size());
        new_I[i].push_back(0);
    }

}

void simulate_one_step(agent* agents, int total_pop, double d_IU, double del_t, double jump_prob, double mobility, int m, int step, int nsteps){

    vector<int> new_I_agents;

    //Update timers:
    for (int i : E_agents){
        agents[i].timer = agents[i].timer + del_t;
    }
    for (int i : I_agents){
        agents[i].timer = agents[i].timer + del_t;
    }

    //Update statuses based on time:
    //Check for infected agents that should become dead/recovered:
    iter = I_agents.begin();
    while (iter != I_agents.end()) {
        int i = *iter;
        if (agents[i].timer >= agents[i].I_time){
            agents[i].status = 3;
            iter = I_agents.erase(iter);
            DR_agents.push_back(agents[i].id);
            agents[i].timer = 0;
        }
        else {
            ++iter;
        }
    }
    //Check for exposed agents that should become infected:
    iter = E_agents.begin();
    while (iter != E_agents.end()) {
        int i = *iter;
        if (agents[i].timer >= agents[i].E_time){
            agents[i].status = 2;
            iter = E_agents.erase(iter);
            I_agents.push_back(agents[i].id);
            agents[i].timer = 0;
            new_I_agents.push_back(agents[i].id);
        }
        else {
            ++iter;
        }
    }
    // std::cout << "here4 " << std::endl;

    //Check for new exposures:
    for (int i: I_agents){
        for (int j : S_agents){
            check_infection(agents[i], agents[j], d_IU);
        }
    }
    // std::cout << "total_pop " <<total_pop<< std::endl;

    //Move agents:
    for (int i = 0; i < total_pop; ++i){
        move(agents[i], jump_prob, mobility, del_t);
    }
    // std::cout << "here6 " << std::endl;

    //Save data:
    for (int i = 0; i < m; i++){
        // // track data infections based on current subpopulation
        // S[i].push_back(count_in_subPop(agents, S_agents, i));
        // E[i].push_back(count_in_subPop(agents, E_agents, i));
        // I[i].push_back(count_in_subPop(agents, I_agents, i));
        // DR[i].push_back(count_in_subPop(agents, DR_agents, i));
        // new_I[i].push_back(count_in_subPop(agents, new_I_agents, i));

        //track data based on agent's original subpopulation
        S[i].push_back(count_from_original_subPop(agents, S_agents, i));
        E[i].push_back(count_from_original_subPop(agents, E_agents, i));
        I[i].push_back(count_from_original_subPop(agents, I_agents, i));
        DR[i].push_back(count_from_original_subPop(agents, DR_agents, i));
        new_I[i].push_back(count_from_original_subPop(agents, new_I_agents, i)); 
    }

    //At last time step, save data to file:
    if (step == nsteps-1){

        save_matrix_data(savename2, new_I, nsteps, m);
        save_matrix_data(savename3, I, nsteps, m);
        save_matrix_data(savename4, S, nsteps, m);
        save_matrix_data(savename5, E, nsteps, m);
        save_matrix_data(savename6, DR, nsteps, m);

    }

}