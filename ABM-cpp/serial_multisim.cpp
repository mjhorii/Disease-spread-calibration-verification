#include "common_multisim.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

using namespace std;

std::random_device rd;
// std::mt19937 gen;
std::uniform_real_distribution<double> rand_11(-1,1);   //uniform distribution between -1,1
std::uniform_real_distribution<double> rand_01(0,1);    //uniform distribution between -1,1
// std::uniform_int_distribution<int> rand_int;            //uniform integer distribution
double d_IU_sq;                                         //radius of infection squared
std::vector<vector<int>> S_agents;                      //vector with entries for each subsim (entry: vector of ids of susceptible agents)
std::vector<vector<int>> E_agents;                      //vector with entries for each subsim (entry: vector of ids of exposed agents)
std::vector<vector<int>> I_agents;                      //vector with entries for each subsim (entry: vector of ids of infected agents)
std::vector<vector<int>> DR_agents;                     //vector with entries for each subsim (entry: vector of ids of dead/recovered agents)
// vector<vector<double>> domain_limits_x;                 //geometric domain limits (x-dir)
// vector<vector<double>> domain_limits_y;                 //geometric domain limits (y-dir)
vector<double>* domain_limits_x;                        //store domain limits for subpopulations (x-direction) (initialize as pointer so data doesn't need to be copied)
vector<double>* domain_limits_y;                        //store domain limits for subpopulations (y-direction)
std::vector<vector<int>> subPops;                       //vector with entries for each subsim (entry: vector of subpop ids (e.g., if there are 3 subpops in subsim 0, subPops[0] = [0,1,2]))
std::vector<int>::iterator iter; 
string savename2;                                       //filenames to save data
string savename3;
string savename4;
string savename5;
string savename6;

int* ms_per_subSim;                                     // pointer to start of array of number of subpops for each subsim
std::vector<int> unique_m_vals;
int* subSim_IDs; //could/should this be done with references instead?

std::mt19937* rng_state;                                // pointer to start of array of rng_states
std::uniform_int_distribution<int>* rand_int;           // pointer to start of array of uniform random integer generators

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
    int test;
    for (int j : agent_IDs){
        // std::cout << j<<"\n";
        test = agents[j].original_subPop;
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
void move(agent& agent, vector<double> &jump_probs, vector<double> &mobilities, double del_t) {
    //sub-population jumping with probability jump_prob
    int subSim = agent.subSim;
    int subSim_m = ms_per_subSim[subSim];
    double jump_prob = jump_probs[agent.subPop];
    double mobility = mobilities[agent.subPop];

    //find the index of subSim_m in unique_m_vals
    iter = std::find(unique_m_vals.begin(), unique_m_vals.end(), subSim_m);
    int index;
    if (iter != unique_m_vals.end()) {
        index = std::distance(unique_m_vals.begin(), iter);
    } else {
        std::cout << "error: the m value of this agent's subsim is not in the vector unique_m_vals \n" << std::endl;
    }
    // std::cout << "move2 \n" << std::endl;

    if (rand_01(rng_state[agent.id])<jump_prob){
        //determine which subpop the agent will move to:
        std::vector<int> choices{subPops[subSim]}; // make a copy of vec
        choices.erase(choices.begin() + agent.subPop - subPops[subSim][0]); // agent.subPop - subPops[subSim][0] should be subPop number within subsim -- if subpop 16 is in subsim_ID=4 with subpop[4] = [14,15,16,17], it is 16-14 = 2, which is index within subPops[subSim]
        if (agent.subPop - subPops[subSim][0] < 0 ||agent.subPop - subPops[subSim][0]>2){
            std::cout<<"INDEX WRONG";
        }
        int ind = rand_int[index](rng_state[agent.id]);
        int new_subPop = choices[ind];
        agent.subPop = new_subPop; 

        // //assign the agent a position within that subpop:
        double rand_num = rand_01(rng_state[agent.id]);
        agent.x = rand_num*(domain_limits_x[agent.subPop][1]-domain_limits_x[agent.subPop][0])+domain_limits_x[agent.subPop][0];
        rand_num = rand_01(rng_state[agent.id]);
        agent.y = rand_num*(domain_limits_y[agent.subPop][1]-domain_limits_y[agent.subPop][0])+domain_limits_y[agent.subPop][0];
        // // std::cout << "move5 \n" << std::endl;
        
    }

    // std::cout << "move6 \n" << std::endl;
    // Move:
    agent.x = agent.x + mobility*del_t*rand_11(rng_state[agent.id]);
    agent.y = agent.y + mobility*del_t*rand_11(rng_state[agent.id]);

    // std::cout << "move7 \n" << std::endl;

    // Bounce from walls
    double x_low = domain_limits_x[agent.subPop][0];
    double x_high = domain_limits_x[agent.subPop][1];
    double y_low = domain_limits_y[agent.subPop][0];
    double y_high = domain_limits_y[agent.subPop][1];

    // if (agent.id == 201){
    //     std::cout<<"subpop: "<<agent.subPop<<"\n";
    //     std::cout<<x_low<<" "<<x_high<<" "<<y_low<<" "<<y_high<<"\n";
    // }

    // std::cout << "move8 \n" << std::endl;

    while (agent.x < x_low || agent.x > x_high) {
        agent.x = agent.x < x_low ? 2 * x_low - agent.x : 2 * x_high - agent.x;
    }

    while (agent.y < y_low || agent.y > y_high) {
        agent.y = agent.y < y_low ? 2 * y_low - agent.y : 2 * y_high - agent.y;
    }
}

// Integrate the ODE
void check_infection(agent& infected, agent& neighbor, double d_IU) { //must pass in susceptible neighbors //not true anymore i think
    // // Calculate Distance
    double dx = neighbor.x - infected.x;
    double dy = neighbor.y - infected.y;
    double r2 = dx * dx + dy * dy;
    int subSim = infected.subSim; 
    // int subSim_neighbor = neighbor.subSim; //debugging

    // if (subSim != subSim_neighbor){ //debugging
    //     std::cout<<"MISMATCHED SIMULATIONS\n";
    // }

    if (r2 < d_IU_sq){
        // neighbor.status = 1; //if within infection radius, set status to exposed

        iter = std::find(S_agents[subSim].begin(), S_agents[subSim].end(), neighbor.id); 
        if (iter != S_agents[subSim].end()){ //if neighbor is already in susceptible vector (aka, not already exposed), iter != S_agents.end() is true //make sure agent wasn't already exposed
            S_agents[subSim].erase(iter);
            // std::cout<<"push back:"<<neighbor.id<<"\n";
            E_agents[subSim].push_back(neighbor.id); //**mark
            neighbor.timer = 0; 
            neighbor.status = 1; //if within infection radius, set status to exposed
        }
    }

}

void init_simulation(agent* agents, int total_pop, double d_IU, int m, vector<vector<double>> &domain_limits_x_, 
        vector<vector<double>> &domain_limits_y_, vector<int> &seeds_per_subSim, int num_of_subSims, 
        vector<int> &subSim_IDs_, vector<int> &ms_per_subSim_, char* savename){

    // void init_simulation(agent* agents, int total_pop, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    domain_limits_x = domain_limits_x_.data(); 
    domain_limits_y = domain_limits_y_.data();
    ms_per_subSim = ms_per_subSim_.data();
    subSim_IDs = subSim_IDs_.data();

    //Initialize random number generators for each agent
    rng_state = (std::mt19937*)malloc(total_pop*sizeof(std::mt19937)); //should i free these somewhere?
    // rng_state = new std::mt19937(total_pop); //should i delete the things i "new"ed somewhere?

    for (int i = 0; i < total_pop; ++i){ //for each agent,
        std::seed_seq seq{seeds_per_subSim[agents[i].subSim], agents[i].id_in_subSim}; //use a seed sequence with subsim seed and specific agent ID within subsim
        rng_state[i].seed(seq);
        double test = rand_01(rng_state[i]); //debugging
        // std::cout << "i value:" << i << "\n";
    }


    //Get unique number of subpopulations to create random number generators for jumping
    // code for if ms_per_subSim is a vector:
    // std::sort(ms_per_subSim.begin(), ms_per_subSim.end());
    // auto last = std::unique(ms_per_subSim.begin(), ms_per_subSim.end());
    // ms_per_subSim.erase(last, ms_per_subSim.end());

    // code for if ms_per_subSim is a pointer:
    std::size_t num_elements = ms_per_subSim_.size();
    std::unordered_set<int> unique_set; // Create an unordered_set to store the unique elements
    for (std::size_t i = 0; i < num_elements; i++) { // Traverse the array and insert the elements into the set
        unique_set.insert(ms_per_subSim[i]); 
    }
    unique_m_vals.assign(unique_set.begin(), unique_set.end()); // Create a vector from the unique elements of the set

    rand_int = new std::uniform_int_distribution<int>(unique_set.size()); //need the same number of random number generators as there are unique m values across subsims (m = number of subpopulations)
    for (int i = 0; i < unique_set.size(); ++i){
        int unique_m = unique_m_vals[i];
        rand_int[i].param(std::uniform_int_distribution<int>::param_type(0, unique_m-2));
    }

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

    // std::mt19937 gen(part_seed ? part_seed : rd());

    // rand_int.param(std::uniform_int_distribution<int>::param_type(0, m-2)); //NEEDS TO BE EDITED FOR MULTISIM TO HAVE DIFFERENT RNGS FOR EACH SUBSIM *** //done
    d_IU_sq = d_IU*d_IU;

    //Resize to number of subsims:
    S_agents.resize(num_of_subSims);
    E_agents.resize(num_of_subSims);
    I_agents.resize(num_of_subSims);
    DR_agents.resize(num_of_subSims);

    //Save agent IDs into vectors based on status
    int subSim;
    for (int i = 0; i < total_pop; ++i){ 
        subSim = agents[i].subSim;
        if(agents[i].status == 0){
            S_agents[subSim].push_back(agents[i].id);
        } else if(agents[i].status == 1){
            E_agents[subSim].push_back(agents[i].id); //**mark
        } else if(agents[i].status == 2){
            I_agents[subSim].push_back(agents[i].id);
        } else if(agents[i].status == 3){
            DR_agents[subSim].push_back(agents[i].id);
        }
    }

    subPops.resize(num_of_subSims);
    for (int i = 0; i < m; i++) {
        int subSim = subSim_IDs[i];
        subPops[subSim].push_back(i);
    }

    S.resize(m);
    E.resize(m);
    I.resize(m);
    DR.resize(m);
    new_I.resize(m);

    //Count and save how many agents have each status
    int current_subSim;
    for (int i = 0; i < m; i++){
        current_subSim = subSim_IDs[i];
        S[i].push_back(count_in_subPop(agents, S_agents[current_subSim], i));
        E[i].push_back(count_in_subPop(agents, E_agents[current_subSim], i));
        I[i].push_back(count_in_subPop(agents, I_agents[current_subSim], i));
        DR[i].push_back(count_in_subPop(agents, DR_agents[current_subSim], i));
        new_I[i].push_back(0);
    }

}

void simulate_one_step(agent* agents, int total_pop, double d_IU, double del_t, vector<double> &jump_probs, vector<double> &mobilities, int m, int step, int nsteps, int num_of_subSims){

    vector<vector<int>> new_I_agents;
    new_I_agents.resize(num_of_subSims);
    // std::cout << "here1 \n " << std::endl;

    for (int subSim = 0; subSim < num_of_subSims; ++subSim){ //this assumes subsim IDs are numbered 0,1,2.... etc ***
        //Update timers:
        for (int i : E_agents[subSim]){
            agents[i].timer = agents[i].timer + del_t;
        }
        for (int i : I_agents[subSim]){
            agents[i].timer = agents[i].timer + del_t;
        }
        // std::cout << "here2 \n " << std::endl;

        //Update statuses based on time:
        //Check for infected agents that should become dead/recovered:
        iter = I_agents[subSim].begin();
        while (iter != I_agents[subSim].end()) {
            int i = *iter;
            if (agents[i].timer >= agents[i].I_time){
                agents[i].status = 3;
                iter = I_agents[subSim].erase(iter);
                DR_agents[subSim].push_back(agents[i].id);
                agents[i].timer = 0;
            }
            else {
                ++iter;
            }
        }
        // std::cout << "here3 \n " << std::endl;
        //Check for exposed agents that should become infected:
        iter = E_agents[subSim].begin();
        while (iter != E_agents[subSim].end()) {
            int i = *iter;
            if (agents[i].timer >= agents[i].E_time){
                agents[i].status = 2;
                iter = E_agents[subSim].erase(iter); //**mark
                I_agents[subSim].push_back(agents[i].id);
                agents[i].timer = 0;
                new_I_agents[subSim].push_back(agents[i].id);
            }
            else {
                ++iter;
            }
        }
        // std::cout << "here4 " << std::endl;

        //Check for new exposures:
        for (int i: I_agents[subSim]){ //for each infected agent in the subsim,
            for (int j : S_agents[subSim]){ //check if in range of all susceptible agents in the subsim
                check_infection(agents[i], agents[j], d_IU);
            }
        }
    }
        // std::cout << "total_pop " <<total_pop<< std::endl;

    //Move agents:
    for (int i = 0; i < total_pop; ++i){
        move(agents[i], jump_probs, mobilities, del_t);
    }
    // std::cout << "here6 " << std::endl;

        //Save data:
        int current_subSim;
        for (int i = 0; i < m; i++){
            current_subSim = subSim_IDs[i];

            // // track data infections based on current subpopulation
            // S[i].push_back(count_in_subPop(agents, S_agents[current_subSim], i));
            // E[i].push_back(count_in_subPop(agents, E_agents[current_subSim], i));
            // I[i].push_back(count_in_subPop(agents, I_agents[current_subSim], i));
            // DR[i].push_back(count_in_subPop(agents, DR_agents[current_subSim], i));
            // new_I[i].push_back(count_in_subPop(agents, new_I_agents[current_subSim], i));

            //track data based on agent's original subpopulation
            // S[i].push_back(count_from_original_subPop(agents, S_agents[current_subSim], i));
            // E[i].push_back(count_from_original_subPop(agents, E_agents[current_subSim], i)); //causing issue
            // I[i].push_back(count_from_original_subPop(agents, I_agents[current_subSim], i));
            // DR[i].push_back(count_from_original_subPop(agents, DR_agents[current_subSim], i));
            new_I[i].push_back(count_from_original_subPop(agents, new_I_agents[current_subSim], i)); 
        }
    // std::cout << "here7 " << std::endl;

    // At last time step, save data to file:
    if (step == nsteps-1){

        std::cout<<"SAVING DATA TO FILE\n";

        save_matrix_data(savename2, new_I, nsteps, m);
        // save_matrix_data(savename3, I, nsteps, m);
        // save_matrix_data(savename4, S, nsteps, m);
        // save_matrix_data(savename5, E, nsteps, m);
        // save_matrix_data(savename6, DR, nsteps, m);

        // delete rand_int;
        // delete rng_state;
        free(rand_int); free(rng_state);
    }
}