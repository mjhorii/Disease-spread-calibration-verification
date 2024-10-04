#include "common_gpu.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>
#include <cuda.h>
#include <numeric>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

using namespace std;

#define NUM_THREADS 256

std::random_device rd;
std::uniform_real_distribution<double> rand_11(-1,1);   //uniform distribution between -1,1
std::uniform_real_distribution<double> rand_01(0,1);    //uniform distribution between -1,1
std::vector<vector<int>> S_agents;                      //vector with entries for each subsim (entry: vector of ids of susceptible agents)
std::vector<vector<int>> E_agents;                      //vector with entries for each subsim (entry: vector of ids of exposed agents)
std::vector<vector<int>> I_agents;                      //vector with entries for each subsim (entry: vector of ids of infected agents)
std::vector<vector<int>> DR_agents;                     //vector with entries for each subsim (entry: vector of ids of dead/recovered agents)
vector<double>* domain_limits_x;                        //store domain limits for subpopulations (x-direction) (initialize as pointer so data doesn't need to be copied)
vector<double>* domain_limits_y;                        //store domain limits for subpopulations (y-direction)
// std::vector<vector<int>> subPops;                       //vector with entries for each subsim (entry: vector of subpop ids (e.g., if there are 3 subpops in subsim 0, subPops[0] = [0,1,2]))

std::vector<int>::iterator iter; 
string savename2;                                       //filenames to save data
string savename3;
string savename4;
string savename5;
string savename6;

int* ms_per_subSim;                                     // pointer to start of array of number of subpops for each subsim
std::vector<int> unique_m_vals;                         // vector of unique m values across subsims (m = number of subpopulations)
int* subSim_IDs; //could/should this be done with references instead?

int* ms_per_subSim_d;                                   // device pointer to start of array of number of subpops for each subsim
int* unique_m_vals_d;                           
int* subSim_IDs_d;                                      //could/should this be done with references instead?

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
int *S_d;
int *E_d;
int *I_d;
int *DR_d;
int *new_I_d;

int* total_pop_d;           // device pointer to total population value
double* d_IU_sq_d;          // device pointer to radius of infection squared
double* del_t_d;            // device pointer to del_t (time step size)
int* m_d;                   // device pointer to number of total subPops
int* num_of_subSims_d;      // device pointer to number of subSims
double* jump_probs_d;       // device address of start of jump_probs array
double* mobilities_d;       // device address of start of mobilities array
double* domain_limits_x_d; //device pointer to array of arrays to store domain limits for subpopulations (x-direction) 
double* domain_limits_y_d; //store domain limits for subpopulations (y-direction)
int* cum_pop_per_subSim_d;  // device pointer to start of total_pop_per_subSim array
int* choices_d;             // device pointer for choices vector, which has size: maximum(number of subpops in any subsim). meant for holding random choice possibilities in move function
int blks;                   // num of thread blocks needed
int *subPops_d;            // (device) vector with entries for each subsim (entry: vector of subpop ids (e.g., if there are 3 subpops in subsim 0, subPops[0] = [0,1,2]))
int *cum_ms_per_subsim_d;     // (device) vector with entries for each subsim (entry: vector of subpop ids (e.g., if there are 3 subpops in subsim 0, subPops[0] = [0,1,2]))
curandState* rng_state_d;   //array of random number generators for each agent
int* time_step_d;           //current time step saved on device
int* max_m_d;                 // maximum number of subPops (m) in any subsimulation

void save_matrix_data(string save_name, std::vector<vector<int>> &matrix, int nsteps, int m){
    //save data from matrix to file with name save_name. nsteps and m (number of subpops) define dimensions of matrix
    std::ofstream fsave(save_name);
    for (int i = 0; i < nsteps+1; ++i) {
        for (int j = 0; j<m-1;++j){
            fsave << matrix[i][j] << " ";
        }
        fsave << matrix[i][m-1];
        fsave << "\n";
    }
}

__global__ void count_from_original_subPop_gpu(agent* agents, int* time_step, int* total_pop_d, int *S_d, int *E_d, int *I_d, int *DR_d, int* m_d) {
    // Get thread (agent) ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= *total_pop_d)
        return;

    int original_subPop = agents[id].original_subPop;

    if (agents[id].status==0){
        atomicAdd(&S_d[(*m_d)*(*time_step)+original_subPop], 1);
        return;
    }
    if (agents[id].status==1){
        atomicAdd(&E_d[(*m_d)*(*time_step)+original_subPop], 1);
        return;
    }
    if (agents[id].status==2){
        atomicAdd(&I_d[(*m_d)*(*time_step)+original_subPop], 1);
        return;
    }
    if (agents[id].status==3){
        atomicAdd(&DR_d[(*m_d)*(*time_step)+original_subPop], 1);
        return;
    }
    //^ there's probably some optimal ordering of these based on what states are most common
}

__global__ void setup_rng_state(curandState *rng_state_d, agent* agents_d, int *seeds_per_subSim_d_ptr, int *total_pop_d){

    int id = threadIdx.x+blockDim.x*blockIdx.x;
    if (id >= *total_pop_d)
        return;

    // printf("%d    %d\n", idx, seeds[idx]);
    curand_init(seeds_per_subSim_d_ptr[agents_d[id].subSim], agents_d[id].id_in_subSim, 0, &rng_state_d[id]);
}

void init_simulation(agent* agents_d, int total_pop, double d_IU, int m, vector<vector<double>> &domain_limits_x_, 
        vector<vector<double>> &domain_limits_y_, vector<int> &seeds_per_subSim, int num_of_subSims, 
        vector<int> &subSim_IDs, vector<int> &ms_per_subSim_, double del_t, vector<double> &jump_probs,
        vector<double> &mobilities, vector<int> &total_pop_per_subSim, int nsteps, char* savename){

    // void init_simulation(agent* agents, int total_pop, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    blks = (total_pop + NUM_THREADS - 1) / NUM_THREADS; //number of thread blocks (round up total_pop/NUM_THREADS)

    //------------------ Moving simulation parameters into GPU memory --------------------
    // Format:
    // cudaMalloc((void**)&ptr, SIZE * sizeof(int))
    // cudaMemcpy(dev_ptr, ptr, SIZE * sizeof(int), cudaMemcpyHostToDevice)

    cudaMalloc(&total_pop_d, sizeof(int));
    cudaMemcpy(total_pop_d, &total_pop, sizeof(int), cudaMemcpyHostToDevice);

    double d_IU_sq = d_IU*d_IU;
    cudaMalloc(&d_IU_sq_d, sizeof(double));
    cudaMemcpy(d_IU_sq_d, &d_IU_sq, sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&del_t_d, sizeof(double));
    cudaMemcpy(del_t_d, &del_t, sizeof(double), cudaMemcpyHostToDevice);  

    cudaMalloc(&m_d, sizeof(int));
    cudaMemcpy(m_d, &m, sizeof(int), cudaMemcpyHostToDevice);  

    cudaMalloc(&num_of_subSims_d, sizeof(int));
    cudaMemcpy(num_of_subSims_d, &num_of_subSims, sizeof(int), cudaMemcpyHostToDevice); 

    cudaMalloc(&jump_probs_d, jump_probs.size()*sizeof(double));
    cudaMemcpy(jump_probs_d, jump_probs.data(), jump_probs.size()*sizeof(double), cudaMemcpyHostToDevice); 

    cudaMalloc(&mobilities_d, mobilities.size()*sizeof(double));
    cudaMemcpy(mobilities_d, mobilities.data(), mobilities.size()*sizeof(double), cudaMemcpyHostToDevice); 

    cudaMalloc(&subSim_IDs_d, subSim_IDs.size()*sizeof(int));
    cudaMemcpy(subSim_IDs_d, subSim_IDs.data(), subSim_IDs.size()*sizeof(int), cudaMemcpyHostToDevice); 

    cudaMalloc(&ms_per_subSim_d, ms_per_subSim_.size()*sizeof(int));
    cudaMemcpy(ms_per_subSim_d, ms_per_subSim_.data(), ms_per_subSim_.size()*sizeof(int), cudaMemcpyHostToDevice);

    auto max_element = std::max_element(ms_per_subSim_.begin(), ms_per_subSim_.end());
    // cudaMalloc(&choices_d, *max_element*sizeof(int));
    cudaMalloc(&choices_d, *max_element*total_pop*sizeof(int)); //this could be more memory efficient, doesn't need to have max_element for every single agent, but would make later indexing more annoying

    int max_val = *max_element;
    cudaMalloc(&max_m_d, sizeof(int));
    cudaMemcpy(max_m_d, &max_val, sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> cumsum(total_pop_per_subSim.size());
    std::partial_sum(total_pop_per_subSim.begin(), total_pop_per_subSim.end(), cumsum.begin());
    cumsum.insert(cumsum.begin(), 0);
    
    cudaMalloc(&cum_pop_per_subSim_d, cumsum.size()*sizeof(int));
    cudaMemcpy(cum_pop_per_subSim_d, cumsum.data(), cumsum.size()*sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> cumsum_ms(ms_per_subSim_.size());
    std::partial_sum(ms_per_subSim_.begin(), ms_per_subSim_.end(), cumsum_ms.begin());
    cumsum_ms.insert(cumsum_ms.begin(), 0);

    cudaMalloc(&cum_ms_per_subsim_d, cumsum_ms.size()*sizeof(int));
    cudaMemcpy(cum_ms_per_subsim_d, cumsum_ms.data(), cumsum_ms.size()*sizeof(int), cudaMemcpyHostToDevice);

    int num_rows = domain_limits_x_.size();
    int num_cols = 2;
    cudaMalloc(&domain_limits_x_d, num_rows*num_cols*sizeof(double));
    for (int i = 0; i < num_rows; ++i){
        cudaMemcpy(&domain_limits_x_d[i*num_cols], domain_limits_x_[i].data(), num_cols * sizeof(double), cudaMemcpyHostToDevice);
    }

    num_rows = domain_limits_y_.size();
    num_cols = 2;
    cudaMalloc(&domain_limits_y_d, num_rows*num_cols*sizeof(double));
    for (int i = 0; i < num_rows; ++i){
        cudaMemcpy(&domain_limits_y_d[i*num_cols], domain_limits_y_[i].data(), num_cols * sizeof(double), cudaMemcpyHostToDevice);
    } 

    num_rows = (nsteps+1);
    num_cols = m;
    cudaMalloc(&S_d, num_rows*num_cols*sizeof(int));
    cudaMemset(S_d, 0, num_rows*num_cols*sizeof(int));

    cudaMalloc(&E_d, num_rows*num_cols*sizeof(int));
    cudaMemset(E_d, 0, num_rows*num_cols*sizeof(int));

    cudaMalloc(&I_d, num_rows*num_cols*sizeof(int));
    cudaMemset(I_d, 0, num_rows*num_cols*sizeof(int));

    cudaMalloc(&DR_d, num_rows*num_cols*sizeof(int));
    cudaMemset(DR_d, 0, num_rows*num_cols*sizeof(int));

    cudaMalloc(&new_I_d, num_rows*num_cols*sizeof(int));
    cudaMemset(new_I_d, 0, num_rows*num_cols*sizeof(int));

    //------------------ Initialize random number generators for each agent --------------------

    thrust::device_vector<int> seeds_per_subSim_d = seeds_per_subSim; //use thrust because too lazy to do memory allocation
    int* seeds_per_subSim_d_ptr = thrust::raw_pointer_cast(seeds_per_subSim_d.data()); //pass this to setup_rng_state

    cudaMalloc(&rng_state_d, total_pop * sizeof(curandState));
    setup_rng_state<<<blks, NUM_THREADS>>>(rng_state_d, agents_d, seeds_per_subSim_d_ptr, total_pop_d);

    //-------- Get unique number of subpopulations to create random number generators for jumping --------
    ms_per_subSim = ms_per_subSim_.data(); 

    // code for if ms_per_subSim is a vector:
    // std::sort(ms_per_subSim.begin(), ms_per_subSim.end());
    // auto last = std::unique(ms_per_subSim.begin(), ms_per_subSim.end());
    // ms_per_subSim.erase(last, ms_per_subSim.end());

    // code for if ms_per_subSim is a pointer: //(easiest to have it be a pointer if you want ms_per_subSim to be a global variable on CPU -- this won't be an issue for GPU code so could change to other version)
    std::size_t num_elements = ms_per_subSim_.size();
    std::unordered_set<int> unique_set; // Create an unordered_set to store the unique elements
    for (std::size_t i = 0; i < num_elements; i++) { // Traverse the array and insert the elements into the set
        unique_set.insert(ms_per_subSim[i]); 
    }
    unique_m_vals.assign(unique_set.begin(), unique_set.end()); // Create a vector from the unique elements of the set

    cudaMalloc(&unique_m_vals_d, unique_m_vals.size()*sizeof(int));
    cudaMemcpy(unique_m_vals_d, unique_m_vals.data(), unique_m_vals.size()*sizeof(int), cudaMemcpyHostToDevice); 

    //filenames:
    if (savename == nullptr){ //**change this later because it's ugly
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

    S.resize(nsteps+1);
    E.resize(nsteps+1);
    I.resize(nsteps+1);
    DR.resize(nsteps+1);
    new_I.resize(nsteps+1);

    for (int i = 0; i < nsteps+1; ++i){
        S[i].resize(m);
        E[i].resize(m);
        I[i].resize(m);
        DR[i].resize(m);
        new_I[i].resize(m);
    }

    //Count and save how many agents have each status
    cudaMalloc(&time_step_d, sizeof(int));
    cudaMemset(time_step_d, 0, sizeof(int));
    count_from_original_subPop_gpu<<<blks, NUM_THREADS>>>(agents_d, time_step_d, total_pop_d, S_d, E_d, I_d, DR_d, m_d); //deals with S, E, I, DR vectors
    cudaDeviceSynchronize();
    //new_I had already been memset to 0 everywhere, can just leave as that for 0th time step

}

__global__ void update_timers(agent* agents_d, double* del_t_d, int* total_pop_d){

    // printf("about to update timers \n");
    // Get thread (agent) ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= *total_pop_d)
        return;

    if (agents_d[id].status == 1 || agents_d[id].status == 2){ //if agent is exposed (1) or infected (2)
        agents_d[id].timer = agents_d[id].timer + *del_t_d; //update timer
    }

    // printf("updated timers \n");
    // if (id == 500){
    // printf("updated timers \n");
    // }

}

__global__ void update_status_based_on_time(agent* agents_d, int* total_pop_d, int time_step, int *new_I_d, int* m_d){

    // printf("about to update status \n");

    // Get thread (agent) ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= *total_pop_d)
        return;

    //Check for infected agents that should become dead/recovered:
    if (agents_d[id].status == 2){ //if agent is infected
        if (agents_d[id].timer >= agents_d[id].I_time){ //and its timer has gone past its infection time period
            agents_d[id].status = 3; //set status to dead/recovered
            agents_d[id].timer = 0; //set timer back to 0 (not technically necessary now, but if there are repeated infections in future would need this)
            return;
        }
    }

    //Check for exposed agents that should become infected:
    if (agents_d[id].status == 1){ //if agent is exposed
        if (agents_d[id].timer >= agents_d[id].E_time){ //and its timer has gone past its exposure time period
            agents_d[id].status = 2; //set status to infected
            agents_d[id].timer = 0; //set timer back to 0
            atomicAdd(&new_I_d[*m_d*time_step+agents_d[id].original_subPop], 1); //add one tally to new_I_d
            // printf("time step: %d \n", time_step);
            // printf("timer: %f \n", agents_d[id].E_time);
            return;
        }
    }
    // if (id == 1){
    // printf("updated statuses \n");
    // }
}

__global__ void check_for_new_exposures(agent* agents_d, int* total_pop_d, int* cum_pop_per_subSim_d, double* d_IU_sq_d){
    // Get thread (agent) ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= *total_pop_d)
        return;

    if (agents_d[id].status != 2){ //if agent is NOT infected
        return;
    }

    //otherwise if agent IS infected:
    agent infected_agent = agents_d[id];
    int subSim = infected_agent.subSim;

    for (int neighbor_id = cum_pop_per_subSim_d[subSim]; neighbor_id < cum_pop_per_subSim_d[subSim+1]; ++neighbor_id){ //loop over all other agents in the same subSimulation
        if (agents_d[neighbor_id].status == 0){ // if the neighbor is susceptible
            // Calculate distance between agents
            double dx = agents_d[neighbor_id].x - infected_agent.x;
            double dy = agents_d[neighbor_id].y - infected_agent.y;
            double r2 = dx * dx + dy * dy;

            if (r2 < *d_IU_sq_d){ //if neighbor is within infection radius
                // agents_d[neighbor_id].status = 1; //set status to exposed
                // agents_d[neighbor_id].timer = 0; //set timer to 0
                atomicCAS(&agents_d[neighbor_id].status, 0, 1); //atomic compare and swap: if neighbor status is equal to 0, store 1 in neighbor status instead. always returns previous value
                // atomicExch(&agents_d[neighbor_id].timer, 0); //atomic Exch: replace current timer val with 0 (atomicCAS not available for non-integer values)
                //** technically not necessary to set timer to 0 since agents cannot become reinfected at the moment
                //would need to change timer type to float to use atomicExch
                //** ^ its fine if multiple infected agents try to run these on a susceptible agent in one time step
                //since its synchronized at the end of a time step, you won't ever be resetting an exposed agent's timer or something
                //since they will always have started the time step susceptible 
            }
        }
    }
    // if (id == 1){
    // printf("chcked for new exposures \n");
    // }
}

__global__ void move_gpu(agent* agents_d, int* choices_d, int* total_pop_d, int* ms_per_subSim_d, curandState* rng_state_d, double* mobilities_d, double* jump_probs_d, double* del_t_d, double* domain_limits_x_d, double* domain_limits_y_d, int *cum_ms_per_subsim_d, int* max_m_d){
    // Get thread (agent) ID
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= *total_pop_d)
        return;

    agent &agent_d = agents_d[id];

    //sub-population jumping with probability jump_prob
    int subPop = agent_d.subPop;
    int subSim = agent_d.subSim;
    int subSim_m = ms_per_subSim_d[subSim]; //number of subpopulations in each subsim
    double jump_prob = jump_probs_d[agent_d.subPop];
    double mobility = mobilities_d[agent_d.subPop];

    // // std::cout << "move2 \n" << std::endl;

    if (curand_uniform(rng_state_d+id)<jump_prob){
        //determine which subpop the agent will move to:
        //update choices with possible subpops the agent can move to:
        int num = 0; //keep track of number of subpops added to the choices_d array
        for (int i = cum_ms_per_subsim_d[subSim]; i < cum_ms_per_subsim_d[subSim+1]; ++i){ //for each subpop in subsim,
            if (i != subPop){ //if the subpop is NOT the current subpop
                choices_d[*max_m_d*id + num] = i; // add it to the choices_d array
                num += 1; //
            }
        }

        //randomly pick a valid index in choices:
        int max_choices_index = subSim_m - 2; //subtract 1 because jumping to itself isn't an option, subtract another to adjust for 0-indexing
        int min_choices_index = 0;
        float randf = curand_uniform(rng_state_d+id);
        randf *= (max_choices_index - min_choices_index + 0.999999);
        // randf += min_choices_index;
        int ind = (int)truncf(randf);

        int new_subPop = choices_d[*max_m_d*id + ind];

        // if(new_subPop >= cum_ms_per_subsim_d[subSim+1] || new_subPop < cum_ms_per_subsim_d[subSim]){
        //         printf("problem, new_subPop is %d \n", new_subPop);
        // }

        agent_d.subPop = new_subPop; //set agent's new subpop

        // //assign the agent a position within that subpop:
        double rand_num = curand_uniform(rng_state_d+id);
        agent_d.x = rand_num*(domain_limits_x_d[2*agent_d.subPop+1]-domain_limits_x_d[2*agent_d.subPop+0])+domain_limits_x_d[2*agent_d.subPop+0]; //domain_limits_x[subpop][neg or pos limit] = domain_limits_x_d[2*subpop+[neg or pos limit]]
        rand_num = curand_uniform(rng_state_d+id);
        agent_d.y = rand_num*(domain_limits_y_d[2*agent_d.subPop+1]-domain_limits_y_d[2*agent_d.subPop+0])+domain_limits_y_d[2*agent_d.subPop+0];
        // // std::cout << "move5 \n" << std::endl;
    }

    // std::cout << "move6 \n" << std::endl;

    // Move:
    double rand_11 = curand_uniform(rng_state_d+id)*2-1; //random number between -1 and 1
    // printf("rand val: %f \n", rand_11);
    agent_d.x = agent_d.x + mobility*(*del_t_d)*rand_11;
    // agents_d[id].x = agents_d[id].x + mobility*(*del_t_d)*rand_11; //**check -- is agent_d properly changing the agent in the agents_d array? or was i just not running in a gpu node?  -- agent_d is fine the problem was that the counting wasn't working
    rand_11 = curand_uniform(rng_state_d+id)*2-1; //random number between -1 and 1
    agent_d.y = agent_d.y + mobility*(*del_t_d)*rand_11;

    // std::cout << "move7 \n" << std::endl;

    // Bounce from walls
    double x_low = domain_limits_x_d[2*agent_d.subPop+0];
    double x_high = domain_limits_x_d[2*agent_d.subPop+1];
    double y_low = domain_limits_y_d[2*agent_d.subPop+0];
    double y_high = domain_limits_y_d[2*agent_d.subPop+1];

    // std::cout << "move8 \n" << std::endl;

    while (agent_d.x < x_low || agent_d.x > x_high) {
        agent_d.x = agent_d.x < x_low ? 2 * x_low - agent_d.x : 2 * x_high - agent_d.x;
    }

    while (agent_d.y < y_low || agent_d.y > y_high) {
        agent_d.y = agent_d.y < y_low ? 2 * y_low - agent_d.y : 2 * y_high - agent_d.y;
    }

    // printf("moved agents\n");
}

void simulate_one_step(agent* agents_d, int total_pop, double d_IU, double del_t, vector<double> &jump_probs, vector<double> &mobilities, int m, int step, int nsteps, int num_of_subSims){
    // std::cout << "here1 \n " << std::endl;

    update_timers<<<blks, NUM_THREADS>>>(agents_d, del_t_d, total_pop_d); //Update timers
    cudaDeviceSynchronize();
    update_status_based_on_time<<<blks, NUM_THREADS>>>(agents_d, total_pop_d, step+1, new_I_d, m_d); //Update statuses based on time (and count number of new infections for new_I vector) //step+1 is so that we don't write over time step 0
    cudaDeviceSynchronize();
    check_for_new_exposures<<<blks, NUM_THREADS>>>(agents_d, total_pop_d, cum_pop_per_subSim_d, d_IU_sq_d); //Check for new exposures
    cudaDeviceSynchronize();

    // //Move agents:
    move_gpu<<<blks, NUM_THREADS>>>(agents_d, choices_d, total_pop_d, ms_per_subSim_d, rng_state_d, mobilities_d, jump_probs_d, del_t_d, domain_limits_x_d, domain_limits_y_d, cum_ms_per_subsim_d, max_m_d);
    cudaDeviceSynchronize();
    // // std::cout << "here6 " << std::endl;
    int time_step = step+1;
    cudaMemcpy(time_step_d, &time_step, sizeof(int), cudaMemcpyHostToDevice); //need to use Memcpy, Memset doesn't seem to work for this (e.g., cudaMemset(time_step_d, step+1, sizeof(int)); doesn't work)
    cudaDeviceSynchronize();
    count_from_original_subPop_gpu<<<blks, NUM_THREADS>>>(agents_d, time_step_d, total_pop_d, S_d, E_d, I_d, DR_d, m_d); //Save data (deals with S, E, I, DR vectors)
    //** ^ currently only have option to count based on original supopulations, not current subpop
    cudaDeviceSynchronize();
    // std::cout << "here7 " << std::endl;

    // At last time step, save data to file:
    if (step == nsteps-1){
        //Transfer data from GPU memory to CPU:
        for (int i = 0; i < nsteps+1; ++i){
            // std::cout<<i<<"\n";
            cudaMemcpy(S[i].data(), S_d+i*m, m*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(E[i].data(), E_d+i*m, m*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(I[i].data(), I_d+i*m, m*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(DR[i].data(), DR_d+i*m, m*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(new_I[i].data(), new_I_d+i*m, m*sizeof(int), cudaMemcpyDeviceToHost);
        }

        // std::cout<<new_I[0][0];

        std::cout<<"SAVING DATA TO FILE\n";

        save_matrix_data(savename2, new_I, nsteps, m);
        save_matrix_data(savename3, I, nsteps, m);
        save_matrix_data(savename4, S, nsteps, m);
        save_matrix_data(savename5, E, nsteps, m);
        save_matrix_data(savename6, DR, nsteps, m);

        std::cout<<"done\n";
    }
}