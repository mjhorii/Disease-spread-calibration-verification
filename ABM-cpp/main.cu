#include "common_gpu.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, agent* agents, int total_pop, double d_IU, double x_min, double x_max, double y_min, double y_max) {
    static bool first = true;

    if (first) {
        fsave << total_pop << " " << d_IU << " " << x_min << " " << x_max << " " << y_min <<" " << y_max<<"\n";
        first = false;
    }

    for (int i = 0; i < total_pop; ++i) {
        fsave << agents[i].x << " " << agents[i].y << " " << agents[i].status <<"\n";
        // cout<<"agent x pos: "<<agents[i].x<<"\n";
        // cout<<"agent y pos: "<<agents[i].y<<"\n";
    }

    fsave << std::endl;
}

template<typename T>
void prettyPrint(const std::vector<std::vector<T>>& v) {
    for (const auto& row : v) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void printVector(const std::vector<T>& v) {
    std::cout << "[";
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it;
        if (it != v.end() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// Particle Initialization
void init_agents(agent* agents, double T_E, double T_E_stdev, double T_I, double T_I_stdev, int m, vector<int> pop_sizes, 
                    vector<vector<double>> status_fractions, vector<vector<double>> domain_limits_x, 
                    vector<vector<double>> domain_limits_y, string dist, int total_pop, vector<int> seeds_per_subSim, 
                    int num_of_subSims, vector<int> total_pop_per_subSim, vector<int> ms_per_subSim) {

    double rand_num;
    std::mt19937 gen;
    int total_pop_so_far = 0;
    int total_subPops_so_far = 0;
    for (int subSim = 0; subSim < num_of_subSims; ++subSim){
        gen.seed(seeds_per_subSim[subSim]);
        std::uniform_real_distribution<double> rand(0.0, 1.0); //uniform distribution between 0,1

        //Generate exposure times and infection times for each agent in this subsim
        if (dist == "Normal"){
            std::normal_distribution<double> dist_E(T_E, T_E_stdev); //normal distribution of exposure times  
            std::normal_distribution<double> dist_I(T_I, T_I_stdev); //normal distribution of infection times
            for (int i = total_pop_so_far; i < total_pop_so_far + total_pop_per_subSim[subSim]; ++i){ 
                agents[i].E_time = dist_E(gen); 
                agents[i].I_time = dist_I(gen);
            }
        }
        // cout<<"here2";
        if (dist == "Gamma"){
            double k = (T_E*T_E)/(T_E_stdev*T_E_stdev);
            double theta = (T_E_stdev*T_E_stdev)/T_E;
            std::gamma_distribution<double> dist_E(k, theta); //gamma distribution of exposure times   

            k = (T_I*T_I)/(T_I_stdev*T_I_stdev);
            theta = (T_I_stdev*T_I_stdev)/T_I;
            std::gamma_distribution<double> dist_I(k, theta); //gamma distribution of infection times

            for (int i = total_pop_so_far; i < total_pop_so_far + total_pop_per_subSim[subSim]; ++i){ 
                agents[i].E_time = dist_E(gen);
                agents[i].I_time = dist_I(gen);
            }
        }
        // cout<<"here4\n";

        int num = total_pop_so_far; //keep track of agent number while looping over subpopulations

        for (int subPop = total_subPops_so_far; subPop < total_subPops_so_far + ms_per_subSim[subSim]; ++subPop){ //loop over subpopulations within subsim
            // Calculate how many agents are S, E, I, and DR:
            int subPopSize = pop_sizes[subPop];
            // cout<<"subPopSize: "<<subPopSize<<"\n";
            int numS = round(subPopSize*status_fractions[subPop][0]);
            int numE = round(subPopSize*status_fractions[subPop][1]);
            int numI = round(subPopSize*status_fractions[subPop][2]);
            int numR = round(subPopSize*status_fractions[subPop][3]);

            // cout<<"numS: "<<numS<<"\n";
            // cout<<"numE: "<<numE<<"\n";
            // cout<<"numI: "<<numI<<"\n";
            // cout<<"numDR: "<<numR<<"\n";

            for (int i = 0; i < subPopSize; ++i){ //loop over agents in the subpop
                //Assign initial positions
                rand_num = rand(gen);
                agents[num].x = rand_num*(domain_limits_x[subPop][1]-domain_limits_x[subPop][0])+domain_limits_x[subPop][0];
                rand_num = rand(gen);
                agents[num].y = rand_num*(domain_limits_y[subPop][1]-domain_limits_y[subPop][0])+domain_limits_y[subPop][0];

                //Initialize all exposure and infection times as 0:
                agents[num].timer = 0;

                //Initialize disease status:
                if (i < numS){
                    agents[num].status = 0;
                } else if (i < numS+numE){
                    agents[num].status = 1;
                } else if (i < numS+numE+numI){
                    agents[num].status = 2;
                } else{
                    agents[num].status = 3;
                }

                //Initialize subpopulation IDs:
                agents[num].subPop = subPop;
                agents[num].original_subPop = subPop;
                agents[num].subSim = subSim;
                agents[num].id_in_subSim = num - total_pop_so_far;

                ++num;
            }
        }

        total_pop_so_far += total_pop_per_subSim[subSim];
        total_subPops_so_far += ms_per_subSim[subSim];
    }

    for (int i = 0; i < total_pop; ++i) {
            agents[i].id = i;
        }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

double find_double_arg(int argc, char** argv, const char* option, double default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stod(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

void per_subPop_to_per_subSim(vector<int> &per_subPop, vector<int> &per_subSim, vector<int> &ms) {
    //turn a vector of properties per subpopulation (per_subPop) into a vector of properties per subsimulation (per_subSim)
    //only applicable to properties that are common across all subpopulations within a subsimulation (e.g., number of subpops, seed)
    //ms is vector of (number of subpops in subsim) for each subpop
    int num_subPops = per_subPop.size();
    for (int num = 0; num < num_subPops; num +=ms[num]){
        per_subSim.push_back(per_subPop[num]);
    }
}

void sum_subPop_per_subSim(vector<int> &per_subPop, vector<int> &per_subSim, vector<int> &ms) {
    //turn a vector of properties per subpopulation (per_subPop) into a vector of properties per subsimulation (per_subSim)
    //only applicable to properties that are common across all subpopulations within a subsimulation (e.g., number of subpops, seed)
    //ms is vector of (number of subpops in subsim) for each subpop
    int num_subPops = per_subPop.size();
    int total;
    for (int num = 0; num < num_subPops; num +=ms[num]){
        total = 0;
        for (int j = num; j < num + ms[num]; ++j){
            total += per_subPop[j];
        }
        per_subSim.push_back(total);
    }
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        // std::cout << "-s <int>: set particle initialization seed" << std::endl;
        std::cout << "-file <string>: set file path to population info csv" << std::endl;
        std::cout << "-T_E <double>: mean exposure time" << std::endl;
        std::cout << "-T_E_stdev <double>: standard deviation of exposure time" << std::endl;
        std::cout << "-T_I <double>: mean infection time" << std::endl;
        std::cout << "-T_I_stdev <double>: standard deviation of infection time" << std::endl;
        std::cout << "-d_IU <double>: infection radius" << std::endl;
        std::cout << "-dist <string>: what type of distribution (Normal or Gamma)" << std::endl;
        std::cout << "-T <double>: total simulation time" << std::endl;
        std::cout << "-del_t <double>: time step size" << std::endl;
        std::cout << "-save_data_for_rendering <string>: True or False" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Initialize Particles
    int part_seed = find_int_arg(argc, argv, "-s", 0); 
    string dist = find_string_option(argc, argv, "-dist", "Gamma");
    string save_rendering_data = find_string_option(argc, argv, "-save_data_for_rendering", "True");
    double T_E = find_double_arg(argc, argv, "-T_E", 11.6);
    double T_E_stdev = find_double_arg(argc, argv, "-T_E_stdev", 1.9);
    double T_I = find_double_arg(argc, argv, "-T_I", 18.49);
    double T_I_stdev = find_double_arg(argc, argv, "-T_I_stdev", 3.71);
    double d_IU = find_double_arg(argc, argv, "-d_IU", 0.005);
    double mobility = find_double_arg(argc, argv, "-mobility", 0.0005);
    double jump_prob = find_double_arg(argc, argv, "-jump_prob", 0);
    double del_t = find_double_arg(argc, argv, "-del_t", 0.1);
    double T = find_double_arg(argc, argv, "-T", 300);

    int nsteps = round(T/del_t);

    // string fname;
    char* fname = find_string_option(argc, argv, "-file", "");
    if (std::strcmp(fname, "") == 0){
        cout<<"No file given\n\n";
    }
 
    //Read population data from external csv:
	vector<vector<string>> content;
	vector<string> row;
	string line, word;
 
	fstream file (fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			stringstream str(line);
 
			while(getline(str, word, ','))
				row.push_back(word);
			content.push_back(row);
		}
	}
	else
		cout<<"Could not open the population data file\n";
 
    // Parse data from population data csv:
    int total_pop = 0;                          //total agents in all subpopulations
    vector<int> pop_sizes;                      //store population sizes for each subpopulation
    vector<vector<double>> status_fractions;    //store info on fractions of S, E, I, DR agents in subpopulation
    vector<vector<double>> domain_limits_x;     //store domain limits for subpopulations (x-direction)
    vector<vector<double>> domain_limits_y;     //store domain limits for subpopulations (y-direction)
    int first_x = 1;                            //first_x and first_y are for calculating x_min, x_max, etc.
    int first_y = 1;
    double x_min;                               //x_min, x_max, y_min, y_max are overall bounds used for plotting
    double x_max;
    double y_min;
    double y_max;
    vector<int> subSim_IDs;                     //which subsim is it
    vector<int> ms;                             //number of subpopulations for the subsim represented in that row
    // vector<vector<double>> jumping_probs;    //jumping probability matrix -- for each subpop, vector of [prob of jumping to subpop 0, prob of jumping to subpop 1, ... etc]
    vector<double> jump_probs;               //jumping probability values for the subsim represented in that row
    vector<double> mobilities;                  //mobility parameter values for each subpop
    vector<int> seeds;                          //seeds for each sub-sim
    int num_of_subSims;

	for(int i=0;i<content.size();i++)
	{
        status_fractions.resize(i+1);
        domain_limits_x.resize(i+1);
        domain_limits_y.resize(i+1);
        // jump_probs.resize(i+1); //for MATRIX jumping probabilities
        // cout<<"line "<<i<<"\n";
        // cout<<content[i].size()<<" \n";
		for(int j=0;j<content[i].size();j++)
		{
            if (i>0){ //skip header row
                // cout<<content[i][j]<<" ";
                // cout<<"ij"<<i<<" "<<j<<" ";
                if(j==0) //If reading the first column of csv (Total population size information)
                {
                    total_pop = total_pop + std::stoi(content[i][j]);
                    pop_sizes.push_back(std::stoi(content[i][j]));
                }
                if(j==1) //If reading the 2nd column of csv
                {
                    status_fractions[i-1].resize(4);
                    domain_limits_x[i-1].resize(2);
                    domain_limits_y[i-1].resize(2);
                }
                if(j>=1 && j<=4) //If reading the 2nd column of csv (information about what percent of each population is S,E,I,R)
                {
                    status_fractions[i-1][j-1]=std::stod(content[i][j]);

                }
                if(j>=5 && j<=6) //If reading the 6th-7th column of csv (x-direction geometric domain limits)
                {
                    domain_limits_x[i-1][j-5] = std::stod(content[i][j]);
                    x_min = min(x_min, std::stod(content[i][j]));
                    x_max = max(x_max, std::stod(content[i][j]));
                    if (first_x==1){
                        x_min = std::stod(content[i][j]);
                        x_max = std::stod(content[i][j]);
                        first_x = 0;
                    }
                }
                if(j>=7 && j<=8) //If reading the 8th-9th column of csv (y-direction geometric domain limits)
                {
                    domain_limits_y[i-1][j-7] = std::stod(content[i][j]);
                    y_min = min(y_min, std::stod(content[i][j]));
                    y_max = max(y_max, std::stod(content[i][j]));
                    if (first_y==1){
                        y_min = std::stod(content[i][j]);
                        y_max = std::stod(content[i][j]);
                        first_y = 0;
                    }
                }
                if(j==9){
                    // cout<<"here1 \n";
                    subSim_IDs.push_back(std::stoi(content[i][j]));
                }
                if(j==10){
                    // cout<<"here2 \n";
                    seeds.push_back(std::stoi(content[i][j]));
                }
                if(j==11){
                    // cout<<"here3 \n";
                    ms.push_back(std::stoi(content[i][j]));
                }
                if(j==12){
                    // cout<<"here3 \n";
                    mobilities.push_back(std::stod(content[i][j]));
                }
                if(j==13){
                    // cout<<"here3 \n";
                    jump_probs.push_back(std::stod(content[i][j]));
                }
                // if(j==12){ //for jumping MATRIX case 
                //     // cout<<"here4 \n";
                //     printVector(ms);
                //     cout<<ms[i-1];
                //     for(int k = 0; k < ms[i-1]; ++k){
                //         cout<<k;
                //         jumping_probs[i].push_back(std::stod(content[i][12+k]));
                //     }
                // }
            }
		}
		// cout<<"\n";
	}

    int m = content.size()-1; //count how many populations were in population info data file

    vector<int> seeds_per_subSim;
    per_subPop_to_per_subSim(seeds, seeds_per_subSim, ms);

    vector<int> ms_per_subSim;
    per_subPop_to_per_subSim(ms, ms_per_subSim, ms);

    vector<int> total_pop_per_subSim;
    sum_subPop_per_subSim(pop_sizes, total_pop_per_subSim, ms);

    // printing info for debugging:
    // cout<<"xmax "<<x_max<< "\n";
    // cout<<"xmin "<<x_min<< "\n";
    // cout<<"ymax "<<y_max<< "\n";
    // cout<<"ymin "<<y_min<< "\n";

    // cout<<"There are "<<m<< " populations\n";
    // cout<<"There are "<<total_pop<< " total agents\n";

    // cout<<"status_fractions:\n";
    // prettyPrint(status_fractions);
    // cout<<"domain_limits_x:\n";
    // prettyPrint(domain_limits_x);
    // cout<<"domain_limits_y:\n";
    // prettyPrint(domain_limits_y);

    // cout<<"T_E: "<<T_E<< "\n";
    // cout<<"T_E_stdev: "<<T_E_stdev<< "\n";
    // cout<<"T_I: "<<T_I<< "\n";
    // cout<<"T_I_stdev: "<<T_I_stdev<< "\n";

    // cout<<"Sub-sim IDs: "<< "\n";
    // printVector(subSim_IDs);
    // cout<<"Seeds: "<< "\n";
    // printVector(seeds);
    // cout<<"Seeds per subsim: "<< "\n";
    // printVector(seeds_per_subSim);
    // cout<<"ms: "<<"\n";
    // printVector(ms);
    // cout<<"ms per subsim: "<< "\n";
    // printVector(ms_per_subSim);
    // cout<<"jumping probs: "<<"\n";
    // printVector(jump_probs);
    // // prettyPrint(jumping_probs); //for jumping_probs matrix
    // cout<<"subsim total population: "<<"\n";
    // printVector(total_pop_per_subSim);
    // cout<<"mobilities: "<<"\n";
    // printVector(mobilities);

    num_of_subSims = ms_per_subSim.size();

    // std::cout<<"here1!\n";
    //Initialize agents
    agent* agents = new agent[total_pop];
    init_agents(agents, T_E, T_E_stdev, T_I, T_I_stdev, m, pop_sizes, status_fractions, domain_limits_x, 
    domain_limits_y, dist, total_pop, seeds_per_subSim, num_of_subSims, total_pop_per_subSim, ms_per_subSim);

    //Move agents to GPU memory
    agent* agents_gpu;
    cudaMalloc((void**)&agents_gpu, total_pop * sizeof(agent));
    cudaMemcpy(agents_gpu, agents, total_pop * sizeof(agent), cudaMemcpyHostToDevice);

    // //Initialize and malloc data storage arrays on CPU (size: nsteps x m):
    // int (*S)[m] = malloc(nsteps * sizeof *S);
    // int (*E)[m] = malloc(nsteps * sizeof *E);
    // int (*I)[m] = malloc(nsteps * sizeof *I);
    // int (*DR)[m] = malloc(nsteps * sizeof *DR);
    // int (*new_I)[m] = malloc(nsteps * sizeof *new_I);

    // //Initialize and malloc data storage arrays on GPU:
    // int (*S_d)[m]; //initializes pointer to array of m integers
    // cudaMalloc((void **)&S_d, nsteps * sizeof(*S_d)); //malloc for nsteps rows of pointers to m size arrays
    // int (*E_d)[m]; 
    // cudaMalloc((void **)&E_d, nsteps * sizeof(*E_d));
    // int (*I_d)[m]; 
    // cudaMalloc((void **)&I_d, nsteps * sizeof(*I_d));
    // int (*DR_d)[m]; 
    // cudaMalloc((void **)&DR_d, nsteps * sizeof(*DR_d));
    // int (*new_I_d)[m]; 
    // cudaMalloc((void **)&new_I_d, nsteps * sizeof(*new_I_d));

    //^^ if i want to use this type of array initialization, would need to pass as void* to function arguments
    //because data type is dependent on array size, which is unknown at compilation time
    //then would need to static cast as correct type before using in function

    // std::cout<<"here!\n";

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    // std::cout<<"here2!\n";
    init_simulation(agents_gpu, total_pop, d_IU, m, domain_limits_x, domain_limits_y, seeds_per_subSim, num_of_subSims, subSim_IDs, ms_per_subSim, del_t, jump_probs,mobilities, total_pop_per_subSim, nsteps, savename);

    // std::cout<<"here3!\n";
    if (fsave.good() && save_rendering_data == "True") {
        save(fsave, agents, total_pop, d_IU, x_min, x_max, y_min, y_max);
    }


    for (int step = 0; step < nsteps; ++step) {
        // std::cout<<"stepping! "<< step <<"\n";
        simulate_one_step(agents_gpu, total_pop, d_IU, del_t, jump_probs, mobilities, m, step, nsteps, num_of_subSims);
        cudaDeviceSynchronize(); //potentially unnecessary

        // Save state if necessary

        if (fsave.good() && save_rendering_data == "True" && (step % savefreq) == 0) {
            // std::cout<<"here "<<step<<"\n";
            cudaMemcpy(agents, agents_gpu, total_pop * sizeof(agent), cudaMemcpyDeviceToHost);
            save(fsave, agents, total_pop, d_IU, x_min, x_max, y_min, y_max);
        }
    }

    cudaDeviceSynchronize(); //potentially unnecessary
    

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " <<total_pop << " particles.\n";
    fsave.close();
    delete[] agents;
}