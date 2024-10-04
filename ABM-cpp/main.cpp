#include "common.h"
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

// Particle Initialization
void init_agents(agent* agents, double T_E, double T_E_stdev, double T_I, double T_I_stdev, int m, vector<int> pop_sizes, 
                    vector<vector<double>> status_fractions, vector<vector<double>> domain_limits_x, 
                    vector<vector<double>> domain_limits_y, string dist, int total_pop, int part_seed) {

    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd()); //***
    std::uniform_real_distribution<double> rand(0.0, 1.0); //uniform distribution between 0,1
    double rand_num;

    //Generate exposure times and infection times for each agent
    if (dist == "Normal"){
        std::normal_distribution<double> dist_E(T_E, T_E_stdev); //normal distribution of exposure times  
        std::normal_distribution<double> dist_I(T_I, T_I_stdev); //normal distribution of infection times
        for (int i = 0; i < total_pop; ++i){
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

        for (int i = 0; i < total_pop; ++i){
            agents[i].E_time = dist_E(gen);
            agents[i].I_time = dist_I(gen);
        }
    }
    // cout<<"here4\n";

    int num = 0;

    for (int subPop = 0; subPop < m; ++subPop){
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

        for (int i = 0; i < subPopSize; ++i){
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

            ++num;
        }
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

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        std::cout << "-file <string>: set file path to population info csv" << std::endl;
        std::cout << "-T_E <double>: mean exposure time" << std::endl;
        std::cout << "-T_E_stdev <double>: standard deviation of exposure time" << std::endl;
        std::cout << "-T_I <double>: mean infection time" << std::endl;
        std::cout << "-T_I_stdev <double>: standard deviation of infection time" << std::endl;
        std::cout << "-d_IU <double>: infection radius" << std::endl;
        std::cout << "-dist <string>: what type of distribution (Normal or Gamma)" << std::endl;
        std::cout << "-jump_prob <double>: probability of jumping to another subpop at a time step (if there is only one subpop, must be 0)" << std::endl;
        std::cout << "-mobility <double>: mobility of agents (max distance able to move in one time unit)" << std::endl;
        std::cout << "-T <double>: total simulation time" << std::endl;
        std::cout << "-del_t <double>: time step size" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Initialize Particles
    int part_seed = find_int_arg(argc, argv, "-s", 0); 
    string dist = find_string_option(argc, argv, "-dist", "Gamma");
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

	for(int i=0;i<content.size();i++)
	{
        status_fractions.resize(i+1);
        domain_limits_x.resize(i+1);
        domain_limits_y.resize(i+1);
        // cout<<"line "<<i<<"\n";
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
            }
		}
		// cout<<"\n";
	}

    int m = content.size()-1; //count how many populations were in population info data file

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

    //Initialize agents
    agent* agents = new agent[total_pop];
    init_agents(agents, T_E, T_E_stdev, T_I, T_I_stdev, m, pop_sizes, status_fractions, domain_limits_x, domain_limits_y, dist, total_pop, part_seed);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    init_simulation(agents, total_pop, d_IU, m, domain_limits_x, domain_limits_y, part_seed, savename);

    if (fsave.good()) {
        save(fsave, agents, total_pop, d_IU, x_min, x_max, y_min, y_max);
    }

#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
    {
        for (int step = 0; step < nsteps; ++step) {
            simulate_one_step(agents, total_pop, d_IU, del_t, jump_prob, mobility, m, step, nsteps);

            // Save state if necessary
#ifdef _OPENMP
#pragma omp master
#endif
            if (fsave.good() && (step % savefreq) == 0) {
                save(fsave, agents, total_pop, d_IU, x_min, x_max, y_min, y_max);
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " <<total_pop << " particles.\n";
    fsave.close();
    delete[] agents;
}
