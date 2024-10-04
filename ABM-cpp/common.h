#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <vector>
using namespace std;

// Program Constants
// #define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
typedef struct agent {
    int id; // Agent ID
    double x;  // Position X
    double y;  // Position Y
    int subPop; //Subpopulation ID
    int original_subPop; //Subpopulation ID of original location
    double E_time; //Exposure time
    double I_time; //Infection time
    double timer; //Current timer (either of infection or exposure depending on disease status)
    int status; //0 = S, 1 = E, 2 = I, 3 = R
    int subSim;
    int id_in_subSim;
    // double del_x;
    // double del_y;
} agent;

// Simulation routine
// void init_simulation(particle_t* parts, int num_parts, double size);
void init_simulation(agent* agents, int total_pop, double d_IU, int m, vector<vector<double>> domain_limits_x_, vector<vector<double>> domain_limits_y_, int part_seed, char* savename);
// void simulate_one_step(particle_t* parts, int num_parts, double size);
void simulate_one_step(agent* agents, int total_pop, double d_IU, double del_t, double jump_prob, double mobility, int m, int step, int nsteps);

#endif
