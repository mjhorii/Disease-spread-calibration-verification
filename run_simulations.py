import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import cProfile
import pickle
import scipy as sci
import pandas as pd
import os
import shutil

from ABM import SEIR_multiple_pops

#============================================================================================================
#==================================Function to generate data==============================================
#============================================================================================================

def simulate_epidemic(m, centers, spread, pop, A_1, A_2, R, d_IU, E_0, I_0, S_0, T_E, T_E_stdev, 
                      T_I, T_I_stdev, del_t, verlet_iter, T, rand_seed, g, al, jumping_times, 
                      jump_prob, spherical, random_seed, dist, num_of_sims, num_of_runs, 
                      RUN_NAME, DIRECTORY):
    
    for i in tqdm(range(num_of_runs)): 
        #run simulation: 
        os.system('ABM-cpp/build/serial_multisim -s 1 -file "'+DIRECTORY+'/Pop info stored in parts/pop-info-part'+str(i)+'.csv" -o '+RUN_NAME+'save_'+str(i)+'.out -d_IU 0.005 -del_t 0.1 -T 300 -save_data_for_rendering False')

        #move data files:
#         shutil.move("S_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/S_save_"+str(i)+".out") #not necessary information, stopped saving in C++ code
#         shutil.move("E_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/E_save_"+str(i)+".out")
#         shutil.move("I_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/I_save_"+str(i)+".out")
#         shutil.move("DR_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/DR_save_"+str(i)+".out")
        shutil.move("new_I_"+RUN_NAME+"save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/new_I_"+RUN_NAME+"save_"+str(i)+".out")
        shutil.move(RUN_NAME+"save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/"+RUN_NAME+"save_"+str(i)+".out")
        
    #repackage data:
    num_of_time_steps = int(T/del_t) + 1
    new_I_all = np.zeros((num_of_sims, num_of_time_steps, 2))
    num_of_sims_per_run = int(num_of_sims/num_of_runs)

    for i in tqdm(range(num_of_runs)):
        temp = pd.read_csv(DIRECTORY+'/Data stored in parts/new_I_'+RUN_NAME+'save_'+str(i)+'.out', header=None, sep = ' ').to_numpy().squeeze()

        for j in range(num_of_sims_per_run):
            new_I_all[i*num_of_sims_per_run+j,:,:] = temp[:, j*2:(j+1)*2]
        if np.max(temp[:, j*2:(j+1)*2]) == 0: #sanity check -- likely should be no or very few simulations that have 0 new infections
            print('zero')

    #Change data type to save memory:
    print('Size before changing data type: ', new_I_all.nbytes)

    if np.max(new_I_all) < 256:
        new_I_all = new_I_all.astype('uint8')
    elif np.max(new_I_all) < 65536:
        new_I_all = new_I_all.astype('uint16')

    print('Size after changing data type: ', new_I_all.nbytes)


    # create csv file of variable parameters across simulations (in this case, mobilities, jumping probabilities, and random seeds)
    full_pop_info = pd.read_csv(DIRECTORY+'/pop-info-FULL.csv').to_numpy()

    variable_parameter_values = np.zeros((num_of_sims,3))

    seeds = full_pop_info[::2,10]
    mobilities = full_pop_info[::2,12]
    jumping_probs = full_pop_info[::2,13]

    variable_parameter_values[:,0] = mobilities
    variable_parameter_values[:,1] = jumping_probs
    variable_parameter_values[:,2] = seeds

    df = pd.DataFrame(variable_parameter_values, columns = ['Mobility', 'Jumping probability', 'Random seed'], dtype=object)

    # create csv file of constant parameters across simulations
    df_constant_params = pd.DataFrame([[m, centers, spread, pop, d_IU, E_0, I_0, S_0, T_E, T_E_stdev, 
                                        T_I, T_I_stdev, del_t, T, jumping_times, dist]], 
                                columns=['Number of populationss (m)', 'Centers', 'Spread', 'Population sizes (pop)','d_IU', 
                                         'Fraction initially exposed (E_0)',
                                   'Fraction initially infected (I_0)','Fraction initially susceptible (S_0)',
                                  'Time of exposure (T_E)','Standard deviation of exposure time (T_E_stdev)',
                                  'Infection time (T_I)','Standard deviation of infection time (T_I_stdev)',
                                  'Time step (del_t)', 'Total time (T)', 'Jumping times (jumping_times)',
                                  'Distribution type (dist)'])

    # Data saving
#     user_input = input("Do you want to save the data? Type 'yes' or 'no': ")
#     if user_input.lower() == 'yes': #So that I don't overwrite the data with zeros
    print('Saving data...')
    with open(DIRECTORY+'/new_I_data_'+RUN_NAME+'.pickle', 'wb') as handle:
        pickle.dump(new_I_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df.to_csv(DIRECTORY+'/variable_parameter_values_'+RUN_NAME+'.csv')
    df_constant_params.to_csv(DIRECTORY+'/constant_parameters_values_'+RUN_NAME+'.csv')
#     else:
#         print('DID NOT SAVE')


def simulate_epidemic_1d(m, centers, spread, pop, A_1, A_2, R, d_IU, E_0, I_0, S_0, T_E, T_E_stdev, 
                      T_I, T_I_stdev, del_t, verlet_iter, T, rand_seed, g, al, jumping_times, 
                      jump_prob, spherical, random_seed, dist, num_of_sims, num_of_runs,
                      RUN_NAME, DIRECTORY):
    
    for i in tqdm(range(num_of_runs)): 
        #run simulation: 
        os.system('ABM-cpp/build/serial_multisim -s 1 -file "'+DIRECTORY+'/Pop info stored in parts/pop-info-part'+str(i)+'.csv" -o '+RUN_NAME+'save_'+str(i)+'.out -d_IU 0.005 -del_t 0.1 -T 300 -save_data_for_rendering False')

        #move data files:
#         shutil.move("S_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/S_save_"+str(i)+".out") #not necessary information, stopped saving in C++ code
#         shutil.move("E_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/E_save_"+str(i)+".out")
#         shutil.move("I_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/I_save_"+str(i)+".out")
#         shutil.move("DR_save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/DR_save_"+str(i)+".out")
        shutil.move("new_I_"+RUN_NAME+"save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/new_I_"+RUN_NAME+"save_"+str(i)+".out")
        shutil.move(RUN_NAME+"save_"+str(i)+".out", DIRECTORY+"/Data stored in parts/"+RUN_NAME+"save_"+str(i)+".out")
        
    #repackage data:
    num_of_time_steps = int(T/del_t) + 1
    new_I_all = np.zeros((num_of_sims, num_of_time_steps))
    num_of_sims_per_run = int(num_of_sims/num_of_runs)

    for i in tqdm(range(num_of_runs)):
        temp = pd.read_csv(DIRECTORY+'/Data stored in parts/new_I_'+RUN_NAME+'save_'+str(i)+'.out', header=None, sep = ' ').to_numpy().squeeze()

        for j in range(num_of_sims_per_run):
            new_I_all[i*num_of_sims_per_run+j,:] = temp[:, j]
        if np.max(temp[:, j]) == 0: #sanity check -- likely should be no or very few simulations that have 0 new infections
            print('zero')

    #Change data type to save memory:
    print('Size before changing data type: ', new_I_all.nbytes)

    if np.max(new_I_all) < 256:
        new_I_all = new_I_all.astype('uint8')
    elif np.max(new_I_all) < 65536:
        new_I_all = new_I_all.astype('uint16')

    print('Size after changing data type: ', new_I_all.nbytes)


    # create csv file of variable parameters across simulations (in this case, mobilities, jumping probabilities, and random seeds)
    full_pop_info = pd.read_csv(DIRECTORY+'/pop-info-FULL.csv').to_numpy()

    variable_parameter_values = np.zeros((num_of_sims,3))

    seeds = full_pop_info[:,10]
    mobilities = full_pop_info[:,12]
    jumping_probs = full_pop_info[:,13]

    variable_parameter_values[:,0] = mobilities
    variable_parameter_values[:,1] = jumping_probs
    variable_parameter_values[:,2] = seeds

    df = pd.DataFrame(variable_parameter_values, columns = ['Mobility', 'Jumping probability', 'Random seed'], dtype=object)

    # create csv file of constant parameters across simulations
    df_constant_params = pd.DataFrame([[m, centers, spread, pop, d_IU, E_0, I_0, S_0, T_E, T_E_stdev, 
                                        T_I, T_I_stdev, del_t, T, jumping_times, dist]], 
                                columns=['Number of populationss (m)', 'Centers', 'Spread', 'Population sizes (pop)','d_IU', 
                                         'Fraction initially exposed (E_0)',
                                   'Fraction initially infected (I_0)','Fraction initially susceptible (S_0)',
                                  'Time of exposure (T_E)','Standard deviation of exposure time (T_E_stdev)',
                                  'Infection time (T_I)','Standard deviation of infection time (T_I_stdev)',
                                  'Time step (del_t)', 'Total time (T)', 'Jumping times (jumping_times)',
                                  'Distribution type (dist)'])

    # Data saving
#     user_input = input("Do you want to save the data? Type 'yes' or 'no': ")
#     if user_input.lower() == 'yes': #So that I don't overwrite the data with zeros
    print('Saving...')
    with open(DIRECTORY+'/new_I_data_'+RUN_NAME+'.pickle', 'wb') as handle:
        pickle.dump(new_I_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df.to_csv(DIRECTORY+'/variable_parameter_values_'+RUN_NAME+'.csv')
    df_constant_params.to_csv(DIRECTORY+'/constant_parameters_values_'+RUN_NAME+'.csv')
#     else:
#         print('DID NOT SAVE')