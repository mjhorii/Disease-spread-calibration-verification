## Agent based model for epidemiological modeling

### Initial set-up
In ABM-cpp, run "mkdir build". Then run "cd build".
Load cmake module using: "module load cmake"
Configure the build: "cmake -DCMAKE_BUILD_TYPE=Release .."
Compile: "make"

### Running serial code
Use -h to see arg options:

-h: see this help
-o <filename>: set the output file name
-s <int>: set particle initialization seed
-file <string>: set file path to population info csv
-T_E <double>: mean exposure time
-T_E_stdev <double>: standard deviation of exposure time
-T_I <double>: mean infection time
-T_I_stdev <double>: standard deviation of infection time
-d_IU <double>: infection radius
-dist <string>: what type of distribution (Normal or Gamma)
-jump_prob <double>: probability of jumping to another subpop at a time step (if there is only one subpop, must be 0)
-mobility <double>: mobility of agents (max distance able to move in one time unit)
-T <double>: total simulation time
-del_t <double>: time step size

pop-info-single.csv and pop-info.csv are examples of population info format.

Example command to run from build folder: ./serial -s 1 -file ../pop-info-single.csv -o "save.out" -d_IU 0.005 -mobility 0.1 -del_t 0.1 -T 300

### Running serial MULTISIM code (preferred)
Use -h to see arg options 

pop-info-multisim.csv is example of population info format.

Example command to run from build folder: ./serial_multisim -file ../pop-info-multisim.csv -o "save.out" -d_IU 0.005 -del_t 0.1 -T 300

### Rendering
Parent directory (Sandia-UCB-Agent-Based-Collab) contains cpp-rendering, which is folder with python rendering code.

If in build folder, use by calling: python3 ../../cpp-rendering/render_ABM.py save.out test_multi.gif
(replace "save.out" with the chosen -o arg, replace file path appropriately if not in build folder)

### Example usage
See ppc function in CalibrationMethod1_process_results.ipynb for an example of full data generation workflow.


### Debugging with LLDB
Uncomment this line in CMakeLists.txt: set(CMAKE_BUILD_TYPE Debug)

Example command line call: lldb -- serial_multisim -s 1 -file ../pop-info-multisim.csv -o "save.out" -d_IU 0.005 -del_t 0.1 -T 300

To alter args within LLDB, can call: process launch -- -s 1 -file ../pop-info-multisim.csv -o "save.out" -d_IU 0.005 -del_t 0.1 -T 300
