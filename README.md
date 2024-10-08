# Instructions

## Code environment

Run the following to set up a conda environment and install required packages:
```
conda create -n ABM_env
conda activate ABM_env
conda install pip
pip install -r requirements.txt
```

Run the following to set up a jupyter notebook kernel in this environment:
```
pip install --user ipykernel
python -m ipykernel install --user --name=ABM_env
```

Select ABM_env as kernel when running any notebooks.

## Required data for reproduction

Download "Data" from https://figshare.com/s/6e703d1b47849779cd22, unzip, and place in folder "Disease-spread-calibration-verification".

For only synthetic data without calibration results, download only "Training Data.zip" and "Test Data.zip." To be consistent with code, place "Training Data" and "Test Data" in "Disease-spread-calibration-verification/Data". 

This dataset contains the synthetic data generated by an agent-based disease spread model, including the training and test data used to produce paper figures.

## Running code for reproduction

### Calibration method 1

#### Example usages:
Example usages of calibration method 1 are found in CalibrationMethod1_example_usage.ipynb and CalibrationMethod1_2D_example_usage.ipynb.

#### Figure reproduction:
To reproduce manuscript figures for the one parameter case, run batch1d_posterior_estimation.py to run all posterior inferences. Then, use CalibrationMethod1_process_results.ipynb to do visualization, calculate ESS values, check SBC rank histograms, and run posterior predictive checks.

To reproduce manuscript figures for the two parameter case, run batch_posterior_estimation.py to run all posterior inferences. Then, use CalibrationMethod1_2D_process_results.ipynb to do visualization, calculate ESS values, and check SBC rank histograms.


### Calibration method 2

Calibration method 2 can be run in ABC_main.ipynb (one-parameter case) and ABC_2D_main.ipynb (two-parameter case). These notebooks contain example usages as well as code to reproduce manuscript figures.


## Generating your own synthetic data

Use ABM-cpp to run the model and create your own synthetic data. Further instructions for running the ABM are found in ABM-cpp/README.md


## License

This work is licensed under CC BY 4.0