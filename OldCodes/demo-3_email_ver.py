from hopfield_map_estimate_Marzen_version import ModifiedHopfieldNet
import numpy as np


# The following is an explanation of all of the arguments of ModifiedHopfieldNet

# in_directory = the relative path to a folder containing the raw .mat files
# out_directory = the path to a folder that you want to store the python data. If not given, defaults to in_directory

# splits: the number of times to split the data to train on each portion (slices sequentially
# --> default: 3 splits = beginning, middle, end thirds of experiment)

# train_percent = the percentage of each chunk of data (the number of chunks as defined by splits) that will be
# used to train a Hopfield network --> default: 0.66

# num_nets: The number of replications to use when training Hopfield networks --> default: 5

# exp_type = 'map' to conduct a MAP estimate analysis, J to analyze the changing connectivity matrices through time.
# and 'MI' for mutual information estimate
# --> default: 'J'

# data_type: use "old" for data stored in the structure of Expt1.mat or "new" for data stored in the structure of
# baseline, baseline carbachol, etc.
# --> default: "old"
# --> WARNING: Currently, this does not support using a folder that has a mixture of data types, so if the program
#              crashes, you can restart it with the data_type parameter set to the opposite value, but you may have to
#              keep switching back and forth

# dt: The time window used when converting the input data into binary sequences
# --> default: 800 (50 ms)
# WARNING: This will vary greatly on your sample rate!

# N: The number of neurons in the experiment
# --> default: 60
# (if left to None it will be set to 60, sorry that part is a little messy)

# n_jobs:: Increase this value from one to load data with multiprocessing
# WARNING: this only works with exp_type="J" currently, and data stored in the "experiments" attribute of the
# ModifiedHopfieldNet will not necessraily be in order. If using this, I would recommend doing it to load and preprocess
# data across more cpu cores, which will then save as as sparse matrices, which can can be loaded back in the correct
# order by restarting the processing after all data is preprocessed and loaded!

# Put all .mat files in a folder, and create an empty output folder for the python-loadable files!
# Alternatively, if the out_directory field is left blank, it will use the in_directory as the out_directory!

# # To conduct an analysis of changing co-activation (functional connectivity) between different stimulation protocols:
# net = ModifiedHopfieldNet(in_directory='toy_data/', out_directory='new_data/', exp_type='J', splits=1, num_nets=5,
#                           train_percent=0.66, data_type='old', stim_shift=10, dt=200)
# net.build_and_train_networks()
# Js = net.get_js()
# # Js will be a numpy array of shape (num_files, splits, num_nets, N, N)
#
# # To estimate MAP and get accuracy of our model with hopfield networks...
# net = ModifiedHopfieldNet( in_directory='toy_data/', out_directory='new_data/', exp_type='map', splits=1,
#                                         num_nets=5, train_percent=0.66)
# net.build_and_train_networks()
# # will print the average accuracy for each split of the data using our model

# To estimate mutual information with different dts...
dts = np.asarray([12])
net = ModifiedHopfieldNet(in_directory='validation_data/', out_directory='test/', exp_type='MI', splits=2, data_type='new', stim_shift=0)
MIs, varMIs = net.get_MI_estimates(dts)
#net = ModifiedHopfieldNet(in_directory='toy_data/', out_directory='new_data/', exp_type='MI', splits=3, data_type='new',
                          #stim_shift=0)
#MIs, varMIs = net.get_MI_estimates(dts)
# will save estimates and variation of the estimates for each file


# To perform MLE with the explicitMLE algorithm:
#net = ModifiedHopfieldNet(in_directory='toy_data/', out_directory='new_data/', exp_type='J', splits=2, data_type='new',
                        #   stim_shift=5)
#thetas, Js = net.run_explicitMLE()
# Note that the experiment_type paramter does not matter if you only want to run the MLE algorithm on the data in
# your in_directory
