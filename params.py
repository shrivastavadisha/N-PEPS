seed = 9

# General training
num_epochs = 100
patience = 20
model_output_path = 'trained_models/E1/N-PEPS/' #'trained_models/E1/GPS_model/'
load_from_checkpoint = False
checkpoint_dir = ''
max_len = None #For debugging set a lower number, use None for running training for full data

# File params (Inference)
global_model_path = 'trained_models/E1/GPS_model/best_model.th' #GPS model path
PE_model_path = 'trained_models/E1/PE_model/best_model.th' #PE model path
result_path = 'results/'

# GPS and PE model training params
train_path = 'data/E1/train_dataset_gps'
val_path = 'data/E1/val_dataset_gps'
learn_rate = 0.001
batch_size = 32
val_iterator_size = 32
lr_scheduler_step_size = 4

#Aggregator training params
att_batch_size = 256

#Inference params
search_method = 'beam'
num_workers = 1
max_beam_size = 819200
dfs_max_width = 50
cab_beam_size = 100
cab_width = 10
cab_width_growth = 10

# DSL params
integer_min = -256
integer_max = 255
integer_range = integer_max - integer_min + 1
max_list_len = 20
num_inputs = 3
num_statements = 1298
num_operators = 38

# Program State Params
max_program_len = 8
max_program_vars = max_program_len + num_inputs
state_len = max_program_vars + 1
state_dim = 256

# H_theta and W_phi network params
type_vector_len = 2
embedding_size = 20
var_encoder_size = 56
dense_output_size = 256
dense_num_layers = 10
dense_growth_size = 56