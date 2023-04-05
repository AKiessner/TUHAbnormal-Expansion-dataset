

import sys
sys.path.insert(0, "/home/code/")

from train_tuheeg_pathology_balanced_saturation_finalrun import * 



model_name = 'Shallow'  # ShallowFBCSPNet


drop_prob=0.5
batch_size=64
lr=0.000625
n_epochs=35
weight_decay=0

seed=20170629


result_folder='/home/results/TUAB/eval/'
train_folder = '/home/data/preprocessed_TUHAbnormal/final_train/'
eval_folder = '/home/data/preprocessed_TUHAbnormal/final_eval/'

    

ids_to_load_train = None 

task_name = 'TUAB_finalrun_' 

train_TUHEEG_pathology(model_name=model_name,
                     task_name = task_name,
                     drop_prob=drop_prob,
                     batch_size=batch_size,
                     lr=lr,
                     n_epochs=n_epochs,
                     weight_decay=weight_decay,
                     result_folder=result_folder,
                     train_folder=train_folder,
                     eval_folder=eval_folder,
                     ids_to_load_train =  ids_to_load_train,
                     cuda = True,
                     seed= seed,
                     )
