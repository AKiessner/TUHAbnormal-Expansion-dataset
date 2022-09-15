import sys
sys.path.insert(0, "/home/code/")

from train_tuheeg_pathology_balanced_saturation_TUAB import * 



only_last_fold_valid = False # True or False
only_fold_x_valid = None # None or fold number 

model_name = 'EEGNet'  # ShallowFBCSPNet


Drop_prob=[0.25]
Batch_size=[64]
Lr=[0.001]
N_epochs=[35]
Weight_decay=[0]



ab_ifold=''

result_folder='/home/results/TUAB/cv/'
train_folder = '/home/data/preprocessed_TUHAbnormal/final_train/'



ids_to_load_train = None 

task_name = 'TUAB_cv' + str(subset_size)

train_TUHEEG_pathology(model_name=model_name,
                     task_name = task_name,
                     n_splits_valid=5,
                     Drop_prob=Drop_prob,
                     Batch_size=Batch_size,
                     Lr=Lr,
                     N_epochs=N_epochs,
                     Weight_decay=Weight_decay,
                     result_folder=result_folder,
                     train_folder=train_folder,
                     ids_to_load_train =  ids_to_load_train,
                     cuda = True,
                     seed= seed,
                     shuffle_folds= False,
                     only_last_fold_valid = only_last_fold_valid,
                     only_fold_x_valid= only_fold_x_valid,
                     ab_ifold = ab_ifold,
                     )