import time
import os
import mne
mne.set_log_level('ERROR')

from warnings import filterwarnings
filterwarnings('ignore')


from IPython.utils import io

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import torch
from torch.nn.functional import relu
from torch.utils.data import WeightedRandomSampler

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
from braindecode.models import Deep4Net,ShallowFBCSPNet,EEGNetv4, TCN
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.serialization import  load_concat_dataset

from braindecode.datasets import BaseConcatDataset
from braindecode.datautil.preprocess import preprocess, Preprocessor, exponential_moving_standardize


from braindecode.training import trial_preds_from_window_preds


from functools import partial 
from skorch.callbacks import LRScheduler, EarlyStopping,Checkpoint, EpochScoring
from skorch.helper import predefined_split


    
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, '/home/kiessnek/TUHEEG_decoding/code/')
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
from ExtendedAdam import ExtendedAdam

from itertools import product
from functools import partial 

#import GPUtil


#######################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def plot_confusion_matrix_paper(confusion_mat, p_val_vs_a=None,
                                p_val_vs_b=None,
                                class_names=None, figsize=None,
                                colormap=cm.bwr,
                                textcolor='black', vmin=None, vmax=None,
                                fontweight='normal',
                                rotate_row_labels=90,
                                rotate_col_labels=0,
                                with_f1_score=False,
                                norm_axes=(0, 1),
                                rotate_precision=False,
                                class_names_fontsize=12):
    
    # TODELAY: split into several functions
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    # then have to transpose pvals also
    if p_val_vs_a is not None:
        p_val_vs_a = p_val_vs_a.T
    if p_val_vs_b is not None:
        p_val_vs_b = p_val_vs_b.T
    n_classes = confusion_mat.shape[0]
    if class_names is None:
        class_names = [str(i_class + 1) for i_class in range(n_classes)]

    # norm by number of targets (targets are columns after transpose!)
    # normed_conf_mat = confusion_mat / np.sum(confusion_mat,
    #    axis=0).astype(float)
    # norm by all targets
    normed_conf_mat = confusion_mat / np.float32(np.sum(confusion_mat, axis=norm_axes, keepdims=True))

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if vmin is None:
        vmin = np.min(normed_conf_mat)
    if vmax is None:
        vmax = np.max(normed_conf_mat)

    # see http://stackoverflow.com/a/31397438/1469195
    # brighten so that black text remains readable
    # used alpha=0.6 before
    def _brighten(x, ):
        brightened_x = 1 - ((1 - np.array(x)) * 0.4)
        return brightened_x

    brightened_cmap = _cmap_map(_brighten, colormap) #colormap #
    ax.imshow(np.array(normed_conf_mat), cmap=brightened_cmap,
              interpolation='nearest', vmin=vmin, vmax=vmax)

    # make space for precision and sensitivity
    plt.xlim(-0.5, normed_conf_mat.shape[0]+0.5)
    plt.ylim(normed_conf_mat.shape[1] + 0.5, -0.5)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in range(width):
        for y in range(height):
            if x == y:
                this_font_weight = 'bold'
            else:
                this_font_weight = fontweight
            annotate_str = "{:d}".format(confusion_mat[x][y])
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
                annotate_str += " *"
            else:
                annotate_str += "  "
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
                annotate_str += u"*"
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
                annotate_str += u"*"

            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
                annotate_str += u" ◊"
            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
                annotate_str += u"◊"
            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
                annotate_str += u"◊"
            annotate_str += "\n"
            ax.annotate(annotate_str.format(confusion_mat[x][y]),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
                        color=textcolor,
                        fontweight=this_font_weight)
            if x != y or (not with_f1_score):
                ax.annotate(
                    "\n\n{:4.1f}%".format(
                        normed_conf_mat[x][y] * 100),
                    #(confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)
            else:
                assert x == y
                precision = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[x, :]))
                sensitivity = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[:, y]))
                f1_score = 2 * precision * sensitivity / (precision + sensitivity)

                ax.annotate("\n{:4.1f}%\n{:4.1f}% (F)".format(
                    (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100,
                    f1_score * 100),
                    xy=(y, x + 0.1),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)

    # Add values for target correctness etc.
    for x in range(width):
        y = len(confusion_mat)
        correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x, :]))
        annotate_str = ""
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    for y in range(height):
        x = len(confusion_mat)
        correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:, y]))
        annotate_str = ""
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    overall_correctness = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat).astype(float)
    ax.annotate("{:5.2f}%".format(overall_correctness * 100),
                xy=(len(confusion_mat), len(confusion_mat)),
                horizontalalignment='center',
                verticalalignment='center', fontsize=12,
                fontweight='bold')

    plt.xticks(range(width), class_names, fontsize=class_names_fontsize,
               rotation=rotate_col_labels)
    plt.yticks(np.arange(0,height), class_names,
               va='center',
               fontsize=class_names_fontsize, rotation=rotate_row_labels)
    plt.grid(False)
    plt.ylabel('Predictions', fontsize=15)
    plt.xlabel('Targets', fontsize=15)

    # n classes is also shape of matrix/size
    ax.text(-1.2, n_classes+0.2, "Specificity /\nSensitivity", ha='center', va='center',
            fontsize=13)
    if rotate_precision:
        rotation=90
        x_pos = -1.1
        va = 'center'
    else:
        rotation=0
        x_pos = -0.8
        va = 'top'
    ax.text(n_classes, x_pos, "Precision", ha='center', va=va,
            rotation=rotation,  # 270,
            fontsize=13)

    return fig

# see http://stackoverflow.com/a/31397438/1469195
def _cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    from matplotlib.colors import LinearSegmentedColormap as lsc
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: list(map(lambda x: x[0], cdict[key])) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_dicts = np.array(list(step_dict.values()))
    step_list = np.unique(step_dicts)
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(list(map(function, y0)))
    y1 = np.array(list(map(function, y1)))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    # Remove alpha, otherwise crashes...
    step_dict.pop('alpha', None)
    return lsc(name, step_dict, N=N, gamma=gamma)

###########################
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



def plot_roc_curve(fper, tper, save_name):
    plt.clf()
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(save_name +'_ROC.png',bbox_inches='tight')
    plt.show()
####################################




def train_TUHEEG_pathology(model_name,
                     drop_prob,
                     batch_size,
                     lr,
                     n_epochs,
                     weight_decay,
                     result_folder,
                     train_folder,
                     eval_folder,
                     task_name,
                     ids_to_load_train,
                     seed,
                     cuda = True,
                     ):

    ###################################
    target_name = None  # Comment Lukas our target is not supported by default, set None and add later 
    add_physician_reports = True

    sfreq = 100  #100
    n_minutes = 20 #

    n_max_minutes = n_minutes+1
    
    #Drop_prob = [drop_prob]
    N_REPETITIONS =1
    ####### MODEL DEFINITION ############
    torch.backends.cudnn.benchmark = True
    ######

    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    
   
    print(task_name)

    print('loading data')
    # load from train
    start = time.time()
    with io.capture_output() as captured:
        train_set = load_concat_dataset(train_folder, preload=False, ids_to_load=ids_to_load_train)

    end = time.time()

    print('finished loading preprocessed trainset ' + str(end-start))

    target = train_set.description['pathological'].astype(int)



    for d, y in zip(train_set.datasets, target):
        d.description['pathological'] = y
        d.target_name = 'pathological'
        d.target = d.description[d.target_name]
    train_set.set_description(pd.DataFrame([d.description for d in train_set.datasets]), overwrite=True)


        
    # load eval set
    
    print('loading eval data')
    # load from train
    start = time.time()
    with io.capture_output() as captured:
        eval_complete = load_concat_dataset(eval_folder, preload=False, ids_to_load=None)

    end = time.time()

    print('finished loading preprocessed trainset ' + str(end-start))

    target = eval_complete.description['pathological'].astype(int)
    
    for d, y in zip(eval_complete.datasets, target):
        d.description['pathological'] = y
        d.target_name = 'pathological'
        d.target = d.description[d.target_name]
    eval_complete.set_description(pd.DataFrame([d.description for d in eval_complete.datasets]), overwrite=True)
    

   

   
    if not os.path.exists(result_folder + model_name + '/' + 'seed'+str(seed)):
            os.mkdir(result_folder + model_name + '/' + 'seed'+str(seed))
            print("Directory " , result_folder + model_name + '/' + 'seed'+str(seed) ,  " Created ")
    else:    
        print("Directory " , result_folder + model_name + '/' + 'seed'+str(seed) ,  " already exists")
            

    dirSave = result_folder + model_name + '/' + 'seed'+str(seed) + '/' + task_name    
    save_name =model_name   +'_trained_' +  str(task_name)+ '_'  


   # cv = 'SKF'    
    if not os.path.exists(dirSave):
        os.mkdir(dirSave)
        print("Directory " , dirSave ,  " Created ")
    else:    
        print("Directory " , dirSave ,  " already exists")


    result_path = dirSave  +'/' + save_name 

    n_classes=2
    # Extract number of chans from dataset
    n_chans = 21
    input_window_samples =6000
    if model_name == 'Deep4Net':
        n_start_chans = 25
        final_conv_length = 1
        n_chan_factor = 2
        stride_before_pool = True
       # input_window_samples =6000
        model = Deep4Net(
                    n_chans, n_classes,
                    n_filters_time=n_start_chans,
                    n_filters_spat=n_start_chans,
                    input_window_samples=input_window_samples,
                    n_filters_2=int(n_start_chans * n_chan_factor),
                    n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                    n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                    final_conv_length=final_conv_length,
                    stride_before_pool=stride_before_pool,
                    drop_prob=drop_prob)
            # Send model to GPU
        if cuda:
            model.cuda()

        to_dense_prediction_model(model)

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    elif model_name == 'Shallow':  
        n_start_chans = 40
        final_conv_length = 25
        input_window_samples =6000
        model = ShallowFBCSPNet(n_chans,n_classes,
                                input_window_samples=input_window_samples,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                final_conv_length= final_conv_length,
                                drop_prob=drop_prob)
        # Send model to GPU
        if cuda:
            model.cuda()

        to_dense_prediction_model(model)

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    elif model_name == 'TCN':  
        n_chan_factor = 2
        stride_before_pool = True
        input_window_samples =6000
        l2_decay = 1.7491630095065614e-08
        gradient_clip = 0.25

        model = TCN(
            n_in_chans=n_chans, n_outputs=n_classes,
            n_filters=55,
            n_blocks=5,
            kernel_size=16,
            drop_prob=drop_prob,
            add_log_softmax=True)

            # Send model to GPU
        if cuda:
            model.cuda()

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

 
    elif model_name == 'EEGNet':    

        input_window_samples =6000
        final_conv_length=18
        drop_prob=0.25
        model = EEGNetv4(
            n_chans, n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=final_conv_length,
            drop_prob=drop_prob)
        if cuda:
            model.cuda()

        to_dense_prediction_model(model)

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    print(model_name + ' model sent to cuda')

    with io.capture_output() as captured:
         window_train_set = create_fixed_length_windows(train_set, 
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,
                                                        preload=True,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=n_preds_per_input,
                                                        drop_last_window=True,)

    with io.capture_output() as captured:
         window_eval_set = create_fixed_length_windows(eval_complete,
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=n_preds_per_input,
                                                        drop_last_window=False,)

    #del  train_set
    print('abnormal train ' + str(len(window_train_set.description[window_train_set.description['pathological']==1])))
    print('normal train ' + str(len(window_train_set.description[window_train_set.description['pathological']==0])))


    print('abnormal eval' + str(len(window_eval_set.description[window_eval_set.description['pathological']==1])))


    print('normal eval ' + str(len(window_eval_set.description[window_eval_set.description['pathological']==0])))



    clf = EEGClassifier(model,cropped=True,
                    criterion=CroppedLoss,
                    criterion__loss_function=torch.nn.functional.nll_loss,
                    optimizer=torch.optim.AdamW,
                    train_split=predefined_split(window_eval_set),
                    optimizer__lr=lr,
                    optimizer__weight_decay=weight_decay,
                    iterator_train__shuffle=True,
                    batch_size=batch_size,
                    callbacks=["accuracy",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),],  #"accuracy",
                    device='cuda')

    print('AdamW')

    print('classifier initialized')

    # Model training for a specified number of epochs. `y` is None as it is already supplied
    clf.fit(window_train_set, y=None, epochs=n_epochs)
    print('end training')
          #
     ####### SAVE ############
    print('saving')
    # save model

    # Extract loss and accuracy values for plotting from history object
    plt.style.use('seaborn')
            
            #plots only loss
            
    results_columns = ['train_loss','valid_loss','train_accuracy','valid_accuracy']#, 'train_balanced_accuracy', 'valid_balanced_accuracy']

    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,index=clf.history[:, 'epoch'])
    df.to_pickle(result_path + '_df_history.pkl')
    #save history
    torch.save(clf.history, result_path + '_clf_history.py')
    
    path = result_path + "model_{}.pt".format(seed)
    torch.save(clf.module, path)
    path = result_path + "state_dict_{}.pt".format(seed)
    torch.save(clf.module.state_dict(), path)

    clf.save_params(f_params=result_path +'model.pkl', f_optimizer= result_path +'opt.pkl', f_history=result_path +'history.json')

    
    pred_win = clf.predict_with_window_inds_and_ys(window_eval_set)



    preds_per_trial= trial_preds_from_window_preds(pred_win['preds'], pred_win['i_window_in_trials'], pred_win['i_window_stops'])
    mean_preds_per_trial = [np.mean(preds, axis=1) for preds in
                                    preds_per_trial]
    mean_preds_per_trial = np.array(mean_preds_per_trial)
    y = window_eval_set.description['pathological']
    column0, column1 = "non-pathological", "pathological"
    a_dict = {column0: mean_preds_per_trial[:, 0],
              column1: mean_preds_per_trial[:, 1],
              "true_pathological": y}

    assert len(y) == len(mean_preds_per_trial)

    # store predictions
    pd.DataFrame.from_dict(a_dict).to_csv(result_path + "predictions_eval_" + str(model_number) +
                                              ".csv")

    
    deep_preds =  mean_preds_per_trial[:, 0] <=  mean_preds_per_trial[:, 1]
    class_label = window_eval_set.description['pathological']
    class_preds =deep_preds.astype(int)



    fig = plot_confusion_matrix_paper(confusion_matrix(class_label, class_preds),class_names=['normal', 'abnormal'])
    fig.savefig(result_path + 'confusion_matrix_eval.png')



    
    del model, clf, window_train_set



