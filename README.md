# TUHAbnormal-Expansion-dataset

This repository contains resources that were used for our study entitled

"An Extended Clinical EEG Dataset with 15,300 Automatically Labelled Recordings for Pathology Decoding".

We present a novel curation of the publicly available TUH EEG Corpus called the TUH Abnormal Expansion EEG Corpus (TUABEX) and its balanced counterpart, the TUH Abnormal Expansion Balanced EEG Corpus (TUABEXB), which we used to train and test four established deep convolutional neural networks (ConvNets) for pathological versus non-pathological classification. It can also be used to train and test other feature-based or end-to-end machine learning apporaches.

## Contents

- dataset_description/ - description of all datasets and train/test data splits. Each description contains the path of the recordings used and the corresponding pathology labels based on automated classification of the medical text reports. 

- code/ - Python scripts and notebooks for processing of the text files and training pipeline of the convolutional neural networks.


## Environments

This repository expects a working installation of [braindecode](https://github.com/braindecode/braindecode).  
Additionally, it requires to install packages listed in `requirements.txt`. So download the reposiory and install:

```
conda env create -f environment.yml
```

## Data

Our study is based on the Temple University Hospital Abnormal EEG Corpus (v2.0.0) and Temple University Hospital EEG Corpus (v1.1.0 and v1.2.0) avilable for download at: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml




## Citing

If you use this code in a scientific publication, please cite us as:

```

@article{kiessner2023tuabexb,
  title={An Extended Clinical EEG Dataset with 15,300 Automatically Labelled Recordings for Pathology Decoding},
  author={Kiessner, Ann-Kathrin and Schirrmeister, Robin Tibor and Gemein, Lukas and Boedecker, Joschka and Ball, Tonio},
  journal={Available at SSRN 4387102},
  doi={http://dx.doi.org/10.2139/ssrn.4387102 }
}

```
