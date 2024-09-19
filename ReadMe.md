# Open-set Plankton Recognition

Implementation of the open-set recognition methods from the paper "Open-set Plankton Recognition".

## Overview
This repository contains implementations of three open-set recognition methods:
- **OpenMax** 
    - A. Bendale and T. Boult, *"Towards Open Set Deep Networks,"* CVPR 2016
- **ArcFace** 
    - J. Deng, et al. *"Arcface: Additive Angular Margin Loss for Deep Face Recognition,"* CVPR 2019
    - O. Mohamed, et al. *"Open-Set Plankton Recognition Using Similarity Learning,"* ISVC 2022
- **Class Anchored Clustering**
    - D. Miller, et al. *"Class Anchor Clustering: A Loss for Distance-Based Open Set Recognition,"* WACV 2021

## Usage

### Datasets
The codes utilize both [SYKE-plankton_IFCB_2022](hhttps://doi.org/10.23728/b2share.abf913e5a6ad47e6baa273ae0ed6617a) and [SYKE-plakton_ZooScan_2024](https://doi.org/10.23729/fa115087-2698-4aa5-aedd-11e260b9694d) datasets.

Initial step is to download both datasets and move them inside data folder if the given data-processing files are used




## Citation

```text
@inproceedings{open-set-plankton,
  title={Open-set Plankton Recognition},
  author={kareinen, Joona and Skyttä, Annaliina and Eerola, Tuomas and Kraft, Kaisa and Lensu, Lasse and Suikkanen, Sanna and Lehtiniemi, Maiju and Kälviäinen, Heikki},
  booktitle={ECCV2024},
  pages={???}
}
```