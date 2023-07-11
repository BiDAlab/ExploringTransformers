Code and explanation:
Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition

This document explains the code used for the above paper. The system details are included in Section V. 
In this folder you will find the files separated by database (whuGAIT and OUISIR).

Each folder contains: 
•	Train/Validation/Test files (.pt) 
  o	These files are created from the original data with GaitDataset.py to have it in PyTorch
  o	PreTrained Models (Vanilla Transformer, Informer, Autoformer, Block-Recurrent Transformer, THAT, and the proposed Transformer)
  o	Evaluation files (Vanilla Transformer, Informer, Autoformer, Block-Recurrent Transformer, THAT, and the proposed Transformer)
  o	Training files (proposed Transformer)
  o	Layers and utils needed (layers and utils folders, BlockRecurrentTransformer.py, and Models_Transformers.py
  
If you have any questions, please contact us at paula.delgado-de-santos@kent.ac.uk or ruben.tolosana@uam.es 
