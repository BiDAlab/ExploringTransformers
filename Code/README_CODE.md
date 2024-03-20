Code and explanation:
Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition

This document explains the code used for the above paper. The system details are included in Section V. 
In this folder you will find the files separated by database (whuGAIT and OUISIR).

Each folder contains: 

-	Train/Validation/Test files(_training/validation/testing_dataset_database.pt_):
  +	These files are created from the original data with GaitDataset.py to have it in PyTorch
- PreTrained Models:
  + Vanilla Transformer (_VanillaTransformer_database_)
  + Informer (_Informer_database_)
  + Autoformer (_Autoformer_database_)
  + Block-Recurrent Transformer (_BlockRecurrentTransformer_database_)
  + THAT (_THAT_database_)
  + Proposed Transformer (_ProposedTransformer_database_) 
- Evaluation files:
  + Vanilla Transformer (_VanillaTransformer_database_Evaluation.py_)
  + Informer (_Informer_database_Evaluation.py_)
  + Autoformer (_Autoformer_database_Evaluation.py_)
  + Block-Recurrent Transformer (_BlockRecurrentTransformer_database_Evaluation.py_)
  + THAT (_THAT_database_Evaluation.py_)
  + Proposed Transformer (_ProposedTransformer_database_Evaluation.py_) 
- Training files:
  + Proposed Transformer (_ProposedTransformer_database_Training.py_)
  + Please, first download the training dataset (_training_dataset_database.py_) as indicated in _data/README_DATA.md_
- Layers and utils needed (layers and utils folders, _BlockRecurrentTransformer.py_, and _Models_Transformers.py_)
  
where database is (_whuGAIT_ or _OUISIR_).

If you have any questions, please contact us at paula.delgadodesantos@telefonica.com or ruben.tolosana@uam.es 
