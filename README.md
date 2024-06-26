
# Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition

![Header](./Images/AllTransformers.png)

# Overview

This repo provides the code and results from the paper ["Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition"](https://www.sciencedirect.com/science/article/pii/S003132032300496X) published in Pattern Recognition (2023).

Our article explores and proposes novel behavioural biometric systems based on Transformers.

Transformers are more recently proposed DL architectures that have already garnered impmense interest due to their effectiveness across a range of application domains such as language assessment, vision, and reinforcement learning. Their main advantages compared with traditional CNN and RNN architectures are: *i)* Transformers are feed-forward models that process all the sequences in parallel, therefore increasing efficiency; *ii)* They apply Self-Attention/Auto-Correlation mechanisms that allows them to operate in long sequences; *iii)* They can be trained efficiently in a single batch since all the sequence is included in every batch; and *iv)* They can attend to the whole sequence, instead of summarising all the previous temporal information.

To the best of our knowledge, this is the first study that explores the potential of Transformers for behavioural biometrics, in particular, gait biometric recognition on mobile devices. Several state-of-the-art Transformer architectures are considered in the evaluation framework (Vanilla, Informer, Autoformer, Block-Recurrent Transformer, and THAT), comparing them with traditional CNN and RNN architectures. In addition, new configurations of the Transformers are proposed to further improve the performance.

# Experimental Protocol

In this repository we include the experimental protocol followed in our experiments for the two popular public databases whuGAIT [\[2\]](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones) and OU-ISIR [\[3\]](https://www.sciencedirect.com/science/article/pii/S003132031300280X). It was presented by Zou *et al.* in [\[2\]](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones), being a predefined division of the database into development and evaluation datasets in order to facilitate the comparison among approaches.


# Benchmark Evaluation on Gait Recognition

The proposed Transformer has outperformed previous Transformer architectures and traditional DL architectures (i.e., CNNs, RNNs, and CNNs + RNNs) when evaluated using both databases whuGAIT [\[2\]](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones) and OU-ISIR [\[3\]](https://www.sciencedirect.com/science/article/pii/S003132031300280X). In particular, for the challenging OU-ISIR database, the proposed Transformer achieves 93.33% accuracy, resulting in accuracy absolute improvements compared with other techniques of 7.59% (THAT), 28.81% (Block-Recurrent Transformer), 30.23% (Autoformer), 33.93% (Informer), and 38.82% (Vanilla Transformer). The proposed Transformer has also been compared with state-of-the-art gait biometric recognition systems, outperforming the results presented in the literature. In addition, it is important to highlight the enhanced time complexity and memory usage of the proposed Transformer compared with traditional DL models.

![Header](./Images/TableResults.png)


# Dependences 

`conda=22.9.0`

`CUDA`

`numpy=1.24.1`

`python=3.9.7`

`torch=1.11.0`


# Code

We provide the evaluation scripts together with their pre-trained models in this repo. 
<!--We provide the evaluation scripts together with their pre-trained models in this repo. -->

| whuGAIT | OU-ISIR |
| --- | --- |
| [Vanilla Transformer](./Code/whuGAIT/VanillaTransformer_whuGAIT_Evaluation.py) | [Vanilla Transformer](./Code/OUISIR/VanillaTransformer_OUISIR_Evaluation.py) |
| [Informer](./Code/whuGAIT/Informer_whuGAIT_Evaluation.py) | [Informer](./Code/OUISIR/Informer_OUISIR_Evaluation.py) |
| [Autoformer](./Code/whuGAIT/Autoformer_whuGAIT_Evaluation.py) | [Autoformer](./Code/OUISIR/Autoformer_OUISIR_Evaluation.py) |
| [Block-Recurrent Transformer](./Code/whuGAIT/BlockRecurrentTransformer_whuGAIT_Evaluation.py) | [Block-Recurrent Transformer](./Code/OUISIR/BlockRecurrentTransformer_OUISIR_Evaluation.py) |
| [THAT](./Code/whuGAIT/THAT_whuGAIT_Evaluation.py) | [THAT](./Code/OUISIR/THAT_OUISIR_Evaluation.py) |
| [Proposed Transformer](./Code/whuGAIT/ProposedTransformer_whuGAIT_Evaluation.py) | [Proposed Transformer](./Code/OUISIR/ProposedTransformer_OUISIR_Evaluation.py) |


# Citation

If you use our code please cite:

```
@article{delgado2023exploring,
  title={{Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition}},
  author={Delgado-Santos, Paula and Tolosana, Ruben and Guest, Richard and Deravi, Farzin and Vera-Rodriguez, Ruben},
  journal={Pattern Recognition},
  volume = {143},
  pages = {109798},
  year = {2023}
}

```

# Contact

If you have any questions, please contact us at [paula.delgadodesantos@telefonica.com](mailto:paula.delgadodesantos@telefonica.com) or [ruben.tolosana@uam.es](mailto:ruben.tolosana@uam.es).
