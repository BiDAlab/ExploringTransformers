
# Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition

![Header](./Images/AllTransformers.png)

# Overview

The present article intends to explore and propose novel behavioural biometric systems based on Transformers. 

Transformers are more recently proposed DL architectures that have already garnered impmense interest due to their effectiveness across a range of application domains such as language assessment, vision, and reinforcement learning. Their main advantages compared with traditional CNN and RNN architectures are: *i)* Transformers are feed-forward models that process all the sequences in parallel, therefore increasing efficiency; *ii)* They apply Self-Attention/Auto-Correlation mechanisms that allows them to operate in long sequences; *iii)* They can be trained efficiently in a single batch since all the sequence is included in every batch; and *iv)* They can attend to the whole sequence, instead of summarising all the previous temporal information.

To the best of our knowledge, this is the first study that explores the potential of Transformers for behavioural biometrics, in particular, gait biometric recognition on mobile devices. Several state-of-the-art Transformer architectures are considered in the evaluation framework (Vanilla, Informer, Autoformer, Block-Recurrent Transformer, and THAT), comparing them with traditional CNN and RNN architectures. In addition, new configurations of the Transformers are proposed to further improve the performance.

# Experimental Protocol

In this repository we include the experimental protocol followed in our experiments for the two popular public databases whuGAIT [\[2\]](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones) and OU-ISIR [\[3\]](https://www.sciencedirect.com/science/article/pii/S003132031300280X). It was presented by Zou *et al.* in [\[2\]](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones), being a predefined division of the database into development and evaluation datasets in order to facilitate the comparison among approaches.


# Benchmark Evaluation on Gait Recognition

The proposed Transformer has outperformed previous Transformer architectures and traditional DL architectures (i.e., CNNs, RNNs, and CNNs + RNNs) when evaluated using both databases whuGAIT [\[2\]](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones) and OU-ISIR [\[3\]](https://www.sciencedirect.com/science/article/pii/S003132031300280X). In particular, for the challenging OU-ISIR database, the proposed Transformer achieves 93.33% accuracy, resulting in accuracy absolute improvements compared with other techniques of 7.59% (THAT), 28.81% (Block-Recurrent Transformer), 30.23% (Autoformer), 33.93% (Informer), and 38.82% (Vanilla Transformer). The proposed Transformer has also been compared with state-of-the-art gait biometric recognition systems, outperforming the results presented in the literature. In addition, it is important to highlight the enhanced time complexity and memory usage of the proposed Transformer compared with traditional DL models.

| Model | Databases |
| ---| --- | --- | --- | --- | --- | --- |
| 30 | TypeNet [\[4\]](https://ieeexplore.ieee.org/document/9539873) | 14.20 | 12.50 | 11.30 | 10.90 | 10.50 |
| 30 | **TypeFormer** [\[1\]](https://arxiv.org/abs/2212.13075) | **9.48** | **7.48** | **5.78** | **5.40** | **4.94** |
| 50 | TypeNet [\[4\]](https://ieeexplore.ieee.org/document/9539873) | 12.60 | 10.70 | 9.20 | 8.50 | 8.00 |
| 50 | Preliminary Transformer [\[3\]](https://arxiv.org/abs/2212.13075) | 6.99 | - | 3.84 | - | 3.15 |
| 50 | **TypeFormer** [\[1\]](https://arxiv.org/abs/2212.13075) | **6.17** | **4.57** | **3.25** | **2.86** | **2.54** |
| 70 | TypeNet [\[4\]](https://ieeexplore.ieee.org/document/9539873) | 11.30 | 9.50 | 7.80 | 7.20 | 6.80 |
| 70 | **TypeFormer** [\[1\]](https://arxiv.org/abs/2212.13075) | **6.44** | **5.08** | **3.72** | **3.30** | **2.96** |
| 100 | TypeNet [\[4\]](https://ieeexplore.ieee.org/document/9539873) | 10.70 | 8.90 | 7.30 | 6.60 | 6.30 |
| 100 | **TypeFormer** [\[1\]](https://arxiv.org/abs/2212.13075) | **8.00** | **6.29** | **4.79** | **4.40** | **3.90** |



# Access and Code

# Citation
If you use DeepFakesON-Phys please cite:

```
@article{delgado2022exploring,
  title={{Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition}},
  author={Delgado-Santos, Paula and Tolosana, Ruben and Guest, Richard and Deravi, Farzin and Vera-Rodriguez, Ruben},
  journal={Under Review in Pattern Recognition},
  year={2022}
}

```

# Contact

If you have any questions, please contact us at [paula.delgado-de-santos@kent.ac.uk](mailto:paula.delgado-de-santos@kent.ac.uk) or [ruben.tolosana@uam.es](mailto:ruben.tolosana@uam.es)
