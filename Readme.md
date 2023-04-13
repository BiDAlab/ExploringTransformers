
# Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition

![Header](./Images/AllTransformers.png)


# Overview

# Benchmark Evaluation on Gait Recognition

# Experimental Protocol

#Access and Code

The new proposed Transformer based on a selection of the best components presented in previous Transformer architectures. 

![]([https://drive.google.com/file/d/1PLjDx34iFugNUUxlw3CuMT8eo0-woYb8/view?usp=sharing](https://drive.google.com/file/d/1PLjDx34iFugNUUxlw3CuMT8eo0-woYb8/view?usp=share_link))

First, we consider a parallel two-stream architecture with Temporal and Channel Modules, similar to the THAT approach presented in [4]. Unlike the THAT model, we consider a Gaussian range encoding as input of both Temporal and Channel Modules. In addition, for the Temporal Module (left branch), we consider a combination of multi-head Auto-Correlation layers, proposed in Autoformer [5], and a recurrent layer in between, proposed in Block-Recurrent Transformer [6]. For the multi-head Auto-Correlation layer, we design a specific multi-scale Gait Biometric Recognition (GBR) CNN sub-layer. Regarding the Channel Module (right branch), we consider a multi-head Auto-Correlation sub-layer together with a multi-scale GBR CNN sub-layer. After each sub-layer, a residual connection is applied followed by a normalisation of the layer, similar to the Vanilla Transformer [7]. The time complexity and memory usage of each layer with Auto-Correlation is O(L log L), whereas for the recurrent layer this is O(L).


<!---# Benchmark Evaluation of our proposed Transformer

We analyse the performance of TypeFormer over an evaluation set of *U* = 1000 subjects unseen in the training and validation phases. The metric chosen for evaluation is the Equal Error Rate (EER). 

We consider a fixed number of 15 acquisition sessions per subject. Out of these, we use a variable number of enrolment sessions (*E* = 1, 2, 5, 7, 10) in order to assess the performance adaptation of the system to reduced availability of enrolment data. Additionally, also the experiments are repeated changing the input sequence length, *L* = 30, 50, 70, 100, to evaluate the optimal keystroke sequence length.

The table below reports the results obtained by TypeFormer in comparison with two recently proposed keystroke verification studies. In [\[3\]](https://arxiv.org/abs/2212.13075), a different Transformer-based architecture was proposed as a preliminary version of the current work. In [\[4\]](https://ieeexplore.ieee.org/document/9539873), TypeNet, a Long Short Term Memory Recurrent Neural Network, was proposed.

The results contained in the table are expressed in terms of EER (%), and obtained according to the same experimental protocol, data subjects, and data acquisition sessions (corresponding to Table 2 in [\[1\]](https://arxiv.org/abs/2212.13075)). 

| Sequence Lenght *L* | Model | *E* = 1 | *E* = 2 | *E* = 5 | *E* = 7 | *E* = 10 |
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


# Experimental Protocol
The genuine and impostor score distributions are subject-specific. 

For each subject, genuine scores are obtained comparing the number enrolment sessions (*E*) with 5 verification sessions. The Euclidean distances are computed for each of the verification sessions with each of the *E* enrolment sessions, and then values are averaged over the enrolment sessions. Therefore, for each subject there are 5 genuine scores, one for each verification session. 

Concerning the impostor score distribution, for every other subject in the evaluation set, the averaged Euclidean distance value is obtained considering 1 verification session and the above-mentioned 5 enrolment sessions. Consequently, for each subject, there are 999 impostor scores. Based on such distributions, the EER score is calculated per subject, and all EER values are averaged across the entire evaluation set. 

# Data Subjects and Data Acquisition Sessions Used for Evaluation

For each subject, the enrolment sessions are the chosen in a orderly fashion from the first 10 sessions. For *E* = 1, the enrolment session chosen will be the first one. For *E* = 2, the enrolment sessions will be the first two, and so on. The verification sessions selected are always the last 5 sessions out of the 15 sessions per subject considered. 

All data sessions used for evaluation, separated by subject, are reported in the "TypeFormer_benchmark_sessions.json" file uploaded. Each key corresponds to a user identified by their "PARTICIPANT_ID" in the raw data of the Aalto Mobile Keystroke Database. For each user keys, each of the list elements correspond to the "TEST_SECTION_ID" of each of the acquisition sessions in the raw data.--->


# References

[\[1\] *Paula Delgado-Santos, Ruben Tolosana, Richard Guest, Ruben Vera-Rodriguez, and Farzin Deravi, “Exploring Transformers for Behavioural Biometrics: A Case Study in Gait Recognition”, arXiv:2206.01441, 2022.*](https://arxiv.org/pdf/2206.01441.pdf)


**Contact: [paula.delgado-de-santos@kent.ac.uk](mailto:paula.delgado-de-santos@kent.ac.uk) or [ruben.tolosana@uam.es](mailto:ruben.tolosana@uam.es)**
