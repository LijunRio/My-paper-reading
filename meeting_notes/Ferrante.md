# Seminar: Enzo Ferrante - "Towards Robust Anatomical Segmentation of Medical Images"

**Date:** Monday, 27 April 2026, 11:00am – 12:00pm  
**Location:** NHB 3532 Room 229  
**Speaker:** Enzo Ferrante

## Overview

Deep learning models for medical image segmentation typically optimize pixel-level objectives, making it difficult to enforce global constraints on shape, topology, and spatial coherence. This talk presents research on building robust and anatomically plausible segmentation models along two interconnected axes:

1. **Hybrid Graph Neural Networks**: HybridGNet and HybridVNet architectures that combine convolutional encoders with graph-based decoders to produce landmark-based segmentations with built-in topological guarantees, including uncertainty quantification via variational formulations.

2. **Fairness & Bias in Medical Image Analysis**: Methods for unsupervised bias discovery based on Reverse Classification Accuracy (RCA) using the CheXmask dataset (676k+ chest X-rays), enabling detection of performance disparities across demographic subgroups without ground-truth annotations.

---

# Meeting Notes

## 1. Graph-based Segmentation (HybridGNet)

The work starts from graph-based segmentation (HybridGNet), where the goal is to move from pixel-wise prediction to structured anatomical representation. Instead of predicting masks directly, a CNN encoder extracts image features and a graph convolutional decoder predicts landmark nodes forming organ contours. The key design is the interaction between image space and graph space: ROIAlign samples image features at node locations, and skip connections inject multi-scale CNN features into the GNN. This allows the model to preserve anatomical structure while still leveraging rich image features.

## 2. Uncertainty Modeling (Variational Framework)

To model uncertainty, the framework is extended with a variational formulation. The encoder predicts a latent distribution parameterized by mean and variance, and sampling from this latent space generates multiple plausible shapes. Passing different samples through the graph decoder produces a distribution of landmark predictions, and node-wise variance is used as uncertainty. This uncertainty is then analyzed through controlled experiments: when Gaussian noise or occlusion is added to the input, the latent variance increases, and uncertainty becomes spatially localized in corrupted regions. A reliability analysis further shows that predicted uncertainty correlates with actual error, indicating that the model uncertainty is reasonably calibrated.

## 3. Anatomical Correspondence & Temporal Extension

The work also shows that the graph-based formulation implicitly learns correspondences across subjects, since nodes are consistently aligned anatomically. This is further extended to temporal data, where spatio-temporal regularization enforces consistency across frames, improving stability in cardiac sequences. In the 3D extension (HybridVNet), multi-view encoders (2D and 3D CNNs) are combined, and the graph decoder predicts 3D meshes instead of 2D contours, still using a variational latent space and graph convolutions.

## 4. Representation Learning for Imaging Genetics

Another line of work focuses on representation learning for imaging genetics. A graph autoencoder is used to learn low-dimensional latent representations of cardiac shape, which serve as phenotypes for downstream genetic association studies. This is further extended by disentangling static (structure) and dynamic (motion) components, enabling more precise identification of gene–phenotype relationships.

## 5. CheXmask: Large-scale Dataset & Evaluation

The CheXmask work shifts focus to large-scale data and evaluation. It constructs a dataset of around 676k chest X-rays with automatically generated segmentation masks and anatomical landmarks across multiple centers. Since ground truth annotations are not always available, Reverse Classification Accuracy (RCA) is used to estimate segmentation quality. The RCA pipeline takes a prediction, propagates it via registration to similar images with ground truth, and computes an average Dice score as a proxy for performance. This allows large-scale quality estimation without manual labels.

## 6. Fairness Analysis

Building on this, fairness analysis is performed by comparing RCA-estimated performance across subgroups such as sex, projection (AP vs PA), and age. The results show systematic differences, for example AP images tend to have lower estimated Dice scores due to acquisition difficulty and artifacts. Importantly, RCA is not modeling uncertainty directly; it provides an estimate of prediction error. In this setting, the observed error is largely driven by data-related factors such as image quality and domain differences.

---

# Questions / Research Interests

- Can error (RCA) be decomposed into epistemic vs. aleatoric components? Can model uncertainty replace RCA as a performance proxy?
- Is uncertainty consistent across subgroups (sex, projection, age)? Do acquisition conditions affect both performance and confidence estimation?
- Can node-level uncertainty distinguish anatomical ambiguity from model failure? Unified framework for data, model, and shape uncertainty?


# Fairness, Bias, and Evaluation in Vision and Language Models

Fairness, bias, and uncertainty.

Main intuition:

- Bias between gender / protected groups.
- Data dominance may increase bias.
- Imbalance is often considered for disease labels, but maybe people think less about protected attributes such as male/female, age, race, etc.
- Intersectional analysis: gender + race, etc.

## 1. Medical Imaging Fairness

Reference / example:

- Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis.
- Back to [2020](https://arxiv.org/abs/2005.10050), not many people looked at gender in this setting.

How to quantify or study gender imbalance:

- Train on the whole dataset, then test separately by male/female.
- Train on male-only / female-only data, then test separately and cross-test.
- Compare balanced data, e.g. 50/50 female and male, with extreme imbalance, e.g. 0 female when testing on female.
- Diversify dataset.

Question:

> What does imbalance cause? Maybe also epistemic uncertainty?

Should we train a classifier per gender?

- My feeling: probably not. Mixing may improve the model.

## 2. Fairness and Uncertainty

Possible connection:

- Aleatoric vs. epistemic uncertainty.
- "World bias" may appear as uncertainty or systematic error.
- Need to understand whether subgroup errors are from task difficulty, under-representation, or model bias.

Reference:

- Fairness of Deep Ensembles: On the Interplay Between Per-Group Task Difficulty and Under-Representation: https://arxiv.org/abs/2501.14551

Question:

- Does improving minority-group performance usually harm majority-group performance?

## 3. Balancing / Mitigation Ideas

How to balance:

- Add noise / dropout from one dominant modality.
- But adding noise is not real difficulty.

References:

- Addressing fairness in artificial intelligence for medical imaging: https://www.nature.com/articles/s41467-022-32186-3
- Fairness of AI in Medical Imaging MICCAI workshop: https://future-ai.eu

## 4. Fairness for Unobserved Characteristics

References:

- Fairness for Unobserved Characteristics: Insights from Technological Impacts on Queer Communities: https://arxiv.org/abs/2102.04257
- Implicit Bias in LLMs for Transgender Populations: https://arxiv.org/abs/2602.13253

Question:

- How to study fairness when the characteristic is unobserved or hard to label?

## 5. Implicit Bias in LLMs

Reference:

- Explicitly unbiased large language models still form biased associations: https://www.pnas.org/doi/10.1073/pnas.2416228122

Thoughts:

- Implicit bias may come from pretraining.
- The bias is human bias from the world.
- Association tests with words can expose implicit bias from LLMs. Is this data bias?
- Need to check how to calculate bias and the formula in the PNAS paper.

Decision connection:

- Bias to decision: agent disease part.
- If the model has bias, it may affect decisions. Need to find the bug/source.

## 6. Modality Bias in LVLMs

Reference:

- Modality Bias in LVLMs: Analyzing and Mitigating Object Hallucination via Attention Lens: https://arxiv.org/abs/2508.02419

Shift focus to text + image:

- The problem: if we give the image and add misleading text, what will happen?

Important:

- MIRAGE: The Illusion of Visual Understanding: https://arxiv.org/abs/2603.21687

EHR system question:

> Is more information always beneficial in multimodal EHR analysis?

Notes:

- Irrelevant context may make some models worse.
- The model may not find the correct context.
- Models are not good at disentangling useful information from irrelevant information.
- Need to find wrong context / misleading context.

## 7. Systematic Bias

Systematic bias may not transfer directly from only one place. It may come from:

- Model design.
- Data.
- Pretraining.
- SFT.

Research questions:

- Bias from the model during decision-making will affect the output. How to find the bug?
- Different models have different biases. Maybe the source is the data.
- How to make/measure model bias and find the problem?
