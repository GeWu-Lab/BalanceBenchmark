# BalanceBenchmark: A Survey for Imbalanced Learning

## Paper
## Overview
![](images/frame6_00.png)

Learning multimodal representations involves integrating information from multiple heterogeneous sources of data. However, it is often hindered by the multimodal imbalance problem, where certain modalities dominate while others remain underutilized. To facilitate addressing this challenge, we release **BalanceMM**, a standardized toolkit that implements various multimodal imbalance learning methods.

BalanceMM provides an automated end-to-end pipeline that simplifies and standardizes data loading, experimental setup, and model evaluation. The toolkit is designed to support research across different multimodal imbalance learning strategies, including data processing, feed-forward adjustments, objective modifications, and optimization approaches.

## Datasets currently supported
+ Audio-Visual: KineticsSounds, CREMA-D, BalancedAV, VGGSound
+ RGB-Optical Flow: UCF-101
+ Image-Text: FOOD-101
+ Audio-Visual-Text: CMU-MOSEI

To add a new dataset:

1. Go to balancemm/datasets/
2. Create a new Python file and a new dataset class
3. Implement the required data loading and preprocessing methods in the corresponding .py file
4. Add configuration file in balancemm/configs/dataset_config.yaml

## Algorithms currently supported
+ Data-level methods: Modality-valuation
+ Feed-forward methods: MLA, OPM, Greedy, AMCo
+ Objective methods: MMCosine, UMT, MBSD, CML, MMPareto, GBlending, LFM
+ Optimization methods: OGM, AGM, PMR, Relearning, ReconBoost

See Section 3 in our paper for detailed descriptions of each method.

![](images/Algorithms.jpeg)

To add a new method:

1. Determine which category your method belongs to:
  + data_methods/ : methods that adjust data processing
  + forward_methods/ : methods that modify network architecture
  + objective_methods/ : methods that adapt learning objectives
  + optimization_methods/ : methods that adjust optimization process
2. Go to balancemm/trainer/
3. Create a new Python file implementing your method
4. Implement the corresponding .py file based on /base_trainer.py, you should rewrite trainer.training_step usually.
5. Other implementation by your method's category:
  + If your method belongs to "Data-level", go to balancemm/datasets/__init.py and modify properly.
  + If your method belongs to "Feed-forward", go to balancemm/models/avclassify_model.py, create a new model class and rewrite specific functions.
  + If your method belongs to "Objective", you mostly don't have to do other modification except traienr.
  + If your method belongs to "Optimization", you may need to modify any combination of the parts mentioned above.
6. Add configuration file in balancemm/configs/trainer_config.yaml
