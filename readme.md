# PyTorch Implementation of  
## Gaussian Adaptive Patching Powered Fully-Connected Spatial-Temporal Graph for Multivariate Time-Series Data

**Authors:**  
[Yucheng Wang](https://frank-wang-oss.github.io), [Yuecong Xu](https://xuyu0010.github.io/), [Jianfei Yang](https://marsyang.site/),  [Min Wu](https://sites.google.com/site/wumincf/), [Xiaoli Li](https://personal.ntu.edu.sg/xlli/), [Lihua Xie](https://personal.ntu.edu.sg/elhxie/), [Zhenghua Chen](https://zhenghuantu.github.io/)

<!-- # :boom: Our paper has been accepted for publication of AAAI 2024 (acceptance rate 23.75%). -->

---

## Overview

This repository provides the official PyTorch implementation of our proposed framework for multivariate time-series (MTS) modeling.

This work extends our previous AAAI 2024 paper:  
👉 [FC-STGNN](https://github.com/Frank-Wang-oss/FCSTGNN)

---

## Requirements

### Environment

We provide the exact environment configuration used in our experiments. You may reproduce the same setup using the following command:

```bash
conda env create -f environment.yml
```

Alternatively, you may refer to the provided environment file as a reference for configuring your own setup.

### Hardware (Recommended)

- A GPU for efficient training  
- CUDA and compatible NVIDIA drivers  

> ⚠️ Training on CPU is possible but may be extremely time-consuming.

---

## Abstract

Multivariate Time-Series (MTS) data is fundamental across diverse domains due to its sequential and multi-source nature (e.g., multi-sensor systems). Such data inherently exhibits **spatial-temporal (ST) dependencies**, including temporal correlations over time and spatial correlations across sensors.

While Graph Neural Networks (GNNs) have been widely adopted for MTS modeling, most existing approaches capture spatial and temporal dependencies separately, overlooking correlations between **different sensors at different time steps (DEDT)**. This limitation restricts comprehensive ST dependency modeling and hinders effective representation learning.

To address this issue, we propose a **Fully-Connected Spatial-Temporal Graph Neural Network (FC-STGNN)**, which introduces:

- **Fully-connected graph construction**, linking all sensors across all time steps with temporally-aware edge weighting  
- **Fully-connected graph convolution**, leveraging moving-pooling GNN layers to capture complex ST dependencies  

However, FC-STGNN relies on fixed-size patching, which may limit its ability to capture optimal local patterns. To overcome this, we further propose **GAP-STGNN**, which incorporates:

- **Gaussian Adaptive Patching (GAP)** for dynamically learning patches with adaptive receptive fields  
- **Adaptive patch selection**, emphasizing informative regions while preserving temporal continuity  

Extensive experiments across multiple MTS datasets demonstrate that FC-STGNN and GAP-STGNN achieve superior performance by effectively modeling comprehensive ST dependencies.

---

## Dataset

We evaluate our methods on both **regression** and **classification** tasks.

---

### Regression Tasks

We use sub-datasets from **C-MAPSS**.

- Download dataset:  
  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

- Place the data in:
  ```
  datasets/CMAPSSData
  ```

- Run experiments:
  ```bash
  python main_REG.py
  ```

---

### Classification Tasks

We evaluate on the following datasets:

- UCI-HAR  
- ISRUC-S3  
- WISDM  
- Opportunity HAR  
- UEA datasets  

Place all processed datasets under:
```
datasets/
```

- Run experiments:
```bash
python main_CLS.py
```

---

## Dataset Preparation

### UCI-HAR

- Download:  
  https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones  

- Preprocess:
```bash
python Data_preprocessing/preprocess_UCI_HAR.py
```

---

### ISRUC-S3

- Download:  
  https://sleeptight.isr.uc.pt/ (select S3)

- Preprocess:
```bash
python Data_preprocessing/preprocess_ISRUC.py
```

---

### WISDM

- Download (processed dataset):  
  https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B  

---

### Opportunity HAR

- Download:  
  https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition  

- Preprocess:
```bash
python Data_preprocessing/preprocess_opphar.py
```

---

### UEA Datasets

- Download:  
  https://www.timeseriesclassification.com/

- Preprocess:
```bash
python Data_preprocessing/preprocess_UEA.py
```

---

## Acknowledgements

We thank the authors of the following repositories for providing preprocessing code:

- UCI-HAR: https://github.com/emadeldeen24/TS-TCC  
- ISRUC-S3: https://github.com/ziyujia/MSTGCN  

---

<!-- ## Citation

If you find this work useful, please consider citing:

@article{Wang2023FullyConnectedSG,
  title={Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data},
  author={Yucheng Wang and Yuecong Xu and Jianfei Yang and Min Wu and Xiaoli Li and Lihua Xie and Zhenghua Chen},
  journal={ArXiv},
  year={2023},
  volume={abs/2309.05305}
}
-->
