# WGAST : Weakly-Supervised Generative Network for Daily 10 m Land Surface Temperature Estimation via Spatio-Temporal Fusion

## Description

<img src="https://github.com/Sofianebouaziz1/WGAST/blob/main/images/WGAST_generator.jpg" width="50%" align="right"/>
<div style="text-align: justify;">
<strong>WGAST</strong> (Weakly-Supervised Generative Network for Daily 10 m Land Surface Temperature Estimation via Spatio-Temporal Fusion) is a novel deep learning framework for spatio-temporal fusion of satellite images to estimate Land Surface Temperature (LST) at 10 m resolution on a daily basis. WGAST addresses the trade-off between spatial and temporal resolution in remote sensing by combining observations from Terra MODIS, Landsat 8, and Sentinel-2. It is built on a conditional generative adversarial architecture and integrates multi-level feature extraction, cosine similarity, normalization, temporal attention mechanisms, and noise suppression within an end-to-end design. WGAST is trained using a weakly-supervised strategy based on physical principles and adversarial learning, and demonstrates strong performance in recovering high-resolution thermal patterns, improving accuracy and robustness over existing methods.
</div>

[**Features**](#Features)
| [**Tutorials**](https://github.com/Sofianebouaziz1/WGAST/tree/main/tutorials)
| [**Structure**](#Code-structure)
| [**Paper**]
| [**ArXiv**]
| [**References**](#How-to-cite)


## Features

WGAST framework offers the following features:
* A novel non-linear generative model specifically tailored for STF of LST, enabling accurate daily estimation at 10 m resolution by integrating coarse 1 km Terra MODIS data with complementary spectral information from multiple satellite RS platforms.
* An effective use of Landsat 8 as an intermediate-resolution bridge, overcoming the large resolution gap between Terra MODIS (1 km) and Sentinel-2 (10 m) to enable more stable and accurate fusion.
* A physically motivated weak supervision strategy that leverages 30 m Landsat-derived LST as proxy ground truth, to bypass the challenge of missing ground truth data at 10 m resolution.
* A training design that avoids dependence on future observations by relying solely on a previous reference date, allowing real-time applicability.
* A significant reduction in cloud-induced gaps at 10 m resolution by leveraging the temporal resilience of Terra MODIS observations.
* Rigorous validation on both satellite-based and in-situ ground measurements, demonstrating WGAST’s superior accuracy, robustness, and generalization compared to existing spatio-temporal fusion methods.

## Paper
WGAST's paper has been submitted to the IEEE Transactions on Geoscience and Remote Sensing (TGRS). Please refer to the arXiv [here] version for the full paper.

## Requirements
WGAST has been implemented and tested with the following versions: 

- Python (v3.12.4).
- Pytorch (v2.5.0).
- Scipy (v1.14.1).
- Earthengine-api (ee) (v1.1.2).
- Geemap (v0.34.5).
- NumPy (v2.0.1).
- Pandas (v2.2.3).
- Rasterio (v1.14.1).

## Code structure

```
WGAST/
├── data_download/ --- Scripts for downloading satellite data from Google Earth Engine
│   ├── Landsat8Processor.py --- Download Landsat 8 data
│   ├── MODISProcessor.py --- Download Terra MODIS data
│   └── Sentinel2Processor.py --- Download Sentinel-2 data
│
├── data_loader/ --- Define the data loader structure and utilities
│   ├── data.py --- Main data loader definition
│   └── utils.py --- Helper functions for loading and processing
│
├── data_preparation/ --- Preprocessing and building Terra MODIS, Landsat 8, and Sentinel-2 triples
│   ├── DataProcessor.py --- Clean and prepare raw satellite data
│   └── GetTriple.py --- Create temporally aligned triples from processed data
│
├── model/ --- Define the WGAST model architecture
│   └── WGAST.py --- Implementation of the WGAST deep learning model
│
├── runner/ --- Manage training and testing phases of the WGAST model
│   └── experiment.py --- Train and test the model
│
└── tutorials/ --- A series of tutorials for each stage of the pipeline
    ├── 01_data_download.ipynb --- Tutorial 01: Downloading satellite data from Google Earth Engine
    ├── 02_data_preparation.ipynb --- Tutorial 02: Preprocessing and building data triples
    ├── 03_data_structuring.ipynb --- Tutorial 03: Structuring and preparing datasets for training
    └── 04_run_model.ipynb --- Tutorial 04: Running training and testing of WGAST
```

## Experimental results

### Quantitative Results

The following table summarizes the results we obtained by comparing WGAST with BicubicI, Ten-ST-GEE, and FuseTen, using multiple standard metrics across four different dates. These metrics include RMSE, SSIM, PSNR, SAM, CC, and ERGAS.

<div style="text-align:center;">
  <img src="https://github.com/Sofianebouaziz1/WGAST/blob/main/figures/Quantitative_results.png" width="100%"/>
</div>

These results highlight the effectiveness of WGAST in achieving a strong trade-off between reducing reconstruction error and preserving quality. In most cases, WGAST outperforms prior approaches, particularly in RMSE, SSIM, PSNR, and ERGAS, validating its robustness and generalizability across diverse temporal scenes.

.



