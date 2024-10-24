
This repository is the implementation for the paper 
# [Learning Similarity Metrics for Volumetric Simulations with Multiscale CNNs](https://arxiv.org/abs/2202.04109)




-----------------------------------------------------------------------------------------------------

## Installation
In the following, Linux is assumed as the OS but the installation on Windows should be similar. First, clone this repository to a destination of your choice.
```
git clone https://github.com/tum-pbs/VOLSIM
cd VOLSIM
```
## Basic Usage
To evaluate the metric on two numpy arrays `arr1, arr2` you only need to load the model and call the `computeDistance` method. Supported input shapes are `[width, height, depth, channels]` or `[batch, width, height, depth, channels]`, with one or three channels.
```python
from volsim.distance_model import *
model = DistanceModel.load("models/VolSiM.pth", useGPU=True)
dist = model.computeDistance(arr1, arr2, normalize=True, interpolate=False)
# resulting shapes: input -> output
# [width, height, depth, channel] -> [1]
# [batch, width, height, depth, channel] -> [batch]
```
The input processing can be modified via the parameters `normalize` and `interpolate`. The `normalize` argument indicates that both input arrays will be normalized to `[-1,1]` via a min-max normalization. In general, this setting is recommended as the metric CNN was trained on this value range, but if the data is already normalized before, it can be omitted. The `interpolate` argument determines if both inputs are interpolated to the input size of `64x64x64` on which the network was trained via a cubic spline interpolation. Since the model is fully convolutional, different input shapes are possible as well, and we determined that the metric still remains stable for spatial input dimensions between `32x32x32 - 128x128x128`. Outside this range the model performance may drop, and too small inputs can cause issues as the feature extractor spatially reduces the input dimensions.

The resulting numpy array `dist` contains distance values with shape `[1]` or `[batch]` depending on the shape of the inputs. If the evaluation should only use the CPU, set `useGPU=False` when loading the model. A simple example is shown in `distance_example_simple.py`, and `distance_example_detailed.py` shows a more advanced usage with a correlation evaluation. To run these examples use:
```
python src/distance_example_simple.py
python src/distance_example_detailed.py
```


-----------------------------------------------------------------------------------------------------

## Data Generation, Download, and Processing

### Downloading our data
Use this: [https://doi.org/10.14459/2023mp1703144](https://doi.org/10.14459/2023mp1703144). Use this command to directly download all data sets (**rsync password: m1703144**):
```
rsync -P rsync://m1703144@dataserv.ub.tum.de/m1703144/* ./data
```
It is recommended to check the .zip archives for corruption, by comparing the SHA512 hash of each downloaded file that can be computed via
```
sha512sum data/*.zip
```
with the corresponding content of the checksum file downloaded to `data/checksums.sha512`. If the hashes do not match, restart the download or try a different download method. Once the download is complete, the data set archives can be extracted with:
```
unzip -o -d data "data/*.zip"
```



### General Data Post-Processing
`plot_data_vis.py` contains simple plotting functionality to visualize individual data samples and the corresponding ground truth distances. `copy_data_lowres.py` can be used to downsample the generation resolution of `128x128x128` to the training and evaluation resolution of `64x64x64`. It processes all .npz data files, while creating copies of all supplementary files in the input directory.

To process custom raw simulation data, `compute_nonlinear_dist_coef.py` can be used to compute the nonlinear distance coefficients that are required for the ground truth distances from the proposed entropy-based similarity model. It creates a .json file with a path to each data file and a corresponding distance coefficient value.



-----------------------------------------------------------------------------------------------------

## Metric Comparison
With the downloaded data sets, the performance of different metrics (element-wise and CNN-based) can be compared using the metric evaluations in `eval_metrics_shallow_tb.py` and `eval_metrics_trained_tb.py`:
```
python src/eval_metrics_shallow_tb.py
python src/eval_metrics_trained_tb.py
```

## Re-training the Model
The metric can be re-trained from scratch with the downloaded data sets via `training.py`:
```
python src/training.py
```

## Backpropagation through the Metric
Backpropagation through the metric network is straightforward by integrating the `DistanceModel` class that derives from `torch.nn.Module` in the target network. Load the trained model weights from the model directory with the `load` method in `DistanceModel` on initialization (see Basic Usage above), and freeze all trainable weights of the metric if required. In this case, the metric model should be called directly (with appropriate data handling beforehand) instead of using `computeDistance` to perform the comparison operation. An example for this process based on a simple Autoencoder can be found in `backprop_example.py`:
```
python src/backprop_example.py
```
