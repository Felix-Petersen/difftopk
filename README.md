# Differentiable Top-k Classification Learning

![difftopk_logo](difftopk_logo.png)

Official implementation for our ICML 2022 Paper "Differentiable Top-k Classification Learning".

The `difftopk` library provides different differentiable sorting and ranking methods as well as a wrapper for using them
in a `TopKCrossEntropyLoss`. `difftopk` builds on PyTorch.

Paper @ [ArXiv](https://arxiv.org/abs/2206.07290),
Video @ [Youtube](https://www.youtube.com/watch?v=J-lZV72DCic).

## 💻 Installation

`difftopk` can be installed via pip from PyPI with
```shell
pip install difftopk
```

### Sparse Computation

For the functionality of evaluating the differentiable topk operators in a sparse way, the package `torch-sparse` has to be installed.
This can be done, e.g., via
```shell
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```
For more information on how to install `torch-sparse`, see [here](https://github.com/rusty1s/pytorch_sparse#installation).

### Example for Full Installation from Scratch and with all Dependencies

<details>
  <summary>(<i>click to expand</i>)</summary>

Depending on your system, the following commands will have to be adjusted.

```shell
virtualenv -p python3 .env_topk
. .env_topk/bin/activate
pip install boto3 numpy requests scikit-learn tqdm 
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install diffsort

# optional for torch-sparse
FORCE_CUDA=1 pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

pip install .
```

</details>

---

## 👩‍💻 Documentation

The `difftopk` library provides of differentiable sorting and ranking methods as well as a wrapper for using them in a
`TopKCrossEntropyLoss`. The differentiable sorting and ranking methods included are:

* Variants of Differentiable Sorting Networks
  * `bitonic` Bitonic Differentiable Sorting Networks (sparse)
  * `bitonic_sort__non_sparse` Bitonic Differentiable Sorting Networks
  * `splitter_selection` Differentiable Splitter Selection Networks (sparse)
  * `odd_even` Odd-Even Differentiable Sorting Networks
* `neuralsort` NeuralSort
* `softsort` SoftSort

Furthermore, this library also includes the smooth top-k loss from Lapin et al. (`SmoothTopKLoss` and `SmoothHardTopKLoss`.)

### `TopKCrossEntropyLoss`
In the center of the library lies the `difftopk.TopKCrossEntropyLoss`, which may be used as a drop-in replacement for
`torch.nn.CrossEntropyLoss`. The signature of `TopKCrossEntropyLoss` is defined as follows:

```python
loss_fn = difftopk.TopKCrossEntropyLoss(
    diffsort_method='odd_even',       # the sorting / ranking method as discussed above
    inverse_temperature=2,            # the inverse temperature / steepness
    p_k=[.5, 0., 0., 0., .5],         # the distribution P_K
    n=1000,                           # number of classes
    m=16,                             # the number m of scores to be sorted (can be smaller than n to make it efficient)
    distribution='cauchy',            # the distribution used for differentiable sorting networks
    art_lambda=None,                  # the lambda for the ART used if `distribution='logistic_phi'`
    device='cpu',                     # the device to compute the loss on
    top1_mode='sm'                    # makes training more stable and is the default value
)
```

It can be used as `loss_fn(outputs, labels)`.

### `DiffTopkNet`

`difftopk.DiffTopkNet` follows the signature of `diffsort.DiffSortNet` from the [`diffsort` package](https://github.com/Felix-Petersen/diffsort#-usage).
However, instead of returning the full differentiable permutation matrices of size `n`x`n`, it returns differentiable top-k attribution matrices of size `n`x`k`.
More specifically, given an input of shape `b`x`n`, the module returns a tuple of `None` and a Tensor of shape `b`x`n`x`k`. 
(It returns a tuple to maintain the signature of `DiffSortNet`.)

```python
sorter = difftopk.DiffTopkNet(
    sorting_network_type='bitonic',
    size=16,                          # Number of inputs
    k=5,                              # k
    sparse=True,                      # whether to use a sparse implementation
    device='cpu',                     # the device
    steepness=10.0,                   # the inverse temperature
    art_lambda=0.25,                  # the lambda for the ART used if `distribution='logistic_phi'`
    distribution='cauchy'             # the distribution used for the differentiable relaxation
)

# Usage example for difftopk on a random input
x = torch.randperm(16).unsqueeze(0).float() * 100.
print(x @ sorter(x)[1][0])
```

### `NeuralSort` / `SoftSort`

```python
sorter = difftopk.NeuralSort(
    tau=2.,     # A temperature parameter
)
sorter = difftopk.SoftSort(tau=2.)
```

---

## 🧪 Experiments

### 🧫 ImageNet Fine-Tuning

We provide pre-computed embeddings for the ImageNet data set. ⚠️ These embedding files are very large (>10 GB each.)
Feel free to also use the embeddings for other fine-tuning experiments.

```shell
# Resnext101 WSL ImageNet-1K (~11GB each)
wget https://nyc3.digitaloceanspaces.com/publicdata1/ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x48d_320.p
wget https://nyc3.digitaloceanspaces.com/publicdata1/ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x32d_320.p
wget https://nyc3.digitaloceanspaces.com/publicdata1/ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x16d_320.p
wget https://nyc3.digitaloceanspaces.com/publicdata1/ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x8d_320.p

# Resnext101 WSL ImageNet-21K-P (~50GB)
wget https://publicdata1.nyc3.digitaloceanspaces.com/ImageNet21K-P_embeddings_labels_train_test_IGAM_Resnext101_32x48d_224_float16.p

# Noisy Student EfficientNet-L2 ImageNet-1K (~14GB)
wget https://publicdata1.nyc3.digitaloceanspaces.com/ImageNet_embeddings_labels_train_test_tf_efficientnet_l2_ns_timm_transform_800_float16.p
```

The following are the hyperparameter combinations for reproducing the tables in the paper.
The DiffSortNet entries in Table 5 can be reproduced using

```shell
python experiments/train_imagenet.py  -d ./ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x48d_320.p  --nloglr 4.5 \
    --p_k .2 .2 .2 .2 .2  --m 16  --method bitonic  --distribution logistic_phi  --inverse_temperature 1.  --art_lambda .5 
python experiments/train_imagenet.py  -d ./ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x48d_320.p  --nloglr 4.5 \
    --p_k .2 .2 .2 .2 .2  --m 16  --method splitter_selection  --distribution logistic_phi  --inverse_temperature 1.  --art_lambda .5

python experiments/train_imagenet.py  -d ./ImageNet_embeddings_labels_train_test_tf_efficientnet_l2_ns_timm_transform_800_float16.p  --nloglr 4.5 \
    --p_k .25 .0 .0 .0 .75  --m 16  --method bitonic  --distribution logistic  --inverse_temperature .5 
python experiments/train_imagenet.py  -d ./ImageNet_embeddings_labels_train_test_tf_efficientnet_l2_ns_timm_transform_800_float16.p  --nloglr 4.5 \
    --p_k .25 .0 .0 .0 .75  --m 16  --method splitter_selection  --distribution logistic  --inverse_temperature .5 
```
and, for the remaining methods and tables, the hyperparameters are defined in the following:

<details>
  <summary>(<i>click to expand</i>)</summary>

```shell
# Tables 2+3 (1K):
python experiments/train_imagenet.py -d ./ImageNet_embeddings_labels_train_test_IGAM_Resnext101_32x48d_320.p  --m 16  --nloglr 4.5

# Tables 2+3 (21K):
python experiments/train_imagenet.py -d ./ImageNet21K-P_embeddings_labels_train_test_IGAM_Resnext101_32x48d_224_float16.p  --m 50  --nloglr 4.  --n_epochs 40 

# combined with one of each of the following

--method softmax_cross_entropy
--method bitonic             --distribution logistic_phi  --inverse_temperature 1.  --art_lambda .5
--method splitter_selection  --distribution logistic_phi  --inverse_temperature 1.  --art_lambda .5
--method neuralsort          --inverse_temperature 0.5
--method softsort            --inverse_temperature 0.5
--method smooth_hard_topk    --inverse_temperature 1. 

--p_k 1.  0. 0. 0. 0.
--p_k 0.  0. 0. 0. 1.
--p_k .5  0. 0. 0. .5
--p_k .25 0. 0. 0. .75
--p_k .1  0. 0. 0. .9
--p_k .2  .2 .2 .2 .2
```

</details>

### 🎆 CIFAR-100 Training from Scratch

In addition to ImageNet fine-tuning, we can also train a ResNet18 on CIFAR-100 from scratch. 

<details>
  <summary>(<i>click to expand</i>)</summary>

```shell
# Tables 1+4:
python experiments/train_cifar100.py

--method softmax_cross_entropy
--method bitonic             --distribution logistic_phi  --inverse_temperature .5  --art_lambda .5
--method splitter_selection  --distribution logistic_phi  --inverse_temperature .5  --art_lambda .5
--method neuralsort          --inverse_temperature 0.0625
--method softsort            --inverse_temperature 0.0625
--method smooth_hard_topk    --inverse_temperature 1.

--p_k 1.  0. 0. 0. 0.
--p_k 0.  0. 0. 0. 1.
--p_k .5  0. 0. 0. .5
--p_k .25 0. 0. 0. .75
--p_k .1  0. 0. 0. .9
--p_k .2  .2 .2 .2 .2
  
# Examples:
python experiments/train_cifar100.py --method softmax_cross_entropy --p_k 1. 0. 0. 0. 0.
python experiments/train_cifar100.py --method splitter_selection --distribution logistic_phi --inverse_temperature .5 --art_lambda .5 --p_k .2 .2 .2 .2 .2
```

</details>

## 📖 Citing

```bibtex
@inproceedings{petersen2022difftopk,
  title={{Differentiable Top-k Classification Learning}},
  author={Petersen, Felix and Kuehne, Hilde and Borgelt, Christian and Deussen, Oliver},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```

## 📜 License

`difftopk` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.


