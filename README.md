# Hierarchical Multimodal Feature Learning and 3D Convolution for Rail Defect Detection

<p align="center">
  <img src="images/prcv2025.png" width="700">
</p>

## Quick Start

1. Create the Python environment

```python
conda create -n HM3DNet python==3.9
conda activate HM3DNet
```

2. Install packages

```python
pip install -r requirements.txt
```

3. Clone the repo

```python
git clone https://github.com/ShifaTang/HM3DNet.git
```

4. Training model

- The train code in the train_and_code folder

```
python train.py
```

5. Testing model

- The test model also in the train_and_code folder and then sailence maps will be generated

```
python test_produce_maps.py
```

## Result

- the result on NEU RSDDS-AUG dataset

<p align="center">
  <img src="images/result_new.png" width="700">
</p>

- the result on NJU2K、NLPR、STERE dataset

<p>
    <img src="images/result_nj.png" width=700>
</p>



- the comparsion between ours and other methods

<p align="center">
    <img src="images/result_comparsion.png">
</p>



## Dataset

Follow previouts the research of Sailent Object Dection, we use NEU RSDDS-AUG、NJU2K、NLPR、STERE as dataset . These datasets have  already been divided into train split、test split and you can find them online.

## Citation

Please cite our paper if you think it's useful. If you have any questions, feel free to contact me!

```
@inproceedings{tang2025hierarchical,
  title={Hierarchical Multimodal Feature Learning and 3D Convolution for Rail Defect Detection},
  author={Tang, Shifa and Zhang, Jinlai and Yu, Shuimiao and Chen, Xi and Wu, Sheng and Zou, Tiefang and Li, Qiqi and Hu, Lin},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={489--503},
  year={2025},
  organization={Springer}
}
```

