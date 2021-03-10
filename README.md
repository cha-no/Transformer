# transformer
transformer 모델을 활용해 영어 - 프랑스어 번역을 수행

## model architecture


## usage
본 코드는 colab환경에서 실행되었음
```python
!git clone https://github.com/cha-no/transformer

cd transformer

%run main.py
# 메모리 부족일 경우
%run main.py --batch_size 256
```

## hyperparameter

hyperparameter|default| 
|:---:|:---:|
|epochs|5|
|batch_size|512|
|lr|0.001|
|num_layers|6|
|features|512|
|num_heads|8|
|fffeatures|2048|

### environment
tensorflow 2.3.0

keras 2.4.0

### 
[논문](https://arxiv.org/pdf/1706.03762.pdf)
