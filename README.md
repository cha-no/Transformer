# Transformer
![image](https://user-images.githubusercontent.com/59329586/110621541-af13d480-81dd-11eb-84b4-f785af375faf.png)

[논문](https://arxiv.org/pdf/1706.03762.pdf)

transformer 모델을 구현하여 영어 - 프랑스어 번역을 수행

## Requirement
- tensorflow == 2.3.0
- keras == 2.4.0

## Running main

```
%run main.py
# 메모리 부족일 경우
%run main.py --batch_size 256
```

### config

hyperparameter|default| 
|:---:|:---:|
|epochs|5|
|batch_size|512|
|lr|0.001|
|num_layers|6|
|features|512|
|num_heads|8|
|fffeatures|2048|
