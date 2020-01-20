# Week 1

## 임베딩: Concept

```python
## Keras default tokenizer 
tokenizer = preprocessing.text.Tokenizer() 
tokenizer.fit_on_texts(samples) 
sequences = tokenizer.texts_to_sequences(samples)
word_index = tokenizer.word_index
```

단어를 인덱스화 해도, 연산할 때 큰 의미를 가지지는 않음
- `3`, `4` 가 인덱스 값으로, 가깝다 해도 실제로 유사한 단어가 아닐 수 있으며
- `4` 의 단어가 더 중요하다고도 볼 수 없음

따라서 단어마다 벡터화 하여 `go` 와 `going` 이 유사한 값을 가지도록 하는 것을 임베딩이라 함

이러한 벡터화된 결과를 딥러닝에서는
- sentence 를 reduce_mean 하면 문장이 의미를 수치화
- word 를 reduce_mean 하면 문장의 의미를 수치화

```python
tf.reduce_mean(embed_input, axis=-1) ## axis 
```

## 패딩 (Padding)

Padding 을 하는 이유는, 

기본적으로는 Vector 연산을 하기 위해 고정된 값으로 제공해야 함. 따라서 Padding 을 통해 매트릭스 형태로 만들어 주는 것. (Dynamic 하게 받는 RNN 도 있긴 하나, 많지는 않다고 함)

이렇게 Padding 을 통해 고정된 사이즈의 매트릭스를 만들게 되면, Batch 를 할 수 있다. Batch 를 하는 이유는
- 복수개의 샘플을 한번에 집어 넣어 연산을 빠르게 할 수도 있으며 (행렬, 벡터 연산은 선형적으로 비용이 증가 X)
- 학습의 수렴을 빠르게 할 수 있다. 하나씩 넣을 때는 비정상적인 샘플로 인해 Optimal 에서 멀어질 수 있으나, 배치를 통해 평균의 loss 를 이용해 수렴하게 됨

Bucketing 이란 기법을 이용해 샘플의 사이즈에 따라 구간을 다르게 할 수도 있음
- 1 ~ 10 은 10
- 11 ~ 100 은 100 등

## 임베딩: Vocab Size 

임베딩 모듈에서 vocab size 에 `word_index 사이즈 + 1` 을 해주는 이유는, Padding 때문.

```python
vocab_size = len(word_index) + 
```

## 임베딩: OOV

OOV (out of vocabulary) 는, 해당 단어가 Training 에는 있는데 Test Set 에 없을 때 발생할 수 있음.

이런 문제를 해결하기 위해, 빈도가 일정 수 이상에 대해서만 (중요한 단어만) Vocab 을 만들고, 나머지는 OOV 처리할 수 있다.

## Keras 기본 모델들

Conv1D, CNN, RNN(LSTM, GRU), Seq2Seq, BERT 에 대해 기본 컨셉을 잡기 위한 아티클 링크 및 간략한 요약

### CNN (Convolutional Nerual Network)

- [CNN 이란? - 1](https://medium.com/@hobinjeong/cnn-convolutional-neural-network-9f600dd3b395)
- [CNN 에서 Pooling 이란? -2 ](https://medium.com/@hobinjeong/cnn%EC%97%90%EC%84%9C-pooling%EC%9D%B4%EB%9E%80-c4e01aa83c83)
- [CNN 역전파](https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/)

filter 하나당, stride 씩 옮겨가며, activation 해서 나온 결과를 모아 놓은 것이 convolution layer (혹은 activation map)

filter 의 사이즈로 인해 (3x3 등), 원본 data (7x7 등) 를 가공한 convolution (5x5 등) 사이즈가 되어 데이터에 손실이 발생할 수 있음. 따라서 원본의 일정 부분을 (7x7 등) 유지하기 위해 (Zero) Padding 을 통해 가공후에도 원본 데이터 사이즈를 유지하도록 할 수 있음. (손실을 막을 수는 있지만, 필요 없던 부분이 생김)

pooling 은 각 pixel 에서 하나의 값을 뽑아내는 과정 (일종의 activation) 으로 max, mean pooling 등 다양하게 존재.

pooling 을 하는 이유는, overfitting 을 방지하기 위함. 예를 들어,
- 96x96 이미지를 8x8, stride=1 의 filter 400개로 convolution 한 경우
- 89 * 89 사이즈의 400 개의 feature 가 존재
- 원본은 96 x 96 이었으나, 89x89x400 = 3,168,400 개 의 feature 가 존재

![](https://miro.medium.com/max/1920/1*5HA3lTFOGyc5TCi4uDCHlw.png)

### RNN (Recurrent Neural Network)

- [RNN Tutorial - Part 1](http://aikorea.org/blog/rnn-tutorial-1/)

RNN 의 기본 아이디어는, 순차적으로 정보를 처리한다는 것. 기존의 신경망 구조에서는 모든 입력과 출력이 각각 독립적이라고 가정했지만, 많은 경우에는 이는 옳지 않은 방법. 예를 들어, 한 문장에서 다음에 나올 단어를 추출한다면 이전에 나올 단어를 아는 것이 큰 도움이 된다.

RNN 이 Recurrent 하다고 불리는 이유는 동일한 태스크를 한 시퀀스의 모든 요소마다 적용하고, 출력 결과는 이전 계산 결과에 영향을 받기 때문. 

- `LSTM (Long Short-Term Memory)`

Vanilla RNN 은 거슬러 올라가는 단계 수가 많아지면, 기울기 소실 문제 때문에 제대로 학습할 수 없어 장기 기억력을 가지도록 개선한 모델이 LSTM

- `GRU (Gated Recurrent Unit)`

LSTM 과 같이 장기 기억이 가능하면서도 계산량은 절감시킨 RNN cell

### Seq2Seq

- [seq2seq 모델로 뉴스 제목 추출하기](https://ratsgo.github.io/natural%20language%20processing/2017/03/12/s2s/)

seq2seq (S2S) 모델은 RNN 의 발전된 형태의 아키텍처로 LSTM, GRU 등의 RNN cell 을 길고 깊게 쌓아 복잡하고 방대한 시퀀스 데이터를 처리하는데 특화된 모델. 

seq2seq 모델은 인코더와 디코더 두 파트로 나뉘는데, 영어를 한국어로 변환하는 기계 번역의 경우를 예로 들면,
- Encoder (입력): `Good morning!` - 한국어 입력
- Decoder (입력): `<go> 좋은 아침입니다!` - 한국어 입력
- Decoder (출력): `좋은 아침입니다!<eos>!` - 한국어 출력 

인코더는 소스 언어 정보를 압축하고, 디코더는 인코더가 압축해 보내준 정보를 받아 타겟 언어로 변환해 출력함. 
- 학습 과정에서는 디코더는 인코더가 보내온 정보와 실제 정답 (`<go> ...`) 을 받아 `... <eos>` 를 출력
- 예측 과정에서는 정답이 없으므로 디코더는 인코더가 보내온 정보만을 이용해 결과물을 차례대로 출력

예측 과정에서의 디코더는 직전에 예측한 결과를 자신의 다음 단계 입력으로 넣어 그 다음 결과를 출력 `좋은` 을 받아 `아침입니다` 를 예측

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2Ft2fZw%2FbtquSx2fTxu%2FLZzzHY1R0mKYZWuF9kIAN1%2Fimg.png)

그러나 seq2seq 는 하나의 고정된 크기 벡터에 모든 정보를 압축하려다 보니 정보 손실이 발생하고, RNN 의 고질적인 문제인 기울기 소실 (Vanishing Gradient) 문제가 발생

- [효과적인 RNN 학습](https://ratsgo.github.io/deep%20learning/2017/10/10/RNNsty/)

### Attention Mechanism

- [딥러닝을 이용한 자연어 처리 입문 - 어텐션 메커니즘](https://wikidocs.net/22893)

Attention 의 기본 아이디어는, 디코더에서 출력 단어를 예측하는 매 시점마다 인코더에서의 전체 입력 문장을 다시 한번 참고하는 것. 단 전체 입력 문장을 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야 할 단어와 연관 있는 단어 부분을 조금 더 집중해서 보는 것. 

- [어텐션 메커니즘 시각화](http://docs.likejazz.com/attention/)

다시 말해, seq2seq 에서 Attention 이란 디코더의 특정 time-step 의 output 이 인코더의 모든 time-step 의 output 중 어떤 time-step 과 가장 유사한가를 보는 것.

**모델로 하여금 중요한 부분만 집중하게 만들자** 는 핵심 이이디어를 구현하기 위해, 디코더가 출력을 생성 할 때 각 단계별로 입력 시퀀스의 각기 다른 부분을 집중하도록 한다. 즉 하나의 고정된 컨텍스트 벡터로 인코딩 하는 대신, 출력의 각 단계별로 컨텍스트 벡터를 생성하는 방법을 학습한다.


- [어텐션 메커니즘과 Transformer](https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225)

그러나 일반적인 seq2seq-attention 모델에서의 번역 태스크의 문제는, 원본 언어 (Source) 와 번역된 언어 (Target) 간의 어느정도 대응 여부는 Attention 을 통해 찾을 수 있었으나, 각 자신의 언어만에 대해서는 관계를 나타낼 수 없었음. 예를 들어, `I love tigher but it is scare` 에서 `it` 이 해당 문장 (같은 언어) 에서 무엇을 나타내는지는 기존 encoder-decoder 기반의 어텐션 메커니즘에서는 찾을 수 없었음.


### Transformer

Transformer 는 RNN, LSTM 없이 time 시퀀스 역할을 하는 모델로, 현재의 BERT 등 모델이 Transformer 구조에 기반하여 구현되었음.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2Fsctry%2FbtquTiSqTlc%2Fm3zep3KgQ8fsGPcLHIgxQ0%2Fimg.png)

![](https://miro.medium.com/max/1072/1*MBc5BeHRr6wtc3R0PU81xg.png)

![](https://miro.medium.com/max/1104/1*FhozOmbyTGv9eR0oeLwhjg.png)


## Keras: Activation, Loss Basic



