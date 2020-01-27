# Classification

## Validation

- https://www.quora.com/What-is-the-difference-between-validation-and-cross-validation

데이터를 Train, Validation, Test 셋으로 나누고
- 이 중 Train 과 셋을 이용해 훈련하며
- Validation 셋을 이용해 여러 모델의 다른 하이퍼 파라미터 조합을 튜닝
가장 좋은 성능을 보이는 하이퍼 파라미터와 모델을 이용해 Test 셋을 이용해 최종 퍼포먼스 테스트

## Cross Validation

데이터를 X 와 Test 셋 두개로 나누고, X 내에서 train, validation 셋을 다양하게 선택해가면서 모델을 튜닝. 일반 검증 방법에 비해 Validation 이 훈련에도 참여할 수 있어 데이터가 적은 경우에 유용.

- https://m.blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=221599517834&proxyReferer=https%3A%2F%2Fwww.google.com%2F

