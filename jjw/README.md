# DACON 모기 비행 궤적 예측 솔루션

데이터 ZIP을 직접 읽어 11개 3D 좌표에서 +80ms 위치를 예측하고, `R-Hit@1cm` 기준으로 물리 외삽 + residual ML 앙상블 제출 파일을 생성합니다. 재검증 후 `R-Hit@1cm` 경계 근처 샘플에 가중치를 둔 weighted tree 후보를 추가했습니다.

## 현재 위치/상태

- 프로젝트 경로: `/Users/jjw/Documents/모기잡기/jjw`
- 현재 Public 최고: `0.6848` (69등)
- 갱신 Public 최고: `0.6868` (48등, `submission_multibase_local_trap_blend.csv`)
- 현재 주력 계열: `hgb_local_trap` / local-frame residual target / hit-aware trapezoid weight

## 실행

이 Mac의 기본 `python3`는 3.14라 현재 과학 패키지가 없을 수 있습니다. 아래처럼 Python 3.13을 사용하세요.

```bash
/opt/homebrew/bin/python3.13 -m src.train --zip-path /Users/jjw/Downloads/open.zip --out-dir outputs --profile quick --folds 5
```

더 무거운 후보까지 학습하려면:

```bash
/opt/homebrew/bin/python3.13 -m src.train --zip-path /Users/jjw/Downloads/open.zip --out-dir outputs_full --profile full --folds 5
```

기존 학습 artifact로 제출 파일만 재생성:

```bash
/opt/homebrew/bin/python3.13 -m src.predict --zip-path /Users/jjw/Downloads/open.zip --model-path outputs/models.pkl --out-dir outputs/recreated
```

## 주요 산출물

- `metrics.json`: 물리 baseline, OOF 모델, shrink, blend 점수
- `models.pkl`: fold별 모델과 blend 설정
- `submission_*.csv`: DACON 업로드 후보
- 현재 다음 실험 후보:
  - `submission_multibase_local_trap_blend.csv` (OOF 0.6695, 현재 최우선)
  - `submission_local_trap_base_accel_c0p58.csv` (OOF 0.6677)
  - `submission_local_trap_base_accel_c0p46.csv` (OOF 0.6675)

## GPU sequence residual 실험

HGB 계열과 다른 inductive bias를 만들기 위해 PyTorch 기반 local-frame sequence residual 모델을 추가했습니다.

```bash
/opt/homebrew/bin/python3.13 -m src.train_torch \
  --out-dir outputs_gpu_seq_c0p58 \
  --residual-base accel_c0.58 \
  --folds 5 \
  --epochs 90 \
  --patience 18 \
  --batch-size 768 \
  --hidden 96 \
  --layers 2 \
  --heads 4 \
  --dropout 0.12 \
  --lr 0.0008 \
  --weight-decay 0.0002 \
  --boundary-loss 0.30 \
  --noise-std 0.012 \
  --device auto \
  --blend-with outputs/submission_multibase_local_trap_blend.csv
```

실험 결과:

- `gpu_seq` 단독 OOF: `0.6658`
- 기존 제출 multibase OOF: `0.6693`
- `multibase + gpu_seq raw` OOF blend:
  - weight `0.45`: `0.6721`
  - weight `0.60`: `0.6725`
- 2-model oracle: `0.7025`

현재 GPU 계열 제출 후보:

- `outputs_gpu_seq_c0p58/submission_gpu_seq_raw_x_multibase_w600.csv`
- `outputs_gpu_seq_c0p58/submission_gpu_seq_raw_x_multibase_w450.csv`

## 규칙 준수

- test 데이터는 예측 생성에만 사용합니다.
- 원격 API/외부 서버 모델을 사용하지 않습니다.
- 기본 환경의 `numpy`, `pandas`, `sklearn`만으로 실행됩니다.
