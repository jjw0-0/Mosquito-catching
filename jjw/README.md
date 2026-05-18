# DACON 모기 비행 궤적 예측 솔루션

데이터 ZIP을 직접 읽어 11개 3D 좌표에서 +80ms 위치를 예측하고, `R-Hit@1cm` 기준으로 물리 외삽 + residual ML 앙상블 제출 파일을 생성합니다. 재검증 후 `R-Hit@1cm` 경계 근처 샘플에 가중치를 둔 weighted tree 후보를 추가했습니다.

## 실행

```bash
python3 -m src.train --zip-path /Users/jjw/Downloads/open.zip --out-dir outputs --profile quick --folds 5
```

더 무거운 후보까지 학습하려면:

```bash
python3 -m src.train --zip-path /Users/jjw/Downloads/open.zip --out-dir outputs_full --profile full --folds 5
```

기존 학습 artifact로 제출 파일만 재생성:

```bash
python3 -m src.predict --zip-path /Users/jjw/Downloads/open.zip --model-path outputs/models.pkl --out-dir outputs/recreated
```

## 주요 산출물

- `metrics.json`: 물리 baseline, OOF 모델, shrink, blend 점수
- `models.pkl`: fold별 모델과 blend 설정
- `submission_*.csv`: DACON 업로드 후보 (`submission_blend_hit_optimized.csv`, `submission_extra_trees_near_shrink.csv` 우선 확인)

## 규칙 준수

- test 데이터는 예측 생성에만 사용합니다.
- 원격 API/외부 서버 모델을 사용하지 않습니다.
- 기본 환경의 `numpy`, `pandas`, `sklearn`만으로 실행됩니다.
