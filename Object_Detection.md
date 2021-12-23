## 객체 탐지 ( Object Detection )
- 한 이미지에서 객체와 그 경계 상자(bounding box)를 탐지
- 객체 탐지 알고리즘은 일반적으로 이미지를 입력으로 받고, 경계 상자와 객체 클래스 리스트를 출력
- 경계 상자에 대해 그에 대응하는 예측 클래스와 클래스의 신뢰도(confidence) 출력

<br>

## Applications
- 자율 주행 자동차에서 다른 자동차와 보행자를 찾을 때
- 의료 분야에서 방사선 사진을 사용해 종양이나 위험한 조직을 찾을 때
- 제조업에서 조립 로봇이 제품을 조립하거나 수립할 때
- 보안 산업에서 위협을 탐지하거나 사람을 셀 때

---
<br>

## 용어 설명

### Bounding Box
- 이미지에서 하나의 객체 전체를 포함하는 가장 작은 직사각형

<br>

### IOU (Intersection Over Union)
- 실측값(Ground Truth)과 모델이 예측한 값이 얼마나 겹치는지를 나타내는 지표
<img src="https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png">

- IOU가 높을수록 잘 예측한 모델
<img src="https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_examples.png">

- 예시
<img src="https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_stop_sign.jpg">

<sub>[이미지 출처]: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/ </sub>
<br>

```python
def compute_iou(pred_box, gt_box):
  x1 = np.maximum(pred_box[0], gt_box[0])
  y1 = np.maximum(pred_box[1], gt_box[1])
  x2 = np.maximum(pred_box[2], gt_box[2])
  y2 = np.maximum(pred_box[3], gt_box[3])

  intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

  pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
  gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

  union = pred_box_area + gt_box_area - intersection_area

  iou = intersection / union

  return iou
 ```
 
<br>

### NMS (Non-Maximum Suppression, 비최댓값 억제)
- 확률이 가장 높은 상자와 겹치는 상자들을 제거하는 과정
- 최댓값을 갖지 않는 상자들을 제거
- 과정
  1. 확률 기준으로 모든 상자를 정렬하고 먼저 가장 확률이 높은 상자를 취함
  2. 각 상자에 대해 다른 모든 상자와의 IOU를 계산
  3. 특정 임계값을 넘는 상자는 제거

<img src="https://www.pyimagesearch.com/wp-content/uploads/2014/10/nms_fast_03.jpg">
<sub>[이미지 출처]: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/ </sub>

<br>

```python
import numpy as np

def non_max_suppression_fast(boxes, overlap_thresh):
  if len(boxes) == 0:
    return []
  
  if boxes.dtype.kind == 'i':
    boxes = boxes.astype('float')

  pick = []
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.maximum(x2[i], x2[idxs[:last]])
    yy2 = np.maximum(y2[i], y2[idxs[:last]])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    overlap = (w * h) / area[idxs[:last]]

    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype('int')
```

---
<br>

## 모델 성능 평가
### 정밀도와 재현율
- 일반적으로 객체 탐지 모델 평가에 사용되지는 않지만, 다른 지표를 계산하는 기본 지표 역할을 함
  - `TP`
    - True Positives
    - 예측이 동일 클래스의 실제 상자와 일치하는지 측정
  - `FP`
    - False Positives
    - 에측이 실제 상자와 일치하지 않는지 측정
  - `FN`
    - False Negatives
    - 실제 분류값이 그와 일치하는 예측을 갖지 못하는지 측정
    
**precision = TP / (TP + FP)**
**recall = TP / (TP + FN)**

- 모델이 안정적이지 않은 특징을 기반으로 객체 존재를 예측하면 거짓긍정(FP)이 많아져서 정밀도가 낮아짐
- 모델이 너무 엄격해서 정확한 조건을 만족할 때만 객체가 탐지된 것으로 간주하면 거짓부정(FN)이 많아져서 재현율이 낮아짐


<br>

### 정밀도-재현율 곡선 (precision-recall curve)
- 신뢰도 임계값마다 모델의 정밀도와 재현율을 시각화
- 모든 bounding box와 함께 모델이 예측의 정확성을 얼마나 확실하는지 0 ~ 1 사이의 숫자로 나타내는 신뢰도를 출력
- 임계값 T에 따라 정밀도와 재현율이 달라짐
  - 임계값 T 이하의 예측은 제거함
  - T가 1에 가까우면 정밀도는 높지만 재현율은 낮음
  놓치는 객체가 많아져서 재현율이 낮아짐. 즉, 신뢰도가 높은 예측만 유지하기 때문에 정밀도는 높아짐
  - T가 0에 가까우면 정밀도는 낮지만 재현율은 높음
  대부분의 예측을 유지하기 때문에 재현율은 높아지고, 거짓긍정(FP)이 많아져서 정밀도가 낮아짐
- 예를 들어, 모델이 보행자를 탐지하고 있으면 특별한 이유 없이 차를 세우더라도 어떤 보행자도 놓치지 않도록 재현율을 높여야 함
모델이 투자 기회를 탐지하고 있다면 일부 기회를 놓치게 되더라도 잘못된 기회에 돈을 거는 일을 피하기 위해 정밀도를 높여야 함

<img src="https://www.researchgate.net/profile/Davide-Chicco/publication/321672019/figure/fig1/AS:614279602511886@1523467078452/a-Example-of-Precision-Recall-curve-with-the-precision-score-on-the-y-axis-and-the.png">
<sub>[이미지 출처] : https://www.researchgate.net/figure/a-Example-of-Precision-Recall-curve-with-the-precision-score-on-the-y-axis-and-the_fig1_321672019 </sub>

<br>

### AP (Average Precision, 평균 정밀도)와 mAP (mean Average Precision)
- 위 그림의 곡선 아래 영역에 해당
- 항상 1x1 정사각형으로 구성되어 있음
즉, 항상 0 ~ 1 사이의 값을 가짐
- 단일 클래스에 대한 모델 성능 정보를 제공
- 전역 점수를 얻기 위해서 mAP를 사용
- 예를 들어, 데이터셋이 10개의 클래스로 구성된다면 각 클래스에 대한 AP를 계산하고, 그 숫자들의 평균을 다시 구함
- (참고)
  - 최소 2개 이상의 객체를 탐지하는 대회인 PASCAL Visual Object Classes와 Common Objects in Context(COCO)에서 mAP가 사용됨
  - COCO 데이터셋이 더 많은 클래스를 포함하고 있기 때문에 보통 Pascal VOC보다 점수가 더 낮게 나옴
  - 예시

<img src="https://www.researchgate.net/profile/Bong-Nam-Kang/publication/328939155/figure/tbl2/AS:692891936649218@1542209719916/Evaluation-on-PASCAL-VOC-2007-and-MS-COCO-test-dev.png">
<sub>[이미지 출처] : https://www.researchgate.net/figure/Evaluation-on-PASCAL-VOC-2007-and-MS-COCO-test-dev_tbl2_328939155</sub>

---

<br>

## Dataset
### VOC
- 2005년부터 2012년까지 진행
- Object Detection 기술의 benchmark로 간주
- 데이터셋에는 20개의 클래스가 존재
(background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike,
person, pottedplant, sheep, sofa, train, tvmonitor)
- 훈련 및 검증 데이터 : 11,530개
- ROI에 대한 27,450개의 Annotation이 존재
- 이미지당 2.4개의 객체 존재
<img src="https://paperswithcode.github.io/sotabench-eval/img/pascalvoc2012.png">
<sub>[이미지 출처]https://paperswithcode.github.io/sotabench-eval/pascalvoc/</sub>

<br>

### COCO Dataset
- Common Objects in Context
- 200,000개의 이미지
- 80개의 카테고리에 500,000개 이상의 객체 Annotation이 존재
<img src="https://cocodataset.org/images/coco-examples.jpg">
<sub>[이미지 출처]https://cocodataset.org/#home</sub>

---

<br>

## YOLO (You Only Look Once)
- 가장 빠른 객체 검출 알고리즘 중 하나
- 256x256 사이즈의 이미지
- GPU 사용 시, 초당 170프레임(170FPS, frames per second),
이는 파이썬, 텐서플로 기반 프레임워크가 아닌 C++로 구현된 코드 기준
- 작은 크기의 물체를 탐지하는 것은 어려움

<br>

### YOLO Backbone
- 백본 모델(backbone model) 기반
- 특징 추출기(Feature Extractor)라고도 불림
- YOLO는 자체 맞춤 아키텍쳐 사용
- 어떤 특징 추출기 아키텍쳐를 사용했는지에 따라 성능 달라짐

<img src="https://www.researchgate.net/publication/335865923/figure/fig1/AS:804106595758082@1568725360777/Structure-detail-of-YOLOv3It-uses-Darknet-53-as-the-backbone-network-and-uses-three.jpg">
<sub>[이미지 출처]https://www.researchgate.net/figure/Structure-detail-of-YOLOv3It-uses-Darknet-53-as-the-backbone-network-and-uses-three_fig1_335865923 </sub>

- 마지막 계층은 크기가 **w x h x D**인 특징 볼륨 출력
- **w x h**는 그리드의 크기, **D**는 특징 볼륨 깊이

<br>

### YOLO의 계층 출력
- 마지막 계층 출력은 **w x h x M** 행렬
  - **M = B x (C + 5)**
    - B: 그리드 셀당 경계 상자 개수
    - C: 클래스 개수
  - 클래스 개수에 5를 더한 이유는 해당 값 만큼의 숫자를 예측해야 함
    - tx, ty는 경계 상자의 중심 좌표를 계산
    - tw, th는 경계 상자의 너비와 높이를 계산
    - c는 객체가 경계 상자 안에 있다고 확신하는 신뢰도
    - p1, p2, ..., pC는 경계 상자가 클래스 1, 2, ..., C의 객체를 포함할 확률

<img src="https://www.researchgate.net/profile/Thi-Le-5/publication/337705605/figure/fig3/AS:831927326089217@1575358339500/Structure-of-one-output-cell-in-YOLO.ppm">
<sub>[이미지 출처]https://www.researchgate.net/figure/Structure-of-one-output-cell-in-YOLO_fig3_337705605</sub>

<br>

### 앵커 박스 (Anchor Box)
- YOLOv2에서 도입
- 사전 정의된 상자(prior box)
- 객체에 가장 근접한 앵커 박스를 맞추고 신경망을 사용해 앵커 박스의 크기를 조정하는 과정 때문에 tx, ty, tw, th이 필요

<img src="https://www.mathworks.com/help/vision/ug/ssd_detection.png">
<sub>[이미지 출처]https://kr.mathworks.com/help/vision/ug/getting-started-with-yolo-v2.html</sub>

---

<br>



