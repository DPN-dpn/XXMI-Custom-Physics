# 2. 가중치 베이킹 알고리즘 및 수학적 원리 (WEIGHTING_ALGORITHM.md)

이 문서는 `source/bone/weighting.py`에 구현된 해부학적 가중치 산출 수식, 다층 메쉬 비침투 제약, 강체 표면 전이 및 찢어짐 방지 동기화 알고리즘을 수학적으로 분석합니다.

---

## 🧮 1. 해부학적 Hann Cosine 이중 감쇠 곡선

단순 선형 거리 감쇠는 끝부분이 뾰족해지거나 기저부에서 각지게 꺾여 부자연스러운 표면 굴곡을 형성합니다. 본 애드온은 전방 깊이(Depth) 축과 방사형(Radial) 축을 합성한 $C^1$ 연속 코사인 곡면을 사용합니다.

```
       Apex (꼭짓점)
          /\
         /  \      w_depth: 흉벽(0.0) -> 꼭짓점(1.0) 코사인 램프
        /    \     w_radial: 중심축(1.0) -> 외곽(0.0) Hann 코사인 감쇠
       /______\
      Chest Wall (흉벽)
```

### 수학적 정의
본의 헤드($\mathbf{H}$)와 테일($\mathbf{T}$), 본의 단위 전방 벡터 $\mathbf{u} = \frac{\mathbf{T} - \mathbf{H}}{\|\mathbf{T} - \mathbf{H}\|}$, 본 길이 $L = \|\mathbf{T} - \mathbf{H}\|$, 유효 반경 $R$에 대해:

1. **전방 깊이 코사인 램프 ($w_{\text{depth}}$)**:
   $$t = (\mathbf{P} - \mathbf{H}) \cdot \mathbf{u}$$
   $$t_{\text{norm}} = \text{clamp}\left(\frac{t}{L}, 0, 1\right)$$
   $$w_{\text{depth}} = \begin{cases} \frac{1}{2} \left(1 - \cos(\pi \cdot t_{\text{norm}})\right) & (t > 0) \\ 0 & (t \le 0) \end{cases}$$

2. **방사형 수직 최단거리 감쇠 ($w_{\text{radial}}$)**:
   본 축선 상의 투영점 $\mathbf{P}_{\text{proj}} = \mathbf{H} + t_{\text{norm}} (\mathbf{T} - \mathbf{H})$과의 3D 수직 거리 $d_{\perp} = \|\mathbf{P} - \mathbf{P}_{\text{proj}}\|$:
   $$R_{\text{max}} = 1.30 \times R$$
   $$\rho = \text{clamp}\left(\frac{d_{\perp}}{R_{\text{max}}}, 0, 1\right)$$
   $$w_{\text{radial}} = \begin{cases} \frac{1}{2} \left(1 + \cos(\pi \cdot \rho)\right) & (d_{\perp} < R_{\text{max}}) \\ 0 & (d_{\perp} \ge R_{\text{max}}) \end{cases}$$

3. **기저 지수 보정 합성 (Power Calibration)**:
   기본 UI 슬라이더 값 $1.0$에서 끝이 뾰족하지 않고 풍성하고 둥근 해부학적 돔 볼륨을 형성하도록 기저 배율 $0.35$를 적용합니다:
   $$p_{\text{eff}} = 0.35 \times \text{falloff\_power}$$
   $$w(\mathbf{P}) = \text{peak\_weight} \times \left(w_{\text{depth}} \times w_{\text{radial}}\right)^{p_{\text{eff}}}$$

---

## 🚫 2. 다층 메쉬 절대 방사형 비침투 배리어 (Strict Radial Non-Penetration)

### 대원칙:
> **"본 중심에 가까운 안쪽 메쉬(속살)는 본 중심에서 더 먼 바깥쪽 메쉬(겉옷/장식)를 절대로 넘지 못한다 ($w_{\text{inner}} \le w_{\text{outer}}$)."**

### ⚠️ 치명적 주의사항: 연결 메쉬 섬(Same Island) 격리
과거 버전에서 가슴 표면에 **대각선 계단식 절벽 단차**와 **얼룩덜룩한 단차 구멍**이 발생했던 이유는, 동일한 하나의 연속된 단일 표면 정점들끼리도 곡면 경사로 인해 1~2mm의 깊이 차이가 발생하여 서로를 '속살 vs 겉옷'으로 오인하고 가중치를 강제로 깎아내렸기 때문입니다.

따라서 **반드시 연결 성분(`label_components`) 분석을 거쳐 동일 컴포넌트 내부 정점 간의 상호 클램프를 엄격히 차단**해야 합니다:

```python
# weighting.py의 필수 규칙
e0, e1 = mesh_edge_arrays(obj.data)
comp_labels = label_components(len(world), e0, e1)

# KD-Tree 탐색 루프
if obj_k == obj_hit and comp_k == comp_hit:
    continue  # 동일 오브젝트의 동일 연속 메쉬 표면 내부 정점은 비교 제외!
```

### Conformal Promotion (겉옷 상향 일치)
- $d_{\text{diff}} = r_{\text{outer}} - r_{\text{inner}} > 0$ 이고 $w_{\text{inner}} > w_{\text{outer}}$인 경우:
  - 겉옷($w_{\text{outer}}$)을 최소한 속살 이상으로 상향 일치시킵니다:
    $$w_{\text{outer}} = \max(w_{\text{outer}}, w_{\text{inner}})$$
  - **속살($w_{\text{inner}}$)을 강제로 깎아내려 분화구 형태의 함몰을 만들지 않습니다.** 겉옷이 속살만큼 움직여주면 속살이 튀어나오지 않으면서도 매끄러운 피부 볼륨이 그대로 보존됩니다.

---

## 🔒 3. 등록 정점 범위 보존 및 바닥값 정리

사용자가 에디트 모드에서 물리 본 슬롯에 특정 부위(예: 가슴, 꼬리)를 선택 등록한 경우:
1. **무단 팽창 금지**: 등록된 정점 목록(`slot_obj_verts`)이 존재할 경우, 반경 기반의 공간 확장(`find_range(0.018)`)을 적용하지 않고 사용자의 선택 영역을 엄격히 고정합니다. 목, 지퍼, 복부로의 가중치 침범을 차단합니다.
2. **인위적 0.002 바닥값 배제**: 계산된 가중치가 $w \le 0.0005$인 외곽 영역은 반드시 버텍스 그룹에서 완전 삭제(`vg.remove([vi])`)해야 합니다. 과거의 `0.002` 하한선 강제 할당은 스무딩 시 외곽 전체로 파란 톱니 얼룩을 번지게 만드는 원인이었습니다.

---

## 🔗 4. 강체 악세서리(Rigid Accessory) 표면 근접 전이

단추, 브로치, 리본, 지퍼 고리 등은 자체적으로 모핑되거나 늘어나서는 안 됩니다:
1. 에디트 모드에서 강체 슬롯에 정점 등록 시, `label_components`로 해당 부품의 **연결된 메쉬 섬 전체를 자동 완결**합니다 (보이지 않는 안감이나 뒷면 정점 누락 방지).
2. **Universal Surface Proximity Transfer**:
   - 강체 부품의 각 정점 위치에서 하부 연질 표면(Soft Surface)의 본별 KD-Tree를 질의합니다.
   - 역거리 가중치($\frac{1}{d}$)로 하부 표면의 움직임 가중치를 샘플링합니다:
     $$w_{\text{rigid}} = \frac{\sum w_i / d_i}{\sum 1 / d_i} \times \text{weight\_scale}$$
   - 부품 내부 엣지 네트워크를 따라 3회 라플라시안 스무딩을 거쳐 금속 부품 전체가 100% 강체 형태를 유지하며 부드럽게 피부를 따라 움직입니다.

---

## ✂️ 5. 경계면 완만화 (Boundary Gradient Taper)

메쉬 상에서 단일 폴리곤 엣지 하나를 사이에 두고 가중치가 $0.40 \to 0.00$으로 급락하면, 인게임 셰이더 변위 적용 시 폴리곤이 길게 늘어나며 **메쉬 찢어짐(Tearing)**이 발생합니다.
- 메쉬의 실제 엣지 연결망(`mesh.edges`)을 따라 3회 반복 검사합니다.
- 연결된 이웃 정점과의 가중치 편차 $\Delta w = w_v - \min(w_{\text{nbr}})$가 임계값($0.12$)을 초과할 경우 완만하게 감쇠합니다:
  $$w_v \leftarrow w_v - (\Delta w - 0.12) \times 0.40$$

---

## 🪡 6. UV 심(Seam) 및 분리 정점 100% 동기화

블렌더에서 1개의 단일 정점으로 보이는 정점이 게임 엔진 익스포트 시 UV 경계선이나 날카로운 모서리(Sharp Edge)에서 여러 개의 버텍스로 분리됩니다. 이 분리점들의 가중치가 $0.001$이라도 다르면 인게임에서 구멍이 뚫리고 메쉬가 갈라집니다.
- **절대 최종 단계**에서 3D 공간 상의 동일 좌표($d < 0.0002\text{m} = 0.2\text{mm}$)를 공유하는 모든 분리 정점을 찾아 최대 가중치로 $100\%$ 일치시킵니다.
- **주의**: 허용 반경을 $1\text{mm}$ 이상으로 넓히면 인접한 서로 다른 정점들까지 덩어리로 묶여 평평하게 뭉개지므로, 반드시 **$0.2\text{mm}$ 이내의 순수 분리 정점만 매칭**해야 합니다.
