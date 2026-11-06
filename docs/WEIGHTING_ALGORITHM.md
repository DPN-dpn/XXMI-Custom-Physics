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

## 🔒 3. 선택 영역 통합 풀(Union Pool) 및 인접 본 브릿지 크로스 블렌딩

타이트한 셔츠, 니트, 바디수트처럼 양쪽 가슴을 하나의 옷감이 덮고 있는(텐트 현상) 메쉬에서 각 본을 자신의 선택 정점(`slot_obj_verts`) 내부에만 가두면, 중앙 경계선($X \approx 0$)에서 양쪽 가중치가 동시에 급락하여 가운데 옷감만 흉벽에 못 박힌 듯 멈춰 있거나 부자연스럽게 접히는 문제가 발생합니다. 이를 해결하기 위해 **하이브리드 통합 풀 + 브릿지 엔벨로프 알고리즘**을 사용합니다:

1. **선택 정점 통합 보호 풀 (Union Pool)**:
   - 팔 안쪽, 목, 등, 배 등 비선택 부위로 가중치가 번지는 것은 원천 차단하되, 등록된 모핑 본들의 정점 합집합($\bigcup_i \text{Slot}_i$)과 슬롯 간 내부 사잇길($\text{Dilate}(\text{Slot}_A, 2) \cap \text{Dilate}(\text{Slot}_B, 2)$)을 하나의 **통합 물리 허용 영역(`union_mask`)**으로 묶습니다.
   - 외곽 경계는 1mm도 팽창하지 않으면서, 통합 영역 내부에서는 좌/우 본의 칸막이가 사라져 스켈레톤 본의 반경(`radius_scale`)에 따라 상대편 영역까지 자유롭게 가중치가 교차 침투합니다.
2. **3D Lofted Bridge Envelope & Raised-Cosine Partition of Unity (`calculate_bridged_pair_weights`)**:
   - 인접한 모핑 본 쌍($A, B$) 사이의 정규화 위치 $s \in [0, 1]$를 따라 보간된 가상의 브릿지 세그먼트 $[H(s), T(s)]$를 구성합니다.
   - 맨살 가슴골처럼 정점이 흉벽($H(s)$)에 밀착되어 있으면 전방 깊이 $t_{\text{proj}} \approx 0$이 되어 자동으로 가중치가 $0$으로 유지되고, 일체형 의상처럼 중앙 옷감이 앞으로 떠 있으면($t_{\text{proj}} > 0$) 좌/우 돔과 동일한 볼륨 곡면 가중치 $W_{\text{loft}}$가 유지됩니다.
   - 산출된 총 가중치 $W_{\text{total}}$을 본 반경 비례 폭($\sigma$)에 따른 코사인 분할 계수 $\alpha_A(s) + \alpha_B(s) = 1.0$으로 배분하여, 중앙에서 합산 가중치($w_A + w_B$)가 꺼지거나 2배로 튀지 않고 매끄럽게 연동되도록 보장합니다.
3. **인위적 0.002 바닥값 배제**: 계산된 가중치가 $w \le 0.0005$인 외곽 영역은 반드시 버텍스 그룹에서 완전 삭제(`vg.remove([vi])`)합니다.

---

## 🔗 4. 강체 악세서리(Rigid Accessory) 다중 오브젝트 일체화 및 표면 근접 전이

단추, 브로치, 벨트 버클뿐만 아니라 **무전기 본체(Hair 오브젝트) + 무전기 파우치(Body 오브젝트)**처럼 여러 메쉬 오브젝트에 걸쳐 나뉜 복합 부품도 하나의 강체 슬롯(`XXMI_PG_rigid_item`)으로 묶여 있다면 **절대로 서로 분리되거나 뚫고 나오지 않아야 합니다.**

### 1. 단일 슬롯 = 단일 완전 강체 조립체 (Single Rigid Assembly)
- 한 슬롯에 등록된 모든 오브젝트의 정점들은 **$100.000\%$ 완벽히 동일한 단일 가중치 벡터 $(w_0, w_1, \dots, w_N)$**를 공유합니다.
- 부품 내부 정점별로 가중치를 다르게 주거나 메쉬 엣지 라플라시안 스무딩을 적용하면, 돌출된 무전기 안테나와 안쪽 케이스의 가중치가 달라져 인게임 모션 시 무전기가 케이스를 뚫고 튀어나오게 됩니다.

### 2. 슬롯 통합 접촉면 가중치 (Mounting Base Proximity) 산출 알고리즘
강체 조립체가 신체/의상에 마운트된 위치의 정확한 움직임을 추종하기 위해, 슬롯 전체의 3D 정점 중 하부 소프트 표면과의 **최단 접촉면(Mounting Base)**을 기준으로 가중치를 산출합니다:
1. 슬롯 내 모든 오브젝트의 정점 $\mathbf{P}_v$에서 하부 소프트 표면까지의 거리 $d_v$를 질의하고 최소 거리 $d_{\text{min}} = \min(d_v)$를 도출합니다 (예: 케이스 뒷면이 가슴 피부와 접촉하는 $d \approx 2\text{mm}$).
2. 접촉면 근방($d_v \le d_{\text{min}} + 15\text{mm}$)에 속하는 접촉 정점들만을 대상으로 역거리 가중 평균을 산출합니다:
   $$W_{\text{base}, k} = \frac{\sum_{v \in \text{contact}} \frac{w(v, \text{soft})}{\max(d_v, 0.002)}}{\sum_{v \in \text{contact}} \frac{1}{\max(d_v, 0.002)}}$$
3. 거리 감쇠 인자를 곱하여 슬롯의 단일 통합 가중치 $W_{\text{slot}, k} = W_{\text{base}, k} \times \text{dist\_factor} \times \text{weight\_scale}$를 결정합니다.
4. 이 단일 스칼라 가중치를 **해당 슬롯에 속한 모든 오브젝트(`Hair`, `Body` 등)의 모든 정점에 100% 균일 할당**합니다.
5. 이를 통해 셰이더 상에서 조립체 전체의 상대 변위($\Delta \mathbf{P}_{\text{radio}} - \Delta \mathbf{P}_{\text{case}}$)가 항등적으로 0이 되어, 뚫림이나 부품 간 분리가 수학적으로 원천 차단됩니다.

### 3. 후속 단계 강체 가중치 보호 규칙
- **오버랩 크로스 정규화(`normalize_overlapping_bone_weights`)**: 비강체 메쉬 정점만 스케일하며, 강체 정점(`rigid_mask`)은 100% 보존.
- **IB 경계면 테이퍼(`smooth_boundary_gradient_taper`, `operators.py`)**: 강체 정점(`rig_out > 0.5`)은 테이퍼 대상에서 엄격히 배제하여, 케이스 외곽이 깎여나가 무전기보다 작게 움직이는 버그를 원천 차단.
- **버퍼 정합 시 소프트 정점과의 블렌딩 금지**: `operators.py`에서 `unified_rigid_weights > 0.5`인 정점은 주변 피부 정점과의 반경 보간 없이 다이렉트 복사.

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
