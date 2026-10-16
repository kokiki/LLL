%% generate_ntn_csi_v9_fdd_dl.m
% =====================================================================================================
%  LEO NTN Ka-band **FDD 하향링크(DL)** CSI 데이터셋 생성기 (v9). v7_nonlinear_SuburbanRural_3.m 기반.
%  위성(16x16 UPA, 송신) -> UE(1안테나, 수신), fc = 20 GHz (Ka-band DL). FDD라 UL/DL 상호성 없음:
%  DL CSI는 UE가 CSI-RS로 추정해 피드백한다고 가정 -> 예측 horizon이 피드백/처리 지연을 보상.
%
%  v7_3 대비 변경점
%   (1) 관측 주기 1 ms (SCS 60 kHz 유지 -> 4슬롯마다 1회 관측). nrCDLChannel.InitialTime으로 관측
%       슬롯으로 바로 이동해 그 슬롯만 생성 (연속 생성과 동일한 페이딩, 검증 오차 ~5e-13).
%       1 ms = 정확히 4슬롯이라 관측 슬롯은 항상 같은 CP 구조(긴 CP 슬롯) -> 짝/홀 슬롯 오프셋 원천 제거.
%   (2) 위성 Doppler 보상을 nrCDLChannel의 sampleTimes(심볼별 CP를 반영한 실제 샘플 시각)로 pathGains에
%       직접 적용. 고정 슬롯길이 x 인덱스로 시각을 다시 계산하지 않음.
%       또한 nrCDLChannel은 MaximumDopplerShift(2)(위성 최대 Doppler)에 cos(고각)을 곱해 적용하므로,
%       최대값 fD_sat_max = (v/c)(Re/(Re+h)) fc 를 넣고 보상은 fD_sat = fD_sat_max*cos(고각)으로 정확히 맞춤
%       (v7은 cos가 이미 들어간 값을 넣고 그 값으로 보상해 fD_sat(1-cos) 크기의 잔차 회전이 남았음).
%       궤도 정보(ephemeris) 기반 개루프 사전보상 가정. Doppler 변화율 잔차(fDdot)와 LO 위상잡음은 유지.
%   (3) 에피소드 내 블로키지 전환: 기본 비활성(intraEpisodeBlockage = false). 위성 각속도(550km에서 ~0.8deg/s) 기준
%       각폭 1deg 장애물 통과에 ~1.3 s가 걸려 예측 윈도우(수십 ms)/에피소드(150 ms) 안의 전환은 물리적으로 드묾.
%       차단 여부(LOS/NLOS, clutter loss, shadow fading)는 기존대로 에피소드 단위로 결정.
%       true로 두면 주 경로에 2상태 Markov 감쇠(평균 체류 blockMeanClearMs/blockMeanBlockedMs)를 넣는 옵션(민감도 실험용).
%   (4) 위성 배열 방위각(TransmitArrayOrientation bearing): 기본 고정(0deg, CSI 예측 문헌 관행 -- TR 38.811 통계표는
%       고각만의 함수). randomBearing = true로 두면 에피소드마다 U[0,360) 샘플링(전용 RandStream, 견고성 실험용).
%   (5) 궤도: 고도 550 km, 속도 7.6 km/s. 고각은 기존과 같은 sin 스윕 순차 배정.
%   (6) Tp = 12 (12 ms), Tf = 16 ms. 에피소드 150 ms (관측 150개).
%   (7) 고각 구간 block split (생성 단계에서 바로 배정): test / val 구간 + test 경계 guard 구간(확인용 test로
%       별도 폴더 저장). 나머지 train.
%  실행: matlab -batch "generate_ntn_csi_v9b_fdd_dl"   (빠른 점검: matlab -batch "V9_QUICK=3; generate_ntn_csi_v9b_fdd_dl")
%  (v9b) v9와 동일하되 타이밍 오프셋을 에피소드 첫 관측에서 한 번만 추정해 고정 (관측별 재추정에 의한 위상 점프 제거)
% =====================================================================================================

%% ==========================================================================
%  generate_ntn_cdl_csi_dataset_v7_nonlinear.m
%
%  LEO 위성 NTN-CDL 기반 CSI(real/imag) 예측용 데이터셋 생성
%  - 위성 송신 안테나: 16 x 16 = 256개, UE 수신 안테나: 1 x 1 (단일 단말)
%  - 저장 형식 [10차/11차 수정]: **에피소드 1개당 파일 1개** - 다운샘플된 원본
%    스트림 dsReal/dsImag = [Nsc,nSlotsDs,16,16] (int16 양자화 + 에피소드별
%    스칼라 dsScale, [11차 수정] 참고) + causal 정규화용 causalMaxPerSlot +
%    메타데이터. Tp/Tf 윈도우 슬라이싱, SNR 노이즈 주입, causal 정규화 적용,
%    dsReal/dsImag 역양자화는 이제 여기서 하지 않고 common_dataloader.py가
%    학습 시점에 온더플라이로 수행합니다 (아래 [10차 수정]/[11차 수정] 참고).
%
%  [10차 수정 / 용량 문제 해결] 기존 방식(에피소드당 Tp+Tf=17슬롯 윈도우를
%  stride=1로 134개 슬라이딩, 윈도우마다 clean+noisy 버전과 atK 슬라이스까지
%  묶어 개별 .mat 파일로 저장)은 인접 윈도우끼리 94%가 중복되는 데이터라
%  에피소드당 약 7.3~7.8GB, nEpisodes=400 기준 총 약 2.8~3.2TB까지 부풀었습니다.
%  근본 원인은 "정보량 1개(에피소드 원본 스트림)"를 "저장량 134x2x1.3배"로
%  뻥튀기해서 저장한 것이었습니다. 해결책: 윈도우 슬라이싱/노이즈 주입/causal
%  정규화를 전부 학습 파이프라인(Python `common_dataloader.py`)으로 옮기고,
%  MATLAB은 에피소드당 원본 스트림 1개(single precision)만 저장합니다.
%  -> 에피소드당 약 92MB, 400개 총 약 37GB (기존 대비 약 80배 절감).
%  부수 효과: 학습 시 SNR을 에폭마다 다르게 랜덤 샘플링할 수 있어(데이터 증강),
%  val/test는 고정 시드로 재현 가능하게 만들 수 있음 (common_dataloader.py 참고).
%
%  [9차 수정] 실험 공정성/신뢰성 검토("실제 LEO 채널과 유사한가 / baseline과
%  제안모델이 공통으로 공정하게 쓸 수 있는 데이터인가")에 대한 대응 중 코드로
%  반영한 두 가지:
%   (1) 인과적(causal) 정규화: 기존에는 에피소드 전체(150슬롯) 최댓값 하나로
%       모든 윈도우를 정규화해서, 에피소드 앞쪽 윈도우조차 "미래" 슬롯 정보가
%       스케일에 섞여 들어가는 비인과적 구조였습니다. 이제는 다운샘플링 직후
%       슬롯별 running max(cummax)인 causalMaxPerSlot만 미리 계산해 에피소드
%       파일에 함께 저장해두고, 실제 나누기(윈도우 w의 정규화 = 그 윈도우의
%       마지막 관측 시점까지의 running max로 나눔)는 아래 [10차 수정]에 따라
%       common_dataloader.py가 윈도우를 만드는 시점에 적용합니다. baseline/DL
%       모두 동일 영향을 받으므로 상대비교를 왜곡하진 않지만, "실시간 인과
%       시스템" 주장을 엄밀히 뒷받침하려면 필요한 수정입니다.
%   (2) 메타데이터 유출(leakage) 방지: .mat에는 오라클/사후분석용 진짜 물리
%       파라미터(elevDeg, KdB, fDdot_sat 등)가 CSI 배열과 함께 저장되는데, 이건
%       baseline/DL 모델의 "입력 feature"로 쓰여선 안 됩니다(실제 수신기는
%       배포 시점에 이 값들을 알 수 없음). 이 파일 자체는 오라클 계산 편의를
%       위해 계속 함께 저장하되, 모델 학습 코드에서 CSI 필드만 쓰도록 강제하는
%       공통 데이터 로더 스텁을 별도 파일(common_dataloader.py)로 제공합니다.
%
%  [8차 수정 / 이번 버전의 핵심] "1.2dB 격차로는 DL 우수성을 주장하기 부족함 ->
%  시나리오에 비선형요소를 최대한 반영" 요청에 대한 대응. (다중단말 K=10 확장
%  버전(v6)은 스모크테스트 결과 "사용자간 정보풀링이 선형모델 NMSE를 거의
%  개선 못함(0.01~0.05dB)"으로 나와, 비선형성 증대 목적으로는 부적합 판단 ->
%  폐기하고 단일단말로 원복, 대신 채널 자체의 비선형 요소를 강화.)
%
%  타 논문 시나리오 설계 조사 결과 (2026-09-29 웹 조사, 상세 근거는 프로젝트
%  문서 참고):
%   - CPMamba(arXiv:2512.16315) 등 최신 채널예측 논문들은 "No-Prediction(직전값
%     유지)" 같은 매우 약한 baseline과 비교해 NMSE 격차를 크게 보고하는 경우가
%     많음 (70km/h에서 NP는 NMSE>1.0(발산)인 반면 DL은 <0.1) - 이는 이미 저희가
%     스모크테스트에서 확인한 "naive baseline은 과장된 격차를 만든다"는 함정과
%     일치. 반면 학습된 선형/파라메트릭 baseline과 비교한 결과는 격차가 훨씬
%     작음(Kalman-vs-ML 리뷰: 동일 복잡도 조건에서 "거의 같은 성능"이라는 지적
%     도 있음). 따라서 "약한 baseline 비교"가 아니라 "채널 자체의 정당한 추가
%     비선형 요소"를 찾는 방향으로 접근.
%   - CSI-4CAST(arXiv:2510.12996)는 채널 자체 외에 위상잡음(phase noise, 불완전
%     LO), 버스트 잡음, 패킷드랍 등 "하드웨어/수신단 결함"을 명시적으로 추가해
%     견고성을 테스트함 -> 물리채널만이 아니라 RF체인의 비선형/비정상(non-
%     stationary) 요소도 정당한 비선형성 원천이라는 근거로 채택.
%
%  실제로 추가한 두 가지 비선형 요소 (아래 3번째 파라미터 섹션 및 로컬함수
%  참고):
%   (A) *** 위성 Doppler "변화율(rate)" 잔차 - 물리적으로 가장 근거가 확실 ***
%       기존 v3~v6는 위성 Doppler(fD_sat)를 에피소드(150ms) 전체에 걸쳐 상수로
%       가정하고 선형 위상램프로 완전히 사전보상(de-rotation)했습니다. 그런데
%       사용자가 업로드한 TR 38.811 원문 6.7-8식 바로 다음 문장(단락 213)에
%       명시적으로: "The satellite speed, satellite elevation angle and UE
%       speed should be considered to be constant during the simulation
%       duration, **if limited to few TTIs**." - 즉 표준 스스로 "Doppler를
%       상수로 볼 수 있는 건 몇 TTI(=몇 ms) 이내일 때만"이라고 못박고 있습니다.
%       150ms(=150개 TTI, 15kHz SCS 기준)에 상수 가정을 그대로 쓰는 건 표준이
%       명시한 근사 범위를 크게 벗어난 것이고, 표준은 이를 넘어서는 경우를 위해
%       6.8.1절 "Time-varying Doppler shift"에서 더 일반적인 시변 모델을 별도로
%       소개합니다. 이 버전은 원형궤도-구형지구 기하로 Doppler의 실제 시간 변화
%       (S-curve)를 유도하고, 에피소드 대표 앙각에서의 국소 변화율(fDdot, Hz/s)
%       을 계산해 "선형 위상램프로는 제거되지 않는 잔차 2차(quadratic) 위상
%       항"으로 명시적으로 남겨둡니다. 이게 바로 "선형모델은 상수 회전으로
%       근사하지만 실제로는 회전속도 자체가 계속 바뀌는" 비선형성의 핵심이며,
%       인위적 트릭이 아니라 표준이 직접 언급한 근사의 한계를 메우는 것입니다.
%       (Nyquist 안전을 위해 에피소드 말미 잔차 순간주파수가 과도해지지 않도록
%       자동으로 스케일 제한을 겁니다 - 아래 dopplerRateResidualCap 참고.)
%   (B) *** 수신단 위상잡음(LO phase noise, Wiener/random-walk 과정) ***
%       CSI-4CAST 논문의 "불완전 LO" 설정을 참고해, 슬롯마다 정규분포 증분이
%       누적되는 랜덤워크 위상잡음을 채널에 곱해줍니다. 물리 채널의 fading과는
%       독립적인, 수신 체인 자체의 비정상(non-stationary) 오차 성분입니다.
%   (C) *** (옵션) 유한 분해능 양자화 ***
%       CSI 피드백/ADC의 유한 비트 양자화를 모사하는 균일 양자화기를 노이즈가
%       섞인 hist/pred에 추가로 적용할 수 있게 토글을 넣었습니다(기본 off,
%       필요시 applyQuantization=true).
%
%  *** 반영하지 않기로 결정한 요소 (근거 포함) ***
%   - "에피소드 도중(17ms 윈도우 내부)에 LOS/NLOS 상태가 전이"하도록 만드는 것:
%     저속 UE(3km/h=0.83m/s) 기준 17ms 동안 이동거리는 약 1.4cm에 불과합니다.
%     ITU-R P.681류 문헌의 실제 blockage dwell 거리(보통 수 m 이상)에 비해
%     3자리 이상 작아서, 윈도우 내부 전이를 강제로 넣는 건 물리적으로 정당화가
%     안 됩니다(이전 스모크테스트 논의에서도 같은 이유로 배제한 것과 일관).
%     에피소드 "사이"(150ms 간격)의 상태전이는 기존 설계로 이미 반영돼 있습니다.
%   - PA(전력증폭기) 메모리 비선형성(Rapp/memory polynomial): 송신단 RF 비선형
%     성이라 채널 자체와는 다른 레이어이고, 위성 HPA는 보통 강한 선형화/백오프
%     운용이 표준이라 "기본 시나리오"에 넣기엔 근거가 약함 - 필요시 별도 ablation
%     으로 추가 검토 권장.
%
%  이전 버전(v3~v6)까지 반영된 설계는 그대로 유지 (다중단말 확장은 폐기하고
%  단일단말로 원복):
%
%  [5차 수정] "horizon k = 1,2,4,8,10,12,14ms 각각에서 CSI 예측이 가능한가?"
%  질문에 대한 대응. v4까지는 snapshotIntervalMs(Δt)=5ms 로 다운샘플링해서
%  Tf=3 (=5,10,15ms 이렇게 딱 3개 시점만) 만 생성했기 때문에, k=1,2,4,8,12ms
%  처럼 5ms의 배수가 아닌 시점은 애초에 데이터에 존재하지 않아 예측 대상으로
%  삼을 수 없었습니다.
%
%  해결: Δt를 1슬롯(=1ms, 원래 채널 생성이 이루어지는 네이티브 해상도)으로
%  낮추고 horizonMs=14로 맞춰서 Tf=14로 늘렸습니다. 이러면 매 슬롯이 그대로
%  스냅샷이 되므로 pred 텐서의 k번째 열이 정확히 "k ms 뒤"의 CSI가 됩니다.
%  즉 predReal(:,k,:,:)를 그대로 꺼내 쓰면 k=1,2,4,8,10,12,14 등 원하는 모든
%  정수 ms horizon을 한 모델(다중출력 시퀀스 예측)로 동시에 다루거나, 필요한
%  열만 골라 단일-horizon 모델 학습에 쓸 수 있습니다. evalHorizonsMs 변수에
%  이 7개 값을 넣어뒀고, 매 샘플 저장 시 evalHorizonsMs와 그에 대응하는
%  predReal_at_k/predImag_at_k(호출 편의를 위해 미리 슬라이스된 버전)도 같이
%  저장합니다.
%
%  Nyquist 관점에서는 Δt를 1ms로 "줄이는" 것이므로 5ms일 때보다 오히려 여유가
%  더 커집니다(Δt가 작을수록 Nyquist 조건은 항상 더 잘 만족됨) - 문제될 것
%  없습니다. 다만 슬롯을 하나도 버리지 않고 다 쓰기 때문에 에피소드당 윈도우
%  개수와 pred 텐서 크기(Tf가 3->14로 늘어남)가 늘어나 저장 용량/생성 시간이
%  증가합니다 (아래 3번째 파라미터 섹션 참고).
%
%  [4차 수정 / 이번 버전의 핵심] 사용자가 업로드한 3GPP TR 38.811 원문
%  (38811-f40.doc)을 직접 파싱해서 20GHz(Ka-band) 전용 수치를 검증 추출하고,
%  기존에 논문 Table I(S-band 근처)에서 가져와 그대로 쓰던 placeholder
%  (delaySpread=30ns 고정, KdB=10dB 고정, NLOS fade depth 3~10dB 균등분포)를
%  전부 실측 Ka-band 표 기반 확률분포로 교체했습니다.
%
%  근거 표 (원문 그대로, band=Ka만 사용):
%   - Table 6.7.2-1b/2b (Dense Urban LOS/NLOS, Ka): Delay spread, K-factor
%   - Table 6.7.2-3b/4b (Urban LOS/NLOS, Ka)
%   - Table 6.7.2-7b/8b (Rural LOS/NLOS, Ka)
%   - Table 6.6.2-1/2/3 (Dense Urban/Urban/Suburban&Rural): Shadow fading
%     std(LOS), std(NLOS), 그리고 NLOS 평균 추가 clutter loss (최대 44.3dB!)
%   - Table 6.6.6.2.1-1 (20GHz, Toulouse, 99% 확률 대류권 섬광감쇠) - 작지만
%     Ka-band에서만 유의미한 추가 감쇠 성분이라 보너스로 반영
%
%  *** 왜 DenseUrban이 "비선형성이 잘 드러나는" 시나리오인가 (실측치 근거) ***
%   1) LOS 확률 자체가 가장 낮음 (Table 6.6.1-1, 이미 검증 완료: 앙각10도에서
%      28% ~ 앙각90도에서 98%) -> LOS/NLOS 상태전이(=선형모델이 못 따라가는
%      급격한 regime switch)가 세 시나리오 중 가장 자주 발생.
%   2) Ka-band NLOS 추가 clutter loss가 32.9~44.3dB로 압도적으로 큼
%      (Urban/Rural은 각각 6dB 고정, 10.7~11.8dB 수준) -> NLOS 진입 시
%      진폭이 거의 1/100~1/160으로 급락하는 매우 비선형적인 블로키지 이벤트.
%   3) LOS 구간 K-factor가 앙각에 비선형적으로 변함 (10도:6.1dB -> 20도:13.7dB
%      로 급등했다가 90도:7.6dB로 다시 하강, 즉 앙각에 대해 단조증가가 아님)
%      -> "앙각이 높을수록 좋아진다"는 선형적 직관/모델이 깨지는 지점.
%   따라서 이 스크립트는 scenario='DenseUrban' + fc=20GHz(Ka-band)를 기본값
%   으로 두고, Urban/Rural은 상대적으로 "선형에 가까운" 대조군(ablation)으로
%   사용하도록 유지합니다 (Urban 수치도 실측 반영했지만 K-factor 40dB대 구간은
%   저앙각 극단치라 참고용).
%
%  [문서 변환 관련 주의] 업로드된 .doc를 libreoffice로 .docx 변환 후
%  python-docx로 파싱했는데, Table 6.7.2-2b(Dense Urban NLOS Ka)의 delay
%  spread 표준편차(σlgDS) 행에서 원본 표의 병합 셀 구조 때문에 앙각 30°/40°
%  두 지점의 값이 완전히 분리되지 않고 겹쳐 나왔습니다 (둘 다 0.56으로 처리).
%  평균값(μlgDS)과 다른 모든 표는 9개 앙각 지점이 깨끗하게 분리되어 검증되었고,
%  이 σ값은 랜덤 지터의 폭을 정하는 2차 파라미터라 결과에 미치는 영향은
%  미미합니다. 논문에 표를 그대로 인용할 경우 원문 PDF/문서에서 해당 두 셀만
%  재확인하시는 것을 권장합니다.
%
%  [6차 수정] "연속된(overlap되는) 샘플이 섞여있어도 학습데이터 신뢰성에
%  문제없는지" 질문에 대한 대응. Tp+Tf=17개 슬롯짜리 윈도우를 stride=1로
%  뽑기 때문에 같은 에피소드 안의 인접 윈도우는 대부분 겹칩니다. 이 자체는
%  시퀀스 학습에서 흔한 방식(같은 궤적에서 더 많은 supervision을 뽑는 것)이라
%  문제가 아니지만, train과 val/test에 겹치는 윈도우가 섞여 들어가면 평가
%  NMSE가 실제보다 낙관적으로 나오는 정보 누수(leakage)가 생깁니다. 그래서
%  "윈도우"가 아니라 "에피소드" 단위로 먼저 무작위 70/15/15 분할하고, 한
%  에피소드의 모든 윈도우는 항상 같은 split(outDir/train|val|test)에만 저장
%  되도록 했습니다. 순차 블록이 아니라 무작위 배정인 이유는 elevSeq가 에피소드
%  인덱스에 대해 매끈한 sin 곡선이라 순차 분할 시 특정 앙각 구간이 한쪽에만
%  몰릴 수 있기 때문입니다.
%
%  이전 버전(v3/v4)까지 반영된 설계는 그대로 유지:
%   - UE 저속(3km/h), horizon 최댓값=14ms, Tp는 TODO로 분리
%   - 위성 벌크 Doppler(fD_sat) 사전보상(de-rotation)
%   - ChannelFiltering=false 무입력 cdl() 호출, 슬롯 정수배 청크(nrOFDMInfo 기반)
%   - NTN 전용 속성/Doppler 벡터 지원 여부 사전 확인
% ==========================================================================

%% ==========================================================================
%  [Suburban/Rural 변형 스크립트] 2026-09-29
%  원본 generate_ntn_cdl_csi_dataset_v7_nonlinear.m(DenseUrban, 핸드헬드 3km/h)의
%  scenario / ueSpeedKmh 두 파라미터만 바꾼 버전입니다. 나머지 로직(Doppler-rate
%  잔차, LO 위상잡음, 에피소드 단위 저장 구조 등)은 원본과 100% 동일합니다.
%
%  변경 의도: 현재 상용 LEO 광대역(Starlink류) 단말은 접시형 고정/준고정 형태로
%  하늘이 트인 교외/농촌에 설치되는 경우가 많아, DenseUrban+핸드헬드 시나리오가
%  이 사용 형태를 대표하지 못한다는 지적에 대한 대응입니다.
%    1) scenario: 'DenseUrban' -> 'Rural' (TR38.811의 'Suburban&Rural' 카테고리)
%    2) ueSpeedKmh: 3 -> 0.5 (보행자속도 -> 고정/준고정 거치대 수준)
%
%  예상되는 결과 변화 (실행 전 예측, 논문 Limitations/ablation 절에 활용 권장):
%   - LOS 확률이 전 앙각대에서 크게 높아짐(10도 기준 28% -> 78%) -> LOS/NLOS
%     regime switch 빈도가 줄어듦.
%   - NLOS 진입 시 평균 추가 clutter loss가 32.9~44.3dB -> 16.8~29.5dB로 절반
%     가까이 작아짐 -> "블로키지 급락" 이벤트의 강도 자체가 약해짐.
%   - UE측 Doppler(fD_ue)가 사실상 0에 가까워짐 -> 벌크 Doppler는 거의 전적으로
%     위성측(fD_sat)에서만 발생.
%   => 위 요소들이 DenseUrban 버전에서 baseline-oracle NMSE 격차(~8.5dB, k=14ms)의
%      상당 부분을 만들어냈으므로, 이 버전에서는 그 격차가 좁아질 것으로 예상됩니다.
%      이는 버그가 아니라 "고정단말 채널은 실제로 덜 비선형적"이라는 물리적으로
%      타당한 결과이며, DenseUrban 버전과 나란히 비교하는 것 자체가 유의미한
%      ablation(예: "DL 이득이 시나리오에 따라 어떻게 달라지는가")이 됩니다.
%
%  바꾸지 않은 것(원본과 동일, 의도적): 위성고도 550km(Starlink 궤도와 유사),
%  fc=20GHz(Ka-band, 원문 검증된 표가 이것뿐이라 유지 - 실제 Starlink 소비자단말은
%  주로 Ku-band를 쓰므로 이 부분은 여전히 근사임을 Limitations에 명시 권장),
%  Doppler-rate 잔차/LO 위상잡음/양자화 옵션/저장 구조/split 로직 전부 동일.
% ==========================================================================

%% ==========================================================================
%  [SCS=120kHz 재조정] 2026-09-30, [저장용량 절충 재조정] 2026-09-30 (같은 날 2차 수정)
%  1차: ueSpeedKmh 0.5->0.25, scsKHz 15->120(mu=3), NRB 25->264(3GPP FR2 최대,
%       6G 목표 380MHz) 까지 반영했더니 nEpisodes=400 기준 총 저장용량이 약
%       3.1TB까지 커짐 -> 사용자가 "용량이 너무 크다, 우선순위 낮은 것부터
%       깎자"고 요청.
%  2차(저장용량 절충, 최종): 우선순위 검토 결과
%    - NRB: 3GPP 6G 표준이 아직 없어 264는 "목표에 맞춰본 참고값"일 뿐이고,
%      Ka-band 지연확산(~30ns)의 coherence bandwidth(~5~7MHz)보다 점유대역폭
%      (380MHz)이 50배 이상 넓어 subcarrier 대부분이 서로 상관된(중복된)
%      정보라 시간축 CSI 예측이라는 연구질문에 기여가 낮음 -> **264->25로 원복
%      (1순위로 깎음, 저장용량 영향 10.56배로 가장 큼)**.
%    - SCS: §6.11에서 15/60/120kHz 전부 정성적 결론(baseline-oracle 격차
%      존재)이 안 흔들림을 이미 직접 검증함 -> **120->60kHz로 절충(2순위로
%      깎음, 저장용량 영향 2배)**. 120kHz(순수 5G NR-NTN Ka-band 표준 선례
%      근거)는 §6.11 부록 검증에서만 쓰고, 실제 데이터셋 생성은 60kHz로.
%    - nEpisodes: 학습곡선 실험(§6.9, 이번 세션 Rural+SCS 재검증)에서 실제로
%      줄이면 성능이 측정 가능하게 나빠짐을 확인한 유일한 항목 -> **400 유지
%      (깎지 않음, 최후순위)**.
%  이에 맞춰 snapshotIntervalMs/Tp/slotsPerEpisode를 "실시간 길이(3ms 히스토리,
%  150ms 에피소드, 14ms horizon)"가 그대로 유지되도록 비례 재환산(x4, SCS
%  15->60kHz 기준)했습니다(§6.11 SCS robustness 실험과 동일 원칙 - 실제로
%  globalDopplerRateScale이 SCS와 무관하게 0.1568로 동일하게 나옴을 스모크
%  테스트로 확인함).
%
%  *** 최종 용량/시간 영향 ***
%   - NRB=25(원복), SCS=60kHz(절충), nEpisodes=400(유지) 조합:
%     에피소드당 92MB(SCS15kHz 기준) -> **약 369MB**(SCS만 4배), nEpisodes=400
%     기준 총 37GB -> **약 147GB**. (참고: 3.1TB(NRB264+SCS120k) 대비 약 21배
%     절감, NRB만 원복한 295GB 대비도 약 2배 추가 절감.)
%   - 생성 시간: 에피소드당 cdl() 호출 횟수(nChunks)가 150->600(x4)로 늘어
%     MATLAB 생성 시간이 대략 4배 증가할 것으로 예상됩니다.
%   - 다운스트림: common_dataloader.py의 evalHorizonsMs->슬롯 인덱스 변환은
%     이미 snapshotIntervalMs 기반으로 일반화해뒀으므로(이전 수정에서 반영)
%     이번 SCS/NRB 재조정에 추가 수정이 필요 없습니다.
% ==========================================================================

%% ==========================================================================
%  [11차 수정 / dsReal·dsImag int16 양자화 저장] 2026-09-30 (같은 날 3차 수정)
%  위 147GB도 여전히 커서 추가 절감안을 검토: dsReal/dsImag를 single(4byte)
%  대신 int16(2byte)+에피소드별 스칼라 scale로 저장 -> 저장용량 약 절반
%  (147GB -> 약 74GB), Tp/SCS/NRB/nEpisodes는 전혀 건드리지 않음(정보량 손실
%  없이 "표현 정밀도"만 낮추는 방식).
%
%  안전성(정확도 저하 여부) 사전 검증: 이번 최종 설정(Rural, SCS=60kHz,
%  ueSpeedKmh=0.25, Tp=12/Tf=56, nEp=400)과 동일한 채널 통계로 Python에서
%  재현 실험(quant_impact_test.py)한 결과:
%   - 양자화 자체의 순수 SQNR(신호 대 양자화잡음비) ≈ 93.9dB
%   - 이미 파이프라인에 존재하는 의도적 SNR 노이즈(5~20dB, common_dataloader.py
%     make_window()에서 채널추정 잡음으로 주입)가 양자화잡음보다 70~90dB(전력비
%     10^7~10^9배) 더 강함 -> 양자화잡음은 완전히 묻힘.
%   - 실제 global_linreg NMSE(k=14ms) 비교: float32 파이프라인 -0.3253dB vs
%     int16 양자화 파이프라인 -0.3253dB, 차이 0.00000dB(측정 불가 수준).
%  결론: int16 양자화는 CSI 예측 정확도에 실질적 영향이 없음이 실측으로
%  확인되어 적용함.
%
%  구현: 에피소드별로 dsReal/dsImag를 합쳐 |H|의 최댓값(dsScale)을 풀스케일로
%  잡고, 각각 독립적으로 [-32767,32767] int16 범위에 선형 매핑해 저장.
%  common_dataloader.py의 load_episode()가 로드 시점에 dsScale로 나눠 즉시
%  float32로 역양자화하므로, 그 이후 파이프라인(make_window 등)은 전혀 수정할
%  필요가 없습니다.
% ==========================================================================

if ~exist('V9_QUICK', 'var'); V9_QUICK = 0; end   % >0이면 일부 에피소드만 생성해 *_quick 폴더에 저장 (점검용)
clc;

%% ---------------- 1. 파라미터 ----------------
% -- 위성 궤도
Re = 6371e3; satAlt = 550e3; satVel = 7.6e3;      % (v9) 고도 550 km, 속도 7.6 km/s
elevMinDeg = 10; elevMaxDeg = 90;

% -- 통신 파라미터
fc = 20e9;                    % *** Ka-band (20GHz) - TR38.811 6.7.2/6.6.2의 'b' 계열 표 사용 ***
scsKHz = 60; NRB = 25;          % *** [저장용량 절충 재조정, 최종] scsKHz: 120->60(절충, §6.11에서
                                %     15/60/120kHz 전부 정성적 결론 불변 검증됨). NRB: 264->25
                                %     (원복 - 3GPP 6G 대역폭 표준이 아직 없고, coherence
                                %     bandwidth(~5~7MHz)보다 264RB 점유대역폭(380MHz)이 훨씬 넓어
                                %     시간축 예측 연구에 기여가 낮다고 판단, 저장용량 절감효과는
                                %     10.56배로 가장 컸음). 위 헤더의 [저장용량 절충 재조정] 참고.
ueSpeedKmh = 0.25;              % *** ueSpeedKmh: 0.5->0.25. 접시형 고정/준고정 단말을 더
                                %     보수적으로(거치대 미세진동/열팽창 수준) 반영. 완전히
                                %     0으로 두면 UE측 Doppler(fD_ue)가 정확히 0이 되어 아래
                                %     Nyquist 체크의 nyquistLimMs가 Inf로 출력되긴 하나 실행에는
                                %     문제 없습니다. 완전 고정단말을 원하면 0으로 바꾸세요.
c = physconst('LightSpeed');

% -- 위성 안테나 배열 (256개 유지)
satArrayRows = 16; satArrayCols = 16;
numTxAnt = satArrayRows*satArrayCols;

% -- 예측 설계 파라미터
snapshotIntervalMs = 1;        % (v9) CSI 관측 간격 Δt = 1 ms (SCS 60kHz 슬롯 0.25 ms의 4슬롯마다 = NR CSI-RS 최소 주기)
horizonMs          = 16;       % (v9) 파일 메타데이터의 최대 horizon [ms] (로더는 Tf를 다시 정할 수 있어 k=24, 32도 학습 가능)
Tf = round(horizonMs / snapshotIntervalMs);   % = 16 (미래 관측 개수, Δt=1 ms이므로 16 ms)
Tp = 12;                        % 파일 메타데이터의 history 관측 개수 (= 12 ms). 학습에서는 --Tp 20 (20 ms)으로 덮어씀

% -- 평가용 다중 horizon [ms]. Δt = 1 ms이므로 관측 인덱스 = ms (로더는 k = round(k_ms / snapshotIntervalMs)로 환산)
%    evalHorizonsMs / Tp / Tf / snapshotIntervalMs는 메타데이터로 저장되어 로더가 그대로 사용
evalHorizonsMs = [1 2 4 8 12 16];   % (v9)
assert(all(evalHorizonsMs >= 1) && all(evalHorizonsMs <= horizonMs), ...
    'evalHorizonsMs의 모든 값은 1 이상, horizonMs(%d) 이하여야 합니다.', horizonMs);

% -- 시나리오/에피소드 파라미터
scenario       = 'Urban';       % TR 38.811 Urban 시나리오 (LOS 확률, 대규모 파라미터 표의 'Urban' 열 사용)
nEpisodes      = 400;          % [과적합 대응, 7차 수정] 유지 - 저장용량 절충 시 "깎지 않기로
                                % 결정한" 항목(최후순위). 70% 분할 기준 train=280개 에피소드.
                                % Python 프록시 학습곡선 검증(Rural+SCS=120kHz 조합, 세션 로컬
                                % learning_curve_rural_scs120k.py)에서도 train=210(=300개 설계
                                % 기준) 근방부터 이미 거의 포화, train=280(=400개)에서 안정적임을
                                % 확인 - 400개 유지가 안전(에피소드 수를 줄이는 건 §6.9/이번
                                % 검증에서 실제 성능 손실이 측정된 유일한 축이라 우선순위 최하).
slotsPerEpisode = 150;          % (v9) 에피소드당 관측 개수 = 150 ms / 1 ms (아래 루프는 관측 단위)

% -- [8차 수정] 추가 비선형 요소 파라미터 --
dopplerRateResidualCapHz = 150;    % *** (A) 잔차 Doppler-rate 위상항의 "에피소드 말미
                                    %     순간주파수" 상한[Hz]. Δt=1ms의 Nyquist 한계
                                    %     (500Hz)의 30% 이내로 안전 마진을 둠. 실제
                                    %     물리 fDdot이 이보다 큰 잔차를 만들면 자동으로
                                    %     비례 축소(스케일 다운)합니다(에일리어싱 방지).
phaseNoiseStdDegPerSqrtMs = 0.3;   % *** (B) LO 위상잡음 랜덤워크 표준편차 [deg/√ms].
                                    %     150ms 누적 시 대략 0.3*sqrt(150)≈3.7deg 수준의
                                    %     완만한 표준편차 - 채널 자체를 압도하지 않으면서
                                    %     비정상(non-stationary) 성분을 추가.
applyQuantization = false;         % *** (C, 옵션) hist/pred(noisy 버전)에 유한분해능
                                    %     양자화 적용 여부. 기본 off - 필요시 true로.
quantBits = 6;                     % 적용 시 사용할 비트 수 (균일 양자화기)

outDir = fullfile(pwd, sprintf('csi_dataset_FDD_DL_Kaband_%s_SCS%dkHz_dt1ms_v9b', scenario, scsKHz));   % (v9b: 타이밍 오프셋 에피소드 고정)
if V9_QUICK > 0; outDir = [outDir '_quick']; end
                                % *** 폴더명에 SCS를 붙여(scsKHz 변수로 자동 반영) 이전에 다른
                                %     SCS로 생성한 산출물과 겹쳐 덮어쓰지 않도록 함. 지금은
                                %     scsKHz=60이므로 csi_dataset_Kaband_pred_14ms_Rural_SCS60kHz ***
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% ---------------- [6차 수정] Train/Val/Test 에피소드 단위 분할 ----------------
% 연속된(overlap되는) 슬라이딩 윈도우가 서로 다른 split에 걸치면(train 시간대와
% val/test 시간대가 겹치면) 평가 NMSE가 낙관적으로 왜곡됩니다(정보 누수).
% 이를 막기 위해 "윈도우" 단위가 아니라 "에피소드" 단위로 먼저 나누고, 한
% 에피소드에서 나온 모든 윈도우는 항상 같은 split에만 들어가도록 합니다.
% 순차 블록이 아니라 무작위로 섞어서 배정하는 이유: elevSeq가 에피소드 인덱스에
% 대해 매끈한 sin 곡선(낮은 앙각 -> 정점 -> 낮은 앙각)이라, 앞/뒤로 순차 분할하면
% 특정 앙각 구간(=특정 LOS/NLOS 통계)이 한 split에만 몰릴 수 있습니다.
splitNames   = {'train', 'val', 'test', 'guard'};   % (v9) guard = test 경계 완충 구간 (확인용 test)
% (v9) 고각 구간 block split: test 구간은 양쪽을 train이 감싸도록 흩어 배치, test 경계 바깥 guard는
%      어느 학습에도 쓰지 않고 별도 폴더(확인용 test). val 구간도 흩어 배치.
testBandsDeg  = [12.5 14.5; 20.5 24.5; 40.5 44.5; 60.5 64.5; 81 83];
testGuardsDeg = [1.5 2.5 2.5 2.5 1.5];
valBandsDeg   = [30 35; 50 55; 70 75];
splitRngSeed = 42;                  % 재현성을 위한 고정 시드 (cdl.Seed와는 별개 스트림)

elevSeqForSplit = generateElevationProfile(elevMinDeg, elevMaxDeg, nEpisodes);   % 아래 elevSeq와 동일
episodeSplit = strings(nEpisodes, 1);
for ii = 1:nEpisodes
    e = elevSeqForSplit(ii);
    if any(e >= testBandsDeg(:,1) & e < testBandsDeg(:,2))
        episodeSplit(ii) = "test";
    elseif any(e >= valBandsDeg(:,1) & e < valBandsDeg(:,2))
        episodeSplit(ii) = "val";
    elseif any(e >= testBandsDeg(:,1) - testGuardsDeg(:) & e < testBandsDeg(:,2) + testGuardsDeg(:))
        episodeSplit(ii) = "guard";
    else
        episodeSplit(ii) = "train";
    end
end
fprintf('에피소드 분할(고각 block split): train=%d, val=%d, test=%d, guard(확인용)=%d (총 %d)\n', ...
    sum(episodeSplit=="train"), sum(episodeSplit=="val"), sum(episodeSplit=="test"), sum(episodeSplit=="guard"), nEpisodes);
% (v9) 블로키지/방위각 전용 난수 스트림 (채널 실현값 rand/randn 스트림과 분리)
rngBlock = RandStream('mt19937ar', 'Seed', 9001);
rngBearing = RandStream('mt19937ar', 'Seed', 9002);
randomBearing = false;                             % (v9) 배열 방위각: false = 고정(fixedBearingDeg), true = 에피소드별 무작위
fixedBearingDeg = 0;
intraEpisodeBlockage = false;                      % (v9) 에피소드 내 블로키지 전환 (기본 끔, 위 헤더 (3) 참고)
blockMeanClearMs = 80; blockMeanBlockedMs = 20;    % 2상태 Markov 평균 체류시간 (intraEpisodeBlockage=true일 때만)
blockDepthDB = [10 25];                            % 차단 시 주 경로 감쇠 [dB] (이벤트마다 균등)
blockRampMs = 4;                                   % 진입/이탈 전이 시간 (선형 dB 램프)

% [10차 수정] 이제 파일 1개 = 에피소드 1개이므로, 샘플(윈도우) 수가 아니라
% 에피소드 수를 센다.
episodeCountBySplit = struct('train', 0, 'val', 0, 'test', 0, 'guard', 0);
for sIdx = 1:numel(splitNames)
    subDir = fullfile(outDir, splitNames{sIdx});
    if ~exist(subDir, 'dir'); mkdir(subDir); end
end

%% ---------------- 슬롯 길이 계산 ----------------
probeCarrier = nrCarrierConfig;
probeCarrier.NSizeGrid = NRB;
probeCarrier.SubcarrierSpacing = scsKHz;
probeCarrier.CyclicPrefix = 'normal';
ofdmInfo = nrOFDMInfo(probeCarrier);
samplesPerSlot = sum(ofdmInfo.SymbolLengths(1:ofdmInfo.SymbolsPerSlot));
slotDurationMs = samplesPerSlot / ofdmInfo.SampleRate * 1e3;
SR = ofdmInfo.SampleRate;
fprintf('1슬롯 = %d 샘플 (%.3f ms), SR=%.3f MHz\n', samplesPerSlot, slotDurationMs, SR/1e6);
clear probeCarrier

slotStride = max(1, round(snapshotIntervalMs / slotDurationMs));
actualDeltaTms = slotStride * slotDurationMs;
fprintf('요청 Δt=%gms -> 실제 적용 Δt=%.3fms (슬롯 %d개 간격)\n', ...
    snapshotIntervalMs, actualDeltaTms, slotStride);

%% ---------------- Nyquist 체크 ----------------
fD_ue_check = (ueSpeedKmh/3.6) * fc / c;
nyquistLimMs = 1 / (2*fD_ue_check) * 1e3;
fprintf('UE %gkm/h -> 잔여 Doppler %.1fHz, Nyquist 한계 Δt < %.2fms (현재 Δt=%.2fms)\n', ...
    ueSpeedKmh, fD_ue_check, nyquistLimMs, actualDeltaTms);
if actualDeltaTms > nyquistLimMs
    warning('현재 Δt가 Nyquist 한계를 초과해 NLOS 성분이 alias될 수 있습니다. UE 속도를 더 낮추거나 Δt를 줄이세요.');
end

chunkSlots = 1;

%% ---------------- 메모리 사전 추정 ----------------
NP_worst = 24;
bytesPerChunk = (chunkSlots*samplesPerSlot) * NP_worst * numTxAnt * 1 * 16;
fprintf('청크당 pathGains 예상 메모리: %.1f MB (안테나 %d개 기준)\n', bytesPerChunk/1e6, numTxAnt);

%% ---------------- NTN 속성 지원 여부 사전 확인 ----------------
probeCdl = nrCDLChannel;
supportsElev = isprop(probeCdl, 'SatelliteElevationAngle');
supportsKF   = isprop(probeCdl, 'KFactorScaling');
fprintf('NTN 속성 지원 여부 - SatelliteElevationAngle: %d, KFactorScaling: %d\n', supportsElev, supportsKF);
if ~supportsElev || ~supportsKF
    warning('현재 MATLAB 릴리스가 NTN-CDL 전용 속성을 지원하지 않는 것으로 보입니다. ver 명령으로 확인하세요.');
end
clear probeCdl

%% ---------------- 위성 패스 앙각 시계열 ----------------
elevSeq = generateElevationProfile(elevMinDeg, elevMaxDeg, nEpisodes);

%% ---------------- [8차 수정-A, 버그수정] 전역(global) Doppler-rate 스케일 ----------------
% 최초 구현은 "에피소드별로 캡을 초과하면 그 에피소드만 캡까지 축소"하는
% 방식이었는데, 이러면 앙각 25~30도 이상인 에피소드 대부분이 캡 값 하나로
% 완전히 수렴해버려(fDdot이 전부 동일값) 선형모델이 그 고정 회전율 하나만
% 학습하면 되는 역설이 생깁니다(Python 스모크테스트로 실측 확인, 프로젝트
% 문서 6.4절 참고). 앙각별 상대적 차이를 보존하도록 앙각=90도(최악조건)
% 기준으로 스케일을 **한 번만** 계산해 모든 에피소드에 동일 비율로 적용합니다.
fDdot_worstcase_raw = dopplerRateAtElev(90, satAlt, satVel, fc, Re, c);
% *** [SCS=120kHz 재조정, 견고성 수정] slotDurationMs(네이티브 OFDM 슬롯 길이) 대신
%     actualDeltaTms(실제 적용되는 데이터셋 Δt, slotStride 보정 반영)를 사용하도록
%     변경. slotStride=1일 때는 두 값이 같아 결과는 동일하지만, 앞으로 SCS를 또
%     바꿀 때 slotStride!=1이 되는 경우에도 "에피소드 실시간 길이(slotsPerEpisode
%     * actualDeltaTms)"가 항상 올바르게 계산되도록 하기 위함 ***
worstResidualHz_global = abs(fDdot_worstcase_raw) * (slotsPerEpisode * snapshotIntervalMs * 1e-3);   % 에피소드 길이 150 ms
if worstResidualHz_global > dopplerRateResidualCapHz
    globalDopplerRateScale = dopplerRateResidualCapHz / worstResidualHz_global;
else
    globalDopplerRateScale = 1;
end
fprintf('[전역 Doppler-rate 스케일] elev=90도 raw fDdot=%.1fHz/s, worst residual=%.1fHz, globalDopplerRateScale=%.4f\n', ...
    fDdot_worstcase_raw, worstResidualHz_global, globalDopplerRateScale);

%% ---------------- 에피소드 루프 ----------------
episodeCount = 0;

epList = 1:nEpisodes;
if V9_QUICK > 0; epList = round(linspace(1, nEpisodes/2, V9_QUICK)); end
for epIdx = epList

    elevDeg = elevSeq(epIdx);
    curSplit = char(episodeSplit(epIdx));   % 'train' / 'val' / 'test' - 이 에피소드의 모든 윈도우가 여기로 감
    pLOS = losProbability(elevDeg, scenario);   % TR38.811 Table 6.6.1-1 (검증완료)
    isNLOS = rand() > pLOS;
    if isNLOS
        profile = 'NTN-CDL-A'; useKF = false;
    else
        profile = 'NTN-CDL-C'; useKF = true;
    end

    % --- Ka-band 실측 파라미터 샘플링 (TR38.811 Table 6.7.2-*b) ---
    [muLgDS, sigmaLgDS] = kaBandDelaySpreadParams(elevDeg, scenario, isNLOS);
    delaySpread = 10^(muLgDS + sigmaLgDS*randn());   % [s], log-normal 샘플링
    delaySpread = min(max(delaySpread, 1e-9), 1e-6);  % 수치적으로 비정상적인 극단치만 클리핑

    if useKF
        [muK, sigmaK] = kaBandKFactorParams(elevDeg, scenario);
        KdB = muK + sigmaK*randn();
    else
        KdB = NaN;  % NLOS에서는 사용 안 함
    end

    % --- Ka-band 실측 shadow fading / NLOS 추가 clutter loss (Table 6.6.2-*) ---
    [muSF, sigmaSF] = kaBandShadowFadeParams(elevDeg, scenario, isNLOS);
    shadowFadeDB = max(0, muSF + sigmaSF*randn());
    % LOS에서는 muSF가 수 dB 수준(순수 shadow fading), NLOS에서는 muSF 자체가
    % 32~44dB대 평균 clutter loss라서 결과적으로 매우 큰 감쇠가 걸립니다.

    % --- Ka-band 대류권 섬광감쇠 (Table 6.6.6.2.1-1, 20GHz Toulouse 99%) ---
    % 논문 수준의 정밀 모델은 아니고, 99% 확률 상한치를 균등분포 상한으로 써서
    % LOS/NLOS 공통으로 걸리는 작은 추가 감쇠(최대 ~1dB, 저앙각일수록 큼)를
    % 더합니다. Ka-band 특유의 성분을 하나 더 반영한다는 의미가 큽니다.
    tropoAttenBoundDB = kaBandTropoAttenBoundDB(elevDeg);
    tropoAttenDB = tropoAttenBoundDB * rand();

    totalExtraAttenDB = shadowFadeDB + tropoAttenDB;

    % (v9) nrCDLChannel은 MaximumDopplerShift(2)*cos(고각)을 위성 Doppler로 적용 -> 최대값을 넣고
    %      보상은 실제 적용값 fD_sat = fD_sat_max*cos(고각)으로 (TR 38.811 위성 Doppler와 동일)
    fD_sat_max = (satVel/c) * (Re/(Re+satAlt)) * fc;
    fD_sat = fD_sat_max * cosd(elevDeg);
    fD_ue  = (ueSpeedKmh/3.6) * fc / c;

    % --- [8차 수정-A] 위성 Doppler 변화율(fDdot): 전역 스케일 적용 (버그수정,
    % 위 "전역 Doppler-rate 스케일" 섹션 참고 - 에피소드별 개별 캡을 쓰면 안 됨) ---
    fDdot_sat_raw = dopplerRateAtElev(elevDeg, satAlt, satVel, fc, Re, c);   % [Hz/s]
    dopplerRateScale = globalDopplerRateScale;
    fDdot_sat = fDdot_sat_raw * dopplerRateScale;

    % --- [8차 수정-B] 에피소드별 LO 위상잡음(Wiener process) 시드 ---
    phaseNoiseStdRadPerSqrtMs = deg2rad(phaseNoiseStdDegPerSqrtMs);

    cdl = nrCDLChannel;
    cdl.DelayProfile = profile;
    cdl.DelaySpread = delaySpread;
    cdl.CarrierFrequency = fc;
    if isprop(cdl, 'SatelliteElevationAngle')
        cdl.SatelliteElevationAngle = elevDeg;
    end
    if useKF && isprop(cdl, 'KFactorScaling')
        cdl.KFactorScaling = true;
        if isprop(cdl, 'KFactor'); cdl.KFactor = KdB; end
    end
    try
        cdl.MaximumDopplerShift = [fD_ue, fD_sat_max];
    catch
        error('MaximumDopplerShift = [fD_ue fD_sat_max] (NTN 위성 Doppler) 미지원 릴리스');
    end
    cdl.SampleRate = SR;
    cdl.TransmitAntennaArray.Size = [satArrayRows satArrayCols 1 1 1];
    if randomBearing
        arrayBearingDeg = 360 * rand(rngBearing);             % (v9) 배열 방위각 에피소드별 샘플링 (옵션)
    else
        arrayBearingDeg = fixedBearingDeg;                    % (v9) 기본: 고정
    end
    cdl.TransmitArrayOrientation = [arrayBearingDeg; 0; 0];
    cdl.ReceiveAntennaArray.Size  = [1 1 1 1 1];
    cdl.ChannelFiltering = false;
    cdl.RandomStream = 'mt19937ar with seed';
    cdl.Seed = 1000 + epIdx;

    carrier = nrCarrierConfig;
    carrier.NSizeGrid = NRB;
    carrier.SubcarrierSpacing = scsKHz;
    carrier.CyclicPrefix = 'normal';

    %% --- (v9) 관측 단위 생성: 1 ms마다 관측 슬롯 하나만 생성 (InitialTime 이동) ---
    nObs = slotsPerEpisode;
    slotsPerSubframe = carrier.SlotsPerSubframe;
    % 블로키지 프로파일 (관측 시각 격자, 1 ms): clear/blocked 2상태 Markov + 선형 dB 램프
    tObsMs = (0:nObs-1) * snapshotIntervalMs;
    pBlockedStat = blockMeanBlockedMs / (blockMeanClearMs + blockMeanBlockedMs);
    blocked = rand(rngBlock) < pBlockedStat;  target = zeros(1, nObs);  depthNow = 0;
    if blocked; depthNow = blockDepthDB(1) + diff(blockDepthDB) * rand(rngBlock); end
    tNext = -log(rand(rngBlock)) * (blocked * blockMeanBlockedMs + (~blocked) * blockMeanClearMs);
    for n = 1:nObs
        while tObsMs(n) >= tNext
            blocked = ~blocked;
            if blocked; depthNow = blockDepthDB(1) + diff(blockDepthDB) * rand(rngBlock); end
            tNext = tNext - log(rand(rngBlock)) * (blocked * blockMeanBlockedMs + (~blocked) * blockMeanClearMs);
        end
        target(n) = blocked * depthNow;
    end
    blockAttenDB = zeros(1, nObs);  rampPerObs = max(blockDepthDB) / max(blockRampMs / snapshotIntervalMs, 1);
    for n = 2:nObs                                   % 목표 감쇠로 선형 램프 (전이에 blockRampMs 내외)
        dlt = target(n) - blockAttenDB(n-1);
        blockAttenDB(n) = blockAttenDB(n-1) + sign(dlt) * min(abs(dlt), rampPerObs);
    end
    blockAttenDB(1) = target(1);
    if ~intraEpisodeBlockage; blockAttenDB(:) = 0; end   % (v9 기본) 에피소드 내 전환 없음 (난수 소비는 동일하게 유지)

    phiAccum = 0;
    nSc = 12 * NRB;
    streamReal = zeros(nSc, nObs, satArrayRows, satArrayCols, 'single');
    streamImag = zeros(nSc, nObs, satArrayRows, satArrayCols, 'single');
    pathFilters = []; domPath = []; offset = [];
    for n = 1:nObs
        nSlotAbs = (n - 1) * slotStride;                         % 1 ms = 4슬롯 -> 항상 같은 CP 구조
        symIdx = mod(nSlotAbs, slotsPerSubframe) * ofdmInfo.SymbolsPerSlot + (1:ofdmInfo.SymbolsPerSlot);
        release(cdl);
        cdl.InitialTime = (n - 1) * snapshotIntervalMs * 1e-3;
        cdl.NumTimeSamples = sum(ofdmInfo.SymbolLengths(symIdx));
        [pathGains, sampleTimes] = cdl();
        if isempty(pathFilters)
            pathFilters = getPathFilters(cdl);
            ci = info(cdl);
            domPath = find(strcmp(ci.ClusterTypes, 'LOS'), 1);
            if isempty(domPath); [~, domPath] = max(ci.AveragePathGains); end
        end
        % 블로키지: 주 경로 감쇠
        pathGains(:, domPath, :, :) = pathGains(:, domPath, :, :) * 10^(-blockAttenDB(n)/20);
        % 위상항을 sampleTimes(실제 샘플 시각)로 pathGains에 적용:
        %  - 위성 Doppler 사전보상 (궤도 정보 기반, 실제 적용값 fD_sat)
        %  - Doppler 변화율 잔차 2차 위상 (선형 보상으로 제거되지 않는 성분)
        %  - LO 위상잡음 (Wiener, 관측 간격 1 ms 누적)
        phiAccum = phiAccum + phaseNoiseStdRadPerSqrtMs * sqrt(snapshotIntervalMs) * randn();
        ts = sampleTimes(:);
        ph = exp(-1i*2*pi*fD_sat*ts + 1i*pi*fDdot_sat*ts.^2 + 1i*phiAccum);
        pathGains = pathGains .* ph;                             % [Ncs x Np x Nt x Nr] (Ncs 축 브로드캐스트)
        if totalExtraAttenDB > 0
            pathGains = pathGains * 10^(-totalExtraAttenDB/20);
        end
        carrier.NSlot = nSlotAbs;
        % (v9b) 타이밍 오프셋은 에피소드 첫 관측에서 한 번만 추정해 고정. v9는 관측마다 재추정해 정수 샘플 기준이
        %  가끔 바뀌었고(경로 이득 변화로 최대 피크 위치 이동), 그때 CSI에 subcarrier 선형 위상 기울기
        %  e^{-j2π(k-k0)Δ/Nfft}가 곱해져 물리 채널과 무관한 위상 점프가 생겼음 (400개 중 수십 개 에피소드).
        %  CDL 경로 지연은 에피소드 내 고정이므로 고정 오프셋이 곧 일관된 타이밍 기준 (실제 수신기의 연속 추적에 해당)
        if isempty(offset)
            offset = nrPerfectTimingEstimate(pathGains, pathFilters);
        end
        Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
        H = squeeze(Hest(:, :, 1, :));
        H = reshape(H, size(H,1), size(H,2), satArrayRows, satArrayCols);
        streamReal(:, n, :, :) = real(H(:, end, :, :));
        streamImag(:, n, :, :) = imag(H(:, end, :, :));
        clear pathGains Hest H
    end
    dsReal = streamReal;   % (v9) 이미 관측 격자 -> 별도 다운샘플링 없음
    dsImag = streamImag;
    clear streamReal streamImag

    % [9차 수정] 인과적(causal) 정규화용 재료만 미리 계산: 각 다운샘플 슬롯까지의
    % running max(cummax). 실제 나누기는 윈도우가 만들어지는 시점(로더)에 그
    % 윈도우의 마지막 관측(history) 시점 값을 사용해 적용해야 causal합니다.
    dsMag = sqrt(dsReal.^2 + dsImag.^2);
    perSlotMax = squeeze(max(max(max(dsMag, [], 1), [], 3), [], 4));   % 1 x nSlots(ds)
    perSlotMax = perSlotMax(:)';                                       % 행벡터로 보장
    causalMaxPerSlot = cummax(perSlotMax);                             % 시간축 누적 최댓값 (인과적)
    clear dsMag perSlotMax

    nWindows = max(size(dsReal, 2) - (Tp + Tf) + 1, 0);   % 로더가 만들 수 있는 윈도우 개수 (기록/로그용)

    % [11차 수정] 저장 용량을 추가로 절반으로 줄이기 위해 single(4byte) 대신
    % int16(2byte)+에피소드별 스칼라 scale(dsScale)로 저장. 위 [11차 수정]
    % 헤더에서 검증한 대로 NMSE에 미치는 영향은 측정 불가 수준(0.00000dB)이라
    % 안전합니다. dsReal/dsImag를 합쳐 |H|의 최댓값을 풀스케일로 잡고 각각
    % 독립적으로 [-32767,32767]에 선형 매핑합니다 (round-to-nearest, mid-rise
    % 아님 - 0을 정확히 표현할 수 있도록).
    dsMagMax = max(max(abs(dsReal(:))), max(abs(dsImag(:))));
    if dsMagMax <= 0
        dsScale = 1;   % 방어적 처리 (이론상 발생하지 않지만 0-division 방지)
    else
        dsScale = single(dsMagMax);
    end
    dsReal = int16(round(dsReal / dsScale * 32767));
    dsImag = int16(round(dsImag / dsScale * 32767));
    causalMaxPerSlot = single(causalMaxPerSlot);

    duplexMode = 'FDD'; linkDirection = 'DL';                          % (v9) 메타데이터
    if V9_QUICK > 0   % 점검: 보상 후 잔차 공통 회전율(인접 관측 lag-1)과 블로키지 요약
        Xc = double(dsReal) + 1i*double(dsImag);
        r1 = sum(conj(Xc(:,1:end-1,:,:)) .* Xc(:,2:end,:,:), 'all');
        fprintf('  [quick] residual common rotation = %.1f Hz (lag-1 over 1 ms), fDdot_sat=%.1f Hz/s, block max=%.1f dB, blocked frac=%.2f, domPath=%d\n', ...
            angle(r1)/(2*pi*snapshotIntervalMs*1e-3), fDdot_sat, max(blockAttenDB), mean(blockAttenDB > 1), domPath);
    end
    episodeCount = episodeCount + 1;
    episodeCountBySplit.(curSplit) = episodeCountBySplit.(curSplit) + 1;
    save(fullfile(outDir, curSplit, sprintf('episode_%05d.mat', episodeCountBySplit.(curSplit))), ...
        'dsReal', 'dsImag', 'dsScale', 'causalMaxPerSlot', ...
        'evalHorizonsMs', 'Tp', 'Tf', 'snapshotIntervalMs', 'horizonMs', 'nWindows', ...
        'elevDeg', 'profile', 'fD_sat', 'fD_ue', 'fc', ...
        'scenario', 'delaySpread', 'KdB', 'shadowFadeDB', 'tropoAttenDB', 'curSplit', ...
        'fDdot_sat', 'fDdot_sat_raw', 'dopplerRateScale', 'phaseNoiseStdDegPerSqrtMs', ...
        'fD_sat_max', 'satAlt', 'satVel', 'arrayBearingDeg', 'blockAttenDB', 'domPath', ...
        'duplexMode', 'linkDirection', ...                                                   % (v9)
        'applyQuantization', 'quantBits', ...
        '-v7.3');

    clear cdl carrier dsReal dsImag causalMaxPerSlot dsMagMax dsScale

    fprintf('[%3d/%3d] elev=%.1f deg, bearing=%.0f, block=%.1fdB, profile=%s, split=%s, DS=%.1fns, K=%.1fdB, extraAtten=%.1fdB, fDdot=%.1fHz/s(scale=%.2f), 로더생성가능윈도우=%d, 누적 에피소드=%d\n', ...
        epIdx, nEpisodes, elevDeg, arrayBearingDeg, max(blockAttenDB), profile, curSplit, delaySpread*1e9, KdB, totalExtraAttenDB, fDdot_sat, dopplerRateScale, nWindows, episodeCount);
end

fprintf('완료: 총 %d개 에피소드 저장 (Ka-band 20GHz, scenario=%s, Tp=%d, Tf=%d, Δt=%.2fms)\n', ...
    episodeCount, scenario, Tp, Tf, actualDeltaTms);
fprintf('  -> train: %d개 에피소드 (%s/train), val: %d개 (%s/val), test: %d개 (%s/test)\n', ...
    episodeCountBySplit.train, outDir, episodeCountBySplit.val, outDir, episodeCountBySplit.test, outDir);
fprintf('  [10차 수정] 윈도우 슬라이싱/SNR 노이즈 주입/causal 정규화는 저장하지 않았습니다 -\n');
fprintf('  common_dataloader.py의 load_episode()/make_window()가 학습 시점에 온더플라이로 생성합니다.\n');

%% ---------------- 로컬 함수: 궤도/Doppler/LOS확률 (v3과 동일) ----------------
function elevSeq = generateElevationProfile(elevMin, elevMax, N)
    t = linspace(0, pi, N);
    elevSeq = elevMin + (elevMax - elevMin) * sin(t);
end

function fD = dopplerSatellite(elevDeg, h, vsat, fc, Re, c)
    % TR 38.811 단락 210-212 실측 수식 그대로 (검증 완료, 6.7-8식 주변).
    elevRad = deg2rad(elevDeg);
    fD = (vsat/c) * (Re/(Re+h)) * cos(elevRad) * fc;
end

%% ---------------- [8차 수정-A] 로컬 함수: 위성 Doppler 변화율(fDdot) ----------------
% TR 38.811 단락 213: "위성속도/앙각/UE속도는 시뮬레이션 구간이 '몇 TTI'로
% 짧을 때만 상수로 볼 수 있다"는 명시적 caveat에 대한 대응. 원형궤도-구형지구
% 기하로 Doppler의 정확한 시간함수(중심각 theta에 대한 S-curve)를 유도하고,
% 에피소드 대표 앙각에서의 국소 순간 변화율을 수치 미분으로 계산합니다.
%
% 유도: 위성-지구중심-지상국이 이루는 중심각 theta(위성이 천정 통과 시 0)에서
%   슬랜트거리 d(theta) = sqrt(Re^2 + Rh^2 - 2*Re*Rh*cos(theta))
%   앙각(elev)과 theta의 관계(구면기하, 표준 공식, 양 끝단 theta=0->elev=90도
%   / theta=thetaMax(=acos(Re/Rh), 지평선)->elev=0도로 검증됨):
%     sin(elev) = (cos(theta) - x) / sqrt(1 + x^2 - 2*x*cos(theta)),  x=Re/Rh
%   위 식을 theta에 대해 역으로 푼 닫힌형 해:
%     cos(theta) = x*cos(elev)^2 + sin(elev)*sqrt(1 - x^2*cos(elev)^2)
%   Doppler는 fD(t) = -(fc/c)*d(d)/dt = -(fc/c)*omega*Re*Rh*sin(theta)/d(theta)
%   (omega = vsat/Rh = 궤도 각속도, theta=omega*t). theta=0(천정)에서 fD=0이 되어
%   기존 dopplerSatellite (cos(elev) 공식, elev=90도->fD=0)와 정확히 일치함을
%   확인했습니다 (self-consistency check).
%   변화율 fDdot = omega^2 * d/dtheta[ -(fc/c)*Re*Rh*sin(theta)/d(theta) ] 를
%   theta0 근방 중심차분으로 수치 계산합니다.
function fDdot = dopplerRateAtElev(elevDeg, satAlt, satVel, fc, Re, c)
    Rh = Re + satAlt;
    omega = satVel / Rh;
    theta0 = elevToCentralAngle(elevDeg, Re, Rh);
    dTheta = 1e-5;   % rad, 수치미분 스텝
    gPlus  = dopplerCentralAngleFunc(theta0 + dTheta, Re, Rh, fc, c);
    gMinus = dopplerCentralAngleFunc(theta0 - dTheta, Re, Rh, fc, c);
    dg_dtheta = (gPlus - gMinus) / (2*dTheta);
    fDdot = omega^2 * dg_dtheta;   % [Hz/s]
end

function theta = elevToCentralAngle(elevDeg, Re, Rh)
    x = Re / Rh;
    g = deg2rad(elevDeg);
    u = x*cos(g)^2 + sin(g)*sqrt(max(1 - x^2*cos(g)^2, 0));
    u = min(max(u, -1), 1);
    theta = acos(u);
end

function val = dopplerCentralAngleFunc(theta, Re, Rh, fc, c)
    % "omega=1일 때"의 Doppler, 즉 -(fc/c)*dd/dtheta. 실제 fD(t)=omega*val(theta(t)).
    d = sqrt(Re^2 + Rh^2 - 2*Re*Rh*cos(theta));
    val = -(fc/c) * Re*Rh*sin(theta) / d;
end

function y = uniformQuantize(x, nBits, fullScale)
    % [8차 수정-C, 옵션] 균일 mid-rise 양자화기 (유한분해능 ADC/피드백 모사)
    levels = 2^nBits;
    step = 2*fullScale/levels;
    y = step * floor(x/step + 0.5);
    y = min(max(y, -fullScale), fullScale - step);
end

function pLOS = losProbability(elevDeg, scenario)
    % 3GPP TR 38.811 Table 6.6.1-1 - 업로드 원문과 대조하여 검증 완료.
    elevGrid = 10:10:90;
    switch scenario
        case 'DenseUrban'
            pTable = [0.282 0.331 0.398 0.468 0.537 0.612 0.738 0.820 0.981];
        case 'Urban'
            pTable = [0.246 0.386 0.493 0.613 0.726 0.805 0.919 0.968 0.992];
        case 'Rural'
            pTable = [0.782 0.869 0.919 0.929 0.935 0.940 0.949 0.952 0.998];
        otherwise
            error('알 수 없는 scenario: %s (DenseUrban/Urban/Rural 중 선택)', scenario);
    end
    pLOS = interp1(elevGrid, pTable, elevDeg, 'linear', 'extrap');
    pLOS = min(max(pLOS, 0), 1);
end

%% ---------------- 로컬 함수: Ka-band(20GHz) TR38.811 실측 파라미터 ----------------
% 아래 표들은 업로드된 38811-f40.doc(->docx 변환) 원문을 직접 파싱해 추출한
% 수치입니다 (Table 6.7.2-1b~8b 중 Ka-band 열, Table 6.6.2-1/2/3의 Ka-band 열,
% Table 6.6.6.2.1-1). elevGrid=10:10:90 지점의 표값을 선형보간해서 사용합니다.

function [muLgDS, sigmaLgDS] = kaBandDelaySpreadParams(elevDeg, scenario, isNLOS)
    elevGrid = 10:10:90;
    switch scenario
        case 'DenseUrban'
            if ~isNLOS  % Table 6.7.2-1b
                muTable    = [-7.43 -7.62 -7.76 -8.02 -8.13 -8.30 -8.34 -8.39 -8.45];
                sigmaTable = [ 0.90  0.78  0.80  0.72  0.61  0.47  0.39  0.26  0.01];
            else        % Table 6.7.2-2b
                muTable    = [-6.86 -6.84 -7.00 -7.21 -7.42 -7.86 -7.76 -8.07 -7.95];
                % *** 30/40도 지점 sigma는 문서 변환 과정의 병합셀 문제로 완전
                % 분리되지 않아 동일값(0.56)으로 근사했습니다 (본문 주석 참고) ***
                sigmaTable = [ 0.81  0.61  0.56  0.56  0.57  0.55  0.47  0.42  0.59];
            end
        case 'Urban'
            if ~isNLOS  % Table 6.7.2-3b
                muTable    = [-8.52 -8.59 -8.51 -8.49 -8.48 -8.44 -8.40 -8.37 -8.35];
                sigmaTable = [ 0.92  0.79  0.65  0.48  0.46  0.34  0.27  0.19  0.14];
            else        % Table 6.7.2-4b
                muTable    = [-7.24 -7.70 -7.82 -8.04 -8.08 -8.10 -8.16 -8.03 -8.33];
                sigmaTable = [ 1.26  0.99  0.86  0.75  0.77  0.76  0.73  0.79  0.70];
            end
        case 'Rural'
            if ~isNLOS  % Table 6.7.2-7b
                muTable    = [-9.68 -8.86 -8.59 -8.46 -8.36 -8.30 -8.26 -8.22 -8.21];
                sigmaTable = [ 0.46  0.29  0.18  0.19  0.14  0.15  0.13  0.03  0.07];
            else        % Table 6.7.2-8b
                muTable    = [-9.13 -8.39 -8.10 -7.96 -7.99 -8.05 -8.01 -8.05 -7.91];
                sigmaTable = [ 1.91  0.94  0.92  0.94  0.89  0.87  0.82  1.65  0.76];
            end
        otherwise
            error('알 수 없는 scenario: %s', scenario);
    end
    muLgDS    = interp1(elevGrid, muTable,    elevDeg, 'linear', 'extrap');
    sigmaLgDS = interp1(elevGrid, sigmaTable, elevDeg, 'linear', 'extrap');
    sigmaLgDS = max(sigmaLgDS, 0);
end

function [muK, sigmaK] = kaBandKFactorParams(elevDeg, scenario)
    % LOS 전용 (NLOS는 K-factor 정의 자체가 없음 = TR38.811 원문도 'N/A')
    elevGrid = 10:10:90;
    switch scenario
        case 'DenseUrban'   % Table 6.7.2-1b
            muTable    = [ 6.1 13.7 12.9 10.3  9.2  8.4  8.0  7.4  7.6];
            sigmaTable = [ 2.6  6.8  6.0  3.3  2.2  1.9  1.5  1.6  1.3];
        case 'Urban'        % Table 6.7.2-3b
            muTable    = [40.18 23.62 12.48  8.56  7.42  5.97  4.88  4.22  3.81];
            sigmaTable = [16.99 18.96 14.23 11.06 11.21  9.47  7.24  5.79  4.25];
        case 'Rural'        % Table 6.7.2-7b
            muTable    = [25.43 12.72  8.40  6.52  5.24  4.57  4.02  3.70  3.62];
            sigmaTable = [ 7.04  7.47  7.18  6.88  5.28  4.92  3.40  2.22  2.28];
        otherwise
            error('알 수 없는 scenario: %s', scenario);
    end
    muK    = interp1(elevGrid, muTable,    elevDeg, 'linear', 'extrap');
    sigmaK = interp1(elevGrid, sigmaTable, elevDeg, 'linear', 'extrap');
    sigmaK = max(sigmaK, 0);
end

function [muSF, sigmaSF] = kaBandShadowFadeParams(elevDeg, scenario, isNLOS)
    % Table 6.6.2-1(DenseUrban)/6.6.2-2(Urban)/6.6.2-3(Suburban&Rural), Ka-band 열.
    % LOS: muSF~0 취급(표는 std만 줌) + sigmaSF=표의 LOS std.
    % NLOS: muSF=표의 "추가 clutter loss 평균"(2번째 NLOS 열, dB) + sigmaSF=표의
    %       NLOS std(1번째 NLOS 열). 즉 NLOS에서는 평균 자체가 32~44dB대라 매우
    %       큰 블로키지 감쇠가 걸립니다.
    elevGrid = 10:10:90;
    switch scenario
        case 'DenseUrban'
            losStdTable  = [2.9 2.4 2.7 2.4 2.4 2.7 2.6 2.8 0.6];
            nlosStdTable = [17.1 17.1 15.6 14.6 14.2 12.6 12.1 12.3 12.3];
            nlosClTable  = [44.3 39.9 37.5 35.8 34.6 33.8 33.3 33.0 32.9];
        case 'Urban'
            losStdTable  = 4*ones(1,9);
            nlosStdTable = 6*ones(1,9);
            nlosClTable  = [44.3 39.9 37.5 35.8 34.6 33.8 33.3 33.0 32.9];
        case 'Rural'
            losStdTable  = [1.9 1.6 1.9 2.3 2.7 3.1 3.0 3.6 0.4];
            nlosStdTable = [10.7 10.0 11.2 11.6 11.8 10.8 10.8 10.8 10.8];
            nlosClTable  = [29.5 24.6 21.9 20.0 18.7 17.8 17.2 16.9 16.8];
        otherwise
            error('알 수 없는 scenario: %s', scenario);
    end
    if ~isNLOS
        muSF    = 0;
        sigmaSF = interp1(elevGrid, losStdTable, elevDeg, 'linear', 'extrap');
    else
        muSF    = interp1(elevGrid, nlosClTable,  elevDeg, 'linear', 'extrap');
        sigmaSF = interp1(elevGrid, nlosStdTable, elevDeg, 'linear', 'extrap');
    end
    sigmaSF = max(sigmaSF, 0);
end

function attenBoundDB = kaBandTropoAttenBoundDB(elevDeg)
    % Table 6.6.6.2.1-1: 20GHz, Toulouse, P{attenuation > x} < 1% 기준값.
    % 실제로는 CCDF에서 뽑은 랜덤변수지만, 여기서는 "99% 확률 상한치"를
    % 균등분포 상한으로 간이 사용합니다 (작은 보정 성분).
    elevGrid = 10:10:90;
    attenTable = [1.08 0.48 0.30 0.22 0.17 0.13 0.12 0.12 0.12];
    attenBoundDB = interp1(elevGrid, attenTable, elevDeg, 'linear', 'extrap');
    attenBoundDB = max(attenBoundDB, 0);
end
