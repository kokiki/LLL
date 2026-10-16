%% generate_ntn_csi_v9_fdd_dl.m
% =====================================================================================================
%  LEO NTN Ka-band FDD 하향링크(DL) CSI 데이터셋 생성기 (v9).
%  위성(16x16 UPA, 송신) -> UE(1안테나, 수신), fc = 20 GHz (Ka-band DL).
%  FDD라 UL/DL 상호성 없음: DL CSI는 UE가 CSI-RS로 추정해 피드백한다고 가정
%  -> 예측 horizon이 피드백/처리 지연을 보상하는 구조.
%
%  v9 최종 설계 (근거/의사결정 히스토리는 프로젝트 문서 "채널데이터_생성_방법론.md" 참고):
%   1) 관측 주기 1 ms (SCS=60kHz -> 4슬롯마다 1회). nrCDLChannel.InitialTime을 관측
%      슬롯으로 직접 이동시켜 그 슬롯만 생성(연속 생성과 동일 페이딩, 오차 ~5e-13).
%      1ms = 정확히 4슬롯이므로 관측 슬롯은 항상 같은 CP 구조 
%   2) 위성 Doppler 보상은 nrCDLChannel이 돌려주는 sampleTimes(심볼별 실제 CP를
%      반영한 샘플 시각)를 pathGains에 직접 적용. MaximumDopplerShift(2)에는 cos(고각) 적용 전의 최댓값(fD_sat_max)을
%      넣고, 보상 위상 : 실제 적용값 fD_sat = fD_sat_max*cos(고각)사용 
%   3) 에피소드 내 블로키지 전환은 기본 비활성(intraEpisodeBlockage=false): 위성
%      각속도(550km에서 ~0.8deg/s) 기준 각폭 1deg 장애물 통과에 ~1.3s가 걸려,
%      예측 윈도우(수십 ms)/에피소드(150ms) 안에서는 전환이 물리적으로 드묾.
%      LOS/NLOS, clutter loss, shadow fading은 기존대로 에피소드 단위로 결정.
%      true로 두면 주 경로에 2상태 Markov 감쇠를 추가하는 민감도-ablation 옵션.
%   4) 위성 배열 방위각(TransmitArrayOrientation bearing)은 기본 고정(0deg).
%      TR 38.811 통계표가 고각만의 함수이고 CSI 예측 선행연구들도 방위각을
%      모델링하지 않는 관행을 따름. randomBearing=true로 두면 에피소드마다
%      U[0,360) 샘플링하는 견고성-ablation 옵션(전용 RandStream으로 분리).
%   5) 궤도: 고도 550km, 속도 7.6km/s. 고각은 sin 스윕으로 에피소드 인덱스에 순차 배정.
%   6) Tp=12ms(히스토리), Tf=16ms(horizon), 에피소드 150ms(=150개 관측, 1ms 간격).
%   7) 고각 구간 block split을 생성 단계에서 바로 배정: test 구간(양쪽을 train이
%      감싸도록 분산 배치) + test 경계 바깥 guard 구간(어느 학습에도 안 쓰고
%      별도 폴더에 저장, 확인용) + val 구간. 나머지는 train.
%   8) 비선형 요소: (A) 위성 Doppler 변화율(fDdot) 잔차 2차 위상 - TR 38.811 단락
%      213의 "Doppler는 few TTI 내에서만 상수로 근사 가능" caveat에 대응해 명시적으로
%      남겨둔 성분. (B) 수신단 LO 위상잡음(Wiener/random-walk). (C, 옵션) 유한분해능
%      양자화(applyQuantization). scenario는 TR 38.811 Table 6.6.1-1/6.6.2-*/6.7.2-*b의
%      Ka-band 열을 그대로 사용(Rural/Urban/DenseUrban 중 선택, 기본 'Urban').
%   9) 저장: 에피소드 1개 = 파일 1개. dsReal/dsImag는 int16 양자화(+에피소드별 스칼라
%      dsScale)로 저장, causal 정규화용 causalMaxPerSlot(러닝맥스)도 함께 저장.
%      윈도우 슬라이싱/SNR 노이즈 주입/causal 정규화 나누기는 저장하지 않고 학습
%      파이프라인(common_dataloader.py)이 로드 시점에 온더플라이로 수행.
%
%  반영하지 않기로 결정한 요소: 에피소드 "도중" LOS/NLOS 전이(위 3번과 동일 근거로
%  배제, 에피소드 "사이" 전이는 기존 설계로 이미 반영됨), PA 메모리 비선형성(위성
%  HPA는 보통 강한 선형화/백오프 운용이 표준이라 근거 약함 - 필요시 별도 ablation).
%
%  실행: matlab -batch "generate_ntn_csi_v9_fdd_dl"
%  빠른 점검: matlab -batch "V9_QUICK=3; generate_ntn_csi_v9_fdd_dl"  (일부 에피소드만 *_quick 폴더에 생성)
% =====================================================================================================

if ~exist('V9_QUICK', 'var'); V9_QUICK = 0; end   % >0이면 일부 에피소드만 생성해 *_quick 폴더에 저장 (점검용)
clc;

%% ---------------- 1. 파라미터 ----------------
% -- 위성 궤도
Re = 6371e3; satAlt = 550e3; satVel = 7.6e3;      % 고도 550 km, 속도 7.6 km/s
elevMinDeg = 10; elevMaxDeg = 90;

% -- 통신 파라미터
fc = 20e9;                     % Ka-band (20GHz) - TR38.811 6.7.2/6.6.2의 'b' 계열(Ka) 표 사용
scsKHz = 60; NRB = 25;          % SCS=60kHz, NRB=25: coherence bandwidth(~5~7MHz) 대비 과도하게 세밀한
                                % 주파수축 샘플링(및 그로 인한 저장용량 폭증)을 피하기 위한 절충값.
ueSpeedKmh = 0.25;              % 접시형 고정/준고정 단말(거치대 미세진동/열팽창 수준). 0으로 두면
                                % fD_ue=0이 되어 Nyquist 체크의 nyquistLimMs가 Inf로 출력됨(정상).
c = physconst('LightSpeed');   % 광속
% -- 위성 안테나 배열
satArrayRows = 16; satArrayCols = 16;
numTxAnt = satArrayRows*satArrayCols;

% -- 예측 설계 파라미터
snapshotIntervalMs = 1;        % CSI 관측 주기 1 ms (SCS=60kHz 기준 4슬롯마다 1회)
horizonMs = 16;                 % 예측 최대 horizon [ms]
Tf = round(horizonMs / snapshotIntervalMs);   % = 16 (매 1ms가 그대로 1스텝)
Tp = 12;                        % 관측(history) 길이 = 12 (ms, dt=1ms 기준)

% -- 평가용 다중 horizon. Δt=1ms이므로 "인덱스(슬롯)=ms"가 그대로 성립해 common_dataloader.py는
%    k_slot = round(k_ms/snapshotIntervalMs)만으로 바로 슬라이스 가능. MATLAB은 evalHorizonsMs/
%    Tp/Tf/snapshotIntervalMs를 메타데이터로 저장하므로 로더가 이 값들로 직접 계산.
evalHorizonsMs = [1 2 4 8 12 16];
assert(all(evalHorizonsMs >= 1) && all(evalHorizonsMs <= horizonMs), ...
    'evalHorizonsMs의 모든 값은 1 이상, horizonMs(%d) 이하여야 합니다.', horizonMs);

% -- 시나리오/에피소드 파라미터
scenario = 'Urban';             % TR38.811 Table 6.6.1-1/6.6.2-2/6.7.2-3b~4b('Urban' 열) 사용.
                                 % 'Rural'/'DenseUrban'도 동일 로컬함수가 지원(severity-sweep ablation 후보).
nEpisodes = 400;                % 70% 분할 기준 train≈280개. 에피소드 수를 줄이면 학습곡선상 측정
                                 % 가능한 성능 손실이 생기는 것으로 확인되어(프로젝트 문서 §6.9) 유지.
slotsPerEpisode = 150;          % 관측 개수 = 150ms / 1ms

% -- 추가 비선형 요소 파라미터
dopplerRateResidualCapHz = 150;    % 잔차 Doppler-rate 위상항의 에피소드 말미 순간주파수 상한[Hz].
                                    % Δt=1ms의 Nyquist 한계(500Hz)의 30% 이내로 안전 마진.
phaseNoiseStdDegPerSqrtMs = 0.3;   % LO 위상잡음 랜덤워크 표준편차 [deg/√ms] (150ms 누적 시 ~3.7deg).
applyQuantization = false;         % (기본 off) hist/pred(noisy 버전)에 유한분해능 양자화 적용 여부
quantBits = 6;                     % 적용 시 사용할 비트 수 (균일 양자화기).

outDir = fullfile(pwd, sprintf('csi_dataset_FDD_DL_Kaband_%s_SCS%dkHz_dt1ms_v9', scenario, scsKHz));
if V9_QUICK > 0; outDir = [outDir '_quick']; end
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% ---------------- 2. Train/Val/Test 에피소드 단위 분할 (고각 block split) ----------------
% "윈도우" 단위가 아니라 "에피소드" 단위로 먼저 split을 정해, 한 에피소드의 모든 윈도우가 항상 같은 split에만 들어가도록 함(정보 누수 방지). 
% 고각 구간을 block 단위로 나눠 test/val을 분산 배치하고, test 경계 바깥에 guard 구간(어느 학습에도 안 쓰고 별도 폴더에 저장, 확인용 test)을 둠.
splitNames    = {'train', 'val', 'test', 'guard'};
testBandsDeg  = [12.5 14.5; 20.5 24.5; 40.5 44.5; 60.5 64.5; 81 83];
testGuardsDeg = [1.5 2.5 2.5 2.5 1.5];
valBandsDeg   = [30 35; 50 55; 70 75];
splitRngSeed = 42;   % 재현성을 위한 고정 시드 (cdl.Seed와는 별개 스트림)

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

% 블로키지/방위각 전용 난수 스트림 (채널 실현값 rand/randn 스트림과 분리)
rngBlock = RandStream('mt19937ar', 'Seed', 9001);
rngBearing = RandStream('mt19937ar', 'Seed', 9002);
randomBearing = false;            % (기본 끔) 배열 방위각: false=고정(fixedBearingDeg), true=에피소드별 무작위(ablation)
fixedBearingDeg = 0;
intraEpisodeBlockage = false;      % (기본 끔, 위 설계 3번 참고) off에피소드 내 블로키지 전환 
blockMeanClearMs = 80; blockMeanBlockedMs = 20;    % 2상태 Markov 평균 체류시간 (intraEpisodeBlockage=true일 때만)
blockDepthDB = [10 25];                            % 차단 시 주 경로 감쇠 [dB] (이벤트마다 균등)
blockRampMs = 4;                                    % 진입/이탈 전이 시간 (선형 dB 램프)

episodeCountBySplit = struct('train', 0, 'val', 0, 'test', 0, 'guard', 0);   % 파일 1개=에피소드 1개라 에피소드 수를 센다
for sIdx = 1:numel(splitNames)
    subDir = fullfile(outDir, splitNames{sIdx});
    if ~exist(subDir, 'dir'); mkdir(subDir); end
end

%% ---------------- 3. 슬롯 길이 계산 / Nyquist 체크 ----------------
probeCarrier = nrCarrierConfig;
probeCarrier.NSizeGrid = NRB; %25
probeCarrier.SubcarrierSpacing = scsKHz; %60kHz
probeCarrier.CyclicPrefix = 'normal';
ofdmInfo = nrOFDMInfo(probeCarrier);
samplesPerSlot = sum(ofdmInfo.SymbolLengths(1:ofdmInfo.SymbolsPerSlot)); %슬롯 1개의 실제 물리 시간계산
slotDurationMs = samplesPerSlot / ofdmInfo.SampleRate * 1e3;
SR = ofdmInfo.SampleRate;
fprintf('1슬롯 = %d 샘플 (%.3f ms), SR=%.3f MHz\n', samplesPerSlot, slotDurationMs, SR/1e6);
clear probeCarrier

%% 요청한 1ms 관측 주기와 슬롯 길이로 실제 적용 Δt 계산, Nyquist 한계 체크
slotStride = max(1, round(snapshotIntervalMs / slotDurationMs)); %4슬롯마다 1회 관측 
actualDeltaTms = slotStride * slotDurationMs;
fprintf('요청 Δt=%gms -> 실제 적용 Δt=%.3fms (슬롯 %d개 간격)\n', ...
    snapshotIntervalMs, actualDeltaTms, slotStride);

%% UE Doppler에 따른 Nyquist 한계 계산 - 관측간격이 NLOS 성분 alias를 방지할 수 있는지 확인
fD_ue_check = (ueSpeedKmh/3.6) * fc / c;
nyquistLimMs = 1 / (2*fD_ue_check) * 1e3;
fprintf('UE %gkm/h -> 잔여 Doppler %.1fHz, Nyquist 한계 Δt < %.2fms (현재 Δt=%.2fms)\n', ...
    ueSpeedKmh, fD_ue_check, nyquistLimMs, actualDeltaTms);
if actualDeltaTms > nyquistLimMs
    warning('현재 Δt가 Nyquist 한계를 초과해 NLOS 성분이 alias될 수 있습니다. UE 속도를 더 낮추거나 Δt를 줄이세요.');
end

chunkSlots = 1; %메모리 청크 단위로 몇 슬롯씩 끊어 버퍼를 할당할지 

%% ---------------- 4. 메모리 사전 추정 / NTN 속성 지원 여부 확인 ----------------
NP_worst = 24;
bytesPerChunk = (chunkSlots*samplesPerSlot) * NP_worst * numTxAnt * 1 * 16;
fprintf('청크당 pathGains 예상 메모리: %.1f MB (안테나 %d개 기준)\n', bytesPerChunk/1e6, numTxAnt);

probeCdl = nrCDLChannel;
supportsElev = isprop(probeCdl, 'SatelliteElevationAngle');
supportsKF   = isprop(probeCdl, 'KFactorScaling');
fprintf('NTN 속성 지원 여부 - SatelliteElevationAngle: %d, KFactorScaling: %d\n', supportsElev, supportsKF);
if ~supportsElev || ~supportsKF
    warning('현재 MATLAB 릴리스가 NTN-CDL 전용 속성을 지원하지 않는 것으로 보입니다. ver 명령으로 확인하세요.');
end
clear probeCdl

%% ---------------- 5. 위성 패스 앙각 시계열 / 전역 Doppler-rate 스케일 ----------------
elevSeq = generateElevationProfile(elevMinDeg, elevMaxDeg, nEpisodes);

% 전역(global) Doppler-rate 스케일: 에피소드별로 개별 캡을 적용하면 앙각 25~30도
% 이상인 에피소드 대부분이 캡 값 하나로 수렴해버려(fDdot이 전부 동일값) 선형모델이
% 그 고정 회전율 하나만 학습하면 되는 역설이 생김(실측 확인, 프로젝트 문서 §6.4).
% 앙각별 상대적 차이를 보존하도록 앙각=90도(최악조건) 기준으로 스케일을 한 번만
% 계산해 모든 에피소드에 동일 비율로 적용한다.
fDdot_worstcase_raw = dopplerRateAtElev(90, satAlt, satVel, fc, Re, c);
worstResidualHz_global = abs(fDdot_worstcase_raw) * (slotsPerEpisode * snapshotIntervalMs * 1e-3);   % 에피소드 길이 150 ms
if worstResidualHz_global > dopplerRateResidualCapHz
    globalDopplerRateScale = dopplerRateResidualCapHz / worstResidualHz_global;
else
    globalDopplerRateScale = 1;
end
fprintf('[전역 Doppler-rate 스케일] elev=90도 raw fDdot=%.1fHz/s, worst residual=%.1fHz, globalDopplerRateScale=%.4f\n', ...
    fDdot_worstcase_raw, worstResidualHz_global, globalDopplerRateScale);

%% ---------------- 6. 에피소드 생성 루프 ----------------
episodeCount = 0;

epList = 1:nEpisodes;
if V9_QUICK > 0; epList = round(linspace(1, nEpisodes/2, V9_QUICK)); end
for epIdx = epList

    elevDeg = elevSeq(epIdx);
    curSplit = char(episodeSplit(epIdx));   % 'train'/'val'/'test'/'guard' - 이 에피소드의 모든 윈도우가 여기로 감
    pLOS = losProbability(elevDeg, scenario);   % TR38.811 Table 6.6.1-1
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
    % 32~44dB대 평균 clutter loss라서 결과적으로 매우 큰 감쇠가 걸림.

    % --- Ka-band 대류권 섬광감쇠 (Table 6.6.6.2.1-1, 20GHz Toulouse 99%) ---
    % 99% 확률 상한치를 균등분포 상한으로 써서 LOS/NLOS 공통의 작은 추가 감쇠를 더함.
    tropoAttenBoundDB = kaBandTropoAttenBoundDB(elevDeg);
    tropoAttenDB = tropoAttenBoundDB * rand();

    totalExtraAttenDB = shadowFadeDB + tropoAttenDB;

    % nrCDLChannel은 MaximumDopplerShift(2)*cos(고각)을 위성 Doppler로 적용 -> 최댓값을 넣고
    % 보상은 실제 적용값 fD_sat = fD_sat_max*cos(고각)으로 (TR 38.811 위성 Doppler와 동일).
    fD_sat_max = (satVel/c) * (Re/(Re+satAlt)) * fc;
    fD_sat = fD_sat_max * cosd(elevDeg);
    fD_ue  = (ueSpeedKmh/3.6) * fc / c;

    % --- 위성 Doppler 변화율(fDdot): 전역 스케일 적용 (위 5번 섹션 참고) ---
    fDdot_sat_raw = dopplerRateAtElev(elevDeg, satAlt, satVel, fc, Re, c);   % [Hz/s]
    dopplerRateScale = globalDopplerRateScale;
    fDdot_sat = fDdot_sat_raw * dopplerRateScale;

    % --- 에피소드별 LO 위상잡음(Wiener process) 표준편차 ---
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
        arrayBearingDeg = 360 * rand(rngBearing);             % 배열 방위각 에피소드별 샘플링 (ablation)
    else
        arrayBearingDeg = fixedBearingDeg;                    % 기본: 고정
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

    %% --- 관측 단위 생성: 1 ms마다 관측 슬롯 하나만 생성 (InitialTime 이동) ---
    nObs = slotsPerEpisode;
    slotsPerSubframe = carrier.SlotsPerSubframe;
    % 블로키지 프로파일 (관측 시각 격자, 1 ms): clear/blocked 2상태 Markov + 선형 dB 램프.
    % intraEpisodeBlockage=false(기본)에서는 아래에서 blockAttenDB를 0으로 덮어쓰므로 실제
    % 채널에는 영향이 없음 - true로 바꾸는 민감도 실험에서 그대로 쓸 수 있도록 생성 로직만 유지.
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
    if ~intraEpisodeBlockage; blockAttenDB(:) = 0; end   % 기본: 에피소드 내 전환 없음 (난수 소비는 동일하게 유지)

    phiAccum = 0;
    nSc = 12 * NRB;
    streamReal = zeros(nSc, nObs, satArrayRows, satArrayCols, 'single');
    streamImag = zeros(nSc, nObs, satArrayRows, satArrayCols, 'single');
    pathFilters = []; domPath = [];
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
        % 위상항을 sampleTimes(실제 샘플 시각)로 pathGains에 직접 적용:
        %  위성 Doppler 사전보상 + Doppler 변화율 잔차 2차 위상 + LO 위상잡음(Wiener, 1ms 누적)
        phiAccum = phiAccum + phaseNoiseStdRadPerSqrtMs * sqrt(snapshotIntervalMs) * randn();
        ts = sampleTimes(:);
        ph = exp(-1i*2*pi*fD_sat*ts + 1i*pi*fDdot_sat*ts.^2 + 1i*phiAccum);
        pathGains = pathGains .* ph;                             % [Ncs x Np x Nt x Nr] (Ncs 축 브로드캐스트)
        if totalExtraAttenDB > 0
            pathGains = pathGains * 10^(-totalExtraAttenDB/20);
        end
        carrier.NSlot = nSlotAbs;
        offset = nrPerfectTimingEstimate(pathGains, pathFilters);
        Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
        H = squeeze(Hest(:, :, 1, :));
        H = reshape(H, size(H,1), size(H,2), satArrayRows, satArrayCols);
        streamReal(:, n, :, :) = real(H(:, end, :, :));
        streamImag(:, n, :, :) = imag(H(:, end, :, :));
        clear pathGains Hest H
    end
    dsReal = streamReal;   % 이미 관측 격자 -> 별도 다운샘플링 없음
    dsImag = streamImag;
    clear streamReal streamImag

    % causal 정규화용 재료만 미리 계산: 각 관측 슬롯까지의 running max(cummax).
    % 실제 나누기(causal하게, 그 윈도우의 마지막 history 시점까지의 max로 정규화)는
    % 윈도우가 만들어지는 학습 파이프라인 시점(common_dataloader.py)에서 적용.
    dsMag = sqrt(dsReal.^2 + dsImag.^2);
    perSlotMax = squeeze(max(max(max(dsMag, [], 1), [], 3), [], 4));   % 1 x nSlots
    perSlotMax = perSlotMax(:)';                                       % 행벡터로 보장
    causalMaxPerSlot = cummax(perSlotMax);                             % 시간축 누적 최댓값 (인과적)
    clear dsMag perSlotMax

    nWindows = max(size(dsReal, 2) - (Tp + Tf) + 1, 0);   % 로더가 만들 수 있는 윈도우 개수 (기록/로그용)

    % dsReal/dsImag를 single(4byte) 대신 int16(2byte)+에피소드별 스칼라 scale(dsScale)로
    % 저장해 용량을 절반으로 줄임. NMSE에 미치는 영향은 측정 불가 수준(0.00000dB, 프로젝트
    % 문서에서 검증)이라 안전. |H|의 최댓값을 풀스케일로 잡고 [-32767,32767]에 선형 매핑.
    dsMagMax = max(max(abs(dsReal(:))), max(abs(dsImag(:))));
    if dsMagMax <= 0
        dsScale = 1;   % 방어적 처리 (이론상 발생하지 않지만 0-division 방지)
    else
        dsScale = single(dsMagMax);
    end
    dsReal = int16(round(dsReal / dsScale * 32767));
    dsImag = int16(round(dsImag / dsScale * 32767));
    causalMaxPerSlot = single(causalMaxPerSlot);

    duplexMode = 'FDD'; linkDirection = 'DL';
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
        'duplexMode', 'linkDirection', ...
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
fprintf('  윈도우 슬라이싱/SNR 노이즈 주입/causal 정규화는 저장하지 않았습니다 -\n');
fprintf('  common_dataloader.py의 load_episode()/make_window()가 학습 시점에 온더플라이로 생성합니다.\n');

%% ==================== 7. 로컬 함수 ====================

%% ---- 궤도/Doppler/LOS확률 ----
function elevSeq = generateElevationProfile(elevMin, elevMax, N)
    t = linspace(0, pi, N);
    elevSeq = elevMin + (elevMax - elevMin) * sin(t);
end

function fD = dopplerSatellite(elevDeg, h, vsat, fc, Re, c)
    % TR 38.811 단락 210-212 실측 수식 (6.7-8식 주변).
    elevRad = deg2rad(elevDeg);
    fD = (vsat/c) * (Re/(Re+h)) * cos(elevRad) * fc;
end

% 위성 Doppler 변화율(fDdot): TR 38.811 단락 213("위성속도/앙각/UE속도는 시뮬레이션
% 구간이 few TTI로 짧을 때만 상수로 볼 수 있다")에 대응해, 원형궤도-구형지구 기하로
% Doppler의 정확한 시간함수(중심각 theta에 대한 S-curve)를 유도하고 에피소드 대표
% 앙각에서의 국소 순간 변화율을 수치 미분으로 계산한다. theta=0(천정)에서 fD=0이 되어
% dopplerSatellite(cos(elev) 공식)와 일치함을 self-consistency check로 확인했음.
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
    % 균일 mid-rise 양자화기 (유한분해능 ADC/피드백 모사, applyQuantization=true일 때 사용)
    levels = 2^nBits;
    step = 2*fullScale/levels;
    y = step * floor(x/step + 0.5);
    y = min(max(y, -fullScale), fullScale - step);
end

function pLOS = losProbability(elevDeg, scenario)
    % 3GPP TR 38.811 Table 6.6.1-1.
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

%% ---- Ka-band(20GHz) TR38.811 실측 파라미터 ----
% 아래 표들은 38811-f40.doc 원문의 Table 6.7.2-1b~8b(Ka-band 열), Table 6.6.2-1/2/3,
% Table 6.6.6.2.1-1에서 추출한 수치. elevGrid=10:10:90 지점을 선형보간해서 사용.
% (DenseUrban NLOS의 σlgDS 30/40도 지점은 원문 표의 병합 셀로 완전히 분리되지 않아
%  동일값(0.56)으로 근사 - 결과에 미치는 영향은 미미한 2차 파라미터.)

function [muLgDS, sigmaLgDS] = kaBandDelaySpreadParams(elevDeg, scenario, isNLOS)
    elevGrid = 10:10:90;
    switch scenario
        case 'DenseUrban'
            if ~isNLOS  % Table 6.7.2-1b
                muTable    = [-7.43 -7.62 -7.76 -8.02 -8.13 -8.30 -8.34 -8.39 -8.45];
                sigmaTable = [ 0.90  0.78  0.80  0.72  0.61  0.47  0.39  0.26  0.01];
            else        % Table 6.7.2-2b
                muTable    = [-6.86 -6.84 -7.00 -7.21 -7.42 -7.86 -7.76 -8.07 -7.95];
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
    % LOS: muSF~0(표는 std만 줌) + sigmaSF=표의 LOS std.
    % NLOS: muSF=표의 "추가 clutter loss 평균"(dB) + sigmaSF=표의 NLOS std
    %       (즉 NLOS에서는 평균 자체가 32~44dB대라 매우 큰 블로키지 감쇠가 걸림).
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
    % CCDF에서 뽑은 랜덤변수지만, 여기서는 "99% 확률 상한치"를 균등분포 상한으로 간이 사용.
    elevGrid = 10:10:90;
    attenTable = [1.08 0.48 0.30 0.22 0.17 0.13 0.12 0.12 0.12];
    attenBoundDB = interp1(elevGrid, attenTable, elevDeg, 'linear', 'extrap');
    attenBoundDB = max(attenBoundDB, 0);
end
