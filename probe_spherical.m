% probe_spherical.m -- 16x16 위성 배열: 정면을 LOS에서 구면 off-nadir eta만큼 기울이고(downtilt = 고각 + eta),
% 방위각은 배열 정면 축 회전(slant)으로 넣었을 때 LOS 위상 기울기 크기 = pi*sin(eta), 방향 = 180 - satAz 인지 확인
fc = 20e9; M = 16; N = 16; Re = 6371e3; Rh = Re + 550e3;
fprintf('%3s %5s %6s %7s | %8s %8s | %8s %8s | %8s %8s\n', 'sl', 'elev', 'satAz', 'eta', 'alpha', 'beta', '|slope|', 'pi*sinE', 'dir', '180-az');
for e = [10 30 60]
    eta = asind(Re/Rh*cosd(e));
    for sl = [1 -1]
        for satAz = [40 -120 170]
            cdl = nrCDLChannel;
            cdl.DelayProfile = 'NTN-CDL-C'; cdl.CarrierFrequency = fc; cdl.SatelliteElevationAngle = e;
            cdl.MaximumDopplerShift = [0 0]; cdl.SampleRate = 30.72e6; cdl.NumTimeSamples = 1;
            cdl.TransmitAntennaArray.Size = [M N 1 1 1]; cdl.ReceiveAntennaArray.Size = [1 1 1 1 1];
            cdl.TransmitArrayOrientation = [0; e + eta; sl*satAz];
            cdl.ChannelFiltering = false; cdl.RandomStream = 'mt19937ar with seed'; cdl.Seed = 1;
            [pg, ~] = cdl(); ci = info(cdl);
            los = find(strcmp(ci.ClusterTypes, 'LOS'), 1);
            H = reshape(squeeze(pg(1, los, :, 1)), M, N);
            a = angle(sum(sum(H(2:end, :) .* conj(H(1:end-1, :)))));
            b = angle(sum(sum(H(:, 2:end) .* conj(H(:, 1:end-1)))));
            fprintf('%3d %5d %6d %7.2f | %8.3f %8.3f | %8.3f %8.3f | %8.1f %8.1f\n', sl, e, satAz, eta, a, b, hypot(a, b), ...
                pi*sind(eta), atan2d(b, a), wrapTo180(180 - satAz));
        end
    end
end
