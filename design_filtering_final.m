% Surya Ghosh
% 52426797
%
% EE3053 Design Exercise 2025

clc;
close all;
clear;

%% 1. Load Data
load("ecg_noisy.mat");
Fs = fs; % Use the sampling frequency loaded from the file

% Ensure data is column vector and rename for clarity
x = x(:);
y_clean = y_clean(:); % Provided clean signal (reference)

% 1.1 Plot raw data
t = (0:length(x)-1) / Fs;

figure;
plot(t,x);
hold on;
plot(t,y_clean);
xlim([0,t(end)]);
xlabel("Time (s)");
ylabel("Amplitude (mV)");
legend("Noisy (x)", "Clean (Reference y\_clean)")
title("ECG Given Data");


%% 2. IIR Band Stop filter for baseline wander (n=3)
Fs = fs;            % Sampling frequency (loaded from .mat file)
f_stop_low = 0.1;   % Lower edge of the stopband (Hz)
f_stop_high = 0.7;  % Upper edge of the stopband (Hz)
N_order = 3;        % Filter Order (3rd order Butterworth for efficiency)

% normalizes cutoff frequencies 
Wn_stop = [f_stop_low, f_stop_high] / (Fs/2);

[num_3, den_3] = butter(N_order, Wn_stop, 'stop');

% Use filtfilt to apply the filter forward and backward, eliminating phase distortion
y_iir_bandstop_filtered_3 = filtfilt(num_3, den_3, x);

figure;
t = (0:length(x)-1) / Fs;
plot(t, y_iir_bandstop_filtered_3);
xlim([5, 7]);
xlabel('Time (s)');
ylabel('Amplitude (mV)');
title('IIR Band-Stop Filter (n=3)');

%% 2.5 Pole/zero plot for IIR band stop filter (n=3)
figure('Position', [100, 100, 600, 600]);

% Get poles and zeros of the filter
poles = roots(den_3);
zeros = roots(num_3);

% Plot unit circle
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1);
hold on;

% Plot real and imaginary axes
plot([-1.2, 1.2], [0, 0], 'k-', 'LineWidth', 0.5);
plot([0, 0], [-1.2, 1.2], 'k-', 'LineWidth', 0.5);

plot(real(poles), imag(poles), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
plot(real(zeros), imag(zeros), 'bo', 'MarkerSize', 6, 'LineWidth', 2);

% Formatting
axis equal;
xlim([-1.2, 1.2]);
ylim([-1.2, 1.2]);
grid on;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Pole-Zero Plot: IIR Band-Stop Filter (n=3)');

% Add legend
legend('', '', '', 'Poles', 'Zeros');


%% 3 IIR Band Stop filter for baseline wander (n=6)
Fs = fs;            % Sampling frequency (loaded from .mat file)
f_stop_low = 0.1;   % Lower edge of the stopband (Hz)
f_stop_high = 0.7;  % Upper edge of the stopband (Hz)
N_order = 6;        % Filter Order (4th order Butterworth for efficiency)

% normalizes cutoff frequencies 
Wn_stop = [f_stop_low, f_stop_high] / (Fs/2);

[num_6, den_6] = butter(N_order, Wn_stop, 'stop');

% Use filtfilt to apply the filter forward and backward, eliminating phase distortion
y_iir_bandstop_filtered_6 = filtfilt(num_6, den_6, x);

figure;
t = (0:length(x)-1) / Fs;
plot(t, y_iir_bandstop_filtered_6);
xlim([5, 7]);
xlabel('Time (s)');
ylabel('Amplitude (mV)');
title('IIR Band-Stop Filter (n=6)');

%% 3.5 Pole/zero plot for IIR band stop filter (n=6)
figure('Position', [100, 100, 600, 600]);

% Get poles and zeros of the filter
poles = roots(den_6);
zeros = roots(num_6);

% Plot unit circle
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1);
hold on;

% Plot real and imaginary axes
plot([-1.2, 1.2], [0, 0], 'k-', 'LineWidth', 0.5);
plot([0, 0], [-1.2, 1.2], 'k-', 'LineWidth', 0.5);

plot(real(poles), imag(poles), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
plot(real(zeros), imag(zeros), 'bo', 'MarkerSize', 6, 'LineWidth', 2);

% Formatting
axis equal;
xlim([-1.2, 1.2]);
ylim([-1.2, 1.2]);
grid on;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Pole-Zero Plot: IIR Band-Stop Filter (n=6)');

% Add legend
legend('', '', '', 'Poles', 'Zeros');

%% 4 IIR Band Stop filter for baseline wander (n=4)
Fs = fs;            % Sampling frequency (loaded from .mat file)
f_stop_low = 0.1;   % Lower edge of the stopband (Hz)
f_stop_high = 0.7;  % Upper edge of the stopband (Hz)
N_order = 4;        % Filter Order (4th order Butterworth for efficiency)

% normalizes cutoff frequencies 
Wn_stop = [f_stop_low, f_stop_high] / (Fs/2);

[num_4, den_4] = butter(N_order, Wn_stop, 'stop');

% Use filtfilt to apply the filter forward and backward, eliminating phase distortion
y_iir_bandstop_filtered = filtfilt(num_4, den_4, x);

figure;
t = (0:length(x)-1) / Fs;
plot(t, y_iir_bandstop_filtered);
xlim([5, 7]);
xlabel('Time (s)');
ylabel('Amplitude (mV)');
title('IIR Band-Stop Filter (n=4)');

%% 4.5 Pole/zero plot for IIR band stop filter (n=4)
figure('Position', [100, 100, 600, 600]);

% Get poles and zeros of the filter
poles = roots(den_4);
zeros = roots(num_4);

% Plot unit circle
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1);
hold on;

% Plot real and imaginary axes
plot([-1.2, 1.2], [0, 0], 'k-', 'LineWidth', 0.5);
plot([0, 0], [-1.2, 1.2], 'k-', 'LineWidth', 0.5);

plot(real(poles), imag(poles), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
plot(real(zeros), imag(zeros), 'bo', 'MarkerSize', 6, 'LineWidth', 2);

% Formatting
axis equal;
xlim([-1.2, 1.2]);
ylim([-1.2, 1.2]);
grid on;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Pole-Zero Plot: IIR Band-Stop Filter (n=4)');

% Add legend
legend('', '', '', 'Poles', 'Zeros');

%% 5 FIR Band Stop filter for baseline wander 

N_fir = 5940;  

% Normalize the cutoff frequencies
Wn_stop_fir = [f_stop_low, f_stop_high] / (Fs/2);

% FIR band stop filter using the Hamming window.
b_fir_bs = fir1(N_fir, Wn_stop_fir, 'stop', hamming(N_fir + 1));

% Apply Filtering
y_fir_bs = filter(b_fir_bs, 1, x);

% FIR Delay Correction: Group Delay = N / 2 samples
delay_samples_fir = N_fir / 2;

% Trim the delayed output and the original signal to align them
y_fir_bandstop_aligned = y_fir_bs(delay_samples_fir + 1 : end);
x_aligned = x(1 : end - delay_samples_fir);
t_aligned = t(1 : end - delay_samples_fir); 


%% 9. Plotting FIR Band-Stop 

figure;
plot(t_aligned, x_aligned, 'DisplayName', 'Original Signal');
hold on;
plot(t_aligned, y_fir_bandstop_aligned, 'DisplayName', 'FIR Band Stop (aligned)');

xlabel('Time (s)');
ylabel('Amplitude (mV)');
title('FIR Filter (Aligned)');
legend('show');
xlim([0 20]);
grid on;

%% 9.1 FIR Band-stop with delay 

figure;
plot(t_aligned, x_aligned, 'DisplayName', 'Original Signal');
hold on;
plot(t, y_fir_bs,'DisplayName', 'FIR Band Stop (unaligned)');

xlabel('Time (s)');
ylabel('Amplitude (mV)');
title('FIR Filter (Unaligned)');
legend('show');
xlim([0 20]);
grid on;

%% 10. FFT of Noisy vs Clean vs FIR band stop vs IIR band stop
% FFT of FIR band stop
N_fft = 2^nextpow2(length(x));
Y_fir = fft(y_fir_bs, N_fft);
Ym_fir = abs(Y_fir(1:N_fft/2+1)) / length(y_fir_bs);
Ym_fir(2:end- (mod(N_fft,2)==0)) = 2 * Ym_fir(2:end- (mod(N_fft,2)==0));

% FFT of IIR band stop n=3
Y_bs3 = fft(y_iir_bandstop_filtered_3, N_fft);
Ym_bs3 = abs(Y_bs3(1:N_fft/2+1)) / length(y_iir_bandstop_filtered_3);
Ym_bs3(2:end- (mod(N_fft,2)==0)) = 2 * Ym_bs3(2:end- (mod(N_fft,2)==0));

% FFT of clean 
Y_ref = fft(y_clean, N_fft);
Ym_ref = abs(Y_ref(1:N_fft/2+1)) / length(y_clean);
Ym_ref(2:end- (mod(N_fft,2)==0)) = 2 * Ym_ref(2:end- (mod(N_fft,2)==0));

% FFT of noisy
X = fft(x, N_fft);
Xm = abs(X(1:N_fft/2+1)) / length(x);
Xm(2:end- (mod(N_fft,2)==0)) = 2 * Xm(2:end- (mod(N_fft,2)==0));

% Update y-axis limits based on all signals
combined_signals = [Xm; Ym_ref; Ym_fir; Ym_bs3];
ylim_max = max(combined_signals) * 1.05;

% Frequency vector 
N_fft = 2^nextpow2(length(x));
f = (0:(N_fft/2)) * (Fs / N_fft);

subplot(4,1,1);
plot(f, Xm);
ylabel('Magnitude (mV)'); 
title('FFT of Noisy ECG Signal');
grid on;
xlim([0, 2]); 
ylim([0, ylim_max]);

subplot(4,1,2);
plot(f, Ym_ref);
ylabel('Magnitude (mV)'); 
title('FFT of Clean ECG Signal');
grid on;
xlim([0, 2]); 
ylim([0, ylim_max]);

subplot(4,1,3);
plot(f, Ym_fir);
ylabel('Magnitude (mV)'); 
title('FFT of FIR Band-Stop Signal (n=5940)');
grid on;
xlim([0, 2]); 
ylim([0, ylim_max]);

subplot(4,1,4);
plot(f, Ym_bs3);
xlabel('Frequency (Hz)');
ylabel('Magnitude (mV)'); 
title('FFT of IIR Band-Stop Signal (n=3)');
grid on;
xlim([0, 2]); 
ylim([0, ylim_max]);

%% 10.1 Time domain plots of IIR band stop vs FIR Band-Stop vs. Clean Reference vs. Noisy vs IIR High pass 
y_clean_aligned = y_clean(1 : length(t_aligned));
y_iir_bandstop_aligned_3 = y_iir_bandstop_filtered_3(1 : length(t_aligned)); 

y_lim = [-1.5, 1.5]; 

figure;

subplot(4,1,1);
plot(t_aligned, x_aligned);
title('Original');
xlim([5, 7]); 
ylim(y_lim);
ylabel('Amplitude (mV)');
grid on;

subplot(4,1,2);
plot(t_aligned, y_fir_bandstop_aligned);
xlim([5, 7]); 
xlabel('Time (s)');
ylabel('Amplitude (mV)');
title('FIR Band Stop');
grid on;

subplot(4,1,3);
plot(t_aligned, y_iir_bandstop_aligned_3);
title('IIR Band Stop (n=3)');
xlim([5, 7]); 
ylim(y_lim);
xlabel('Time (s)');
ylabel('Amplitude (mV)');
grid on;

subplot(4,1,4);
plot(t_aligned, y_clean_aligned);
title('Clean Signal');
xlim([5, 7]); 
ylim(y_lim);
ylabel('Amplitude (mV)');
grid on;

%% 11. Removing mains hum at 50 Hz

% 50 Hz notch filter
f_notch = 50; 
Q = 25; 

% bandwidth needed from Q factor 
BW = f_notch / Q; % 1.43 Hz

% Cutoff frequencies 
f_low = f_notch - (BW / 2);   % 49.285 Hz
f_high = f_notch + (BW / 2);  % 50.715 Hz

% creating filter 
N_order_notch = 2;
Wn_notch = [f_low, f_high] / (Fs/2);
[b_n50, a_n50] = butter(N_order_notch, Wn_notch, 'stop');

% phase correction
y_50hz_removed = filtfilt(b_n50, a_n50, y_iir_bandstop_aligned_3); 

% FFT of the 50 Hz Removed Signal
Y_notch = fft(y_50hz_removed, N_fft);
Ym_notch = abs(Y_notch(1:N_fft/2+1)) / length(y_50hz_removed);
Ym_notch(2:end- (mod(N_fft,2)==0)) = 2 * Ym_notch(2:end- (mod(N_fft,2)==0));

% Plotting the 50 Hz Removal Effect (FFT Comparison)
figure;

subplot(2,1,1);
plot(f, Xm);
title('FFT of Noisy ECG Signal');
xlim([0, 120]); 
ylim([0, 0.26]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (mV)');
hold on;
grid on;

subplot(2,1,2);
plot(f, Ym_notch);
title('FFT of IIR band stop + notch at 50 Hz');
xlim([0, 120]);
ylim([0, 0.1]); 
xlabel('Frequency (Hz)');
ylabel('Amplitude (mV)');
grid on;

%% 12. Removing harmonic at 100 Hz
f_notch_100 = 100; 
Q = 35; 

% bandwidth needed from Q factor 
BW_100 = f_notch_100 / Q; 

% Cutoff frequencies 
f_low_100 = f_notch_100 - (BW_100 / 2);
f_high_100 = f_notch_100 + (BW_100 / 2); 

% Creating filter
N_order_notch = 2; 
Wn_notch_100 = [f_low_100, f_high_100] / (Fs/2);
[b_n100, a_n100] = butter(N_order_notch, Wn_notch_100, 'stop');


y_100hz_removed = filtfilt(b_n100, a_n100, y_50hz_removed); 

% FFT of the 100 Hz Removed Signal
Y_notch_100 = fft(y_100hz_removed, N_fft);
Ym_notch_100 = abs(Y_notch_100(1:N_fft/2+1)) / length(y_100hz_removed);
Ym_notch_100(2:end- (mod(N_fft,2)==0)) = 2 * Ym_notch_100(2:end- (mod(N_fft,2)==0));

%% 13. Final low pass filter
fc_lpf = 20;       
N_order_lpf = 6;

Wn_lpf = fc_lpf / (Fs/2);
[b_lpf, a_lpf] = butter(N_order_lpf, Wn_lpf, 'low');

y_final_denoised = filtfilt(b_lpf, a_lpf, y_100hz_removed); 

Y_final_fft = fft(y_final_denoised, N_fft);
Y_final = abs(Y_final_fft(1:N_fft/2+1)) / length(y_final_denoised);

Y_final(2:end- (mod(N_fft,2)==0)) = 2 * Y_final(2:end- (mod(N_fft,2)==0));

%% 13.5 Plotting FFTs

figure;
subplot(3,1,1);
plot(f, Ym_notch_100);
title('FFT of IIR bandstop and notches at 50/100 Hz');
xlim([0, 120]);
ylim([0, 0.12]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (mV)');
grid on;

subplot(3,1,2);
plot(f, Y_final);
xlabel('Frequency (Hz)');
ylabel('Magnitude (mV)');
title('FFT of Fully Filtered Signal');
grid on;
xlim([0, 120]); 
ylim([0, 0.12]);

subplot(3,1,3);
plot(f, Ym_ref);
xlabel('Frequency (Hz)');
ylabel('Magnitude (mV)'); 
title('FFT of Clean ECG Signal');
grid on;
xlim([0, 120]); 
ylim([0, 0.12]);


%% 14. Fully filtered signals time domain plot
figure;

subplot(2,1,1);
plot(t, y_clean);
title('Clean Signal');
xlim([0, 10]); 
ylim([-1.0, 1.0]);
ylabel('Amplitude (mV)');
grid on;

subplot(2,1,2);
plot(t_aligned(1:length(y_final_denoised)), y_final_denoised);
title('Fully Filtered Signal');
xlim([0, 10]); 
ylim([-1.0, 1.0]);
xlabel('Time (s)');
ylabel('Amplitude (mV)');
grid on;


%% 15. Performace evaluation (RMSE and Percent Error)

N_final = length(y_final_denoised); 
N_clean = length(y_clean);

y_clean_for_rmse = y_clean(1:N_final);
rmse = sqrt(mean((y_clean_for_rmse - y_final_denoised).^2));
avg_clean_amplitude = mean(abs(y_clean_for_rmse));

percent_error = (rmse / avg_clean_amplitude) * 100;

fprintf('RMSE: %.4f mV\n', rmse);
fprintf('Percent RMSE Error: %.4f%%\n', percent_error);

initial_noise = x(1:N_clean) - y_clean;
initial_noise_power = mean(initial_noise.^2);

final_noise = y_final_denoised - y_clean_for_rmse;
final_noise_power = mean(final_noise.^2);

SNR_improvement = 10 * log10(initial_noise_power / final_noise_power);


fprintf('Initial Noise Power: %.6f mV^2\n', initial_noise_power);
fprintf('Final Noise Power: %.6f mV^2\n', final_noise_power);
fprintf('SNR Improvement: %.4f dB\n', SNR_improvement);

%% 15. (AI generated) Wavelet Denoising for White Noise Removal 

% 15.1 Signal Preparation - Use ALIGNED signals
% We need to use the aligned signal to match time vectors
y_for_wavelet = y_100hz_removed;  % Use your final denoised signal

% Create corresponding time vector
t_for_wavelet = t_aligned(1:length(y_for_wavelet));

% 15.2 Wavelet Parameters
wname = 'sym8'; % Symlet 8 wavelet (good for ECG)
level = 5;      % Decomposition level

% 15.3 Decomposition (Discrete Wavelet Transform - DWT)
[C, L] = wavedec(y_for_wavelet, level, wname);

% 15.4 Threshold Calculation (Universal Threshold/VisuShrink)
% Estimate noise from finest detail coefficients
detail_coeffs = detcoef(C, L, 1);  % Get level 1 detail coefficients
sigma = median(abs(detail_coeffs)) / 0.6745; 
lambda = sigma * sqrt(2 * log(length(y_for_wavelet)));

% 15.5 Apply soft thresholding to DETAIL coefficients only
% (preserve approximation coefficients)
C_denoised = C;  % Start with all coefficients

% Apply threshold to each detail level
for i = 1:level
    % Get detail coefficients at level i
    coeffs = detcoef(C, L, i);
    
    % Apply soft thresholding with scaled threshold
    coeffs_thresh = wthresh(coeffs, 's', lambda/(2^(i/2)));
    
    % Put back into coefficient vector
    start_idx = sum(L(1:end-i)) + 1;
    end_idx = sum(L(1:end-i)) + L(end-i);
    C_denoised(start_idx:end_idx) = coeffs_thresh;
end

% 15.6 Reconstruction (Inverse DWT)
y_wavelet_denoised = waverec(C_denoised, L, wname);

% Ensure same length
if length(y_wavelet_denoised) > length(y_for_wavelet)
    y_wavelet_denoised = y_wavelet_denoised(1:length(y_for_wavelet));
elseif length(y_wavelet_denoised) < length(y_for_wavelet)
    y_wavelet_denoised(end+1:length(y_for_wavelet)) = 0;
end

%% (AI generated) 15.7 Time Domain Comparison
figure();

% Before wavelet
subplot(2,1,1);
plot(t_for_wavelet, y_for_wavelet);
title('Before Wavelet Denoising');
xlim([5, 10]);  % Zoom on a few beats
ylabel('Amplitude (mV)');
grid on;

% After wavelet
subplot(2,1,2);
plot(t_for_wavelet, y_wavelet_denoised);
title(['After Wavelet Denoising (Symlet 8, Level ' num2str(level) ')']);
xlim([5, 10]);  % Same zoom
ylabel('Amplitude (mV)');
grid on;
