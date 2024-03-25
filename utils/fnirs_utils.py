"""

Extract feature from time series data of Hemoglobin to classify HCs and MDD.


"""


# load the pretreatment data 
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
import os
from scipy.signal import welch
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew
from xgboost import XGBClassifier
import pandas as pd 
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold

def train_model_using_loocv(data, label, model):
    loo = LeaveOneOut()
    result = []

    # Loop over each train/test split
    for train_index, test_index in loo.split(data):
        # Split the data into training and testing sets
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        # Train the classifier
        model.fit(X_train, y_train)

        # Predict the label for the test set
        y_pred = model.predict(X_test)

        # Append the accuracy to the list
        result.append([y_pred, y_test])

    return np.array(result), model

def print_md_table(model_name, set, metrics):
    print()
    print('| Model Name | Val/Test Set | Accuracy | Sensitivity | Specificity | F1 Score |')
    print('|------------|--------------|----------|-------------|-------------|----------|')
    print(f'| {model_name} | {set} |', end = '')
    for i in range(4):
        print(f" {metrics[i]:.4f} |", end = '')
    print()
    print(''*10)
    

def get_metrics(y_true, y_pred):
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 明确指定labels参数
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 现在cm是一个2x2矩阵，即使数据只包含一个类别
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, f1


def get_activity_start_time(data, index_start):
    gradient = np.gradient(data)
    max_gradient = np.argmax(gradient[0:int(index_start*1.2)])  # 0:index_start*4 # current index_start = 400,
    if max_gradient <= index_start:
        max_gradient = index_start
    return max_gradient


"""
Calulate the left slope based on the data sample time from activity_start_time and  activity_start_time + task_duration//2
"""


def get_left_slope(data, activity_start, task_duration):
    activity_start = int(activity_start)
    data = data[activity_start: activity_start+task_duration//2]
    slope, _ = np.polyfit(np.arange(data.shape[0]), data, 1)
    return slope


"""
Calulate the right slope based on the data sample time from activity_start_time + task_duration//2 and activity_start_time + task_duration  
"""


def get_right_slope(data, activity_start, task_duration):
    activity_start = int(activity_start)
    data = data[activity_start+task_duration//2: activity_start+task_duration]
    slope, _ = np.polyfit(np.arange(data.shape[0]), data, 1)
    return slope


"""
For calculating FWHM
"""


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def get_FWHM(y_values, activity_start, task_duration):
    # make sure the peak value is situated in the task duration period
    task = y_values[activity_start:activity_start+task_duration]
    max_task = np.max(task)  # Find the maximum y value
    half_max_y = max_task / 2.0
    max_index_task = np.argmax(task)
    # if max_index_task is in the first two values, set left_index to 0
    if max_index_task <= 1:
        left_index = 0
    else:
        left_index = find_nearest(y_values[:max_index_task], half_max_y)
    # if max_index_task is in the last two values, set right_index to the last value
    if max_index_task >= activity_start+task_duration-1:
        right_index = task_duration-1
    else:
        right_index = find_nearest(
            y_values[max_index_task:], half_max_y) + max_index_task

    return right_index - left_index


"""Get all 10 features"""


def get_10_features_xgboost_zhu(hbo, index_start, task_duration):
    feature_shape = hbo.shape[:2]
    # Feature 1 mean
    mean = np.mean(hbo, axis=2)  # feature shape is (subject, channel)

    # Feature 2 variance
    variance = np.std(hbo, axis=2)  # feature shape is (subject, channel)

    # Feature 3 activity_start_time
    activity_start_time = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            activity_start_time[sub, ch] = get_activity_start_time(
                hbo[sub, ch], index_start=index_start)

    # # Feature 4 left_slope
    left_slope = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            left_slope[sub, ch] = get_left_slope(
                hbo[sub, ch], activity_start=activity_start_time[sub, ch], task_duration=task_duration)
    # # Feature 5  right_slope
    right_slope = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            right_slope[sub, ch] = get_right_slope(
                hbo[sub, ch], activity_start=activity_start_time[sub, ch], task_duration=task_duration)

    # # Feature 6 kurtosis
    kurt = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            kurt[sub, ch] = kurtosis(hbo[sub, ch])
    # There might be some nan in kurtosis calucaltion because of all 0-value array
    kurt = np.nan_to_num(kurt)

    # # Feature 7 skewness
    skewness = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            skewness[sub, ch] = skew(hbo[sub, ch])
    # There might be some nan in skewness calucaltion because of all 0-value array
    skewness = np.nan_to_num(skewness)

    # # Feature 8 area under the curve AUC Based on the sample time from activity_start_time + task_duration
    AUC = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            activity_start = int(activity_start_time[sub, ch])
            AUC[sub, ch] = np.sum(
                hbo[sub, ch][activity_start:activity_start+task_duration])
    # for sub in range(10):
    #     plt.plot(AUC[sub])

    # # Feature 9 full width half maximum (FWHM)
    # FWHM
    FWHM = np.empty_like(mean)
    for sub in range(feature_shape[0]):
        for ch in range(feature_shape[1]):
            activity_start = int(activity_start_time[sub, ch])
            FWHM[sub, ch] = get_FWHM(
                hbo[sub, ch], activity_start, task_duration)
    # for sub in range(10):
    #     plt.plot(FWHM[sub])

    # # Feature 10 peak
    peak = np.max(hbo, axis=2)

    features = np.stack((mean, variance, activity_start_time,
                        left_slope, right_slope, kurt, skewness, AUC, FWHM, peak), axis=2)

    return features

# Normalize 
def normalize(data):
    # Iterate over each subject
    normalized_data = np.empty_like(data)
    for i in range(data.shape[0]):
        # Calculate the mean and standard deviation for the current subject
        mean = np.mean(data[i, :])
        std = np.std(data[i, :])

        # Perform z-normalization for the current subject
        normalized_data[i, :] = (data[i, :] - mean) / std
    return normalized_data


def li_svm_compute_10_fetures(hb, index_start=10, index_end=70):

    hb_task = hb[:, :, index_start:index_end]

    # 1. Integral Raw
    feature_1 = normalize(np.sum(hb_task, axis=2))
    # print(f' feature_1 - {feature_1.shape}')

    # 2. Integral Positive
    feature_2 = normalize(np.sum(np.where(hb_task < 0, 0, hb_task), axis=2))
    # print(f'feature_2 - {feature_2.shape}')

    # 3. Integral Zero-Norm
    feature_3 = normalize(np.sum(hb_task - np.min(hb_task, axis=(0, 1, 2)), axis=2))
    # print(f'feature_3 - {feature_3.shape}')

    # 4. Integral Absolute
    feature_4 = normalize(np.sum(np.abs(hb_task), axis=2))

    # 5. Integral (CUM)
    cum_hb_task = np.cumsum(hb_task, axis=2)
    feature_5 = normalize(np.sum(cum_hb_task, axis=2))
    # print(f'feature_5 - {feature_5.shape}')

    # 6. Integral (CUM) Positive
    cum_hb_task = np.cumsum(hb_task, axis=2)
    cum_hb_task = np.where(cum_hb_task < 0, 0, cum_hb_task)
    feature_6 = normalize(np.sum(cum_hb_task, axis=2))
    # print(f'feature_6 - {feature_6.shape}')

    # 7. Integral (CUM) Zero-Norm
    cum_hb_task = np.cumsum(hb_task, axis=2)
    cum_hb_task = cum_hb_task - np.min(cum_hb_task, axis=(0, 1, 2))
    feature_7 = normalize(np.sum(cum_hb_task, axis=2))
    # print(f'feature_7 - {feature_7.shape}')

    # 8. Integral (CUM) Absolute
    cum_hb_task = np.cumsum(hb_task, axis=2)
    cum_hb_task = np.abs(cum_hb_task)
    feature_8 = normalize(np.sum(cum_hb_task, axis=2))
    # print(f'feature_8 - {feature_8.shape}')

    # 9. Centroid (CUM) Positive
    cum_hb_task = np.cumsum(hb_task, axis=2)
    cum_hb_task = np.abs(cum_hb_task)
    hb_task_sum = np.sum(cum_hb_task, axis=2)/2
    abs_cum_hb_task_minus_sum_2 = np.abs(
        cum_hb_task - hb_task_sum[:, :, np.newaxis])
    feature_9 = normalize(np.argmin(abs_cum_hb_task_minus_sum_2, axis=2))
    # print(f'feature_9 - {feature_9.shape}')

    # 10. Centroid (CUM) Zero-Norm
    cum_hb_task = np.cumsum(hb_task, axis=2)
    cum_hb_task = cum_hb_task - np.min(cum_hb_task, axis=(0, 1, 2))
    hb_task_sum = np.sum(cum_hb_task, axis=2)/2
    abs_cum_hb_task_minus_sum_2 = np.abs(
        cum_hb_task - hb_task_sum[:, :, np.newaxis])
    feature_10 = normalize(np.argmin(abs_cum_hb_task_minus_sum_2, axis=2))
    # print(f'feature_10 - {feature_10.shape}')

    feature_sum = np.concatenate((feature_1[:,:,np.newaxis],
                                feature_2[:,:,np.newaxis],
                                feature_3[:,:,np.newaxis],
                                feature_4[:,:,np.newaxis],
                                feature_5[:,:,np.newaxis],
                                feature_6[:,:,np.newaxis],
                                feature_7[:,:,np.newaxis],
                                feature_8[:,:,np.newaxis],
                                feature_9[:,:,np.newaxis],
                                feature_10[:,:,np.newaxis]), axis=2)

    # print(res.shape)
    return feature_sum


def wang_alex_feature_selection(input, index_task_start,index_task_end,fs):


    index_task_start = 10
    index_task_end = 70
    fs = 1  # sampling rate

    # 1. Total: Sum of hemoglobin concentration changes in the whole process.
    feature_1 = np.sum(input, axis=2)[..., np.newaxis]
    # print(f'feature 1 shape -> {feature_1.shape}')

    # 2. Peak: Peak value of hemoglobin concentration changes in four periods.
    feature_2 = np.concatenate(
        (np.max(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.max(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.max(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.max(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    # print(f'feature 2 shape -> {feature_2.shape}')

    # 3. Valley: Valley value of hemoglobin concentration changes in four periods.
    feature_3 = np.concatenate(
        (np.min(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.min(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.min(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.min(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    # print(f'feature 3 shape -> {feature_3.shape}')

    # 4. Average: Mean value of hemoglobin concentration changes in four periods.
    feature_4 = np.concatenate(
        (np.mean(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.mean(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.mean(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.mean(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    # print(f'feature 4 shape -> {feature_4.shape}')

    # 5. Variance: Variance of hemoglobin concentration changes in four periods.
    feature_5 = np.concatenate(
        (np.var(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.var(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.var(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.var(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    # print(f'feature 5 shape -> {feature_5.shape}')

    # 6. Integral: Integral of hemoglobin concentration changes in four periods.
    feature_6 = np.concatenate(
        (np.sum(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.sum(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.sum(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.sum(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    # print(f'feature 6 shape -> {feature_6.shape}')


    def compute_linear_fitting(input):
        time = np.array(list(range(input.shape[-1])))
        n = len(time)
        sum_x = np.sum(time)
        sum_y = np.sum(input, axis=2)
        sum_x_squared = np.sum(time ** 2)
        sum_xy = np.sum(time * input, axis=2)

        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)

        b = (sum_y - m * sum_x) / n

        concatenate_m_b = np.concatenate(
            (m[..., np.newaxis], b[..., np.newaxis]), axis=-1)
        return concatenate_m_b  # slope and intercept


    feature_7 = np.concatenate(
        (compute_linear_fitting(input[..., :index_task_start]),
        compute_linear_fitting(input[..., index_task_start:index_task_end]),
        compute_linear_fitting(input[..., :index_task_end:])),
        axis=2
    )
    # print(f'feature 7 shape -> {feature_7.shape}')

    time = np.array(list(range(input.shape[-1])))
    feature_8 = np.zeros((input.shape[0], input.shape[1], 9))
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            # Fit a quadratic function (degree 2 polynomial) to the data
            # coefficients[0] is a, coefficients[1] is b, and coefficients[2] is c in the equation ax^2 + bx + c
            # the 2 specifies a quadratic function
            a1, b1, c1 = np.polyfit(
                time[:index_task_start], input[i, j, :index_task_start], 2)
            a2, b2, c2 = np.polyfit(
                time[index_task_start:index_task_end], input[i, j, index_task_start:index_task_end], 2)
            a3, b3, c3 = np.polyfit(
                time[index_task_end:], input[i, j, index_task_end:], 2)
            feature_8[i, j] = [a1, b1, c1, a2, b2, c2, a3, b3, c3]
    # print(f'feature 8 shape -> {feature_8.shape}')


    def compute_power_features(signal, fs):
        # Compute power spectral density using Welch's method
        f, Pxx_den = welch(signal, fs)

        # Find the indices corresponding to the frequency bands
        idx_band1 = np.where((f >= 0.01) & (f <= 0.25))
        idx_band2 = np.where((f >= 0.25) & (f <= 0.5))

        # Feature 1: maximum power in the first frequency band
        Pnu1 = np.max(Pxx_den[idx_band1])

        # Feature 2: maximum power in the second frequency band
        Pnu2 = np.max(Pxx_den[idx_band2])

        # Find index closest to 0.01 Hz and 0.25 Hz
        idx_closest_001 = np.argmin(np.abs(f - 0.01))
        idx_closest_025 = np.argmin(np.abs(f - 0.25))

        # Feature 3: subtract power at 0.01 Hz from maximum power in first frequency band
        Pnu10 = Pxx_den[idx_closest_001]
        feature3 = Pnu1 - Pnu10

        # Feature 4: subtract power at 0.25 Hz from maximum power in second frequency band
        Pnu20 = Pxx_den[idx_closest_025]
        feature4 = Pnu2 - Pnu20

        # Feature 5: ratio of feature 4 to feature 3
        feature5 = feature4 / feature3

        return Pnu1, Pnu2, feature3, feature4, feature5


    feature_9 = np.zeros((input.shape[0], input.shape[1], 5))
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            feature_9[i, j] = compute_power_features(input[i, j], fs)
    # print(f'feature 9 shape -> {feature_9.shape}')


    q = np.mean(input)  # threhold
    # print(f'threshold is set to {q} ')


    def entropy_wsh(y):
        return -np.sum(y*y * np.log10(y*y), axis=1)


    def entropy_we(y):
        return np.sum(y, axis=1)


    def entropy_wp(y):
        t1 = np.abs(y) ** 2
        t2 = np.linalg.norm(y, ord=2, axis=1) ** 2 / y.shape[1]
        # original should be
        # t2 = np.linalg.norm(y, ord=norm_q) ** norm_q
        # but this will cause the value to be 1000x time compared to other entropy values.
        # So I added / y.shape[1]
        return np.sum(t1-t2[..., np.newaxis], axis=1)


    def entropy_wt(y):
        b = np.where(y > q, 1, 0)
        return np.sum(b, axis=1)


    def entropy_wsu(y):
        t1 = y.shape[0]
        t2 = np.sum(np.where(y > q, 0, 1), axis=1)
        t3 = min(np.min(y**2), q**2)
        return t1 - t2 + t3

    # Bacuse using wavelet decomposition will

    def calculte_entropy(x):
        coeffs = pywt.wavedec(x, 'db6', level=4)
        max_length = max(len(coeff) for coeff in coeffs)
        y = np.array([np.pad(coeff, (0, max_length - len(coeff)),
                    constant_values=(q)) for coeff in coeffs])
        return np.concatenate((entropy_wsh(y), entropy_we(y), entropy_wp(y), entropy_wt(y), entropy_wsu(y)), axis=0)

    feature_10 = np.zeros((input.shape[0], input.shape[1], 25))
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            feature_10[i, j] = calculte_entropy(input[i, j])
    # print(f'feature 10 shape -> {feature_10.shape}')

    nor_all_feature = np.concatenate(
        (normalize(feature_1),
        normalize(feature_2),
        normalize(feature_3),
        normalize(feature_4),
        normalize(feature_5),
        normalize(feature_6),
        normalize(feature_7),
        normalize(feature_8),
        normalize(feature_9),
        normalize(feature_10)),
        axis=2
    )

    return nor_all_feature




def extract_hb_core_temporal_features(Hb):
    # 1.Average
    feature_average = np.mean(Hb, axis=2)

    # 2.Maximum
    feature_maximum = np.max(Hb, axis=2)

    # 3.Minimum
    feature_minimum = np.min(Hb, axis=2)

    # 4.Variance
    feature_variance = np.var(Hb, axis=2)

    # 5.Skewness
    feature_skewness = np.mean((Hb - feature_average[:, :, np.newaxis]) ** 3)
    feature_skewness /= np.std(Hb, axis=2) ** 3

    # 6.Kurtosis
    n = Hb.shape[-1]
    # Standard deviation of the data
    std_dev = np.std(Hb, axis=2)

    # Calculate the fourth moment
    fourth_moment = np.mean(
        (Hb - feature_average[:, :, np.newaxis]) ** 4, axis=2)

    # Calculate the kurtosis
    feature_kurtosis = (n * (n + 1) * fourth_moment) / ((n - 1) * (n - 2)
                                                        * (n - 3) * (std_dev ** 4)) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    all_feature = np.concatenate((normalize(feature_average[:, :, np.newaxis]),
                                  normalize(feature_maximum[:, :, np.newaxis]),
                                  normalize(feature_minimum[:, :, np.newaxis]),
                                  normalize(feature_variance[:, :, np.newaxis]),
                                  normalize(feature_skewness[:, :, np.newaxis]),
                                  normalize(feature_kurtosis[:, :, np.newaxis])), axis=2)
    return all_feature


def temporal_feature_extract_yu_gnn_full(input, index_start, index_end, hbo_type, hbr_type):

    feature_silent_1_hbo = extract_hb_core_temporal_features(
        input[:, :, :index_start, hbo_type])#, hbo_type
    feature_silent_1_hbr = extract_hb_core_temporal_features(
        input[:, :, :index_start, hbr_type])

    feature_task_hbo = extract_hb_core_temporal_features(
        input[:, :, index_start:index_end, hbo_type])#, hbo_type
    feature_task_hbr = extract_hb_core_temporal_features(
        input[:, :, index_start:index_end, hbr_type])

    feature_silent_2_hbo = extract_hb_core_temporal_features(
        input[:, :, index_end:, hbo_type])#, hbo_type
    feature_silent_2_hbr = extract_hb_core_temporal_features(
        input[:, :, index_end:, hbr_type])
    # output = np.concatenate(
    #         (feature_silent_1_hbo, feature_task_hbo, feature_silent_2_hbo), axis=2)
    output = np.concatenate(
        (feature_silent_1_hbo, feature_task_hbo, feature_silent_2_hbo, feature_silent_1_hbr, feature_task_hbr, feature_silent_2_hbr), axis=2)
    return output


def temporal_feature_extract_yu_gnn(input, index_start, index_end, hbo_type, hbr_type):

    feature_silent_1_hbo = extract_hb_core_temporal_features(
        input[:, :, :index_start])#, hbo_type
    # feature_silent_1_hbr = extract_hb_core_temporal_features(
    #     input[:, :, :index_start, hbr_type])

    feature_task_hbo = extract_hb_core_temporal_features(
        input[:, :, index_start:index_end])#, hbo_type
    # feature_task_hbr = extract_hb_core_temporal_features(
    #     input[:, :, index_start:index_end, hbr_type])

    feature_silent_2_hbo = extract_hb_core_temporal_features(
        input[:, :, index_end:])#, hbo_type
    # feature_silent_2_hbr = extract_hb_core_temporal_features(
    #     input[:, :, index_end:, hbr_type])
    output = np.concatenate(
            (feature_silent_1_hbo, feature_task_hbo, feature_silent_2_hbo), axis=2)
    # output = np.concatenate(
    #     (feature_silent_1_hbo, feature_task_hbo, feature_silent_2_hbo, feature_silent_1_hbr, feature_task_hbr, feature_silent_2_hbr), axis=2)
    return output


def feature_extraction(data, start_time=10, task_duration=60):
    extracted_features1 = get_10_features_xgboost_zhu(data, start_time, task_duration)
    print('extracted_features1', extracted_features1.shape)
    # extracted_features2 = li_svm_compute_10_fetures(data, index_start=start_time, index_end=task_duration+start_time)
    # print('extracted_features2', extracted_features2.shape)
    # extracted_features3 = wang_alex_feature_selection(data, index_task_start=start_time, index_task_end=task_duration+start_time, fs=1)
    # print('extracted_features3', extracted_features3.shape)
    # extracted_features4 = temporal_feature_extract_yu_gnn(data, index_start=start_time, index_end=task_duration+start_time, hbo_type=0, hbr_type=1)
    # print('extracted_features4', extracted_features4.shape)

    # extracted_features = np.concatenate((extracted_features1, extracted_features2, extracted_features3, extracted_features4), axis=2)
    # extracted_features = np.concatenate((extracted_features1, extracted_features3), axis=2)
    extracted_features = extracted_features1
    extracted_features = np.nan_to_num(extracted_features, 0)
    print('maximum value in extracted_features ->', np.max(extracted_features))
    print(f'extracted_features shape: {extracted_features.shape}')
    return extracted_features


def specify_feature_extraction(data, feature_selection_method='4_yu', start_time=10, task_duration=60):
    if feature_selection_method == '1_zhu':
        extracted_features = get_10_features_xgboost_zhu(data, start_time, task_duration)
    elif feature_selection_method == '2_zhifei':
        extracted_features = li_svm_compute_10_fetures(data, index_start=start_time, index_end=task_duration+start_time)
    elif feature_selection_method == '3_wang':
        extracted_features = wang_alex_feature_selection(data, index_task_start=start_time, index_task_end=task_duration+start_time, fs=1)
    elif feature_selection_method == '4_yu':
        extracted_features = temporal_feature_extract_yu_gnn(data, index_start=start_time, index_end=task_duration+start_time, hbo_type=0, hbr_type=1)
    else:
        raise ValueError('feature_selection_method is not defined')
    extracted_features = np.nan_to_num(extracted_features, 0)
    extracted_features = np.reshape(extracted_features, (extracted_features.shape[0], -1))
    return extracted_features

def avg_every_ten_point_in_last_dimension(data):
    data = np.reshape(data, (data.shape[0], 52, -1, 10))
    data = np.mean(data, axis=-1)
    return data 
import random
def generate_random_params(seed):
    random.seed(seed)
    params = {
        'learning_rate': round(random.uniform(0.01, 0.3), 2),  # 生成0.01到0.3之间的浮点数，保留两位小数
        'n_estimators': random.randint(50, 150),  # 生成50到150之间的整数
        'max_depth': random.randint(3, 10),  # 生成3到10之间的整数
        'min_child_weight': random.randint(1, 10),  # 生成1到10之间的整数
        'gamma': round(random.uniform(0, 0.5), 2),  # 生成0到0.5之间的浮点数，保留两位小数
        'subsample': round(random.uniform(0.5, 1.0), 2),  # 生成0.5到1.0之间的浮点数，保留两位小数
        'colsample_bytree': round(random.uniform(0.5, 1.0), 2),  # 生成0.5到1.0之间的浮点数，保留两位小数
        'objective': 'binary:logistic',  # 固定值
        'nthread': 4,  # 固定值
        'scale_pos_weight': 1,  # 固定值
        'seed': int(time.time()) # 生成1到100之间的整数
    }
    return params

def specify_model_and_train(data, label, model_name, seed):
    
    print(f'Define model {model_name} with default setting and seed {seed}')
    print(f'current seed: {seed}')
    para = ''
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    if model_name == 'XGBoost':
        # parameters should be set according to random seed
        para = generate_random_params(seed)
        model = XGBClassifier(**para)
        
    model.random_state = seed

    result,model = train_model_using_loocv(data, label, model)
    res_metrics = get_metrics(result[:, 1], result[:, 0])
    print_md_table(model_name, 'test', res_metrics)
    return res_metrics, para, model

def AutoML_data_dict_for_DMFC(model_name, best_res_metrics, best_seed, iteration_time, para, hb_type):
    
    data = {
        'model': [model_name],
        'hb_type': [hb_type], # 'hbo', 'hbr', 'hbt
        'seed': [best_seed],
        'accuracy': [best_res_metrics[0]],
        'sensitivity': [best_res_metrics[1]],
        'specificity': [best_res_metrics[2]],
        'F1_score': [best_res_metrics[3]],
        'total_itr': [iteration_time],
        'para': [para],
    }
    
    return data 

def AutoML_data_dict_for_DEMO(model_name, best_res_metrics, best_seed, iteration_time, para, num_features):
    
    data = {
        'model': [model_name],
        'amount_demographic': [num_features], # 'hbo', 'hbr', 'hbt
        'seed': [best_seed],
        'accuracy': [best_res_metrics[0]],
        'sensitivity': [best_res_metrics[1]],
        'specificity': [best_res_metrics[2]],
        'F1_score': [best_res_metrics[3]],
        'total_itr': [iteration_time],
        'para': [para],
    }
    
    return data 

def save_autodl(data, save_path):
    
    df = pd.DataFrame(data)
    if not os.path.exists(save_path):
        df.to_csv(save_path, index=False, mode='w')
    else:
        df.to_csv(save_path, index=False, mode='a', header=False)
    print(f"Model: is save to {save_path}!")

def remove_nan_for_demographic_data(demographic_data):
    for index, sub_value in enumerate(demographic_data):
        try: 
            demographic_data[index] = sub_value.astype(int)
            pass
        except:
            for i, v in enumerate(sub_value):
                if type(v) is not int:
                    print('index:', i, 'value:', v)
                    sub_value[i] = 1
                    print('index:', i, 'fixed - value:', sub_value[i])
                    print('there should be a subject whose handedness is empty, the above steps set it to be 1')
            demographic_data[index] = sub_value.astype(int)

    demographic_data = demographic_data.astype(int)
    return demographic_data


def plot_model_importance(model):
    feature_importances = model.feature_importances_
    feature_importances = feature_importances.reshape(52, -1)
    plt.imshow(feature_importances)
    
def reshape_to_matrix(data):
    reshape_data = np.reshape(data, (data.shape[0], -1))
    return reshape_data

"""
input shape: (sub, 52, 125, 2) last dimensio is hbo and hbr 
output shape: (sub, 26000)
"""
def get_chao_cfnn_novel_feature(hbo, hbr):
    CBV = (hbo + hbr) / np.sqrt(2)
    print(CBV.shape)

    # COE 
    COE = (hbo - hbr) / np.sqrt(2)
    print(COE.shape)
    # L 
    Mag_L = np.sqrt((np.square(hbo) + np.square(hbr))) / np.sqrt(2)
    print(Mag_L.shape)

    # Angle K 
    Ang_K = np.arctan(COE/CBV)
    print(Ang_K.shape)


    r_CBV = reshape_to_matrix(CBV) # (458, 6500)
    r_COE = reshape_to_matrix(COE) # (458, 6500)
    r_Mag_L = reshape_to_matrix(Mag_L) # (458, 6500)
    r_Ang_K = reshape_to_matrix(Ang_K) # (458, 6500)
    novel_features = np.concatenate((r_CBV, r_COE, r_Mag_L, r_Ang_K), axis=1)
    return novel_features


def get_duan_rsfc_data(hbo, index_start=10, index_end=70):
    feature_shape = hbo.shape
    def compute_resting_state_period(data, index_start, index_end):
        first_resting_data = data[:,:,:index_start] # (458, 52, 10)
        end_resting_data = data[:,:,index_end+10:] # (458, 52, 45)
        resting_data = np.concatenate((first_resting_data, end_resting_data), axis=2) # (458, 52, 55)
        return resting_data
    data = compute_resting_state_period(hbo, index_start, index_end)


    RSFC = np.zeros((data.shape[0], data.shape[1], data.shape[1]))

    def compute_correlation(x, y):
        corr, _ = stats.pearsonr(x, y)
        return corr

    for sub in range(feature_shape[0]):
        for ch_1 in range(feature_shape[1]):
            for ch_2 in range(feature_shape[1]):
                if ch_2 < ch_1: continue
                corr = compute_correlation(
                    data[sub, ch_1],data[sub, ch_2])
                RSFC[sub, ch_1, ch_2] = corr
                RSFC[sub, ch_2, ch_1] = corr
    return RSFC

def retrieve_model(model_name, seed):
    para = ''
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    if model_name == 'XGBoost':
        para = generate_random_params()
        model = XGBClassifier(**para)
        

    model.random_state = seed
    
    return model, para


def print_md_table_val_test(model_name, test_result, val_result):
    print('| Model Name | Testing Set |             |             |             | Validation Set |             |             |             |')
    print('|------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|')
    print('|            | Accuracy | Sensitivity | Specificity | F1 Score | Accuracy | Sensitivity | Specificity | F1 Score |')


    # print('| Dataset | Model Name | Accuracy | Sensitivity | Specificity | F1 Score |')
    # print('|------------|------------|----------|-------------|-------------|----------|')
    print(f'| {model_name}   |', end='')
    
    for val in test_result:
        print(f' {val:.4f}  |', end='')
    for val in val_result:
        print(f' {val:.4f}  |', end='')        
        
    print()

# Define custom scoring functions
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity
    
def train_model_with_stratified_kfold(data, label, model, seed, num_folds=5):
    # Create scorers dictionary
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'sensitivity': make_scorer(recall_score),  # Sensitivity is the same as recall
        'specificity': make_scorer(specificity_score),
        'f1_score': make_scorer(f1_score)
    }
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    results = cross_validate(model, data, label, cv=cv, scoring=scorers, return_train_score=False)
    
    # Calculate mean of each metric
    mean_metrics = [
        np.mean(results['test_accuracy']),
        np.mean(results['test_sensitivity']),
        np.mean(results['test_specificity']),
        np.mean(results['test_f1_score'])
    ]    
    return mean_metrics


def train_model_with_CV_and_LOOCV(data, label, model, seed, num_folds=5):
    loo = LeaveOneOut()
    result = []
    all_val_result = []
    # Loop over each train/test split
    for train_index, test_index in loo.split(data):
        # Split the data into training and testing sets
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        val_result_fold = train_model_with_stratified_kfold(X_train, y_train, model, num_folds)
        all_val_result.append(val_result_fold)
        
        # Train the classifier
        model.fit(X_train, y_train)

        # Predict the label for the test set
        y_pred = model.predict(X_test)

        # Append the accuracy to the list
        result.append([y_pred, y_test])
    all_val_result = np.array(all_val_result)
    all_val_result = np.mean(all_val_result, axis=0)
    return np.array(result), all_val_result, model