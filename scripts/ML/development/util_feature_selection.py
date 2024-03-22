import numpy as np

def normalize(data):
    # Iterate over each subject
    normalized_data = np.empty_like(data)
    for i in range(data.shape[0]):
        # Calculate the mean and standard deviation for the current subject
        mean = np.mean(data[i, :, :])
        std = np.std(data[i, :, :])

        # Perform z-normalization for the current subject
        normalized_data[i, :, :] = (data[i, :, :] - mean) / std
    return normalized_data




def wang_alex_feature_selection(input, index_task_start,index_task_end,fs):


    index_task_start = 10
    index_task_end = 70
    fs = 1  # sampling rate

    # 1. Total: Sum of hemoglobin concentration changes in the whole process.
    feature_1 = np.sum(input, axis=2)[..., np.newaxis]
    print(f'feature 1 shape -> {feature_1.shape}')

    # 2. Peak: Peak value of hemoglobin concentration changes in four periods.
    feature_2 = np.concatenate(
        (np.max(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.max(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.max(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.max(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    print(f'feature 2 shape -> {feature_2.shape}')

    # 3. Valley: Valley value of hemoglobin concentration changes in four periods.
    feature_3 = np.concatenate(
        (np.min(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.min(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.min(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.min(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    print(f'feature 3 shape -> {feature_3.shape}')

    # 4. Average: Mean value of hemoglobin concentration changes in four periods.
    feature_4 = np.concatenate(
        (np.mean(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.mean(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.mean(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.mean(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    print(f'feature 4 shape -> {feature_4.shape}')

    # 5. Variance: Variance of hemoglobin concentration changes in four periods.
    feature_5 = np.concatenate(
        (np.var(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.var(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.var(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.var(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    print(f'feature 5 shape -> {feature_5.shape}')

    # 6. Integral: Integral of hemoglobin concentration changes in four periods.
    feature_6 = np.concatenate(
        (np.sum(input[..., :index_task_start], axis=2)[..., np.newaxis],
        np.sum(input[..., index_task_start:index_task_end],
                axis=2)[..., np.newaxis],
        np.sum(input[..., :index_task_end:], axis=2)[..., np.newaxis],
        np.sum(input[..., :], axis=2)[..., np.newaxis]),
        axis=2
    )
    print(f'feature 6 shape -> {feature_6.shape}')


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
    print(f'feature 7 shape -> {feature_7.shape}')

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
    print(f'feature 8 shape -> {feature_8.shape}')


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
    print(f'feature 9 shape -> {feature_9.shape}')


    q = np.mean(input)  # threhold
    print(f'threshold is set to {q} ')


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
    print(f'feature 10 shape -> {feature_10.shape}')

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

    # all_feature = np.concatenate(
    #     (feature_1,
    #     feature_2,
    #     feature_3,
    #     feature_4,
    #     feature_5,
    #     feature_6,
    #     feature_7,
    #     feature_8,
    #     feature_9,
    #     feature_10),
    #     axis=2
    # )
    return nor_all_feature