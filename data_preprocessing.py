import nibabel as nib
import numpy as np
import scipy.signal

def load_fmri_data(file_path):
    """
    加载4D rs-fMRI数据。
    :param file_path: rs-fMRI数据文件路径。
    :return: 4D numpy数组。
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def extract_roi_time_series(fmri_data, roi_mask):
    """
    从fMRI数据中提取ROI的时间序列。
    :param fmri_data: 4D rs-fMRI数据。
    :param roi_mask: ROI掩码（每个感兴趣区域一个掩码）。
    :return: 时间序列数组。
    """
    time_series = []
    for mask in roi_mask:
        masked_data = fmri_data[mask]
        roi_series = np.mean(masked_data, axis=0)
        time_series.append(roi_series)
    return np.array(time_series)

def compute_connectivity_matrix(time_series):
    """
    计算ROI之间的连接矩阵。
    :param time_series: ROI时间序列。
    :return: 连接矩阵。
    """
    correlation_matrix = np.corrcoef(time_series)
    return correlation_matrix

def flatten_feature_matrix(matrix):
    """
    展平连接矩阵以形成特征向量。
    :param matrix: 连接矩阵。
    :return: 展平的特征向量。
    """
    return matrix.flatten()


if __name__ == "__main__":
    # 加载数据
    fmri_data = load_fmri_data("path_to_your_fmri_data.nii")

    # ROI掩码
    # roi_masks = [np.zeros(fmri_data.shape[:3], dtype=bool), ...]

    # 提取时间序列
    # time_series = extract_roi_time_series(fmri_data, roi_masks)

    # 计算连接矩阵
    # connectivity_matrix = compute_connectivity_matrix(time_series)

    # 展平特征矩阵
    # feature_vector = flatten_feature_matrix(connectivity_matrix)
