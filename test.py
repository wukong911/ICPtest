import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def icp(source, target, max_iterations=10, tolerance=5.0):
    """
    使用ICP算法将源点云配准到目标点云。

    :param source: 源点云，形状为 (N, 3) 的 NumPy 数组
    :param target: 目标点云，形状为 (M, 3) 的 NumPy 数组
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛容差
    :return: 配准后的源点云，旋转矩阵和平移向量
    """
    assert source.shape[1] == 3 and target.shape[1] == 3, "点云应为 (N, 3) 形状的数组"

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    transformation = np.eye(4)
    for i in range(max_iterations):
        # 1. 找到最近点对
        distances = np.linalg.norm(source_centered[:, None, :] - target_centered[None, :, :], axis=2)
        closest_indices = np.argmin(distances, axis=1)
        closest_points = target_centered[closest_indices]

        # 2. 计算最优旋转矩阵和平移向量
        W = np.dot(source_centered.T, closest_points)
        U, S, Vt = np.linalg.svd(W)
        R = np.dot(Vt.T, U.T)
        t = target_mean - np.dot(R, source_mean)

        # 3. 更新源点云
        source_centered = np.dot(source_centered, R.T)
        source_mean = np.dot(source_mean, R.T) + t
        source = source_centered + source_mean

        # 可视化配准过程
        visualize_registration(source, target, i)

        # 4. 检查收敛
        if np.linalg.norm(t) < tolerance:
            break

    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return source, transformation[:3, :3], transformation[:3, 3]


def visualize_registration(source, target, iteration):
    """
    可视化源点云和目标点云。

    :param source: 当前配准的源点云
    :param target: 目标点云
    :param iteration: 当前迭代次数
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(target[:, 0], target[:, 1], target[:, 2], color='b', label='Target', s=30)
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], color='r', label='Source', s=30)

    ax.set_title(f'ICP Iteration {iteration + 1}')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.xlim([-10, 30])
    plt.ylim([-10, 30])
    ax.set_zlim([-10, 30])

    plt.pause(0.5)
    plt.clf()


if __name__ == "__main__":
    # 生成示例点云数据
    source = np.array([[1.0, 6.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 15.0, 10.0],
                       [100.0, 11.0, 12.0],
                       [13.0, 9.0, 15.0],
                       [16.0, 33.0, 18.0],
                       [44.0, 20.0, 11.0],
                       [22.0, 23.0, 2.0]])

    # 假设目标点云经过了旋转和平移
    rotation = R.from_euler('xyz', [30, 20, 10], degrees=True).as_matrix()
    translation = np.array([1.0, -2.0, 3.0])
    target = np.dot(source, rotation.T) + translation

    # 显示初始点云
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], color='b', label='Target', s=30)
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], color='r', label='Source', s=30)
    ax.set_title('Initial Point Clouds')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show(block=False)

    # 使用ICP进行配准
    registered_source, R_final, t_final = icp(source, target)
    print('R_final:\n',R_final)
    print('t_final:\n',t_final)
    # 显示最终配准结果
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], color='b', label='Target', s=30)
    ax.scatter(registered_source[:, 0], registered_source[:, 1], registered_source[:, 2], color='r',
               label='Registered Source', s=30)
    ax.set_title('Final Registered Point Clouds')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()