import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 画像ディレクトリ
image_dir = 'Datasets/fountain-P11'

# 画像ファイルの取得
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])

# カメラ行列
with open('Datasets/fountain-P11/K.txt') as f:
    K = np.array([list(map(float, line.split())) for line in f])

# カメラ間のベースラインの読み込み
baseline = np.loadtxt(os.path.join(image_dir, 'res/fountain-P11_baselines.txt'))


"""
視差マップは、画像間のピクセルのズレを計算することで得られる
"""
# 画像ファイルの取得
def compute_disparity_map(img1, img2):
    # ステレオマッチングの設定
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    # 画像をグレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 視差マップの計算
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # 視差マップのノイズをフィルタリング
    disparity = cv2.medianBlur(disparity, 5)

    return disparity


# 視差マップの表示
def display_disparity_map(disparity):
    plt.imshow(disparity, cmap='gray')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()



'''
視差マップから深度マップを生成
カメラパラメータが必要
'''
# 視差マップから深度マップを生成
def disparity_to_depth(disparity, K, baseline):
    # カメラの焦点距離
    fx = K[0, 0]
    fy = K[1, 1]

    # ゼロ視差の処理
    with np.errstate(divide='ignore'):
        depth = np.where(disparity > 0, (fx * baseline) / disparity, 0)

    return depth


def display_depth_map(depth):
    plt.imshow(depth, 'gray')
    plt.colorbar()
    plt.title('Depth Map')
    plt.show()




"""
深度マップを利用し3Dポイントクラウドを生成
"""
def generate_point_cloud(depth, K):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    i = i.reshape(-1)
    j = j.reshape(-1)
    d = depth.reshape(-1)

    # 有効な深度値のみを使用
    valid = d > 0
    i = i[valid]
    j = j[valid]
    d = d[valid]

    # カメラ行列の逆行列
    K_inv = np.linalg.inv(K)

    # 3Dポイントの計算
    points_3d = np.dot(K_inv, np.vstack((i * d, j * d, d)))
    points_3d = points_3d.T

    return points_3d


# 3Dポイントクラウドの表示
def display_point_cloud(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
    plt.title('3D Point Cloud')
    plt.show()



# 3dポイントクラウドを.plyファイルに保存
def save_point_cloud_to_ply(points_3d, filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory): # filenameフォルダが存在するか確認
        os.makedirs(directory) # なかったら作成

    with open(filename, 'w') as f:
        # .plyのヘッダー
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points_3d)}\n')
        f.write('property float x \n')
        f.write('property float y \n')
        f.write('property float z \n')
        f.write('end_header\n')

        # 各ポイントの座標を書き込み
        for point in points_3d:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')


# メイン処理
all_points_3d = []

for i in range(len(image_files) - 1):
    img1 = cv2.imread(image_files[i])
    img2 = cv2.imread(image_files[i + 1])

    disparity = compute_disparity_map(img1, img2)
    display_disparity_map(disparity)

    depth = disparity_to_depth(disparity, K, baseline)
    display_depth_map(depth)

    points_3d = generate_point_cloud(depth, K)
    all_points_3d.append(points_3d)

    print(f'Processed images {i + 1} and {i + 2}')

# すべてのポイントを統合
all_points_3d = np.concatenate(all_points_3d, axis=0)

# display_point_cloud(all_points_3d)

# ポイントクラウドを.plyファイルに保存
save_point_cloud_to_ply(all_points_3d, 'mvs.ply')

print('finish')