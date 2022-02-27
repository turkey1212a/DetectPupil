import cv2
import dlib_utils
import glob
import numpy as np
import os
import sys
import time

# dlibの座標の出力形式を(x, y)のタプルに変換する
def part_to_coordinates(part):
    return (part.x, part.y)


def shape_to_landmark(shape):
    landmark = []
    for i in range(shape.num_parts):
        landmark.append(part_to_coordinates(shape.part(i)))
    return landmark


# 目の部分の切り出し
def cut_out_eye_img(img_cv2, eye_points):
    height, width = img_cv2.shape[:2]
    x_list = []; y_list = []
    for point in eye_points:
        x_list.append(point[0])
        y_list.append(point[1])
    x_min = max(min(x_list) - 3, 0)
    x_max = min(max(x_list) + 4, width)
    y_min = max(min(y_list) - 3, 0)
    y_max = min(max(y_list) + 4, height)
    eye_img = img_cv2[y_min : y_max, x_min : x_max]
    eye_points_local = [(x - x_min, y - y_min) for (x, y) in zip(x_list, y_list)]
    return eye_img, x_min, x_max, y_min, y_max, eye_points_local


# 重み（工夫すれば精度が向上するかも）
WEIGHT_H = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
WEIGHT_V = np.array([0.1, 0.2, 0.4, 0.2, 0.1])


def detect_pupil(img_negative, eye_points):
    # 目の周りを切り出す
    eye_img_negative, x_min, _, y_min, _, eye_points_local = cut_out_eye_img(img_negative, eye_points)
    # print(f'x_min={x_min}')
    # print(f'y_min={y_min}')

    # 目の部分をマスク処理
    eye_mask = np.zeros_like(eye_img_negative)
    eye_mask = cv2.fillConvexPoly(eye_mask, np.array(eye_points_local), True, 1)
    eye_img_negative_masked = np.where(eye_mask == 1, eye_img_negative, 0)

    # 列ごとに縦方向の和を求める
    sum_x = np.sum(eye_img_negative_masked, axis=0)

    # 重み付き移動平均を求める
    weighing_moving_ave_x = np.convolve(sum_x, WEIGHT_H, mode='same')

    # 横方向の和を求める
    sum_y = np.sum(eye_img_negative_masked, axis=1)

    # 重み付き移動平均を求める
    weighing_moving_ave_y = np.convolve(sum_y, WEIGHT_V, mode='same')

    pupil_x = np.argmax(weighing_moving_ave_x) + x_min
    pupil_y = np.argmax(weighing_moving_ave_y) + y_min
    # print(f'pupil_x:{np.argmax(weighing_moving_ave_x)}->{pupil_x}')
    # print(f'pupil_y:{np.argmax(weighing_moving_ave_y)}->{pupil_y}')
    return (pupil_x, pupil_y)


def detect_pupil_2(img_negative, eye_points):
    # マスク処理によって目の部分だけを取り出す
    eye_mask = np.zeros_like(img_negative)
    eye_mask = cv2.fillConvexPoly(eye_mask, np.array(eye_points), True, 1)
    eye_img_negative_masked = np.where(eye_mask == 1, img_negative, 0)

    # 列ごとに縦方向の和を求める
    sum_x = np.sum(eye_img_negative_masked, axis=0)

    # 重み付き移動平均を求める
    weighing_moving_ave_x = np.convolve(sum_x, WEIGHT_H, mode='same')

    # 横方向の和を求める
    sum_y = np.sum(eye_img_negative_masked, axis=1)

    # 重み付き移動平均を求める
    weighing_moving_ave_y = np.convolve(sum_y, WEIGHT_V, mode='same')

    pupil_x = np.argmax(weighing_moving_ave_x)
    pupil_y = np.argmax(weighing_moving_ave_y)

    return (pupil_x, pupil_y)


# 瞳（特に瞳孔）を検出し、(x, y)を返す
# 動作未確認
def detect_pupils(img_cv2, eye_points_list):
    # グレースケール化
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # ポジ→ネガ反転
    img_negative = 255 - img_gray

    # 瞳（瞳孔）を検出
    start_time = time.time()
    pupils = []
    for eye_points in eye_points_list:
        pupil_x, pupil_y = detect_pupil(img_negative, eye_points)
        # pupil_x, pupil_y = detect_pupil_2(img_negative, eye_points)
        pupils.append((pupil_x, pupil_y))

    required_time = time.time() - start_time
    print(f'required time (with cutting-out) : {required_time}')

    return pupils


'''
テスト実施方法：

ターミナルで下記を実行
python3 detect_pupils.py [画像フォルダ]

画像フォルダ下のjpgファイルを開いて瞳位置を検出、検出結果を図示した画像を新規フォルダに格納する。
'''


if __name__ == '__main__':
    args = sys.argv 
    img_dir = args[1]

    output_dir = time.strftime('output_%Y%m%d_%H%M%S')
    os.makedirs(output_dir)

    files = glob.glob(f'{img_dir}/*.jpg')
    for file in files:
        print(f'[{file}]')
        img = cv2.imread(file)
        landmark = dlib_utils.get_face_landmark(img)
        eye_points = [landmark[36:42], landmark[42:48]]
        pupils = detect_pupils(img, eye_points)
        img_copy = img.copy()
        for pupil in pupils:
            print(pupil)
            cv2.drawMarker(img_copy, pupil, (255, 0, 255),markerType=cv2.MARKER_CROSS)
        cv2.imwrite(f'{output_dir}/pupil_{os.path.basename(file)}', img_copy)

        # 目の部分を切り出す
        for eye_no, eye_point in enumerate(eye_points):
            img_eye, _, _, _, _, _ = cut_out_eye_img(img_copy, eye_point)
            # 見やすいように拡大する
            n = 8
            img_eye_extended = img_eye.repeat(n, axis=0)
            img_eye_extended = img_eye_extended.repeat(n, axis=1)
            cv2.imwrite(f'{output_dir}/pupil_{eye_no}_{os.path.basename(file)}', img_eye_extended)
