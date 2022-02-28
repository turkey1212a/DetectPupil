import dlib


# dlibの座標の出力形式を(x, y)のタプルに変換する
def part_to_coordinates(part):
    return (part.x, part.y)


def shape_to_landmark(shape):
    landmark = []
    for i in range(shape.num_parts):
        landmark.append(part_to_coordinates(shape.part(i)))
    return landmark


class dlib_detector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

    def get_face_landmark(self, img_cv2):
        # 顔の位置を検出
        CUT_OFF = -0.1
        rects, scores, types = self.detector.run(img_cv2, 1, CUT_OFF)

        # 顔の部位を検出
        shape = self.predictor(img_cv2, rects[0])

        # 検出したshapeをlandmark（座標のリスト）に変換
        landmark = shape_to_landmark(shape)

        return landmark
