import os
import cv2
import argparse

global point1, point2


def on_mouse(event, x, y, flags, param):

    global point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:                                       # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)
        cv2.namedWindow(pic, cv2.WINDOW_NORMAL)
        cv2.imshow(pic, img2)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.namedWindow(pic, cv2.WINDOW_NORMAL)
        cv2.imshow(pic, img2)
        cv2.imwrite(new_old, img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 2)
        cv2.namedWindow(pic, cv2.WINDOW_NORMAL)
        cv2.imshow(pic, img2)
    elif event == cv2.EVENT_LBUTTONUP:                                        # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 2)
        cv2.namedWindow(pic, cv2.WINDOW_NORMAL)
        cv2.imshow(pic, img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = img[min_y:min_y + height, min_x:min_x + width]
        cv2.imwrite(new_dir, cut_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="C:\\Users\\Brandon Han\\Downloads\\original")
    parser.add_argument('-o', '--output', type=str, default="E:\\Umich\\Dataset\\unlabeled")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    pics = os.listdir(input_path)
    count = 0

    for pic in pics:
        point1 = (0, 0)
        point2 = (0, 0)
        pic_type = os.path.splitext(pic)[1]
        new_dir = os.path.join(output_path, str(count).zfill(4) + pic_type)
        new_old = os.path.join(output_path, pic)
        old_dir = os.path.join(input_path, pic)

        try:
            img = cv2.imread(old_dir)
        except:
            continue
        cv2.namedWindow(pic, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(pic, on_mouse, [img, new_dir, pic, new_old])
        try:
            cv2.imshow(pic, img)
        except:
            cv2.destroyAllWindows()
            continue
        key = cv2.waitKey()
        if key == 27:
            break
        cv2.destroyAllWindows()

        count += 1

    cv2.destroyAllWindows()
