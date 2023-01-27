import glob
import os.path
import shutil

import cv2

if __name__ == '__main__':
    labels = sorted(glob.glob("*.txt", root_dir="dataset/labels_raw"))

    for label in labels:
        basename = label.rstrip(".txt")

        img = cv2.imread("dataset/images_raw/" + basename + ".jpg")
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)

        left = img[0:100, 220:860]
        right = img[0:100, 1060:-220]

        img_con = cv2.vconcat([left, right])

        in_file = open("dataset/labels_raw/" + label, "r")
        out_file = open("dataset/labels/" + label, "w")
        lines = in_file.readlines()

        score_need = 2

        for line in lines:
            c, x, y, w, h = line.split()
            x, y, w, h = float(x) * 1920, float(y) * 1080, float(w) * 1920, float(h) * 1080
            if x < 960:
                x = x - 220
            else:
                x = x - 1060
                y = y + 100
            if int(c) <= 30 and w < 40:
                score_need -= 1
            # pt1 = (int(x - w/2), int(y-h/2))
            # pt2 = (int(x + w/2), int(y+h/2))
            # cv2.rectangle(img_con, pt1, pt2, (0, 0, 255), 2)
            out_file.write("{} {} {} {} {}\n".format(c, x / 640, y / 200, w / 640, h / 200))

        in_file.close()
        out_file.close()
        cv2.imwrite("dataset/images/" + basename + ".jpg", img_con, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

        if score_need > 0:
            if not os.path.exists("dataset/needs_work/" + basename + ".jpg"):
                shutil.copy("dataset/images/" + basename + ".jpg", "dataset/needs_work/" + basename + ".jpg")
            if not os.path.exists("dataset/needs_work/" + label):
                shutil.copy("dataset/labels/" + label, "dataset/needs_work/" + label)

        # cv2.imshow("test", img_con)
        # cv2.waitKey(0)
