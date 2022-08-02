import cv2


def draw_bboxes(imgs, bboxes):
    for i, row in bboxes.iterrows():
        x, y = row.left, row.top
        x2, y2 = row.right, row.bottom
        for img in imgs:
            cv2.rectangle(img, (x, y), (x2, y2), (0,255,0), thickness=2)
