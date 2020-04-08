import cv2
from skimage.transform import rescale
import matplotlib.pyplot as plt
import numpy as np


def plot_image(img):
    plt.imshow(img)
    plt.show()


def open_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    new_img = []
    for line in im:
        new_img.append(np.mean(line, axis=1))

    new_img = np.array(new_img)
    return new_img


def rescale_image(data, factor):
    data_size = int(data.shape[0] ** 0.5)
    img = rescale(data.reshape(data_size, data_size), factor, mode='reflect')
    x = img.shape[0] ** 2
    return img.reshape(x, 1).ravel()


def take_photo():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    # cam.set(3,28)
    # cam.set(3,28)

    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, (28, 28), 0, 0, cv2.INTER_CUBIC)
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "live_images/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def main():
    # take_photo()
    open_image('live_images/opencv_frame_0.png')


if __name__ == '__main__':
    main()
