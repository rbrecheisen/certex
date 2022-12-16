import cv2


def main():
    image = cv2.imread('data/example.png')
    print(image.shape)
    cv2.imshow('image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
