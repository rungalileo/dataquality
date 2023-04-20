import sys


def main():
    print(sys.argv)
    arguments = sys.argv
    if len(arguments) < 2:
        raise ValueError(
            "\nYou need to pass at least two argument to the command line.\n"
            "For example dqyolo data=coco128.yaml model=yolov8n.pt\n"
            "Or for training dqyolo train data=coco128.yaml model=yolov8n.pt "
            "epochs=1 lr0=0.01"
        )


if __name__ == "__main__":
    main()
