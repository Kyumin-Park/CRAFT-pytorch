import ground_truth
import train
import test
import argparse


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser(description='CRAFT Text Detection Trainer')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                        help='pretrained refiner model')
    parser.add_argument('--train', default=False, action='store_true', help='train or test')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=10, type=int, help='max epoch')
    parser.add_argument('--gt', default=False, action='store_true', help='generate ground truth')

    args = parser.parse_args()

    print(args)

    if args.gt:
        ground_truth.ground_truth(args)
    else:
        if args.train:
            train.train(args)
        else:
            test.test(args)


if __name__ == '__main__':
    main()
