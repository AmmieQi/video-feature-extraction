import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--input', default='input', type=str, help='Input file path')
    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--frame_root', default='', type=str, help='Root path of frames of video')
    parser.add_argument('--fps', default=10, type=int, help='Frames per second')
    parser.add_argument('--feature_type', default='2D', type=str, help='Type of feature to extract')
    # model
    parser.add_argument('--model', default='', type=str, help='Model file path')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    # train
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    args = parser.parse_args()

    return args
