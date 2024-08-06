import argparse

from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v2/subsample_v1 - wo pre-filter/')
    parser.add_argument('--dtr_type', default='DetectorWrapperV3NPY')
    parser.add_argument('--db_key', default=None)
    parser.add_argument('--dbo_type', default='DataV4')
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/demo_arc_detector_save')
    return parser.parse_args()


def run(args):
    detector_wrapper = eval(args.dtr_type)(args.addr, args.dir_plot_save, args.db_key, args.dbo_type)
    detector_wrapper.run()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
