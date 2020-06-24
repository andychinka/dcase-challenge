#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2019
# Task 1A: Acoustic Scene Classification
# Dataset downloader
# ---------------------------------------------
# Author: Toni Heittola ( toni.heittola@tuni.fi ), Tampere University / Audio Research Group
# License: MIT

import sys
import os
import argparse
import textwrap
import pkg_resources

try:
    import dcase_util

    if pkg_resources.parse_version(pkg_resources.get_distribution("dcase_util").version) < pkg_resources.parse_version('0.2.14'):
        raise AssertionError(
            'Please update your dcase_util module, version >= 0.2.14 required. \n'
            'You can update dcase_util it with `pip install --upgrade dcase_util`.\n'
        )

except ImportError:
    raise ImportError('Unable to import dcase_util module. You can install it with `pip install dcase_util`.\n')

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            '''\
            DCASE 2020 
            {app_title}
            Dataset downloader
            ---------------------------------------------            
            Author:  Toni Heittola ( toni.heittola@tuni.fi )
            Tampere University / Audio Research Group
            '''.format(app_title='Task 1A: Acoustic Scene Classification')
        )
    )

    parser.add_argument(
        '-d', '--dataset',
        help='Dataset name to download ["dev", "leaderboard", "eval", "train", "test"]',
        dest='dataset_name',
        required=False,
        type=str
    )

    parser.add_argument(
        '-o', '--output_path',
        help='Output path',
        dest='output_path',
        required=False,
        type=str
    )

    # Application information
    parser.add_argument(
        '-v', '--version',
        help='Show version number and exit',
        action='version',
        version='%(prog)s ' + __version__
    )

    # Parse arguments
    args = parser.parse_args()

    # Get output path
    if not args.output_path:
        output_path = os.path.join('datasets')

    else:
        output_path = args.output_path

    # Make sure given path exists
    dcase_util.utils.Path().create(
        paths=output_path
    )

    # Get dataset class name
    dataset_map = {
        'dev': 'TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet',
        # 'leaderboard': 'TAUUrbanAcousticScenes_2019_LeaderboardSet',
        # 'eval': 'TAUUrbanAcousticScenes_2019_EvaluationSet',
        # 'train': 'TAUUrbanAcousticScenes_2019_DevelopmentSet',
        # 'test': 'TAUUrbanAcousticScenes_2019_LeaderboardSet',
    }

    if args.dataset_name:
        if args.dataset_name in dataset_map:
            dataset_class_name_list = [dataset_map[args.dataset_name]]

        else:
            dataset_class_name_list = [args.dataset_name]
    else:
        dataset_class_name_list = [
            'TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet',
            # 'TAUUrbanAcousticScenes_2019_LeaderboardSet'
        ]

    for dataset_class_name in dataset_class_name_list:
        if dcase_util.datasets.dataset_exists(dataset_class_name=dataset_class_name):
            log.section_header(
                'Processing dataset [{dataset_class_name}]'.format(dataset_class_name=dataset_class_name)
            )

            # Get dataset and initialize
            dcase_util.datasets.dataset_factory(
                dataset_class_name=dataset_class_name,
                data_path=output_path,
            ).initialize().log()

            log.foot()

        else:
            log.line(
                'Unknown dataset class [{dataset_class_name}]'.format(dataset_class_name=dataset_class_name)
            )


if __name__ == "__main__":
    import traceback

    retry_cnt = 0
    while True:
        try:
            sys.exit(main(sys.argv))
            break
        except (ValueError, IOError) as e:
            print(traceback.format_exc())
            retry_cnt += 1
            print("retry now: ", retry_cnt)
            # sys.exit(e)