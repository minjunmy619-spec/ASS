# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from typing import Callable

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path

import torch

from rich.progress import track


def separate_one(args: Namespace, unk_args: list[str], initialize: Callable, separate: Callable):
    ctx = initialize(args, unk_args)
    separate(args.src_filename, args.dst_filename, ctx, args, unk_args)


def separate_batch(args: Namespace, unk_args: list[str], initialize: Callable, separate: Callable):
    args.dst_path.mkdir(exist_ok=True)

    src_fname_list = sorted(list(args.src_path.rglob(f"mix*.{args.ext}")))
    if "SGE_TASK_ID" in os.environ and os.environ["SGE_TASK_ID"] != "undefined":
        start = int(os.environ["SGE_TASK_ID"]) - 1
        end = start + int(os.environ["SGE_TASK_STEPSIZE"])
        src_fname_list = src_fname_list[start:end]

    ctx = initialize(args, unk_args)

    for src_filename in track(src_fname_list):
        parent1, parent2 = src_filename.parent.name, src_filename.parent.parent.name
        dst_filename = args.dst_path / parent2 / parent1 / "est.wav"
        dst_filename.parent.mkdir(exist_ok=True, parents=True)

        separate(src_filename, dst_filename, ctx, args, unk_args)


@torch.inference_mode()
def main(
    add_common_args: Callable[[ArgumentParser], None],
    initialize: Callable,
    separate: Callable,
):
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("one")
    sub_parser.add_argument("model_path", type=Path)
    sub_parser.add_argument("src_filename", type=Path)
    sub_parser.add_argument("dst_filename", type=Path)
    add_common_args(sub_parser)
    sub_parser.set_defaults(handler=separate_one)

    sub_parser = sub_parsers.add_parser("batch")
    sub_parser.add_argument("model_path", type=Path)
    sub_parser.add_argument("src_path", type=Path)
    sub_parser.add_argument("dst_path", type=Path)
    add_common_args(sub_parser)
    sub_parser.add_argument("--ext", type=str, default="wav")
    sub_parser.set_defaults(handler=separate_batch)

    args, unk_args = parser.parse_known_args()

    print("=" * 32)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 32)

    if hasattr(args, "handler"):
        args.handler(args, unk_args, initialize, separate)
    else:
        parser.print_help()
