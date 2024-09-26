"""
An auxiliary script to generate HTML files for image visualization.
"""

import argparse
import json
import os
import random
import shutil

import dominate
from dominate.tags import h3, img, table, td, tr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--caption_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--aliases", type=str, default=None, nargs="+")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--hard_copy", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.aliases is not None:
        assert len(args.image_dirs) == len(args.aliases)
    else:
        args.aliases = [str(i) for i in range(len(args.image_dirs))]
    return args


def check_existence(image_dirs, filename):
    for image_dir in image_dirs:
        if not os.path.exists(os.path.join(image_dir, filename)):
            print(os.path.join(image_dir, filename))
            return False
    return True


if __name__ == "__main__":
    args = get_args()
    filenames = sorted(os.listdir(args.image_dirs[0]))
    filenames = [
        filename
        for filename in filenames
        if filename.endswith(".png")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
    ]
    if args.max_images is not None:
        random.seed(args.seed)
        random.shuffle(filenames)
        filenames = filenames[: args.max_images]
        filenames = sorted(filenames)
    doc = dominate.document(title="Visualization" if args.title is None else args.title)
    if args.title:
        with doc:
            h3(args.title)
    t_main = table(border=1, style="table-layout: fixed;")
    prompts = json.load(open(args.caption_path, "r"))
    for i, filename in enumerate(filenames):
        bname = os.path.splitext(filename)[0]
        if not check_existence(args.image_dirs, filename):
            continue
        title_row = tr()
        _tr = tr()
        title_row.add(td(f"{bname}"))
        _tr.add(td(prompts[int(bname)]))
        for image_dir, alias in zip(args.image_dirs, args.aliases):
            title_row.add(td(f"{alias}"))
            _td = td(style="word-wrap: break-word;", halign="center", valign="top")
            source_path = os.path.abspath(os.path.join(image_dir, filename))
            target_path = os.path.abspath(
                os.path.join(args.output_root, "images", alias, filename)
            )
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            if args.hard_copy:
                shutil.copy(source_path, target_path)
            else:
                os.symlink(source_path, target_path)
            _td.add(
                img(
                    style="width:256px",
                    src=os.path.relpath(target_path, args.output_root),
                )
            )
            _tr.add(_td)
        t_main.add(title_row)
        t_main.add(_tr)
    with open(os.path.join(args.output_root, "index.html"), "w") as f:
        f.write(t_main.render())
