import os
import argparse
from pathlib import Path
from shutil import rmtree as rm_folder

parser = argparse.ArgumentParser('Tensorboard cleaner (Remove empty sessions)')
parser.add_argument("--path", type=str, default='tensorboard', help="Path to tensorboard log directory")

if __name__ == "__main__":
    args = parser.parse_args()

    paths_to_remove = []
    for f in Path(args.path).glob("**/events.out.tfevents.*"):
        if os.path.getsize(f) == 0:
            to_remove_path = "/".join(f.parts[:-2])
            paths_to_remove.append(to_remove_path)

    # Remove duplicates (There is 2 events file for each experiments, we remove the experiment instead of
    #                    those individuals folders)
    paths_to_remove = list(set(paths_to_remove))

    if len(paths_to_remove) > 0:
        for path in paths_to_remove:
            print(f"Removing {path}")
            rm_folder(path)

        print("All done")
    else:
        print(f"No empty tensorboard session found in '{os.getcwd()}/{args.path}'")

