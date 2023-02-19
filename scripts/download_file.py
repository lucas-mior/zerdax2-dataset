import os
from Ipython import display, FileLink


def download_file(file_name):
    base_name = os.path.basename(file_name)
    k_info_file = ".download_file_info.txt"

    if os.path.isfile(k_info_file):
        with open(k_info_file, "r") as fin:
            previous_file = fin.read()

        if os.path.isfile(previous_file):
            print("Removing previous file link.")
            os.remove(previous_file)

    with open(k_info_file, "w") as fout:
        fout.write(base_name)
    os.symlink(file_name, base_name)
    display(FileLink(base_name))

# download_file('yolov5/runs/train/exp/weights/best.pt')
