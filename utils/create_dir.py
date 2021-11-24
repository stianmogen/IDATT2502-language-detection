import os


def create_dir(parent, child):
    path = parent + child + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path