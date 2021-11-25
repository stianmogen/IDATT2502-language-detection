import os

"""
Method to create directory with child and parent
Checks if directory does not already exist
"""
def create_dir(parent, child):
    path = parent + child + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path