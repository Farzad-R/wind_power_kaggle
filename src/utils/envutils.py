import yaml
import os


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def check_directory(dir: str):
    """Create directory if it does not exist.
    Args:
    dir (str): relative address (from the root directory of the project)

    Returns:
    print a statement about the existance of the directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("The new directory is created!")
    else:
        print("Path already exists!")