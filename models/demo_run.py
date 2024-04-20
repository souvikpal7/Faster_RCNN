import os.path as osp
import sys
# import config
# import RPN
# from . import config
# from . import RPN
from models import config
from models import RPN


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


if __name__ == "__main__":
    # this_dir = osp.dirname(__file__)
    #
    # # Add lib to PYTHONPATH
    # lib_path = osp.join(this_dir, 'lib')
    # add_path(lib_path)
    #
    # coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
    # add_path(coco_path)

    config.backbone = "haku"
    r = RPN.RPN()
