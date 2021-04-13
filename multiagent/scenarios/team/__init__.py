import importlib.machinery
import os.path as osp
import sys


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    if name not in sys.modules:
        module_loaded = importlib.machinery.SourceFileLoader('', pathname).load_module()
    else:
        module_loaded = sys.modules.get(name)
    return module_loaded
