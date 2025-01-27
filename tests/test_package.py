import pyirk as p
import unittest
from pathlib import Path
from os.path import join as pjoin


PACKAGE_ROOT_PATH = Path(__file__).parent.parent.absolute().as_posix()
olaa = p.irkloader.load_mod_from_path(pjoin(PACKAGE_ROOT_PATH, "olaa.py"), prefix="olaa")


class TestPackage01(unittest.TestCase):

    def test_001_subclass_relationship(self):

        pass
