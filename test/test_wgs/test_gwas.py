import unittest
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from heig.wgs.gwas import parse_ldr_col


class Test_parse_ldr_col(unittest.TestCase):
    def test_good_cases(self):
        target = (0, 3)
        input_string = "1:3"
        self.assertEqual(target, parse_ldr_col(input_string))

        target = (2, 5)
        input_string = "3:5"
        self.assertEqual(target, parse_ldr_col(input_string))

        target = (2, 5)
        input_string = "3,4,5"
        self.assertEqual(target, parse_ldr_col(input_string))

        target = (0, 10)
        input_string = "1:8,9,10"
        self.assertEqual(target, parse_ldr_col(input_string))

        target = (0, 10)
        input_string = "1,2,3:10"
        self.assertEqual(target, parse_ldr_col(input_string))

    def test_bad_cases(self):
        input_string = "1:3,5"
        with self.assertRaises(ValueError):
            parse_ldr_col(input_string)

        input_string = "0:3"
        with self.assertRaises(ValueError):
            parse_ldr_col(input_string)
