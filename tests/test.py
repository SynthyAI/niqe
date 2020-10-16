import unittest
import cv2
from niqe import niqe


class NIQE(unittest.TestCase):

    def test(self):
        img_clean = cv2.imread("tests/pristine/car.png")
        img_degraded = cv2.imread("tests/degraded/car.png")
        print(img_clean)

        self.assertGreater(niqe(img_degraded), niqe(img_clean), "clean image NIQE is higher than degraded")

if __name__ == '__main__':
    unittest.main()
