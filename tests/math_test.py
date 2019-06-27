import unittest
import pymagsac

#ToDo: add real tests
class MainTest(unittest.TestCase):
    def test_fundamental(self):
        self.assertEqual(2, 2)

    def test_homography(self):
        self.assertEqual(2, 0)

if __name__ == '__main__':
    unittest.main()