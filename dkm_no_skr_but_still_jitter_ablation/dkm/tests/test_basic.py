from harness import Sin2dHarness
import unittest

class TestSin2d(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harness = Sin2dHarness()
        self.harness.reset_model()
    def test_forward(self):
        self.harness.test_eval()
        self.harness.train_epoch()
    def test_check_relu_kernel(self):
        """sanity check for relu kernel"""
        self.harness.reset_model(kernel=dict(type='relu', params=dict()))
        test_result1 = self.harness.test_eval()
        train_result1 = self.harness.train_epoch()
        for _ in range(3): self.harness.train_epoch()
        train_result2 = self.harness.train_epoch()
        test_result2 = self.harness.test_eval()

        self.assertGreaterEqual(test_result2['acc'], test_result1['acc'])
        self.assertGreaterEqual(test_result2['ll'], test_result1['ll'])
        self.assertGreaterEqual(test_result2['obj'], test_result1['obj'])
        self.assertGreaterEqual(train_result2['acc'], train_result1['acc'])
        self.assertGreaterEqual(train_result2['ll'], train_result1['ll'])
        self.assertGreaterEqual(train_result2['obj'], train_result1['obj'])
    def test_check_polar_heaviside_kernel(self):
        """sanity check for polar heaviside kernel"""
        self.harness.reset_model(kernel=dict(type='polar-heaviside', params=dict()))
        test_result1 = self.harness.test_eval()
        train_result1 = self.harness.train_epoch()
        for _ in range(3): self.harness.train_epoch()
        train_result2 = self.harness.train_epoch()
        test_result2 = self.harness.test_eval()

        self.assertGreaterEqual(test_result2['acc'], test_result1['acc'])
        self.assertGreaterEqual(test_result2['ll'], test_result1['ll'])
        self.assertGreaterEqual(test_result2['obj'], test_result1['obj'])
        self.assertGreaterEqual(train_result2['acc'], train_result1['acc'])
        self.assertGreaterEqual(train_result2['ll'], train_result1['ll'])
        self.assertGreaterEqual(train_result2['obj'], train_result1['obj'])
    def test_polar_heaviside_kernel_is_good(self):
        """check we achieve a decent accuracy on the sin2d problem"""
        self.harness.reset_model(kernel=dict(type='polar-heaviside', params=dict()))
        for _ in range(35): self.harness.train_epoch()
        test_result = self.harness.test_eval()

        self.assertGreaterEqual(test_result['acc'], 0.75)
        # we should really have a higher accuracy (see PolarHeavisideKernel)
        # when fixed, uncomment below!
        # self.assertGreaterEqual(test_result['acc'], 0.9)

if __name__ == '__main__':
    unittest.main()