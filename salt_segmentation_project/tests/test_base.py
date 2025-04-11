import unittest
import torch

class TestBase(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.img_size = 101
        self.dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        self.dummy_mask = torch.randint(0, 2, (self.batch_size, 1, self.img_size, self.img_size)).float()

    def to_device(self, tensor):
        return tensor.to(self.device)
