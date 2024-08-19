import torch
import unittest
from diffusers.models.attention import FeedForward
from xfuser.model_executor.layers.feedforward import xFuserFeedForwardWrapper
from xfuser.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from torch import distributed as dist


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        init_distributed_environment()

        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()

        initialize_model_parallel(tensor_parallel_degree=self.world_size)

    def test_feedforward(self):
        torch.manual_seed(0)
        self.input_data = torch.ones(1, 20).cuda(self.local_rank)
        dist.broadcast(self.input_data, src=0)

        torch.manual_seed(0)
        self.model1 = FeedForward(20, 5, bias=True, activation_fn="geglu").cuda(
            self.local_rank
        )

        # Broadcast the parameters
        for param in self.model1.parameters():
            dist.broadcast(param.data, src=0)

        output1 = self.model1(self.input_data)

        self.model2 = xFuserFeedForwardWrapper(self.model1)
        output2 = self.model2(self.input_data)

        print(output1 - output2)
        self.assertTrue(torch.allclose(output1, output2, atol=1e-2))


# torchrun --nproc_per_node=2 ./tests/layers/ffn_test.py
if __name__ == "__main__":
    unittest.main()
