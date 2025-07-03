import pytest
from unittest.mock import patch, MagicMock

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import bucketize_kjt_before_all2all

class TestBucketizeKJTBeforeAll2All:
    @pytest.fixture
    def mock_kjt(self):
        kjt = MagicMock(spec=KeyedJaggedTensor)
        kjt.keys.return_value = ["feature1", "feature2"]
        kjt.values.return_value = torch.tensor([1, 2, 3, 4])
        kjt.lengths.return_value = torch.tensor([2, 2])
        kjt.weights_or_none.return_value = None
        return kjt

    @patch("torchrec.distributed.embedding_sharding._fx_wrap_tensor_to_device_dtype")
    @patch("hybrid_torchrec.modules.ids_process.block_bucketize_sparse_features_cpu")
    def test_basic_bucketize(self, mock_bucketize, mock_wrap, mock_kjt):
        # Setup mock returns
        mock_wrap.return_value = torch.tensor([1, 1])
        mock_bucketize.return_value = (
            torch.tensor([1, 1]),  # bucketized_lengths
            torch.tensor([1, 2]),  # bucketized_indices
            None,  # bucketized_weights
            None,  # pos
            torch.tensor([0, 1]),  # unbucketize_permute
            None,  # _
        )

        # Call function
        result, permute = bucketize_kjt_before_all2all(
            kjt=mock_kjt,
            num_buckets=2,
            block_sizes=torch.tensor([1, 1])
        )

        # Assertions
        assert isinstance(result, KeyedJaggedTensor)
        assert permute is not None
        mock_bucketize.assert_called_once()

    @patch("torchrec.distributed.embedding_sharding._fx_wrap_tensor_to_device_dtype")
    def test_block_sizes_mismatch(self, mock_wrap, mock_kjt):
        mock_wrap.return_value = torch.tensor([1, 1])

        with pytest.raises(AssertionError) as excinfo:
            bucketize_kjt_before_all2all(
                kjt=mock_kjt,
                num_buckets=2,
                block_sizes=torch.tensor([1])  # Only 1 block size for 2 features
            )

        assert "Expecting block sizes for 2 features" in str(excinfo.value)

    @patch("torchrec.distributed.embedding_sharding._fx_wrap_tensor_to_device_dtype")
    @patch("hybrid_torchrec.modules.ids_process.block_bucketize_sparse_features_cpu")
    def test_with_bucketize_pos(self, mock_bucketize, mock_wrap, mock_kjt):
        mock_wrap.return_value = torch.tensor([1, 1])
        mock_bucketize.return_value = (
            torch.tensor([1, 1]),
            torch.tensor([1, 2]),
            None,
            torch.tensor([0.5, 0.5]),  # pos
            torch.tensor([0, 1]),
            None,
        )

        result, _ = bucketize_kjt_before_all2all(
            kjt=mock_kjt,
            num_buckets=2,
            block_sizes=torch.tensor([1, 1]),
            bucketize_pos=True
        )

        assert result.weights() is not None

    @patch("torchrec.distributed.embedding_sharding._fx_wrap_tensor_to_device_dtype")
    @patch("hybrid_torchrec.modules.ids_process.block_bucketize_sparse_features_cpu")
    def test_with_output_permute(self, mock_bucketize, mock_wrap, mock_kjt):
        mock_wrap.return_value = torch.tensor([1, 1])
        mock_bucketize.return_value = (
            torch.tensor([1, 1]),
            torch.tensor([1, 2]),
            None,
            None,
            torch.tensor([0, 1]),  # unbucketize_permute
            None,
        )

        _, permute = bucketize_kjt_before_all2all(
            kjt=mock_kjt,
            num_buckets=2,
            block_sizes=torch.tensor([1, 1]),
            output_permute=True
        )

        assert permute is not None
