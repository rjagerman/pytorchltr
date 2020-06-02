import os

import pytest
from pytorchltr.datasets.example3 import Example3
from tests.datasets.test_svmrank import mock_svmrank_dataset


pkg = "pytorchltr.datasets.example3"


def test_wrong_split_raises_error():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        with pytest.raises(ValueError):
            Example3(tmpdir, split="nonexisting")


def test_call_validate_download():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        Example3(tmpdir, split="train")
        mock_vali.called_once()
        args, kwargs = mock_vali.call_args
        assert kwargs["location"] == tmpdir
        assert kwargs["validate_checksums"]
        assert isinstance(kwargs["expected_files"], list)


def test_call_super_train():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        Example3(tmpdir, split="train")
        mock_super.called_once()
        args, kwargs = mock_super.call_args
        assert kwargs["file"] == os.path.join(tmpdir, "example3", "train.dat")
        assert kwargs["normalize"]
        assert not kwargs["filter_queries"]


def test_call_super_test():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        Example3(tmpdir, split="test")
        mock_super.called_once()
        args, kwargs = mock_super.call_args
        assert kwargs["file"] == os.path.join(tmpdir, "example3", "test.dat")
        assert kwargs["normalize"]
        assert kwargs["filter_queries"]
