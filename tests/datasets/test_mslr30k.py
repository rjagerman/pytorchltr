import os
from unittest import mock

import pytest
from pytorchltr.datasets.mslr30k import MSLR30K
from tests.datasets.test_svmrank import mock_svmrank_dataset


pkg = "pytorchltr.datasets.mslr30k"


def test_wrong_split_raises_error():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        with pytest.raises(ValueError):
            MSLR30K(tmpdir, split="nonexisting")


def test_wrong_fold_raises_error():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        with pytest.raises(ValueError):
            MSLR30K(tmpdir, split="train", fold=99)


def test_call_validate_download():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        MSLR30K(tmpdir, split="train")
        mock_vali.called_once()
        args, kwargs = mock_vali.call_args
        assert kwargs["location"] == tmpdir
        assert kwargs["validate_checksums"] == True
        assert isinstance(kwargs["expected_files"], list)


def test_call_super_train():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        MSLR30K(tmpdir, split="train", fold=1)
        mock_super.called_once()
        args, kwargs = mock_super.call_args
        assert kwargs["file"] == os.path.join(tmpdir, "Fold1", "train.txt")
        assert kwargs["normalize"] == True
        assert kwargs["filter_queries"] == False


def test_call_super_vali():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        MSLR30K(tmpdir, split="vali", fold=2)
        mock_super.called_once()
        args, kwargs = mock_super.call_args
        assert kwargs["file"] == os.path.join(tmpdir, "Fold2", "vali.txt")
        assert kwargs["normalize"] == True
        assert kwargs["filter_queries"] == True


def test_call_super_test():
    with mock_svmrank_dataset(pkg) as (tmpdir, mock_super, mock_vali):
        MSLR30K(tmpdir, split="test", fold=5)
        mock_super.called_once()
        args, kwargs = mock_super.call_args
        assert kwargs["file"] == os.path.join(tmpdir, "Fold5", "test.txt")
        assert kwargs["normalize"] == True
        assert kwargs["filter_queries"] == True
