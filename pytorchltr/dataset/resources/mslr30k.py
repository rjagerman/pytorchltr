import os

from pytorchltr.dataset.resources.downloader import DefaultProgress
from pytorchltr.dataset.resources.downloader import Downloader
from pytorchltr.dataset.resources.resource import DownloadableResource
from pytorchltr.dataset.resources.util import extract_zip
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.dataset.svmrank import svmranking_dataset


class MSLR30K(DownloadableResource):
    """
    Utility class for downloading and using the MSLR-WEB30K	dataset:
    https://www.microsoft.com/en-us/research/project/mslr/.

    Attributes:
        normalize (bool): Whether to perform query-level feature normalization.
    """
    default_downloader = Downloader(
        url="https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbXGPBlwD1rnFdBY/root/content",  # noqa: E501
        target="MSLR-WEB30K.zip",
        sha256_checksum="08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae",  # noqa: E501
        progress_fn=DefaultProgress(),
        postprocess_fn=extract_zip)

    per_fold_expected_files = {
        1: [
            {"path": "Fold1/train.txt", "sha256": "40b8eee4d1221cf8205e81603441c1757dd024a3baac25a47756210b03c031d6"},  # noqa: E501
            {"path": "Fold1/test.txt", "sha256": "9a4668fd2615e6772d2e5c4d558d084b2daaf2405571eaf3e4d0526f4da096c7"},  # noqa: E501
            {"path": "Fold1/vali.txt", "sha256": "7647834b84c849a61e5cf3c999a2f72a4785613286fd972a5615e9fcb58f94d8"}  # noqa: E501
        ],
        2: [
            {"path": "Fold2/train.txt", "sha256": "a6d12dc4cc8c2dd0743b58c49cad12bd0f6f1cbeda54b54a91e5b54e0b96e7ca"},  # noqa: E501
            {"path": "Fold2/test.txt", "sha256": "d192c023baebdae148d902d716aeebf3df2f2f1ce5aee12be4f0b8bb76a8c04a"},  # noqa: E501
            {"path": "Fold2/vali.txt", "sha256": "9a4668fd2615e6772d2e5c4d558d084b2daaf2405571eaf3e4d0526f4da096c7"}  # noqa: E501
        ],
        3: [
            {"path": "Fold3/train.txt", "sha256": "3d0eb52a6702b2c48750a6de89e757cfac499c8ec1d38e3843cb531c059f2f74"},  # noqa: E501
            {"path": "Fold3/test.txt", "sha256": "ba2487a27c21dceea0b9afaadbb6b392d5f652eb0fb8649cbd201bb894c47c12"},  # noqa: E501
            {"path": "Fold3/vali.txt", "sha256": "d192c023baebdae148d902d716aeebf3df2f2f1ce5aee12be4f0b8bb76a8c04a"}  # noqa: E501
        ],
        4: [
            {"path": "Fold4/train.txt", "sha256": "21e389656be3c2bfe92eb4fb898d2f30dc24990fc524aada7471b949f176778d"},  # noqa: E501
            {"path": "Fold4/test.txt", "sha256": "a7ba03d708ae6b21556a8a6859d52e37cc0dcb2cbff992c630ef6173bb02122a"},  # noqa: E501
            {"path": "Fold4/vali.txt", "sha256": "ba2487a27c21dceea0b9afaadbb6b392d5f652eb0fb8649cbd201bb894c47c12"}  # noqa: E501
        ],
        5: [
            {"path": "Fold5/train.txt", "sha256": "a19da1799650b08c0dc3baaec6c590469ef1773148082e1a10b8504f2f5e9a8b"},  # noqa: E501
            {"path": "Fold5/test.txt", "sha256": "7647834b84c849a61e5cf3c999a2f72a4785613286fd972a5615e9fcb58f94d8"},  # noqa: E501
            {"path": "Fold5/vali.txt", "sha256": "a7ba03d708ae6b21556a8a6859d52e37cc0dcb2cbff992c630ef6173bb02122a"}  # noqa: E501
        ]
    }

    def __init__(self, location, fold=1, normalize=True,
                 downloader=default_downloader, validate_checksums=True):
        """
        Args:
            location: Directory where the dataset is located.
            fold: The fold to use (1...5) in the MSLR dataset.
            normalize: Whether to perform query-level feature normalization.
            downloader: A downloader for downloading the dataset.
            validate_checksums: Whether to validate the dataset files via
                sha256.
        """
        super().__init__("MSLR-WEB30K", location,
                         expected_files=MSLR30K.per_fold_expected_files[fold],
                         downloader=downloader,
                         validate_checksums=validate_checksums)
        self.normalize = normalize
        self._fold = fold

    def fold_folder(self):
        return "Fold%d" % self._fold

    def collate_fn(self):
        return create_svmranking_collate_fn()

    def train(self):
        return svmranking_dataset(
            os.path.join(self.location, self.fold_folder(), "train.txt"),
            sparse=False, normalize=self.normalize, filter_queries=False)

    def test(self):
        return svmranking_dataset(
            os.path.join(self.location, self.fold_folder(), "test.txt"),
            sparse=False, normalize=self.normalize, filter_queries=True)

    def vali(self):
        return svmranking_dataset(
            os.path.join(self.location, self.fold_folder(), "vali.txt"),
            sparse=False, normalize=self.normalize, filter_queries=True)
