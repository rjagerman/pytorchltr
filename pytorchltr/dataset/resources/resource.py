from pytorchltr.dataset.svmrank import svmranking_dataset
from urllib.request import urlopen
import os
import shutil
import tarfile
import logging


class Resource:
    """A resource provides access to a publically available dataset.

    A resource provides convenient access to the train, vali and/or test splits
    provided by publically available datasets and can provide download
    functionality if available.
    """
    def __init__(self, name, location, download=False, url=None):
        """
        Initializes the resource with given name in directory at root.

        Arguments:
            name: The name of the resource.
            location: The directory to load the resource from.
            download: If True, this will force a download of the dataset into
                given root folder.
            url: The URL to download from if download is True.
        """
        self.name = name
        self.location = location
        self._url = url
        self._cache = {}
        if download and url:
            self.download_and_extract()

    def download_and_extract(self):
        """
        Downloads and extracts the resource.
        """

        # Download
        req = urlopen(self._url)
        dest = os.path.join(self.location, "data.tar.gz")
        logging.info("downloading %s dataset from %s to %s",
                     self.name, self._url, dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as fp:
            shutil.copyfileobj(req, fp, 32 * 1024)

        # Extract
        logging.info("extracting %s dataset from %s to %s",
                     self.name, dest, self.location)
        with tarfile.open(dest) as f:
            f.extractall(self.location)

    def load_svmrank_dataset(self, path, cache_name=None, normalize=False,
                             filter_queries=False, sparse=False):
        """
        Loads an svmrank-style dataset from given path, while storing it in an
        in-memory cache to speed-up future access.

        Arguments:
            path: The path to load the svmrank dataset from
            cache_name: The name to use to cache the loaded dataset
            normalize: Whether to perform query-level feature normalization
            filter_queries: Whether to filter out queries that have no relevant
                documents.
            sparse: Boolean indicating whether to load this svmrank dataset as
                a sparse dataset.
        """
        if cache_name and cache_name in self._cache:
            return self._cache[cache_name]
        else:
            logging.info("loading svmrank dataset from %s", path)
            data = svmranking_dataset(
                path, normalize=normalize, filter_queries=filter_queries,
                sparse=sparse)
            if cache_name:
                self._cache[cache_name] = data
            return data

    def train(self):
        raise NotImplementedError("%s provides no train split" % self.name)

    def test(self):
        raise NotImplementedError("%s provides no test split" % self.name)

    def vali(self):
        raise NotImplementedError("%s provides no vali split" % self.name)
