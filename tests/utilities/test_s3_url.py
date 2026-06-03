from languagemodelcommon.utilities.s3_url import S3Url


class TestS3Url:
    def test_basic_url(self) -> None:
        s = S3Url(url="s3://my-bucket/path/to/file.json")
        assert s.bucket == "my-bucket"
        assert s.key == "path/to/file.json"
        assert s.url == "s3://my-bucket/path/to/file.json"

    def test_url_with_query_string(self) -> None:
        s = S3Url(url="s3://bucket/hello/world?version=2")
        assert s.bucket == "bucket"
        assert s.key == "hello/world?version=2"

    def test_url_with_fragment(self) -> None:
        s = S3Url(url="s3://bucket/hello/world#section")
        assert s.bucket == "bucket"
        assert s.key == "hello/world#section"

    def test_url_with_query_and_fragment(self) -> None:
        s = S3Url(url="s3://bucket/path?qwe1=3#ddd")
        assert s.bucket == "bucket"
        assert s.key == "path?qwe1=3#ddd"

    def test_root_key(self) -> None:
        s = S3Url(url="s3://bucket/")
        assert s.bucket == "bucket"
        assert s.key == ""

    def test_nested_path(self) -> None:
        s = S3Url(url="s3://bucket/a/b/c/d.txt")
        assert s.bucket == "bucket"
        assert s.key == "a/b/c/d.txt"
