import io
import zipfile
from unittest.mock import patch
import pytest
from scripts.inventory_video_mme import HTTPRangeFile


class Response(io.BytesIO):
    status=206
    def __init__(self,data,start,end,total):
        super().__init__(data)
        self.headers={'Content-Range':f'bytes {start}-{end}/{total}'}


def test_zip_reads_only_ranges_and_extracts_exact_content():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('video.mp4',b'example bytes'*100)
    data=buffer.getvalue()
    def serve(request,timeout):
        start,end=map(int,request.get_header('Range').split('=')[1].split('-'))
        return Response(data[start:end+1],start,end,len(data))
    with patch('scripts.inventory_video_mme.urllib.request.urlopen',side_effect=serve):
        remote=HTTPRangeFile('https://example.invalid/archive',len(data))
        with zipfile.ZipFile(remote) as archive:
            assert archive.read('video.mp4')==b'example bytes'*100
        assert remote.transferred>0


def test_server_ignoring_range_is_rejected_before_payload_read():
    response=Response(b'not an archive',0,13,14);response.status=200
    with patch('scripts.inventory_video_mme.urllib.request.urlopen',return_value=response), patch.object(response,'read',side_effect=AssertionError('must not read whole archive')):
        with pytest.raises(ValueError,match='range'):
            HTTPRangeFile('https://example.invalid/archive',100).read(4)


def test_unbounded_download_refused():
    with pytest.raises(ValueError,match='unbounded'):
        HTTPRangeFile('https://example.invalid/archive',100_000_000).read()


def test_transient_range_failure_retries_same_bytes():
    import urllib.error
    response=Response(b'abcd',0,3,4)
    with patch('scripts.inventory_video_mme.urllib.request.urlopen',side_effect=[urllib.error.URLError('timed out'),response]) as request:
        assert HTTPRangeFile('https://example.invalid/archive',4).read(4)==b'abcd'
        assert request.call_count==2
        assert request.call_args_list[0].args[0].full_url != request.call_args_list[1].args[0].full_url
        assert request.call_args_list[0].args[0].get_header('Range')==request.call_args_list[1].args[0].get_header('Range')


def test_exhausted_transport_failure_is_not_misreported_as_bad_zip():
    import urllib.error
    with patch('scripts.inventory_video_mme.urllib.request.urlopen',side_effect=urllib.error.URLError('timed out')) as request:
        with pytest.raises(RuntimeError,match='transport'):
            HTTPRangeFile('https://example.invalid/archive',4).read(4)
        assert request.call_count==3
