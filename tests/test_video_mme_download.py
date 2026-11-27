import zlib
import hashlib
import pytest
from scripts.download_video_mme_subset import unpack,checksum


@pytest.mark.parametrize('compression',[0,8])
def test_exact_uncompressed_bytes_crc_and_sha(tmp_path,compression):
    data=b'example video payload'*100
    encoder=zlib.compressobj(wbits=-15)
    raw=encoder.compress(data)+encoder.flush() if compression==8 else data
    source=tmp_path/'compressed';source.write_bytes(raw)
    target=tmp_path/'video.mp4'
    member={'compression':compression,'size':len(data),'crc':zlib.crc32(data)}
    digest=unpack(source,target,member)
    assert target.read_bytes()==data
    assert digest==hashlib.sha256(data).hexdigest()
    assert checksum(target)==(len(data),zlib.crc32(data),digest)


def test_corrupt_payload_never_published(tmp_path):
    source=tmp_path/'compressed';source.write_bytes(b'bad bytes')
    target=tmp_path/'video.mp4'
    with pytest.raises(ValueError,match='CRC'):
        unpack(source,target,{'compression':0,'size':9,'crc':123})
    assert not target.exists()


def test_oversize_payload_never_published(tmp_path):
    source=tmp_path/'compressed';source.write_bytes(b'too big')
    target=tmp_path/'video.mp4'
    with pytest.raises(ValueError,match='exceeds'):
        unpack(source,target,{'compression':0,'size':1,'crc':123})
    assert not target.exists()


def test_resume_fetches_only_remaining_compressed_payload(tmp_path):
    import io,json,zipfile,struct
    from unittest.mock import patch
    from scripts.download_video_mme_subset import download
    data=b'unit test payload'*1000
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data/test.mp4',data)
    raw=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive: info=archive.getinfo('data/test.mp4')
    header=struct.unpack('<4s5H3I2H',raw[:30]);begin=30+header[9]+header[10]
    prefix=info.compress_size//2
    (tmp_path/'001.compressed.part').write_bytes(raw[begin:begin+prefix])
    class Reader(io.BytesIO):
        def __init__(self): super().__init__(raw);self.seeks=[]
        def seek(self,offset,whence=0): self.seeks.append(offset);return super().seek(offset,whence)
    remote=Reader()
    member={'name':info.filename,'compression':8,'crc':info.CRC,'size':info.file_size,
            'compressed_size':info.compress_size,'archive':'a.zip','archive_bytes':len(raw),'header_offset':0}
    probe={'format':{'duration':'10'},'streams':[{'codec_type':'video'},{'codec_type':'audio'}]}
    with patch('scripts.download_video_mme_subset.HTTPRangeFile',return_value=remote), patch('scripts.download_video_mme_subset.subprocess.check_output',return_value=json.dumps(probe)):
        result=download({'repo':'example/repo','revision':'pinned'},{'video_id':'001','member':member},tmp_path)
    assert begin+prefix in remote.seeks
    assert (tmp_path/'001.mp4').read_bytes()==data
    assert result['sha256']==hashlib.sha256(data).hexdigest()
