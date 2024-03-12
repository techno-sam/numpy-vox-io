from struct import unpack_from as unpack, calcsize
import logging

from .models import Vox, Size, Voxel, Color, Model, Material

# File format:
# https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt
# https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt

log = logging.getLogger(__name__)

class ParsingException(Exception): pass

def bit(val, offset):
    mask = 1 << offset
    return(val & mask)

def read_string(buf: bytes | bytearray | memoryview, offset: int) -> tuple[str, int]:
    """
    :param buf: data buffer
    :param offset: start offset in buffer
    :return: (string, new offset)
    """
    size, = unpack('I', buf, offset)
    offset += 4
    string, = unpack(f'{size}s', buf, offset)#sum(chr(v) for v in unpack(f"{size}B", buf, offset))
    offset += size
    return string, offset

def read_dict(buf: bytes | bytearray | memoryview, offset: int) -> tuple[dict[str, str], int]:
    """
    :param buf: data buffer
    :param offset: start offset in buffer
    :return: (dict, new offset)
    """
    count, = unpack('I', buf, offset)
    offset += 4
    out = {}
    for _ in range(count):
        k, offset = read_string(buf, offset)
        v, offset = read_string(buf, offset)
        out[k] = v
    return out, offset

def read_rotation(buf: bytes | bytearray | memoryview, offset: int) -> tuple[int, int]:
    """
    :param buf: data buffer
    :param offset: start offset in buffer
    :return: (encoded_rotation, new offset)
    """
    out, = unpack('B', buf, offset)
    offset += 1
    return out, offset

class Chunk(object):
    def __init__(self, chunk_id, content=None, chunks=None):
        self.chunk_id = chunk_id
        self.content = content or b''
        self.chunks = chunks or []

        if chunk_id == b'MAIN':
            if len(self.content): raise ParsingException('Non-empty content for main chunk')
        elif chunk_id == b'PACK':
            self.models = unpack('I', content)[0]
        elif chunk_id == b'SIZE':
            self.size = Size(*unpack('III', content))
        elif chunk_id == b'XYZI':
            n = unpack('i', content)[0]
            log.debug('xyzi block with %d voxels (len %d)', n, len(content))
            self.voxels = []
            self.voxels = [ Voxel(*unpack('BBBB', content, 4+4*i)) for i in range(n) ]
        elif chunk_id == b'RGBA':
            self.palette = [ Color(*unpack('BBBB', content, 4*i)) for i in range(255) ]
            # Docs say:  color [0-254] are mapped to palette index [1-255]
            # hmm
            # self.palette = [ Color(0,0,0,0) ] + [ Color(*unpack('BBBB', content, 4*i)) for i in range(255) ]
        elif chunk_id == b'MATT':
            _id, _type, weight, flags = unpack('IIfI', content)
            props = {}
            offset = 16
            for b,field in [ (0, 'plastic'),
                             (1, 'roughness'),
                             (2, 'specular'),
                             (3, 'IOR'),
                             (4, 'attenuation'),
                             (5, 'power'),
                             (6, 'glow'),
                             (7, 'isTotalPower') ]:
                if bit(flags, b) and b<7: # no value for 7 / isTotalPower
                    props[field] = unpack('f', content, offset)
                    offset += 4

            self.material = Material(_id, _type, weight, props)
        # todo actually parse these
        elif chunk_id == b'nTRN':
            self.todo = True
        elif chunk_id == b'nGRP':
            self.todo = True
        elif chunk_id == b'nSHP':
            self.todo = True
        elif chunk_id == b'MATL':
            self.material_id, = unpack('I', content)
            self.material_properties, _ = read_dict(content, 4)
        elif chunk_id == b'LAYR':
            self.todo = True
        elif chunk_id == b'rOBJ':
            self.rendering_attributes, _ = read_dict(content, 0)
        elif chunk_id == b'rCAM':
            self.todo = True
        elif chunk_id == b'NOTE':
            count, = unpack('I', content)
            offset = 4
            self.color_names: list[str] = []
            for _ in range(count):
                name, offset = read_string(content, offset)
                self.color_names.append(name)
        elif chunk_id == b'IMAP':
            self.todo = True
        else:
            raise ParsingException('Unknown chunk type: %s' % self.chunk_id)

    @property
    def __filtered_dict(self) -> dict:
        out = {}
        banned = ["chunk_id", "content", "chunks"]
        for k, v in self.__dict__.items():
            if k not in banned:
                out[k] = v
        return out

    def __repr__(self):
        return f"Chunk({self.chunk_id}, {self.__filtered_dict}, {self.chunks})"

    def __str__(self):
        return repr(self)

class VoxParser(object):

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.content = f.read()

        self.offset = 0

    def unpack(self, fmt):
        r = unpack(fmt, self.content, self.offset)
        self.offset += calcsize(fmt)
        return r

    def _parseChunk(self):

        _id, N, M = self.unpack('4sII')

        log.debug("Found chunk id %s / len %s / children %s", _id, N, M)

        content = self.unpack('%ds'%N)[0]

        start = self.offset
        chunks = [ ]
        while self.offset<start+M:
            chunks.append(self._parseChunk())

        return Chunk(_id, content, chunks)

    def parse(self):

            header, version = self.unpack('4sI')

            if header != b'VOX ': raise ParsingException("This doesn't look like a vox file to me")

            if version not in [150, 200]: raise ParsingException("Unknown vox version: %s expected 150 or 200"%version)

            main = self._parseChunk()

            if main.chunk_id != b'MAIN': raise ParsingException("Missing MAIN Chunk")

            chunks: list[Chunk] = list(reversed(main.chunks))
            if chunks[-1].chunk_id == b'PACK':
                models: int = chunks.pop().models
            else:
                models: int = 1

            log.debug("file has %d models", models)

            models: list[Model] = [ self._parseModel(chunks.pop(), chunks.pop()) for _ in range(models) ]

            palette = None
            for c in chunks:
                if c.chunk_id == b'RGBA':
                    palette = c.palette
                    break

            materials: list[Material] = [ c.material for c in chunks if c.chunk_id == b'MATT' ]
            if len(materials) == 0:
                materials: list[dict[str, str]|None] = [None]*256
                for c in chunks:
                    if c.chunk_id == b'MATL':
                        materials[c.material_id - 1] = c.material_properties

            return Vox(models, palette, materials)



    def _parseModel(self, size, xyzi):
        if size.chunk_id != b'SIZE': raise ParsingException('Expected SIZE chunk, got %s', size.chunk_id)
        if xyzi.chunk_id != b'XYZI': raise ParsingException('Expected XYZI chunk, got %s', xyzi.chunk_id)

        return Model(size.size, xyzi.voxels)




if __name__ == '__main__':

    import sys
    import coloredlogs

    coloredlogs.install(level=logging.DEBUG)


    VoxParser(sys.argv[1]).parse()
