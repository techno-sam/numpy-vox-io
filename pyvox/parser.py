from struct import unpack_from as unpack, calcsize
import logging

from .models import Vox, Size, Voxel, Color, Model, Material, nTrn, nGrp, nShp, ShapeModelEntry, Layer, Camera, Frame, Transform, Group, Shape

# File format:
# https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt
# https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt

log = logging.getLogger(__name__)
#log.setLevel('DEBUG')
log.addHandler(logging.StreamHandler())

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
    string: bytes = unpack(f'{size}s', buf, offset)[0]#sum(chr(v) for v in unpack(f"{size}B", buf, offset))
    string: str = string.decode('utf-8')
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
            self.models = unpack('i', content)[0]
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
        elif chunk_id == b'nTRN':
            _id, = unpack('i', content)
            offset = 4
            attributes, offset = read_dict(content, offset)
            child_id, = unpack('i', content, offset)
            offset += 4
            reserved_id, = unpack('i', content, offset)
            offset += 4
            layer_id, = unpack('i', content, offset)
            offset += 4
            frame_count, = unpack('i', content, offset)
            offset += 4
            frames: list[Frame] = []
            for _ in range(frame_count):
                frame_attributes, offset = read_dict(content, offset)
                frames.append(Frame(frame_attributes))
            self.node_transform = nTrn(_id, attributes, child_id, reserved_id, layer_id, frame_count, frames)
        elif chunk_id == b'nGRP':
            _id, = unpack('i', content)
            offset = 4
            attributes, offset = read_dict(content, offset)
            num_children, = unpack('i', content, offset)
            offset += 4
            child_ids: list[int] = []
            for _ in range(num_children):
                child_ids.append(unpack('i', content, offset)[0])
                offset += 4
            self.node_group = nGrp(_id, attributes, num_children, child_ids)
        elif chunk_id == b'nSHP':
            _id, = unpack('i', content)
            offset = 4
            attributes, offset = read_dict(content, offset)
            num_models, = unpack('i', content, offset)
            offset += 4
            models: list[ShapeModelEntry] = []
            for _ in range(num_models):
                model_id, = unpack('i', content, offset)
                offset += 4
                model_attributes, offset = read_dict(content, offset)
                models.append(ShapeModelEntry(model_id, model_attributes))
            self.node_shape = nShp(_id, attributes, num_models, models)
        elif chunk_id == b'MATL':
            self.material_id, = unpack('I', content)
            self.material_properties, _ = read_dict(content, 4)
        elif chunk_id == b'LAYR':
            _id, = unpack('i', content)
            offset = 4
            attributes, offset = read_dict(content, offset)
            reserved_id, = unpack('i', content, offset)
            self.layer = Layer(_id, attributes, reserved_id)
        elif chunk_id == b'rOBJ':
            self.rendering_attributes, _ = read_dict(content, 0)
        elif chunk_id == b'rCAM':
            _id, = unpack('i', content)
            offset = 4
            attributes, offset = read_dict(content, offset)
            self.camera = Camera(_id, attributes)
        elif chunk_id == b'NOTE':
            count, = unpack('i', content)
            offset = 4
            self.color_names: list[str] = []
            for _ in range(count):
                name, offset = read_string(content, offset)
                self.color_names.append(name)
        # todo parse
        elif chunk_id == b'IMAP':
            self.todo = True
        else:
            raise ParsingException('Unknown chunk type: %s' % self.chunk_id)

    @property
    def __filtered_dict(self) -> dict:
        out = {}
        banned = ["chunk_id", "content", "chunks"]
        for k, v in self.__dict__.items():
            if k == "voxels":
                out[k] = "[...]"
            elif k not in banned:
                out[k] = v
        return out

    def __repr__(self):
        return f"Chunk({self.chunk_id}, {self.__filtered_dict}, {self.chunks})"

    def __str__(self):
        return repr(self)


def _parse_transform(id_: int, elements: dict[int, nTrn | nGrp | nShp]) -> Transform:
    dat = elements[id_]
    child = elements[dat.child_id] if dat.child_id in elements else None
    if isinstance(child, nTrn):
        child = _parse_transform(child.id, elements)
    elif isinstance(child, nGrp):
        child = _parse_group(child.id, elements)
    elif isinstance(child, nShp):
        child = _parse_shape(child.id, elements)
    return Transform(dat.id, dat.attributes, child, dat.reserved_id, dat.layer_id, dat.frames)

def _parse_group(id_: int, elements: dict[int, nTrn | nGrp | nShp]) -> Group:
    dat = elements[id_]
    children = [ _parse_transform(child_id, elements) for child_id in dat.child_ids ]
    return Group(dat.id, dat.attributes, children)

def _parse_shape(id_: int, elements: dict[int, nTrn | nGrp | nShp]) -> Shape:
    dat = elements[id_]
    models = [ ShapeModelEntry(model_id, model_attributes) for model_id, model_attributes in dat.models ]
    return Shape(dat.id, dat.attributes, models)


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
            log.debug("remaining chunks: [%s]", "\n\t".join(str(c) for c in chunks))

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

            root_transform = None
            transform_elements: dict[int, nTrn|nGrp|nShp] = {}
            for c in chunks:
                if c.chunk_id == b'nTRN':
                    transform_elements[c.node_transform.id] = c.node_transform
                elif c.chunk_id == b'nGRP':
                    transform_elements[c.node_group.id] = c.node_group
                elif c.chunk_id == b'nSHP':
                    transform_elements[c.node_shape.id] = c.node_shape

            root_transform = _parse_transform(0, transform_elements)

            return Vox(models, palette, materials, root_transform)

    def _parseModel(self, size, xyzi):
        if size.chunk_id != b'SIZE': raise ParsingException('Expected SIZE chunk, got %s', size.chunk_id)
        if xyzi.chunk_id != b'XYZI': raise ParsingException('Expected XYZI chunk, got %s', xyzi.chunk_id)

        return Model(size.size, xyzi.voxels)




if __name__ == '__main__':

    import sys
    import coloredlogs

    coloredlogs.install(level=logging.DEBUG)


    VoxParser(sys.argv[1]).parse()
