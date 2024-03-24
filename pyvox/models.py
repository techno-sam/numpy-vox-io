from collections import namedtuple

from .defaultpalette import default_palette
from .utils import chunks

Size = namedtuple('Size', 'x y z')
Color = namedtuple('Color', 'r g b a')
Voxel = namedtuple('Voxel', 'x y z c')
Model = namedtuple('Model', 'size voxels')
Material = namedtuple('Material', 'id type weight props')
nTrn = namedtuple('nTrn', 'id attributes child_id reserved_id layer_id num_frames frames')
nGrp = namedtuple('nGrp', 'id attributes num_children child_ids')
nShp = namedtuple('nShp', 'id attributes num_models models')
ShapeModelEntry = namedtuple('ShapeModelEntry', 'model_id attributes')
Layer = namedtuple('Layer', 'id attributes reserved_id')
Camera = namedtuple('Camera', 'id attributes')
Frame = namedtuple('Frame', 'attributes')

class Transform:
    def __init__(self, id_: int, attributes: dict[str, str], child: 'Transform|Group|Shape', reserved_id: int, layer_id: int, frames: list[Frame]):
        self.id_ = id_
        self.attributes = attributes
        self.child = child
        self.reserved_id = reserved_id
        self.layer_id = layer_id
        self.frames = frames

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Transform({self.id_}, {self.attributes}, {type(self.child)}#{self.child.id_})"

class Group:
    def __init__(self, id_: int, attributes: dict[str, str], children: list['Transform']):
        self.id_ = id_
        self.attributes = attributes
        self.children = children

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Group({self.id_}, {self.attributes}, [{', '.join([f'{type(child)}#{child.id_}' for child in self.children])}])"

class Shape:
    def __init__(self, id_: int, attributes: dict[str, str], models: list[ShapeModelEntry]):
        self.id_ = id_
        self.attributes = attributes
        self.models = models

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Shape({self.id_}, {self.attributes}, {self.models})"

def get_default_palette():
    return [ Color( *tuple(i.to_bytes(4,'little')) ) for i in default_palette ]


class Vox(object):

    def __init__(
            self, models: list[Model], palette: list[Color]|None = None,
            materials: list[Material]|None = None, root_transform: Transform|None = None
    ):
        self.models: list[Model] = models
        self.default_palette: bool = not palette
        self._palette: list[Color] = palette or get_default_palette()
        self.materials: list[Material] = materials or []
        self.root_transform: Transform|None = root_transform

    @property
    def palette(self) -> list[Color]:
        return self._palette

    @palette.setter
    def palette(self, val: list[Color]):
        self._palette = val
        self.default_palette = False

    def to_dense_rgba(self, model_idx=0):

        import numpy as np
        m = self.models[model_idx]
        res = np.zeros(( m.size.y, m.size.z, m.size.x, 4 ), dtype='B')

        for v in m.voxels:
            res[v.y, m.size.z-v.z-1, v.x] = self.palette[v.c]

        return res

    def to_dense(self, model_idx=0):

        import numpy as np
        m = self.models[model_idx]
        res = np.zeros(( m.size.y, m.size.z, m.size.x ), dtype='B')

        for v in m.voxels:
            res[v.y, m.size.z-v.z-1, v.x] = v.c

        return res

    def __str__(self):
        return 'Vox(%s)'%(self.models)

    @staticmethod
    def from_dense(a, black=[0,0,0]):

        palette = None

        if len(a.shape) == 4:
            from PIL import Image
            import numpy as np

            mask = np.all(a == np.array([[black]]), axis=3)

            x,y,z,_ = a.shape

            # color index 0 is reserved for empty, so we get 255 colors
            img = Image.fromarray(a.reshape(x,y*z,3)).quantize(255)
            palette = img.getpalette()
            palette = [ Color(*c, 255) for c in chunks(palette, 3) ]
            a = np.asarray(img, dtype='B').reshape(x,y,z).copy() + 1
            a[mask] = 0


        if len(a.shape) != 3: raise Exception("I expect a 4 or 3 dimensional matrix")

        y,z,x = a.shape

        nz = a.nonzero()

        voxels = [ Voxel( nz[2][i], nz[0][i], z-nz[1][i]-1, a[nz[0][i], nz[1][i], nz[2][i]] ) for i in range(nz[0].shape[0]) ]

        return Vox([ Model(Size(x,y,z), voxels)], palette)
