# pointcept/models/ditr/__init__.py
from .pditr import PDITR_PTv3
from .ditr_utils import DINOFeatureExtractor, DITRInjector # 也可以不在这里导入，如果只在 pditr.py 中使用