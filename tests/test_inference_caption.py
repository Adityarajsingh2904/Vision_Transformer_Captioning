from contextlib import nullcontext
from types import ModuleType
from unittest import mock
import sys


def _add_stub(name: str) -> ModuleType:
    """Create a minimal stub module and register it in sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    sys.modules[name] = module
    parent_name, _, child = name.rpartition('.')
    if parent_name:
        parent = _add_stub(parent_name)
        setattr(parent, child, module)
    return module

# Stub out heavy or missing dependencies before importing the module under test
for mod in [
    "hydra",
    "numpy",
    "datasets",
    "datasets.caption",
    "datasets.caption.field",
    "datasets.caption.coco",
    "datasets.caption.transforms",
    "models",
    "models.caption",
    "models.caption.detector",
    "models.common",
    "models.common.attention",
    "engine",
    "engine.caption_engine",
    "engine.utils",
    "torch",
    "torch.distributed",
    "torch.multiprocessing",
    "torch.nn",
    "torch.nn.parallel",
]:
    _add_stub(mod)

# Provide simple behavior for attributes accessed during import
sys.modules["hydra"].initialize = lambda *a, **k: nullcontext()
sys.modules["hydra"].compose = lambda *a, **k: {}
sys.modules["hydra"].main = lambda *a, **k: (lambda f: f)

sys.modules["datasets.caption.field"].TextField = object
sys.modules["datasets.caption.coco"].build_coco_dataloaders = lambda *a, **k: None
sys.modules["datasets.caption.transforms"].get_transform = lambda cfg: {"valid": lambda x: x}

sys.modules["models.caption"].Transformer = object
sys.modules["models.caption"].GridFeatureNetwork = object
sys.modules["models.caption"].CaptionGenerator = object
sys.modules["models.caption.detector"].build_detector = lambda cfg: None
sys.modules["models.common.attention"].MemoryAttention = object

sys.modules["engine.utils"].nested_tensor_from_tensor_list = lambda x: x
sys.modules["torch.nn.parallel"].DistributedDataParallel = object

import inference_caption


def test_generate_caption_with_mocked_config():
    mock_cfg = {"dummy": "config"}
    with mock.patch.object(inference_caption, "initialize", return_value=nullcontext()):
        with mock.patch.object(inference_caption, "compose", return_value=mock_cfg):
            with mock.patch.object(inference_caption, "_inference_from_config", return_value="caption") as inf_mock:
                result = inference_caption.generate_caption("img.jpg")
                assert result == "caption"
                inf_mock.assert_called_once_with(mock_cfg)

