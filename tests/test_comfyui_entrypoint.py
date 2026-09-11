"""
Regressao: o pack precisa registrar seus nodes quando carregado como o ComfyUI carrega.

O ComfyUI executa `__init__.py` com `spec_from_file_location(<pasta>, ..., 
submodule_search_locations=[<pasta>])` e a raiz do ComfyUI — que contem um
`nodes.py` proprio, o registro core — esta no `sys.path`. Um import absoluto
`from nodes import NODE_CLASS_MAPPINGS` resolve para o core: o pack carrega sem
erro e nao registra nada. Este teste falha se alguem reintroduzir isso.
"""

import importlib.util
import os
import sys
import types
import unittest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ComfyUiEntrypointTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("nodes",)}
        core = types.ModuleType("nodes")
        core.NODE_CLASS_MAPPINGS = {"LoadImage": object, "SaveImage": object}
        core.NODE_DISPLAY_NAME_MAPPINGS = {"LoadImage": "Load Image"}
        sys.modules["nodes"] = core

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        sys.modules.pop("aw-comfyui-nodes", None)

    def _load_as_comfyui(self):
        spec = importlib.util.spec_from_file_location(
            "aw-comfyui-nodes",
            os.path.join(PACK_DIR, "__init__.py"),
            submodule_search_locations=[PACK_DIR],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_registers_own_node_not_comfyui_core(self):
        module = self._load_as_comfyui()

        self.assertIn("GeminiImageInteractionsNode", module.NODE_CLASS_MAPPINGS)
        self.assertNotIn("LoadImage", module.NODE_CLASS_MAPPINGS)
        self.assertNotIn("SaveImage", module.NODE_CLASS_MAPPINGS)

    def test_display_name_mapping_is_ours(self):
        module = self._load_as_comfyui()

        self.assertIn("GeminiImageInteractionsNode", module.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertNotIn("LoadImage", module.NODE_DISPLAY_NAME_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
