"""
Regressao: o pack precisa registrar seus nodes quando carregado como o ComfyUI carrega.

O ComfyUI executa `__init__.py` com `spec_from_file_location(<pasta>, ...,
submodule_search_locations=[<pasta>])` e a raiz do ComfyUI — que contem um
`nodes.py` proprio, o registro core — esta no `sys.path`. Um import absoluto
`from nodes import NODE_CLASS_MAPPINGS` resolve para o core: o pack carrega sem
erro e nao registra nada. Este teste falha se alguem reintroduzir isso.

Nodes esperados (v0.2.0):
  - GeminiImageInteractionsNode
  - AwGptImageNode
  - AwKlingVideoNode
"""

import importlib.util
import os
import sys
import types
import unittest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Nodes que o pack DEVE registrar (todos os tres)
_EXPECTED_NODES = {
    "GeminiImageInteractionsNode",
    "AwGptImageNode",
    "AwKlingVideoNode",
}

# Nodes do core do ComfyUI que NAO devem vazar para os mappings do pack
_CORE_NODES = {"LoadImage", "SaveImage"}


class ComfyUiEntrypointTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("nodes",)}
        core = types.ModuleType("nodes")
        core.NODE_CLASS_MAPPINGS = {n: object for n in _CORE_NODES}
        core.NODE_DISPLAY_NAME_MAPPINGS = {"LoadImage": "Load Image"}
        sys.modules["nodes"] = core

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        # Remove modulos do pack para evitar cache entre testes
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("aw-comfyui-nodes", "aw_comfyui_nodes",
                            "aw_gpt_image", "aw_kling_video",
                            "gemini_image_interactions"):
                sys.modules.pop(mod_name, None)

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

    def test_registers_all_three_nodes(self):
        """Os tres nodes do pack devem aparecer em NODE_CLASS_MAPPINGS."""
        module = self._load_as_comfyui()
        for node_name in _EXPECTED_NODES:
            self.assertIn(
                node_name, module.NODE_CLASS_MAPPINGS,
                f"{node_name} deve estar registrado em NODE_CLASS_MAPPINGS",
            )

    def test_does_not_leak_comfyui_core_nodes(self):
        """Nodes do core do ComfyUI nao devem aparecer nos mappings do pack."""
        module = self._load_as_comfyui()
        for core_node in _CORE_NODES:
            self.assertNotIn(
                core_node, module.NODE_CLASS_MAPPINGS,
                f"Node do core '{core_node}' nao deve vazar para o pack",
            )

    def test_display_names_for_all_three_nodes(self):
        """Os tres nodes devem ter display names registrados."""
        module = self._load_as_comfyui()
        for node_name in _EXPECTED_NODES:
            self.assertIn(
                node_name, module.NODE_DISPLAY_NAME_MAPPINGS,
                f"{node_name} deve ter display name registrado",
            )

    # Mantidos por compatibilidade com o nome da suite anterior
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
