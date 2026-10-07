# -*- Python -*-

# Configuration shared by the `test/` and `examples/` suites. Each suite's
# generated `lit.site.cfg.py` sets its name, roots and tool directories.

import lit.formats

from lit.llvm import llvm_config

config.test_format = lit.formats.ShTest()

# `.mlir` files run the interpreter, and `.test` files run a C++ test binary.
config.suffixes = [".mlir", ".test"]

llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    ["interpreter", "engine-lifetime-test", "eval-cache-test"],
    config.interpreter_tools_dirs,
)
