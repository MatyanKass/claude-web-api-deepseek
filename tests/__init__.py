"""Test package setup.

The runtime opens its telemetry database and loads the control config when
the package is imported, so the redirect to temporary files has to happen
before any test module imports claude_web_api. unittest imports this package
first, which is the only place that ordering is guaranteed.

The data root is redirected too: otherwise the suite reads the developer's
real ``control_config.json`` and its results depend on which provider and
profile happen to be active on that machine.
"""

import os
import tempfile
from pathlib import Path

_TEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="claude-web-api-tests-")
os.environ["CLAUDE_WEB_API_DATA"] = str(Path(_TEST_DIRECTORY.name) / "data")
os.environ["OPENCLAUDE_TELEMETRY_DB"] = str(
    Path(_TEST_DIRECTORY.name) / "telemetry.sqlite3"
)
