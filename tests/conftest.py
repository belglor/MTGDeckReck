"""Test-wide guards.

Chroma's product telemetry posts to a third party by default. The suite must
stay offline, so switch it off in the environment before anything imports
chromadb — `store.open_client` sets the same flag on the clients it builds, but
a test constructing its own client should not have to remember.
"""

from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
