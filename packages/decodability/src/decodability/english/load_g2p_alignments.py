"""Load grapheme-to-phoneme alignments for the CMU pronouncing dictionary.

The ``g2p.json`` file is not committed to the repo. It is downloaded
once from ``_G2P_URL`` on first import and cached in ``DATA_DIR`` for
subsequent runs.
"""

import json
from urllib.request import urlopen

from decodability.english.paths import DATA_DIR

_G2P_URL = "https://raw.githubusercontent.com/ckw017/aligned-cmudict/master/g2p.json"
_G2P_PATH = DATA_DIR / "g2p.json"


def load_g2p_alignments() -> dict:
    """Return the CMU G2P alignment dict, downloading and caching if missing."""
    if not _G2P_PATH.exists():
        _G2P_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(_G2P_URL) as response, open(_G2P_PATH, "wb") as f:
            f.write(response.read())
    with open(_G2P_PATH) as f:
        return json.load(f)


alignments = load_g2p_alignments()
