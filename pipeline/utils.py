"""Utilities: PubChem lookups (language detection removed — language is provided by input)."""

import re
from typing import Optional, Dict, Any
from urllib.parse import quote

import requests


class PubChemClient:
    """PubChem REST client for name↔SMILES lookups."""

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def get_smiles_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        encoded_name = quote(name, safe="")
        url = (
            f"{self.BASE_URL}/compound/name/{encoded_name}/property/"
            "CanonicalSMILES,IsomericSMILES/JSON"
        )
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    p = props[0]
                    smiles = (
                        p.get("CanonicalSMILES")
                        or p.get("IsomericSMILES")
                        or p.get("SMILES")
                        or p.get("ConnectivitySMILES")
                    )
                    if smiles:
                        return {"smiles": smiles, "cid": p.get("CID")}
                return None
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                if attempt < self.max_retries - 1:
                    continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    continue
        return None

    def get_iupac_name_by_smiles(self, smiles: str) -> Optional[str]:
        if not smiles:
            return None
        try:
            encoded = quote(smiles, safe="")
            url = f"{self.BASE_URL}/compound/smiles/{encoded}/property/IUPACName/JSON"
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IUPACName")
        except Exception:
            pass
        return None
