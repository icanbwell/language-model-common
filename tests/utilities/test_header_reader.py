from typing import Mapping

from languagemodelcommon.utilities.header_reader.header_reader import HeaderReader


def test_get_client_id_from_headers_prefers_bwell_header() -> None:
    headers: Mapping[str, str] = {
        "bwell-managing-organization": "client-primary",
        "x-client-id": "client-secondary",
        "client-id": "client-tertiary",
    }

    assert HeaderReader.get_client_id_from_headers(headers=headers) == "client-primary"


def test_get_client_id_from_headers_prefers_x_client_id_when_bwell_missing() -> None:
    headers: Mapping[str, str] = {
        "X-Client-Id": "client-secondary",
        "Client-Id": "client-tertiary",
    }

    assert (
        HeaderReader.get_client_id_from_headers(headers=headers) == "client-secondary"
    )


def test_get_client_id_from_headers_falls_back_to_client_id() -> None:
    headers: Mapping[str, str] = {"Client-Id": "client-tertiary"}

    assert HeaderReader.get_client_id_from_headers(headers=headers) == "client-tertiary"


def test_get_client_id_from_headers_returns_none_when_missing() -> None:
    assert HeaderReader.get_client_id_from_headers(headers={}) is None


def test_get_client_id_from_headers_is_case_insensitive() -> None:
    headers: Mapping[str, str] = {
        "BWell-Managing-Organization": "client-primary",
    }

    assert HeaderReader.get_client_id_from_headers(headers=headers) == "client-primary"


def test_has_debug_mode_enabled_accepts_truthy_values() -> None:
    assert HeaderReader.has_debug_mode_enabled(headers={"Debug-Mode": "true"})
    assert HeaderReader.has_debug_mode_enabled(headers={"Debug-Mode": "1"})


def test_has_debug_mode_enabled_rejects_non_truthy_values() -> None:
    assert not HeaderReader.has_debug_mode_enabled(headers={"Debug-Mode": "false"})
    assert not HeaderReader.has_debug_mode_enabled(headers={"Debug-Mode": "0"})
    assert not HeaderReader.has_debug_mode_enabled(headers={"Debug-Mode": ""})
    assert not HeaderReader.has_debug_mode_enabled(headers={})
