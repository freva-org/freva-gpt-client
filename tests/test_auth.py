from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock
from pytest_mock import MockerFixture

import climateclaw_client._auth as auth

# =============================================================================
# Fixtures
# =============================================================================


def make_oidc_token(
    expires: datetime,
    refresh_expires: datetime,
) -> dict[str, Any]:
    """Create a mock OIDC token dictionary with the given expiration times."""
    return {
        "access_token": "XYZ",
        "token_type": "Bearer",
        "expires": expires.timestamp(),
        "refresh_token": "ABC",
        "refresh_expires": refresh_expires.timestamp(),
        "scope": "openid",
        "headers": {
            "Authorization": "Bearer XYZ",
        },
    }


@pytest.fixture
def base_url() -> httpx.URL:
    """Base URL used for all auth tests."""
    return httpx.URL("https://myinstance.com")


@pytest.fixture
def default_store_path() -> str:
    """Default path for token store."""
    return "/default/path/token.json"


@pytest.fixture
def prepare_identity(base_url) -> auth.AuthIdentity:
    return auth.AuthIdentity(
        host=str(base_url),
        backend="oidc",
        grant=auth.Grant.DEVICE_CODE,
        scopes=["openid"],
    )


@pytest.fixture
def mock_token_store(mocker, default_store_path):
    """Mock TokenStore class with sensible defaults."""
    mock_cls = mocker.patch.object(auth, "TokenStore", spec=True)
    mock_instance = mock_cls.return_value
    mock_instance.path = default_store_path
    return mock_cls, mock_instance


@pytest.fixture(
    params=[(True, 1), (False, 1), (True, 0), (True, -2)],
    ids=[
        "with_valid_token",
        "without_valid_token",
        "with_expired_access_valid_refresh",
        "with_expired_both",
    ],
)
def token_store_entry(request, prepare_identity):
    """
    Fixture providing different token store states.

    Returns a dict mapping base_url to token data, or empty dict if no token.
    The parametrization covers:
    - with_valid_token: valid access and refresh tokens
    - without_valid_token: no token in store
    - with_expired_access_valid_refresh: access token expired, refresh still valid
    - with_expired_both: both tokens expired
    """
    has_token, expiry_factor = request.param
    if not has_token:
        return None

    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=1) * (expiry_factor - 0.5)
    refresh_expires = now + timedelta(hours=1) * (expiry_factor + 1)

    return auth.StoreEntry.from_dict(
        {
            "identity": prepare_identity.to_dict(),
            "token": make_oidc_token(expires, refresh_expires),
            "stored_at": (now - timedelta(hours=4)).timestamp(),
            "parent": None,
        }
    )


@pytest.fixture(
    params=[[True, None], [True, "/path/to/token.json"], [False, None]],
    ids=["no_store_path", "custom_store_path", "no_initial_token"],
)
def token_auth_instance(request, base_url, mock_token_store, token_store_entry):
    """
    Fixture providing a TokenAuth instance with mocked dependencies.

    Parametrization covers:
    - no_store_path: no custom token store path, uses default
    - custom_store_path: explicit token store path
    - no_initial_token: no token pre-set
    """
    token_set, token_store_path = request.param
    _, mock_instance = mock_token_store

    if token_store_path:
        mock_instance.path = token_store_path

    # Configure mock to return token data based on token_store_entry
    mock_instance.get.return_value = token_store_entry.token if token_store_entry else None
    mock_instance.get_entry.return_value = token_store_entry if token_store_entry else None
    mock_instance.put.return_value = token_store_entry

    token_auth = auth.TokenAuth(
        base_url=base_url,
        store_path=token_store_path,
        timeout=10,
        app_name="auth-test",
        backend="oidc",
    )
    token_auth.store_entry = token_store_entry
    if not token_set:
        token_auth.token = None

    return token_auth


# =============================================================================
# Tests for __init__
# =============================================================================


def test_token_auth_init(token_auth_instance, mock_token_store):
    """Test __init__ method sets up TokenAuth correctly with various configurations."""
    mock_cls, mock_instance = mock_token_store

    # Check timeout and app_name are always set correctly
    assert token_auth_instance.timeout == 10
    assert token_auth_instance.app_name == "auth-test"

    # TokenStore should be instantiated with correct app_name
    assert all(call[1]["app_name"] == "auth-test" for call in mock_cls.call_args_list)

    # Check token_store_path is set correctly on the instance
    assert token_auth_instance.store_path == mock_instance.path


# =============================================================================
# Tests for _authenticate and _async_authenticate
# =============================================================================


def test_authenticate_calls_oidc_client(token_auth_instance, mock_token_store, mocker, base_url):
    """Test _authenticate calls the py_oidc_auth_client.authenticate function correctly."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth, "authenticate")

    token_auth_instance._authenticate()

    mocked_authenticate.assert_called_once_with(
        host=f"{base_url}/api/freva-nextgen",
        store=mock_instance,
        app_name="auth-test",
        timeout=10,
    )


@pytest.mark.asyncio
async def test_async_authenticate_calls_oidc_client(
    mocker: MockerFixture, token_auth_instance, mock_token_store, base_url
):
    """Test _async_authenticate calls the py_oidc_auth_client.authenticate function correctly."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth, "authenticate_async")
    await token_auth_instance._async_authenticate()

    mocked_authenticate.assert_called_once_with(
        host=f"{base_url}/api/freva-nextgen",
        store=mock_instance,
        app_name="auth-test",
        timeout=10,
    )


# =============================================================================
# Tests for Token and TokenStore validation (sync and async)
# =============================================================================


def test_update_token_or_store(token_auth_instance, mock_token_store, prepare_identity):
    """Test _update_token_or_store syncs token with token store correctly."""
    _, mock_instance = mock_token_store
    identity = prepare_identity

    # Start with no auth token
    token_auth_instance.token = None
    token_auth_instance._update_token_or_store()

    # Check behavior based on whether store has a token
    stored_token = mock_instance.get(identity)

    if stored_token:
        # If store has token and token was None, token should be updated from store
        assert token_auth_instance.token == stored_token
    else:
        # If store has no token, token should be written to store
        mock_instance.put.assert_called_once_with(identity=identity, token=None)


def test_validate_token_store(token_auth_instance, mock_token_store, mocker, prepare_identity):
    """Test _validate_token_store loads token from store or triggers authentication."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth.TokenAuth, "_authenticate")
    mocked_update = mocker.patch.object(token_auth_instance, "_update_token_or_store")
    identity = prepare_identity

    # Get the token from store (if any)
    store_entry = mock_instance.get_entry(identity)

    # token_auth_instance starts with token=None (from fixture)
    token_auth_instance._validate_token_store()

    # Check the correct branch was taken based on stored token state
    if store_entry:
        # Token in store, no token initially -> should load from store
        assert token_auth_instance.token == store_entry.token
        mocked_authenticate.assert_not_called()
    else:
        # No token in store, no auth_token initially -> should call _authenticate
        mocked_authenticate.assert_called_once()

    # _update_token_or_store should always be called
    mocked_update.assert_called_once()


def test_validate_token_no_token_non_interactive_raises(
    token_auth_instance, mock_token_store, base_url
):
    token_auth_instance._interactive = False
    _, mock_instance = mock_token_store
    stored_token = mock_instance.get(str(base_url))
    if not (stored_token or token_auth_instance.token):
        with pytest.raises(
            auth.AuthError,
            match=r"Token store does not contain token for.*. New token can only be generated in interactive mode.",
        ):
            token_auth_instance._validate_token_store()


@pytest.mark.asyncio
async def test_async_validate_token_store(
    mocker: MockerFixture, token_auth_instance, mock_token_store, prepare_identity
):
    """Test _async_validate_token_store loads token from store or triggers authentication."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(
        auth.TokenAuth, "_async_authenticate", new_callable=mocker.AsyncMock
    )
    mocked_update = mocker.patch.object(token_auth_instance, "_update_token_or_store")
    identity = prepare_identity

    # Get the token from store (if any)
    store_entry = mock_instance.get_entry(identity)

    # token_auth_instance starts with token=None (from fixture)
    await token_auth_instance._async_validate_token_store()

    # Check the correct branch was taken based on stored token state
    if store_entry:
        # Token in store, no token initially -> should load from store
        assert token_auth_instance.token == store_entry.token
        mocked_authenticate.assert_not_called()
    else:
        # No token in store, no token initially -> should call _authenticate
        mocked_authenticate.assert_called_once()

    # _update_token_or_store should always be called
    mocked_update.assert_called_once()


@pytest.mark.asyncio
async def test_async_validate_token_no_token_non_interactive_raises(
    token_auth_instance, mock_token_store, base_url
):
    token_auth_instance._interactive = False
    _, mock_instance = mock_token_store
    stored_token = mock_instance.get(str(base_url))
    if not (stored_token or token_auth_instance.token):
        with pytest.raises(
            auth.AuthError,
            match=r"Token store does not contain token for.*. New token can only be generated in interactive mode.",
        ):
            await token_auth_instance._async_validate_token_store()


def test_validate_token_refreshes_expired_tokens(
    token_auth_instance, mock_token_store, mocker, token_store_entry
):
    """Test _validate_token refreshes tokens when expired and raises on invalid refresh tokens."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth.TokenAuth, "_authenticate")
    spy_logger = mocker.spy(auth, "logger")

    # Create a fresh token for _authenticate to return
    now = datetime.now(timezone.utc)
    fresh_token = make_oidc_token(
        expires=now + timedelta(days=1),
        refresh_expires=now + timedelta(days=2),
    )
    mocked_authenticate.return_value = fresh_token

    # Determine which token scenario we're testing
    stored_token_data = token_store_entry.token if token_store_entry else None

    if stored_token_data:
        token = stored_token_data
    else:
        token = fresh_token

    refresh_expires_dt = datetime.fromtimestamp(token["refresh_expires"], tz=timezone.utc)
    expires_dt = datetime.fromtimestamp(token["expires"], tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)

    if refresh_expires_dt < now_dt:
        # Both token and refresh token expired -> Should prompt user to authenticate
        token_auth_instance._validate_token()
        spy_logger.debug_assert_called_once_with(
            "Both auth and refresh token expired. Prompting user to log in to generate new token."
        )
        mocked_authenticate.assert_called_once()
    elif expires_dt < now_dt:
        # Access token expired but refresh valid -> should refresh
        token_auth_instance._validate_token()
        spy_logger.debug.assert_called_once_with(
            "Freva auth token expired. Using refresh token to generate new token and updating token store."
        )
        mocked_authenticate.assert_called_once()
        # Original token validation + new token storage = 2 put calls
        assert mock_instance.put.call_count == 2
    else:
        # Both tokens valid -> should return stored token
        returned_token = token_auth_instance._validate_token()
        assert returned_token == token


@pytest.mark.asyncio
async def test_async_validate_token_refreshes_expired_tokens(
    mocker: MockerFixture, token_auth_instance, mock_token_store, token_store_entry
):
    """Test _validate_token refreshes tokens when expired and raises on invalid refresh tokens."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth.TokenAuth, "_async_authenticate")
    spy_logger = mocker.spy(auth, "logger")

    # Create a fresh token for _authenticate to return
    now = datetime.now(timezone.utc)
    fresh_token = make_oidc_token(
        expires=now + timedelta(days=1),
        refresh_expires=now + timedelta(days=2),
    )
    mocked_authenticate.return_value = fresh_token

    # Determine which token scenario we're testing
    stored_token_data = token_store_entry.token if token_store_entry else None

    if stored_token_data:
        token = stored_token_data
    else:
        token = fresh_token

    refresh_expires_dt = datetime.fromtimestamp(token["refresh_expires"], tz=timezone.utc)
    expires_dt = datetime.fromtimestamp(token["expires"], tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)

    if refresh_expires_dt < now_dt:
        # Both token and refresh token expired -> Should prompt user to authenticate
        await token_auth_instance._async_validate_token()
        spy_logger.debug_assert_called_once_with(
            "Both auth and refresh token expired. Prompting user to log in to generate new token."
        )
        mocked_authenticate.assert_called_once()
    elif expires_dt < now_dt:
        # Access token expired but refresh valid -> should refresh
        await token_auth_instance._async_validate_token()
        spy_logger.debug.assert_called_once_with(
            "Freva auth token expired. Using refresh token to generate new token and updating token store."
        )
        mocked_authenticate.assert_called_once()
        # Original token validation + new token storage = 2 put calls
        assert mock_instance.put.call_count == 2
    else:
        # Both tokens valid -> should return stored token
        returned_token = await token_auth_instance._async_validate_token()
        assert returned_token == token


@pytest.mark.parametrize(argnames="interactive", argvalues=[True, False])
def test_validate_token_raises_on_auth_failure(
    token_auth_instance, mock_token_store, mocker, base_url, token_store_entry, interactive
):
    """Test _validate_token raises AuthError when token refresh fails."""
    _, _ = mock_token_store
    token_auth_instance._interactive = interactive
    mocked_authenticate = mocker.patch.object(auth, "authenticate")

    now = datetime.now(timezone.utc)
    fresh_token = make_oidc_token(
        expires=now + timedelta(days=1),
        refresh_expires=now + timedelta(days=2),
    )
    mocked_authenticate.return_value = fresh_token

    stored_token_data = token_store_entry.token if token_store_entry else None
    token = stored_token_data or fresh_token

    refresh_expires_dt = datetime.fromtimestamp(token["refresh_expires"], tz=timezone.utc)
    expires_dt = datetime.fromtimestamp(token["expires"], tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)
    if token_auth_instance._interactive:
        mocked_authenticate.side_effect = Exception("Auth failed")
    if (refresh_expires_dt < now_dt or expires_dt < now_dt) and token_auth_instance._interactive:
        with pytest.raises(auth.AuthError, match=r"Could not generate a new token.*"):
            token_auth_instance._validate_token()
    elif refresh_expires_dt < now_dt and not token_auth_instance._interactive:
        with pytest.raises(
            auth.AuthError,
            match="Refresh token has expired. New one can only be generated in interactive mode.",
        ):
            token_auth_instance._validate_token()


@pytest.mark.asyncio
@pytest.mark.parametrize(argnames="interactive", argvalues=[True, False])
async def test_async_validate_token_raises_on_auth_failure(
    token_auth_instance, mock_token_store, mocker, token_store_entry, interactive
):
    """Test _async_validate_token raises AuthError when token refresh fails."""
    _, _ = mock_token_store
    token_auth_instance._interactive = interactive
    mocked_authenticate = mocker.patch.object(auth, "authenticate_async")

    now = datetime.now(timezone.utc)
    fresh_token = make_oidc_token(
        expires=now + timedelta(days=1),
        refresh_expires=now + timedelta(days=2),
    )
    mocked_authenticate.return_value = fresh_token

    stored_token_data = token_store_entry.token if token_store_entry else None
    token = stored_token_data or fresh_token

    refresh_expires_dt = datetime.fromtimestamp(token["refresh_expires"], tz=timezone.utc)
    expires_dt = datetime.fromtimestamp(token["expires"], tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)

    if (refresh_expires_dt < now_dt or expires_dt < now_dt) and token_auth_instance._interactive:
        mocked_authenticate.side_effect = Exception("Auth failed")
        with pytest.raises(auth.AuthError, match=r"Could not generate a new token.*"):
            await token_auth_instance._async_validate_token()
    elif refresh_expires_dt < now_dt and not token_auth_instance._interactive:
        with pytest.raises(
            auth.AuthError,
            match="Refresh token has expired. New one can only be generated in interactive mode.",
        ):
            await token_auth_instance._async_validate_token()


# =============================================================================
# Tests for Auth Header retrieval (sync and async)
# =============================================================================


def test_get_auth_headers(
    token_auth_instance, mock_token_store, mocker, base_url, token_store_entry
):
    """Test get_auth_headers returns correct headers and handles token refresh."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth.TokenAuth, "_authenticate")

    now = datetime.now(timezone.utc)
    fresh_token = make_oidc_token(
        expires=now + timedelta(days=1),
        refresh_expires=now + timedelta(days=2),
    )
    mocked_authenticate.return_value = fresh_token

    stored_token_data = token_store_entry.token if token_store_entry else None
    token = stored_token_data or fresh_token

    refresh_expires_dt = datetime.fromtimestamp(token["refresh_expires"], tz=timezone.utc)
    expires_dt = datetime.fromtimestamp(token["expires"], tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)

    if refresh_expires_dt < now_dt:
        # Refresh token expired -> should prompt user to reauthenticate
        token_auth_instance.get_auth_headers()
        mocked_authenticate.assert_called_once()
    elif expires_dt < now_dt:
        # Token expired -> should refresh and return headers
        auth_headers = token_auth_instance.get_auth_headers()
        mocked_authenticate.assert_called_once()
        assert mock_instance.put.call_count == 2
        assert "Authorization" in auth_headers
        assert "Bearer" in auth_headers["Authorization"]
        assert token["access_token"] in auth_headers["Authorization"]
    else:
        # Valid token -> should return headers directly
        auth_headers = token_auth_instance.get_auth_headers()
        assert "Authorization" in auth_headers
        assert "Bearer" in auth_headers["Authorization"]
        assert token["access_token"] in auth_headers["Authorization"]


@pytest.mark.asyncio
async def test_async_get_auth_headers(
    mocker: MockerFixture, token_auth_instance, mock_token_store, base_url, token_store_entry
):
    """Test get_auth_headers returns correct headers and handles token refresh."""
    _, mock_instance = mock_token_store
    mocked_authenticate = mocker.patch.object(auth.TokenAuth, "_async_authenticate")

    now = datetime.now(timezone.utc)
    fresh_token = make_oidc_token(
        expires=now + timedelta(days=1),
        refresh_expires=now + timedelta(days=2),
    )
    mocked_authenticate.return_value = fresh_token

    stored_token_data = token_store_entry.token if token_store_entry else None
    token = stored_token_data or fresh_token

    refresh_expires_dt = datetime.fromtimestamp(token["refresh_expires"], tz=timezone.utc)
    expires_dt = datetime.fromtimestamp(token["expires"], tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)

    if refresh_expires_dt < now_dt:
        # Refresh token expired -> should prompt user to reauthenticate
        await token_auth_instance.async_get_auth_headers()
        mocked_authenticate.assert_called_once()
    elif expires_dt < now_dt:
        # Token expired -> should refresh and return headers
        auth_headers = await token_auth_instance.async_get_auth_headers()
        mocked_authenticate.assert_called_once()
        assert mock_instance.put.call_count == 2
        assert "Authorization" in auth_headers
        assert "Bearer" in auth_headers["Authorization"]
        assert token["access_token"] in auth_headers["Authorization"]
    else:
        # Valid token -> should return headers directly
        auth_headers = await token_auth_instance.async_get_auth_headers()
        assert "Authorization" in auth_headers
        assert "Bearer" in auth_headers["Authorization"]
        assert token["access_token"] in auth_headers["Authorization"]


# =============================================================================
# Tests for Auth Flow (sync and async)
# =============================================================================


def test_sync_auth_flow_passes_through_non_401(mocker, httpx_mock: HTTPXMock):
    """Test sync_auth_flow does not add auth headers for non-401 responses."""
    mocked_get_auth_headers = mocker.patch.object(auth.TokenAuth, "get_auth_headers")
    httpx_mock.add_response(status_code=200)

    with httpx.Client(
        base_url="https://myinstance.com",
        auth=auth.TokenAuth(
            base_url="https://myinstance.com",
            store_path=None,
            timeout=10,
            app_name="auth-test",
        ),
    ) as client:
        response = client.get("/test")
        response.raise_for_status()
        mocked_get_auth_headers.assert_not_called()
        assert "authorization" not in response.request.headers


@pytest.mark.asyncio
async def test_async_auth_flow_passes_through_non_401(mocker, httpx_mock: HTTPXMock):
    """Test async_auth_flow does not add auth headers for non-401 responses."""
    mocked_get_auth_headers = mocker.patch.object(auth.TokenAuth, "async_get_auth_headers")
    httpx_mock.add_response(status_code=200)

    async with httpx.AsyncClient(
        base_url="https://myinstance.com",
        auth=auth.TokenAuth(
            base_url="https://myinstance.com",
            store_path=None,
            timeout=10,
            app_name="auth-test",
        ),
    ) as client:
        response = await client.get("/test")
        response.raise_for_status()
        mocked_get_auth_headers.assert_not_called()
        assert "authorization" not in response.request.headers


def test_sync_auth_flow_retry_on_401(mocker, httpx_mock: HTTPXMock):
    """Test sync_auth_flow adds auth headers and retries on 401 responses."""
    mocked_get_auth_headers = mocker.patch.object(auth.TokenAuth, "get_auth_headers")
    mocked_get_auth_headers.return_value = {"Authorization": "Bearer XYZ"}
    httpx_mock.add_response(status_code=401)
    httpx_mock.add_response(status_code=200)

    with httpx.Client(
        base_url="https://myinstance.com",
        auth=auth.TokenAuth(
            base_url="https://myinstance.com",
            store_path=None,
            timeout=10,
            app_name="auth-test",
        ),
    ) as client:
        response = client.get("/test")
        response.raise_for_status()
        mocked_get_auth_headers.assert_called_once()
        assert "authorization" in response.request.headers
        assert "Bearer XYZ" in response.request.headers["authorization"]


@pytest.mark.asyncio
async def test_async_auth_flow_retry_on_401(mocker, httpx_mock: HTTPXMock):
    """Test async_auth_flow adds auth headers and retries on 401 responses."""
    mocked_get_auth_headers = mocker.patch.object(auth.TokenAuth, "async_get_auth_headers")
    mocked_get_auth_headers.return_value = {"Authorization": "Bearer XYZ"}
    httpx_mock.add_response(status_code=401)
    httpx_mock.add_response(status_code=200)

    async with httpx.AsyncClient(
        base_url="https://myinstance.com",
        auth=auth.TokenAuth(
            base_url="https://myinstance.com",
            store_path=None,
            timeout=10,
            app_name="auth-test",
        ),
    ) as client:
        response = await client.get("/test")
        response.raise_for_status()
        mocked_get_auth_headers.assert_called_once()
        assert "authorization" in response.request.headers
        assert "Bearer XYZ" in response.request.headers["authorization"]
