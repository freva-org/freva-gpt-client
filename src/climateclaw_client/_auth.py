import logging
from pathlib import Path
from typing import Dict

import httpx
from py_oidc_auth_client import (  # type: ignore
    AuthError,
    AuthIdentity,
    DeviceCode,
    DeviceFlow,
    Grant,
    StoreEntry,
    Token,
    TokenStore,
    authenticate,
    authenticate_async,
)

from ._constants import DEFAULT_AUTH_TIMEOUT

logger = logging.getLogger(__name__)


class TokenAuth(httpx.Auth):
    """Authentication handler for ClimateClaw API using OIDC tokens.

    This class manages token authentication for HTTP requests, including
    token storage, validation, and automatic refresh when tokens expire.

    Attributes:
        base_url: The base URL of the ClimateClaw API.
        app_name: The application name used for token storage.
        timeout: Timeout in seconds for authentication requests.
        store_path: Path to the token store file.
        store: TokenStore instance for managing tokens.
        token: Current authentication token.
    """

    def __init__(
        self,
        base_url: httpx.URL,
        store_path: str | Path | None = None,
        timeout: float = DEFAULT_AUTH_TIMEOUT,
        app_name: str = "climateclaw-client",
        backend: str = "oidc",
        interactive: bool = True,
    ):
        """Initializes TokenAuth with base URL and token store configuration.

        Args:
            base_url: The base URL of the ClimateClaw API.
            token_store_path: Optional path to the token store file.
            timeout: Timeout in seconds for authentication requests.
            app_name: Application name for token storage identification.
            backend: Which provider backend drives the flow, ``"py-oidc-auth"`` or ``"oidc"``.
            interactive: Boolean to determine if authentication can be performed interactively (prompting the user to log in if necessary).
        """
        self.base_url: httpx.URL = base_url
        self.app_name: str = app_name
        self.timeout: float = timeout
        self.identity: AuthIdentity = AuthIdentity(
            host=f"{self.base_url}/api/freva-nextgen",
            backend=backend,
            grant=Grant.DEVICE_CODE,
            scopes=["openid"],
        )
        self.store: TokenStore = TokenStore(app_name=app_name, path=store_path)
        self.store_path: str | Path = (
            store_path or TokenStore(app_name=app_name, path=store_path).path
        )
        self.store_entry: StoreEntry | None = self.store.get_entry(self.identity)
        self.token: Token | None = self.store.get(self.identity)
        self._interactive = interactive

    def _authenticate(self) -> Token:
        """Authenticates with the OIDC provider and returns a new token."""
        try:
            return authenticate(
                host=f"{self.base_url}/api/freva-nextgen",
                store=self.store,
                app_name=self.app_name,
                timeout=self.timeout,
            )
        except Exception as e:
            raise AuthError(f"Could not generate a new token. Error: {e}")

    def _update_token_or_store(self) -> None:
        """Updates the token store with the current auth token."""
        store_entry = self.store.get_entry(self.identity)
        if store_entry and not self.token:
            self.store_entry = store_entry
            self.token = store_entry.token
        else:
            self.store_entry = self.store.put(identity=self.identity, token=self.token)

    def _validate_token_store(self) -> TokenStore:
        """Validates and initializes the token store."""
        # load token store
        store_entry = self.store.get_entry(self.identity)
        if not self.token:
            # if auth token is not set, but token store contains correct token, update token from token store
            if store_entry:
                self.token = store_entry.token
                self.store_entry = store_entry
            # if not in interactive mode, and no auth token set, raise AuthError
            elif not self._interactive:
                raise AuthError(
                    "Token store does not contain token for current identity. New token can only be generated in interactive mode."
                ) from None
            # else start oidc device flow and prompt user to login
            else:
                self.token = self._authenticate()
                self.store_entry = self.store.put(identity=self.identity, token=self.token)
        self._update_token_or_store()
        return self.store

    def _validate_token(self) -> Token:
        """Validates the current authentication token."""
        self.store = self._validate_token_store()
        if self.store_entry:
            if self.store_entry.expired():
                if self._interactive:
                    logger.debug(
                        "Both auth and refresh token expired. Prompting user to log in to generate new token."
                    )
                    self.token = self._authenticate()
                    self.store_entry = self.store.put(identity=self.identity, token=self.token)
                else:
                    raise AuthError(
                        "Refresh token has expired. New one can only be generated in interactive mode."
                    ) from None
            elif self.store_entry.usefulness() == 1:
                logger.debug(
                    "Freva auth token expired. Using refresh token to generate new token and updating token store."
                )
                self.token = self._authenticate()
                self.store_entry = self.store.put(identity=self.identity, token=self.token)
        return self.token

    def get_auth_headers(self) -> Dict[str, str]:
        """Gets the authentication headers for HTTP requests.

        Returns:
            Dictionary containing authentication headers.
        """
        auth_token = self._validate_token()
        return auth_token.get("headers")

    def sync_auth_flow(self, request: httpx.Request):
        """HTTP authentication flow handler.

        This generator yields requests with authentication headers added
        when a 401 Unauthorized response is received.

        Args:
            request: The HTTP request to authenticate.

        Yields:
            httpx.Request: The authenticated request.
        """
        response: httpx.Response = yield request
        if response.status_code == 401:
            # If the server issues a 401 response then resend the request,
            # with custom authentication headers.
            request.headers.update(self.get_auth_headers())
            yield request

    async def _async_authenticate(self) -> Token | tuple[DeviceFlow, DeviceCode]:
        """Authenticates asynchronously with the OIDC provider and returns a new token."""
        try:
            return await authenticate_async(
                host=f"{self.base_url}/api/freva-nextgen",
                store=self.store,
                app_name=self.app_name,
                timeout=self.timeout,
            )
        except Exception as e:
            raise AuthError(f"Could not generate a new token. Error: {e}")

    async def _async_validate_token_store(self) -> TokenStore:
        """Validates and initializes the token store."""
        # load token store
        store_entry = self.store.get_entry(self.identity)
        if not self.token:
            # if auth token is not set, but token store contains correct token, update token from token store
            if store_entry:
                self.token = store_entry.token
                self.store_entry = self.store_entry
            # if not in interactive mode, and no auth token set, raise AuthError
            elif not self._interactive:
                raise AuthError(
                    "Token store does not contain token for current identity. New token can only be generated in interactive mode."
                ) from None
            # else start oidc device flow and prompt user to login
            else:
                self.token = await self._async_authenticate()
                self.store_entry = self.store.put(identity=self.identity, token=self.token)
        self._update_token_or_store()
        return self.store

    async def _async_validate_token(self) -> Token:
        """Validates the current authentication token."""
        self.store = await self._async_validate_token_store()
        if self.store_entry:
            if self.store_entry.expired():
                if self._interactive:
                    logger.debug(
                        "Both auth and refresh token expired. Prompting user to log in to generate new token."
                    )
                    self.token = await self._async_authenticate()
                    self.store_entry = self.store.put(identity=self.identity, token=self.token)
                else:
                    raise AuthError(
                        "Refresh token has expired. New one can only be generated in interactive mode."
                    ) from None
            elif self.store_entry.usefulness() == 1:
                logger.debug(
                    "Freva auth token expired. Using refresh token to generate new token and updating token store."
                )
                self.token = await self._async_authenticate()
                self.store_entry = self.store.put(identity=self.identity, token=self.token)
        return self.token

    async def async_get_auth_headers(self) -> Dict[str, str]:
        """Gets the authentication headers for HTTP requests.

        Returns:
            Dictionary containing authentication headers.
        """
        auth_token = await self._async_validate_token()
        return auth_token.get("headers")

    async def async_auth_flow(self, request: httpx.Request):
        """Async HTTP authentication flow handler.

        This generator yields requests with authentication headers added
        when a 401 Unauthorized response is received.

        Args:
            request: The HTTP request to authenticate.

        Yields:
            httpx.Request: The authenticated request.
        """
        response: httpx.Response = yield request
        if response.status_code == 401:
            # If the server issues a 401 response then resend the request,
            # with custom authentication headers.
            auth_headers = await self.async_get_auth_headers()
            request.headers.update(auth_headers)
            yield request
