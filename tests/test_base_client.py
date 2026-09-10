import ipaddress
import socket

import httpx
import pytest
from pytest_httpx import HTTPXMock, IteratorStream
from pytest_mock import MockerFixture

import climateclaw_client._base_client
from climateclaw_client._base_client import AsyncAPIClient, BaseClient, SyncAPIClient  # noqa
from climateclaw_client._constants import DEFAULT_TIMEOUT
from climateclaw_client._streaming import StreamResponse

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_url():
    return "https://myinstance.com"


@pytest.fixture
def base_client_config(base_url):
    return {
        "version": "1.0.0",
        "base_url": base_url,
        "token_store_path": "/path/to/token/store.json",
        "follow_redirects": True,
        "max_retries": 5,
        "timeout": 0.5,
        "custom_headers": {},
        "auth_url": "http://test-auth:7777",
    }


@pytest.fixture
def make_base_client(mocker: MockerFixture, base_client_config):
    mock_validate_base_url = mocker.patch.object(BaseClient, "_validate_base_url")
    mock_validate_base_url.side_effect = lambda url: httpx.URL(url)
    return BaseClient(**base_client_config)


@pytest.fixture
def make_sync_api_client(mocker: MockerFixture, base_client_config):
    def prep_api_client(http_client: httpx.Client | None = None) -> SyncAPIClient:
        mocked_token_auth = mocker.patch.object(climateclaw_client._base_client, "TokenAuth")
        mocked_token_auth.return_value = None
        mock_validate_base_url = mocker.patch.object(SyncAPIClient, "_validate_base_url")
        mock_validate_base_url.side_effect = lambda url: httpx.URL(url)
        return SyncAPIClient(**base_client_config, http_client=http_client)

    return prep_api_client


@pytest.fixture
def make_async_api_client(mocker: MockerFixture, base_client_config):
    def prep_api_client(http_client: httpx.AsyncClient | None = None) -> AsyncAPIClient:
        mocked_token_auth = mocker.patch.object(climateclaw_client._base_client, "TokenAuth")
        mocked_token_auth.return_value = None
        mock_validate_base_url = mocker.patch.object(AsyncAPIClient, "_validate_base_url")
        mock_validate_base_url.side_effect = lambda url: httpx.URL(url)
        return AsyncAPIClient(**base_client_config, http_client=http_client)

    return prep_api_client


# =============================================================================
# BaseClient Tests
# =============================================================================


class TestBaseClient:
    """BaseClient Tests"""

    def test_init(self, mocker, base_client_config, make_base_client):
        """Test that __init__ method of BaseClient class initializes properties correctly."""
        base_client: BaseClient = make_base_client
        assert base_client.base_url == base_client_config["base_url"]
        assert base_client.follow_redirects == base_client_config["follow_redirects"]
        assert base_client.max_retries == base_client_config["max_retries"]
        assert base_client.timeout == base_client_config["timeout"]
        assert base_client.headers == base_client._build_headers(
            base_client_config["custom_headers"]
        )
        assert base_client._version == base_client_config["version"]
        assert base_client.auth_url == base_client_config["auth_url"]

    def test_auth(self, mocker: MockerFixture, make_base_client):
        """Test that _auth property can be accessed correctly."""
        base_client = make_base_client
        mocked_token_auth = mocker.patch.object(climateclaw_client._base_client, "TokenAuth")
        mocked_token_auth.return_value = (
            {}
        )  # mocked return value to check that _auth property returns the same value
        assert base_client._auth == {}
        assert base_client._auth == {}
        # because the auth token is cached, the underlying TokenAuth should only be called once.
        mocked_token_auth.assert_called_once_with(
            base_url=base_client.auth_url,
            store_path=base_client._token_store_path,
            interactive=True,
        )

    def test_default_headers(self, make_base_client):
        """Test that default_headers are set correctly for a given instance of the base client"""
        base_client = make_base_client
        default_headers = base_client.default_headers
        assert "content-type" in default_headers
        assert default_headers["content-type"] == "application/json"
        assert "accept" in default_headers
        assert default_headers["accept"] == "application/json"
        assert "user-agent" in default_headers
        assert "climate-claw-python" in default_headers["user-agent"]
        assert "x-freva-vault-url" in default_headers
        assert str(base_client.base_url) in default_headers["x-freva-vault-url"]
        assert "x-freva-rest-url" in default_headers
        assert str(base_client.base_url) in default_headers["x-freva-rest-url"]
        assert "x-freva-config-path" in default_headers

    def test_build_headers(self, make_base_client):
        base_client = make_base_client
        # assert that _build_headers simply returns the default headers (as an httpx.Headers object)
        headers = base_client._build_headers()
        assert headers == httpx.Headers(base_client.default_headers)
        # assert that _build_headers adds any custom headers to the default headers and that keys of headers are invariant to case
        custom_headers = {"Header1": "info", "header2": "token"}
        headers = base_client._build_headers(custom_headers)
        for key, value in custom_headers.items():
            assert key.lower() in headers  # key in headers should be case-invariant
            assert (
                value == headers[key]
            )  # accessing value from headers should also be case-invariant (for key)
        # make sure that default key-values are still present in headers
        for key, value in base_client.default_headers.items():
            assert key in headers
            assert headers[key] == value
        # assert that custom header can override key-value pairs present in default headers
        headers = base_client._build_headers()
        assert headers["accept"] == "application/json"
        custom_headers = {"accept": "Any"}
        headers = base_client._build_headers(custom_headers)
        assert headers["accept"] == custom_headers["accept"]

    @pytest.mark.parametrize(
        argnames="url",
        argvalues=[
            "https://myinstance.com",
            "http://myinstance.com",
            "https://myinstance.com/api",
            "http://127.0.0.1:5802",
            "https://[1080::8:800:200c:417a]:1234",
            "http://[::1]:5682/api",
            "ftp://myftpserver.com:21",
        ],
        ids=[
            "url with https",
            "url with http",
            "url with https and path",
            "ipv4 address with http and port",
            "ipv6 address with https and port",
            "ipv6 localhost address with http, port and path",
            "url with ftp protocol and port",
        ],
    )
    def test_validate_base_url_with_protocol(self, mocker: MockerFixture, url):
        """Test that addresses in the form of '{protocol}://{domain_name}' are parsed correctly"""
        mock_socket = mocker.patch.object(climateclaw_client._base_client, "socket", spec=True)
        validated_url = BaseClient._validate_base_url(url)
        assert validated_url == httpx.URL(url)
        try:
            ipaddress.ip_address(validated_url.host)
            mock_socket.gethostbyname.assert_not_called()
        except ValueError:
            mock_socket.gethostbyname.assert_called_once()

    @pytest.mark.parametrize(
        argnames="url",
        argvalues=[
            "myinstance.com",
            "myinstance.com/api",
            "127.0.0.1:5802",
            "[1080::8:800:200c:417a]:1234",
            "[::1]:5682/api",
        ],
        ids=[
            "url",
            "url with path",
            "ipv4 address with port",
            "ipv6 address with port",
            "ipv6 localhost address with port and path",
        ],
    )
    def test_validate_base_url_without_protocol(self, mocker: MockerFixture, url):
        """Test that addresses in the form of '{domain_name}' are parsed correctly."""
        mock_socket = mocker.patch.object(climateclaw_client._base_client, "socket", spec=True)
        validated_url = BaseClient._validate_base_url(url)
        assert validated_url == httpx.URL(f"http://{url}")
        try:
            ipaddress.ip_address(validated_url.host)
            mock_socket.gethostbyname.assert_not_called()
        except ValueError:
            mock_socket.gethostbyname.assert_called_once()

    def test_validate_base_url_ipv4_localhost_only_port(self, mocker: MockerFixture):
        """Test that addresses with just a port are parsed correctly as localhost+port."""
        url = ":8502"
        mock_socket = mocker.patch.object(climateclaw_client._base_client, "socket", spec=True)
        validated_url = BaseClient._validate_base_url(url)
        assert validated_url == httpx.URL(f"http://127.0.0.1{url}")
        mock_socket.gethostbyname.assert_not_called()

    def test_validate_base_url_name_resolution_exception(self, mocker: MockerFixture):
        """Test that domain name resolution error triggers correctly."""
        url = "https://missinginstance.com"
        mock_socket = mocker.patch.object(climateclaw_client._base_client.socket, "gethostbyname")
        mock_socket.side_effect = socket.gaierror
        with pytest.raises(ConnectionError, match="Temporary failure in name resolution of host"):
            BaseClient._validate_base_url(url)


# =============================================================================
# SyncAPIClient Tests
# =============================================================================


class TestSyncAPIClient:
    """Tests for SyncAPIClient"""

    def test_init(self, make_sync_api_client, base_client_config):
        """Test init method of SyncAPIClient"""
        api_client: SyncAPIClient = make_sync_api_client()
        assert api_client.base_url == base_client_config["base_url"]
        assert api_client.follow_redirects == base_client_config["follow_redirects"]
        assert api_client.max_retries == base_client_config["max_retries"]
        assert api_client.timeout == base_client_config["timeout"]
        assert api_client.headers == api_client._build_headers(base_client_config["custom_headers"])
        assert api_client._version == base_client_config["version"]
        assert api_client._client is not None

    def test_init_with_client(self, make_sync_api_client):
        """Test that custom http client is passed correctly during init."""
        test_client = httpx.Client(base_url="https://anotherinstance.com")
        api_client: SyncAPIClient = make_sync_api_client(test_client)
        assert api_client._client == test_client

    def test_init_invalid_http_client(self, make_sync_api_client):
        """Test that type of custom http client is enforced by init method."""
        not_a_client = "This is not a httpx.Client instance"
        with pytest.raises(TypeError, match="Invalid `http_client`"):
            make_sync_api_client(not_a_client)

    def test_default_client(self, mocker: MockerFixture, make_sync_api_client):
        """Test that default client is initialized correctly and cached."""
        spy_client = mocker.spy(climateclaw_client._base_client.httpx, "Client")
        api_client: SyncAPIClient = make_sync_api_client()
        default_client = api_client._default_client
        assert api_client._client == default_client
        api_client._default_client
        # as default client is cached, httpx.Client should only be called once
        spy_client.assert_called_once_with(
            base_url=api_client.base_url,
            follow_redirects=api_client.follow_redirects,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=api_client.timeout),
            headers=api_client.headers,
            auth=api_client._auth,
        )

    def test_is_closed(self, make_sync_api_client):
        """Test that SyncAPIClient is_closed attribute correctly inherits underlying httpx.Client closed state."""
        with httpx.Client() as client:
            api_client: SyncAPIClient = make_sync_api_client(client)
            assert not api_client.is_closed
        assert api_client.is_closed

    def test_sleep_for_retry_first_attempt(self, mocker: MockerFixture, make_sync_api_client):
        """Test that _sleep_for_retry takes the correct branch if the number of retries taken has not reached the maximum number."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_logger = mocker.spy(climateclaw_client._base_client, "logger")
        mocked_time = mocker.patch.object(climateclaw_client._base_client, "time", spec=True)

        retries_taken = 0
        api_client._sleep_for_retry(retries_taken)
        spy_logger.debug.assert_called_once()
        assert "Retrying connection after sleeping" in spy_logger.debug.call_args.args[0]
        mocked_time.sleep.assert_called_once()

    def test_sleep_for_retry_final_attempt(self, mocker: MockerFixture, make_sync_api_client):
        """Test that _sleep_for_retry takes the correct branch if the number of retries taken has reached the maximum number."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_logger = mocker.spy(climateclaw_client._base_client, "logger")
        mocked_time = mocker.patch.object(climateclaw_client._base_client, "time", spec=True)

        retries_taken = api_client.max_retries
        api_client._sleep_for_retry(retries_taken)
        spy_logger.debug.assert_called_once()
        assert "Retrying connection after sleeping" in spy_logger.debug.call_args.args[0]
        assert "Final attempt." in spy_logger.debug.call_args.args[0]
        mocked_time.sleep.assert_called_once()

    def test_sleep_for_retry_exceeded_maximum(self, mocker: MockerFixture, make_sync_api_client):
        """Test that _sleep_for_retry takes the correct branch if the number of retries taken has exceeded the maximum number."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_logger = mocker.spy(climateclaw_client._base_client, "logger")
        mocked_time = mocker.patch.object(climateclaw_client._base_client, "time", spec=True)

        retries_taken = api_client.max_retries + 1
        api_client._sleep_for_retry(retries_taken)
        spy_logger.debug.assert_called_once()
        assert (
            "Maximum number of retries exceeded. Skipping timeout."
            in spy_logger.debug.call_args.args[0]
        )
        mocked_time.sleep.assert_not_called()

    def test_stream_success(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _stream returns a StreamResponse object in case of a successful request and output of stream is as expected."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        stream_iterator = [b'{"a":1}', b'{"b":2}']
        expected_output = [{"a": 1}, {"b": 2}]
        httpx_mock.add_response(status_code=200, stream=IteratorStream(stream_iterator))
        stream_response = api_client._stream(method="GET", url="/api")
        assert isinstance(stream_response, StreamResponse)
        spy_build_request.assert_called_once_with(method="GET", url="/api")
        spy_send_request.assert_called_once()

        for response, expected in zip(stream_response.iter_json_objects(), expected_output):
            assert response == expected

    def test_stream_success_with_retry(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _stream handles temporary timeouts by sleeping and retrying."""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._sleep_for_retry = mocker.MagicMock()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        stream_iterator = [b'{"a":1}', b'{"b":2}']
        httpx_mock.add_exception(httpx.TimeoutException(message="Temporary time out"))
        httpx_mock.add_response(status_code=200, stream=IteratorStream(stream_iterator))
        stream_response = api_client._stream(method="GET", url="/api")
        assert isinstance(stream_response, StreamResponse)
        spy_build_request.assert_called_with(method="GET", url="/api")
        assert spy_build_request.call_count == 2
        assert spy_send_request.call_count == 2
        api_client._sleep_for_retry.assert_called_once()

    def test_stream_http_status_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _stream handles status errors differently from timeouts."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        httpx_mock.add_response(status_code=401)
        with pytest.raises(ConnectionError, match="Error connecting"):
            api_client._stream(method="GET", url="/api")
        spy_build_request.assert_called_once_with(method="GET", url="/api")
        spy_send_request.assert_called_once()

    def test_stream_http_general_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _stream handles http errors that are neither status nor timeouts differently."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        httpx_mock.add_exception(httpx.ProxyError("Proxy error."))
        with pytest.raises(httpx.ProxyError, match="Proxy error."):
            api_client._stream(method="GET", url="/api")
        spy_build_request.assert_called_once_with(method="GET", url="/api")
        spy_send_request.assert_called_once()

    def test_stream_timeout_error_after_retries(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _stream raises an error after encountering the maximum number of allowed timeouts."""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._sleep_for_retry = mocker.MagicMock()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        httpx_mock.add_exception(
            httpx.TimeoutException(message="Temporary time out"), is_reusable=True
        )
        with pytest.raises(ConnectionError, match="Failed to connect"):
            api_client._stream(method="GET", url="/api")
        spy_build_request.assert_called_with(method="GET", url="/api")
        assert spy_build_request.call_count == api_client.max_retries + 1
        assert spy_send_request.call_count == api_client.max_retries + 1
        assert api_client._sleep_for_retry.call_count == api_client.max_retries

    def test_request_raw_success(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _request_raw returns a Response object in case of a successful request and output is as expected."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_request = mocker.spy(api_client._client, "request")
        response_json = [{"a": 1}, {"b": 2}]
        httpx_mock.add_response(status_code=200, json=response_json)
        response = api_client._request_raw(method="GET", url="/api")
        assert isinstance(response, httpx.Response)
        spy_request.assert_called_once_with(method="GET", url="/api")
        assert response.json() == response_json

    def test_request_raw_success_with_retry(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _request_raw handles temporary timeouts by sleeping and retrying."""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._sleep_for_retry = mocker.MagicMock()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_exception(httpx.TimeoutException(message="Temporary time out"))
        httpx_mock.add_response(status_code=200)
        response = api_client._request_raw(method="GET", url="/api")
        assert isinstance(response, httpx.Response)
        spy_request.assert_called_with(method="GET", url="/api")
        assert spy_request.call_count == 2
        api_client._sleep_for_retry.assert_called_once()

    def test_request_raw_http_status_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _request_raw handles status errors differently from timeouts."""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._sleep_for_retry = mocker.MagicMock()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_response(status_code=401)
        with pytest.raises(ConnectionError, match="Error connecting"):
            api_client._request_raw(method="GET", url="/api")
        spy_request.assert_called_once_with(method="GET", url="/api")

    def test_request_raw_http_general_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _request_raw handles http errors that are neither status nor timeouts differently."""
        api_client: SyncAPIClient = make_sync_api_client()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_exception(httpx.ProxyError("Proxy error."))
        with pytest.raises(httpx.ProxyError, match="Proxy error."):
            api_client._request_raw(method="GET", url="/api")
        spy_request.assert_called_once_with(method="GET", url="/api")

    def test_request_raw_timeout_error_after_retries(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_sync_api_client
    ):
        """Test that _request_raw raises an error after encountering the maximum number of allowed timeouts."""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._sleep_for_retry = mocker.MagicMock()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_exception(
            httpx.TimeoutException(message="Temporary time out"), is_reusable=True
        )
        with pytest.raises(ConnectionError, match="Failed to connect"):
            api_client._request_raw(method="GET", url="/api")
        spy_request.assert_called_with(method="GET", url="/api")
        assert spy_request.call_count == api_client.max_retries + 1
        assert spy_request.call_count == api_client.max_retries + 1
        assert api_client._sleep_for_retry.call_count == api_client.max_retries

    def test_request_stream(self, mocker: MockerFixture, make_sync_api_client):
        """Test request with stream=True takes the correct branch and passes arguments correctly to _stream"""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._stream = mocker.MagicMock()
        api_client._request_raw = mocker.MagicMock()
        api_client.request(method="GET", url="/api", stream=True)
        api_client._stream.assert_called_once_with(method="GET", url="/api")
        api_client._request_raw.assert_not_called()

    def test_request_non_stream(self, mocker: MockerFixture, make_sync_api_client):
        """Test request with stream=False takes the correct branch and passes arguments correctly to _request_raw"""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client._stream = mocker.MagicMock()
        api_client._request_raw = mocker.MagicMock()
        api_client.request(method="GET", url="/api", stream=False)
        api_client._request_raw.assert_called_once_with(method="GET", url="/api")
        api_client._stream.assert_not_called()

    def test_get(self, mocker: MockerFixture, make_sync_api_client):
        """Test get triggers a call to request with method=GET and passes on arguments correctly"""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client.request = mocker.MagicMock()
        api_client.get(path="/api", stream=True, params={"test": "hello"})
        api_client.request.assert_called_with(
            url="/api", method="GET", stream=True, params={"test": "hello"}
        )

    def test_post(self, mocker: MockerFixture, make_sync_api_client):
        """Test post triggers a call to request with method=POST and passes on arguments correctly"""
        api_client: SyncAPIClient = make_sync_api_client()
        api_client.request = mocker.MagicMock()
        api_client.post(path="/api", stream=True, json={"test": "hello"})
        api_client.request.assert_called_with(
            url="/api", method="POST", stream=True, json={"test": "hello"}
        )


# =============================================================================
# AsyncAPIClient Tests
# =============================================================================


class TestAsyncAPIClient:
    """Tests for AsyncAPIClient"""

    def test_init(self, make_async_api_client, base_client_config):
        """Test init method of AsyncAPIClient"""
        api_client: AsyncAPIClient = make_async_api_client()
        assert api_client.base_url == base_client_config["base_url"]
        assert api_client.follow_redirects == base_client_config["follow_redirects"]
        assert api_client.max_retries == base_client_config["max_retries"]
        assert api_client.timeout == base_client_config["timeout"]
        assert api_client.headers == api_client._build_headers(base_client_config["custom_headers"])
        assert api_client._version == base_client_config["version"]
        assert api_client._client is not None

    def test_init_with_client(self, make_async_api_client):
        """Test that custom http client is passed correctly during init."""
        test_client = httpx.AsyncClient(base_url="https://anotherinstance.com")
        api_client: AsyncAPIClient = make_async_api_client(test_client)
        assert api_client._client == test_client

    def test_init_invalid_http_client(self, make_async_api_client):
        """Test that type of custom http client is enforced by init method."""
        not_a_client = "This is not a httpx.AsyncClient instance"
        with pytest.raises(TypeError, match="Invalid `http_client`"):
            make_async_api_client(not_a_client)

    def test_default_client(self, mocker: MockerFixture, make_async_api_client):
        """Test that default client is initialized correctly and cached."""
        spy_client = mocker.spy(climateclaw_client._base_client.httpx, "AsyncClient")
        api_client: AsyncAPIClient = make_async_api_client()
        default_client = api_client._default_client
        assert api_client._client == default_client
        api_client._default_client
        # as default client is cached, httpx.AsyncClient should only be called once
        spy_client.assert_called_once_with(
            base_url=api_client.base_url,
            follow_redirects=api_client.follow_redirects,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=api_client.timeout),
            headers=api_client.headers,
            auth=api_client._auth,
        )

    @pytest.mark.asyncio
    async def test_is_closed(self, make_async_api_client):
        """Test that AsyncAPIClient is_closed attribute correctly inherits underlying httpx.AsyncClient closed state."""
        async with httpx.AsyncClient() as client:
            api_client: AsyncAPIClient = make_async_api_client(client)
            assert not api_client.is_closed
        assert api_client.is_closed

    @pytest.mark.asyncio
    async def test_sleep_for_retry_first_attempt(
        self, mocker: MockerFixture, make_async_api_client
    ):
        """Test that _sleep_for_retry takes the correct branch if the number of retries taken has not reached the maximum number."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_logger = mocker.spy(climateclaw_client._base_client, "logger")
        mocked_time = mocker.patch.object(climateclaw_client._base_client, "asyncio", spec=True)

        retries_taken = 0
        await api_client._sleep_for_retry(retries_taken)
        spy_logger.debug.assert_called_once()
        assert "Retrying connection after sleeping" in spy_logger.debug.call_args.args[0]
        mocked_time.sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_sleep_for_retry_final_attempt(
        self, mocker: MockerFixture, make_async_api_client
    ):
        """Test that _sleep_for_retry takes the correct branch if the number of retries taken has reached the maximum number."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_logger = mocker.spy(climateclaw_client._base_client, "logger")
        mocked_time = mocker.patch.object(climateclaw_client._base_client, "asyncio", spec=True)

        retries_taken = api_client.max_retries
        await api_client._sleep_for_retry(retries_taken)
        spy_logger.debug.assert_called_once()
        assert "Retrying connection after sleeping" in spy_logger.debug.call_args.args[0]
        assert "Final attempt." in spy_logger.debug.call_args.args[0]
        mocked_time.sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_sleep_for_retry_exceeded_maximum(
        self, mocker: MockerFixture, make_async_api_client
    ):
        """Test that _sleep_for_retry takes the correct branch if the number of retries taken has exceeded the maximum number."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_logger = mocker.spy(climateclaw_client._base_client, "logger")
        mocked_time = mocker.patch.object(climateclaw_client._base_client, "time", spec=True)

        retries_taken = api_client.max_retries + 1
        await api_client._sleep_for_retry(retries_taken)
        spy_logger.debug.assert_called_once()
        assert (
            "Maximum number of retries exceeded. Skipping timeout."
            in spy_logger.debug.call_args.args[0]
        )
        mocked_time.sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_success(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _stream returns a StreamResponse object in case of a successful request and output of stream is as expected."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        stream_iterator = [b'{"a":1}', b'{"b":2}']
        expected_output = [{"a": 1}, {"b": 2}]
        httpx_mock.add_response(status_code=200, stream=IteratorStream(stream_iterator))
        stream_response = await api_client._stream(method="GET", url="/api")
        assert isinstance(stream_response, StreamResponse)
        spy_build_request.assert_called_once_with(method="GET", url="/api")
        spy_send_request.assert_called_once()

        actual_output = []
        async for response in stream_response.aiter_json_objects():
            actual_output.append(response)
        for response, expected in zip(actual_output, expected_output):
            assert response == expected

    @pytest.mark.asyncio
    async def test_stream_success_with_retry(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _stream handles temporary timeouts by sleeping and retrying."""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._sleep_for_retry = mocker.AsyncMock()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        stream_iterator = [b'{"a":1}', b'{"b":2}']
        httpx_mock.add_exception(httpx.TimeoutException(message="Temporary time out"))
        httpx_mock.add_response(status_code=200, stream=IteratorStream(stream_iterator))
        stream_response = await api_client._stream(method="GET", url="/api")
        assert isinstance(stream_response, StreamResponse)
        spy_build_request.assert_called_with(method="GET", url="/api")
        assert spy_build_request.call_count == 2
        assert spy_send_request.call_count == 2
        api_client._sleep_for_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_http_status_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _stream handles status errors differently from timeouts."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        httpx_mock.add_response(status_code=401)
        with pytest.raises(ConnectionError, match="Error connecting"):
            await api_client._stream(method="GET", url="/api")
        spy_build_request.assert_called_once_with(method="GET", url="/api")
        spy_send_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_http_general_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _stream handles http errors that are neither status nor timeouts differently."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        httpx_mock.add_exception(httpx.ProxyError("Proxy error."))
        with pytest.raises(httpx.ProxyError, match="Proxy error."):
            await api_client._stream(method="GET", url="/api")
        spy_build_request.assert_called_once_with(method="GET", url="/api")
        spy_send_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_timeout_error_after_retries(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _stream raises an error after encountering the maximum number of allowed timeouts."""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._sleep_for_retry = mocker.AsyncMock()
        spy_build_request = mocker.spy(api_client._client, "build_request")
        spy_send_request = mocker.spy(api_client._client, "send")
        httpx_mock.add_exception(
            httpx.TimeoutException(message="Temporary time out"), is_reusable=True
        )
        with pytest.raises(ConnectionError, match="Failed to connect"):
            await api_client._stream(method="GET", url="/api")
        spy_build_request.assert_called_with(method="GET", url="/api")
        assert spy_build_request.call_count == api_client.max_retries + 1
        assert spy_send_request.call_count == api_client.max_retries + 1
        assert api_client._sleep_for_retry.call_count == api_client.max_retries

    @pytest.mark.asyncio
    async def test_request_raw_success(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _request_raw returns a Response object in case of a successful request and output is as expected."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_request = mocker.spy(api_client._client, "request")
        response_json = [{"a": 1}, {"b": 2}]
        httpx_mock.add_response(status_code=200, json=response_json)
        response = await api_client._request_raw(method="GET", url="/api")
        assert isinstance(response, httpx.Response)
        spy_request.assert_called_once_with(method="GET", url="/api")
        assert response.json() == response_json

    @pytest.mark.asyncio
    async def test_request_raw_success_with_retry(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _request_raw handles temporary timeouts by sleeping and retrying."""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._sleep_for_retry = mocker.AsyncMock()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_exception(httpx.TimeoutException(message="Temporary time out"))
        httpx_mock.add_response(status_code=200)
        response = await api_client._request_raw(method="GET", url="/api")
        assert isinstance(response, httpx.Response)
        spy_request.assert_called_with(method="GET", url="/api")
        assert spy_request.call_count == 2
        api_client._sleep_for_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_raw_http_status_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _request_raw handles status errors differently from timeouts."""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._sleep_for_retry = mocker.AsyncMock()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_response(status_code=401)
        with pytest.raises(ConnectionError, match="Error connecting"):
            await api_client._request_raw(method="GET", url="/api")
        spy_request.assert_called_once_with(method="GET", url="/api")

    @pytest.mark.asyncio
    async def test_request_raw_http_general_error(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _request_raw handles http errors that are neither status nor timeouts differently."""
        api_client: AsyncAPIClient = make_async_api_client()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_exception(httpx.ProxyError("Proxy error."))
        with pytest.raises(httpx.ProxyError, match="Proxy error."):
            await api_client._request_raw(method="GET", url="/api")
        spy_request.assert_called_once_with(method="GET", url="/api")

    @pytest.mark.asyncio
    async def test_request_raw_timeout_error_after_retries(
        self, mocker: MockerFixture, httpx_mock: HTTPXMock, make_async_api_client
    ):
        """Test that _request_raw raises an error after encountering the maximum number of allowed timeouts."""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._sleep_for_retry = mocker.AsyncMock()
        spy_request = mocker.spy(api_client._client, "request")
        httpx_mock.add_exception(
            httpx.TimeoutException(message="Temporary time out"), is_reusable=True
        )
        with pytest.raises(ConnectionError, match="Failed to connect"):
            await api_client._request_raw(method="GET", url="/api")
        spy_request.assert_called_with(method="GET", url="/api")
        assert spy_request.call_count == api_client.max_retries + 1
        assert spy_request.call_count == api_client.max_retries + 1
        assert api_client._sleep_for_retry.call_count == api_client.max_retries

    @pytest.mark.asyncio
    async def test_request_stream(self, mocker: MockerFixture, make_async_api_client):
        """Test request with stream=True takes the correct branch and passes arguments correctly to _stream"""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._stream = mocker.AsyncMock()
        api_client._request_raw = mocker.AsyncMock()
        await api_client.request(method="GET", url="/api", stream=True)
        api_client._stream.assert_called_once_with(method="GET", url="/api")
        api_client._request_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_non_stream(self, mocker: MockerFixture, make_async_api_client):
        """Test request with stream=False takes the correct branch and passes arguments correctly to _request_raw"""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client._stream = mocker.AsyncMock()
        api_client._request_raw = mocker.AsyncMock()
        await api_client.request(method="GET", url="/api", stream=False)
        api_client._request_raw.assert_called_once_with(method="GET", url="/api")
        api_client._stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_get(self, mocker: MockerFixture, make_async_api_client):
        """Test get triggers a call to request with method=GET and passes on arguments correctly"""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client.request = mocker.AsyncMock()
        await api_client.get(path="/api", stream=True, params={"test": "hello"})
        api_client.request.assert_called_with(
            url="/api", method="GET", stream=True, params={"test": "hello"}
        )

    @pytest.mark.asyncio
    async def test_post(self, mocker: MockerFixture, make_async_api_client):
        """Test post triggers a call to request with method=POST and passes on arguments correctly"""
        api_client: AsyncAPIClient = make_async_api_client()
        api_client.request = mocker.AsyncMock()
        await api_client.post(path="/api", stream=True, json={"test": "hello"})
        api_client.request.assert_called_with(
            url="/api", method="POST", stream=True, json={"test": "hello"}
        )
