import asyncio
import getpass
import json
import logging
from functools import cached_property
from importlib import metadata
from typing import Any, Dict, List, Literal, Tuple, cast

import httpx
import nest_asyncio
from httpx import URL

from ._base_client import AsyncAPIClient, SyncAPIClient
from ._constants import (
    CLIMATECLAW_API_ENDPOINTS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    OPENAPI_SPEC_PATH,
)
from ._streaming import StreamResponse
from .models import Conversation, MessageModel, StreamConversation

try:
    __version__ = metadata.version("climateclaw-client")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

logger = logging.getLogger(__name__)

nest_asyncio.apply()


class ClimateClaw(SyncAPIClient):
    """Synchronous client for interacting with the ClimateClaw API.

    This class provides a high-level interface for communicating with the
    ClimateClaw chatbot API, including methods for managing threads, sending
    prompts, and retrieving conversation history.

    Attributes:
        thread_id: Current active thread ID.
        model: Selected chatbot model.
    """

    _root_api_path: str = "/api/chatbot"
    _user: str = getpass.getuser()
    thread_id: str | None
    model: str | None

    def __init__(
        self,
        *,
        base_url: str | URL,
        store_path: str = "",
        follow_redirects: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
        thread_id: str | None = None,
        model: str | None = None,
        custom_headers: dict[str, Any] = {},
        auth_url: str | URL = "",
        interactive_auth: bool = True,
    ):
        """Initializes the ClimateClaw client.

        Args:
            base_url: Base URL for the ClimateClaw API.
            store_path: Path to store authentication tokens.
            follow_redirects: Whether to follow HTTP redirects.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            http_client: Optional pre-configured httpx.Client.
            thread_id: Optional thread ID for an existing conversation.
            model: Optional chatbot model to use for requests.
            auth_url: Optional url to determine location of auth endpoint.
            interactive_auth: Optional boolean to determine if authentication should be performed interactively.

        Raises:
            ValueError: If the specified model is not in available_models.
        """
        super().__init__(
            version=__version__,
            base_url=base_url,
            store_path=store_path,
            follow_redirects=follow_redirects,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=custom_headers,
            auth_url=auth_url,
            interactive_auth=interactive_auth,
        )
        self._validate_backend_endpoints()
        self.thread_id = thread_id
        self.model = model

    @cached_property
    def available_models(self) -> List[str]:
        """Gets the list of available chatbot models.

        Returns:
            List[str]: List of available model names.
        """
        response = self.get(path=self._construct_path("chatbots"), stream=False)
        response = cast(httpx.Response, response)
        available_models = response.json()
        return available_models

    @property  # type: ignore[no-redef]
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, new_model: str):
        if new_model and new_model not in self.available_models:
            raise ValueError(
                f"Model {new_model} is not a valid selection. Please select from available models: {self.available_models} instead."
            )
        self._model = new_model

    def _validate_backend_endpoints(self) -> None:
        """Validates chatbot endpoints available on the backend against those expected by the client.

        Fetches the OpenAPI spec from the backend and verifies that all expected
        endpoints are available. Also checks for unexpected endpoints and logs warnings.

        Raises:
            KeyError: If required keys ('paths', 'info') are missing from the OpenAPI spec,
                or if expected endpoints are not found in the backend specification.
        """
        r = self.get(path=f"{self._root_api_path}/{OPENAPI_SPEC_PATH}", stream=False)
        r = cast(httpx.Response, r)
        openapi_spec = r.json()
        if (key := "paths") not in openapi_spec or (key := "info") not in openapi_spec:
            raise KeyError(
                f"Key '{key}' cannot be found in openapi spec file located under {self.base_url.join(f'{self._root_api_path}/{OPENAPI_SPEC_PATH}')}. Make sure backend is configured correctly."
            )
        # get version info from openapi spec file
        if "version" not in openapi_spec["info"]:
            raise KeyError(
                "ClimateClaw backend version information could not be retrieved from openapi spec file. Make sure backend is configured correctly."
            )
        cc_backend_version = openapi_spec["info"]["version"]
        # filter relevant paths specific to chatbot
        chatbot_paths = list(
            filter(lambda p: self._root_api_path in p, openapi_spec["paths"].keys())
        )
        # first check that all endpoints located on the backend are included in the client specification, raise warnings if unexpected enpoints are encountered
        for found_path in chatbot_paths:
            if found_path not in map(self._construct_path, CLIMATECLAW_API_ENDPOINTS.keys()):
                logger.warning(
                    f"API endpoint {found_path} not included in client specification. The client (version {self._version}) might not be compatible with the backend (version {cc_backend_version})."
                )
        # now check that all endpoints expected by the client are available on the backend, raise a KeyError if at least one cannot be found.
        for endpoint in CLIMATECLAW_API_ENDPOINTS.keys():
            if (expected_endpoint := self._construct_path(endpoint)) not in chatbot_paths:
                raise KeyError(
                    f"ClimateClaw client expected endpoint {expected_endpoint} could not be found in backend specification. Make sure that client (version {self._version}) and backend (version {cc_backend_version}) are up-to-date."
                )

    def authenticate(self) -> None:
        """Authenticates the client with the ClimateClaw API.

        Triggers the OIDC authentication flow.
        """
        self._auth._authenticate()

    def newthread(self) -> str:
        """Creates a new conversation thread and updates the current thread id.

        Returns:
            str: The ID of the newly created thread.
        """
        response = self.get(path=self._construct_path("newthread"), stream=False)
        response = cast(httpx.Response, response)
        thread_id = response.json()
        self.thread_id = thread_id
        return thread_id

    def prompt(
        self,
        input: str,
        model: str | None = None,
        thread_id: str | None = None,
        stream: bool = False,
        store_thread: bool = True,
    ) -> Conversation | StreamConversation:
        """Sends a prompt to the chatbot and gets the response.

        Args:
            input (str): The user input/prompt to send.
            model (str | None, optional): Model to use for this request. Falls back to instance attribute if not specified.
            thread_id (str | None, optional): Thread ID for the conversation. Creates a new thread if not specified and no active thread exists. Defaults to None.
            stream (bool): If True, returns a `StreamConversation` for streaming, else returns a complete `Conversation`. Defaults to False.
            store_thread (bool): If True, stores thread to the backend database. Note: only threads stored to the database can be accessed in a later session. Defaults to True.

        Examples:
            Send a prompt to the backend and print the result to get an overview of the response:

            .. code-block:: python

                from climateclaw_client import ClimateClaw

                cc = ClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                response = cc.prompt(
                    input="Please calculate and plot the average yearly temperature over Germany for the years 1990-2020!"
                )
                print(response)

            After having received the response, you can extract certain information from it, such as code cells (if any are part of the response):

            .. code-block:: python

                from climateclaw_client import ClimateClaw

                cc = ClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                response = cc.prompt(
                    input="Please calculate and plot the average yearly temperature over Germany for the years 1990-2020!"
                )
                for cell in response.code_cells:
                    print(cell)

            Additionally, if the output includes any (base64-encoded) images, these can be extracted and saved to file:

            .. code-block:: python

                from climateclaw_client import ClimateClaw

                cc = ClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                response = cc.prompt(
                    input="Please calculate and plot the average yearly temperature over Germany for the years 1990-2020!"
                )
                for i, image in enumerate(response.images):
                    image.save_to_file(f"plot_average_temperature_germany_{i+1}.png")

            If ``stream=True``, the response can be streamed incrementally, either as a string ready for markdown-rendering:

            .. code-block:: python

                from climateclaw_client import ClimateClaw

                cc = ClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                with cc.prompt(
                    input="Please explain the phenomenon knows as the ENSO to me!", stream=True
                ) as stream:
                    for variant, md in stream.iter_for_markdown():
                        print(variant, md)

            Alternatively the raw json-like response can be streamed as dicts:

            .. code-block:: python

                from climateclaw_client import ClimateClaw

                cc = ClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                with cc.prompt(
                    input="Please explain the phenomenon knows as the ENSO to me!", stream=True
                ) as stream:
                    for part in stream.iter_raw():
                        print(part)

            Once a streamed response has been consumed, the entire response can be accessed as a ``Conversation`` object:

            .. code-block:: python

                from climateclaw_client import ClimateClaw

                cc = ClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                stream = cc.prompt(
                    input="Please explain the phenomenon knows as the ENSO to me!", stream=True
                )
                for part in stream.iter_raw():
                    pass
                print(stream.conversation)

        Returns:
            Conversation | StreamConversation: Conversation containing all messages or StreamConversation, allowing for incremental streaming.

            * If ``stream=False`` a ``Conversation`` instance is returned, containing entire response for a given prompt.
            * If ``stream=False`` a ``StreamConversation`` instance is returned, which can be used to stream the response as a markdown-ready string or in its raw json-like form.

        Raises:
            ValueError: If model is specified but not in available_models.
            TypeError: If model is not specified and instance has no model set.
        """

        if not model and self.model:
            model = self.model
        elif model and model not in self.available_models:
            raise ValueError(
                f"Value '{model}' is not a valid selection for param 'model'. Please select from available models: {self.available_models} instead."
            )
        elif not model and not self.model:
            raise TypeError(
                f"Argument 'model' has to be specified, unless instance attribute '{self.__class__.__name__}.model' is set."
            )

        if not (self.thread_id or thread_id):
            thread_id = self.newthread()
        elif not thread_id:
            thread_id = self.thread_id
        try:
            response: httpx.Response | StreamResponse = self.post(
                path=self._construct_path("streamresponse"),
                json={
                    "input": input,
                    "thread_id": thread_id,
                    "chatbot": model,
                    "store_thread": store_thread,
                },
                stream=stream,
            )
        except KeyboardInterrupt:
            logger.debug("Registered keyboard-interrupt. Stopping thread.")
            self.stop(thread_id=thread_id)
            raise
        if not stream:
            response = cast(httpx.Response, response)
            messages = [
                MessageModel(message=json.loads(el)) for el in response.text.split("\n") if el
            ]
            return Conversation(raw_messages=messages)
        else:
            response = cast(StreamResponse, response)
            return StreamConversation(
                stream=response, on_exit_callback=lambda: self.stop(thread_id)
            )

    def getthread(self, thread_id: str | None = None) -> Conversation:
        """Retrieves a conversation thread by ID.

        Args:
            thread_id (str | None, optional): The ID of the thread to retrieve. If not specified,
                uses the current active thread ID. Defaults to None.

        Returns:
            Conversation: Conversation containing all messages in the thread (including user messages).

        Raises:
            TypeError: If thread_id is not specified and no active thread exists.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        response = self.post(
            path=self._construct_path("getthread"),
            json={"thread_id": thread_id},
            stream=False,
        )
        response = cast(httpx.Response, response)
        messages = [MessageModel(message=m) for m in response.json()]
        return Conversation(raw_messages=messages)

    def getuserthreads(
        self, num_threads: int = 20
    ) -> Tuple[int, List[Dict[str, str | Conversation]]]:
        """Retrieve the most recent conversation threads of the authenticated user, limited by the requested number.

        Args:
            num_threads (int): The maximum number of recent threads to return. Defaults to 20.

        Returns:
            Tuple[int, List[Dict[str, str | Conversation]]]: A tuple containing the total number of threads available for the user and a list containing dictionaries
            that include information on each thread retrieved, such as "thread_id", "date", and "topic", as well as "content",
            which includes the thread's messages as a Conversation object.

        Raises:
            ValueError: Raised if integer num_threads is 0 or smaller.
        """
        if num_threads <= 0:
            raise ValueError("Value 'num_threads' has to be at least 1.")
        response = self.post(
            path=self._construct_path("getuserthreads"),
            json={
                "num_threads": num_threads,
                "page": 0,  # currently hardcoded to be 0 (other values seem to always return an empty list)
            },
            stream=False,
        )
        response = cast(httpx.Response, response)
        user_threads: List[Dict[str, Any]] = response.json()[0]
        n_threads: int = response.json()[1]

        def map_key_value(k: str, v: Any):
            return (
                Conversation(raw_messages=[MessageModel(message=m) for m in v])
                if k.lower() == "content"
                else str(v)
            )

        thread_data = [
            {key: map_key_value(key, value) for key, value in ut.items()} for ut in user_threads
        ]
        return n_threads, thread_data

    def deletethread(self, thread_id: str | None = None) -> None:
        """
        Delete a given thread by the authenticated user.

        Args:
            thread_id (str | None, optional): The ID of the thread to be deleted on the backend. If not specified, uses the current active thread ID (and resets active thread ID to None). Defaults to None.

        Raises:
            TypeError: Raised, if no thread_id is specified and no previous conversation was started.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        self.post(
            path=self._construct_path("deletethread"), json={"thread_id": thread_id}, stream=False
        )
        # reset self.thread_id in case it is identical to id of deleted thread
        self.thread_id = None if self.thread_id == thread_id else self.thread_id

    def setthreadtopic(self, new_topic: str, thread_id: str | None = None) -> str:
        """Sets the topic of a given thread.
        Thread topics can be used to search threads (see `searchthreads`).

        Args:
            new_topic (str): String describing the new thread topic.
            thread_id (str | None, optional): The ID of the thread which topic should be changed. If not specified, uses the current active thread ID. Defaults to None.

        Raises:
            TypeError: Raised, if no thread_id is specified and no previous conversation was started.

        Returns:
            str: The new thread topic.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        self.post(
            path=self._construct_path("setthreadtopic"),
            json={"thread_id": thread_id, "topic": new_topic},
            stream=False,
        )
        return new_topic

    def searchthreads(
        self, query: str, num_threads: int = 20
    ) -> Tuple[int, List[Dict[str, str | Conversation]]]:
        """Search the authenticated user's conversation threads using a query string. Supports only topic-based search.

        Args:
            query (str): The search query string.
            num_threads (int, optional): The maximum number of results to return. Defaults to 20.

        Returns:
            Tuple[int, List[Dict[str, str | Conversation]]]: A tuple containing the total number of threads available for the user and a list containing dictionaries
            that include information on each thread retrieved, such as "thread_id", "date", and "topic", as well as "content",
            which includes the thread's messages as a Conversation object.

        Raises:
            ValueError: Raised if integer num_threads is 0 or smaller.
        """
        if num_threads <= 0:
            raise ValueError("Value 'num_threads' has to be at least 1.")
        response = self.post(
            path=self._construct_path("searchthreads"),
            json={
                "query": query,
                "num_threads": num_threads,
            },
            stream=False,
        )
        response = cast(httpx.Response, response)
        user_threads, n_threads = response.json()

        def map_key_value(k: str, v: Any):
            return (
                Conversation(raw_messages=[MessageModel(message=m) for m in v])
                if k.lower() == "content"
                else str(v)
            )

        thread_data = [
            {key: map_key_value(key, value) for key, value in ut.items()} for ut in user_threads
        ]
        return n_threads, thread_data

    def stop(self, thread_id: str | None = None) -> bool:
        """Stop an active streaming conversation, cancels any in-flight tool executions on the backend.

        Args:
            thread_id (str | None, optional): The ID of the thread which should be stopped. If not specified, uses the current active thread ID.

        Raises:
            TypeError: Raised if thread_id is not specified and no active thread exists.
            ConnectionError: Raised if the request results in an internal server error.

        Returns:
            bool: True if thread was stopped successfully (without an error).
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        try:
            self.post(
                path=self._construct_path("stop"), json={"thread_id": thread_id}, stream=False
            )
            return True
        except ConnectionError as e:
            if e.errno == 404:
                logger.warning(f"No active thread could be found under thread_id {thread_id}.")
                return True
            elif e.errno == 505:
                raise ConnectionError(
                    "Could not stop thread due to an internal server error."
                ) from e
            raise

    def editthread(
        self, user_index: int, source_thread_id: str | None = None
    ) -> Tuple[str, Conversation]:
        """Fork an existing conversation thread at a given message index.
        This causes the endpoint to create a new thread by copying the message history of an existing thread up to the edited message.
        The specified message and all subsequent messages are discarded in the new branch, allowing the client to replace or modify the conversation from that point onward.

        Args:
            user_index (int): The (zero-based) index from which to fork the conversation.
            source_thread_id (str | None, optional): The ID of the thread which should be forked. If not specified, uses the current active thread ID. Defaults to None.

        Raises:
            TypeError: Raised if thread_id is not specified and no active thread exists.
            ValueError: Raised if given thread cannot be found by the backend.
            IndexError: Raised if the user index is out of bounds, as indicated by the backend.
            ConnectionError: Raised in case server responds with an internal error message.
            KeyError: Raised if the response does not include the keys 'new_thread_id' or 'history'.

        Returns:
            Tuple[str, Conversation]: A tuple containing both the new thread id and a Conversation object that encapsulates the message history from which the new thread starts from.
        """
        if not source_thread_id and self.thread_id:
            source_thread_id = self.thread_id
        elif not (source_thread_id or self.thread_id):
            raise TypeError(
                "Argument 'source_thread_id' has to be specified, if no conversation was started previously."
            )
        try:
            response = self.post(
                path=self._construct_path("editthread"),
                json={
                    "source_thread_id": source_thread_id,
                    "user_index": user_index,
                },
                stream=False,
            )
            response = cast(httpx.Response, response)
        except ConnectionError as e:
            if e.errno == 404:
                raise ValueError(f"No thread found for id '{source_thread_id}'!")
            elif e.errno == 422:
                raise IndexError(f"User message index {user_index} out of bounds!")
            else:
                raise ConnectionError(
                    "Editing thread failed due to an internal server error."
                ) from e
        response_dict: Dict[str, Any] = response.json()
        if not response_dict.keys() >= (expected_keys := {"new_thread_id", "history"}):
            raise KeyError(
                f"The response to editing thread '{source_thread_id}' did not include keys {expected_keys}."
            )
        new_thread_id = response_dict["new_thread_id"]
        history = Conversation(
            raw_messages=[MessageModel(message=m) for m in response_dict["history"]]
        )
        return new_thread_id, history

    def userfeedback(
        self,
        feedback_index: int,
        feedback: Literal["up", "down", "remove"],
        thread_id: str | None = None,
    ) -> str:
        """Submit or modify to a specific message within a thread.

        Args:
            feedback_index (int): The (zero-based) index of the (Code, Assistant) message within a thread where feedback should be added or removed.
            feedback (str): Feedback to be submitted. Must be one of 'up' (positive), 'down' (negative) or 'remove' (remove any existing feedback from message).
            thread_id (str | None, optional): The ID of the thread containing the message feedback should be submitted for. If not specified, uses the current active thread ID. Defaults to None.

        Raises:
            TypeError: Raised if thread_id is not specified and no active thread exists.
            ValueError: Raised if feedback string is not one of 'up', 'down', 'remove', or if thread cannot be found by backend.
            IndexError: Raised if feedback cannot be found at given index (in case of removal) or if index is out of bounds.
            ConnectionError: Raised if feedback submission results in a internal server error on the backend.

        Returns:
            str: Detail message returned by the backend, or warning if the response was empty.

        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        if feedback not in (allowed_feedback := ["up", "down", "remove"]):
            raise ValueError(f"Feedback string must be one of {allowed_feedback}.")
        try:
            response = self.post(
                path=self._construct_path("userfeedback"),
                params={
                    "thread_id": thread_id,
                    "feedback_index": feedback_index,
                    "feedback": feedback,
                },
                stream=False,
            )
            response = cast(httpx.Response, response)
            response_dict: Dict[str, str] = response.json()
            message: str = response_dict.get(
                "detail",
                "Empty message was returned. User feedback was possibly not correctly processed by the backend.",
            )
            return message
        except ConnectionError as e:
            if e.errno == 404:
                if e.strerror and "thread not found" in e.strerror.lower():
                    raise ValueError(f"No thread found for id '{thread_id}'.")
                elif e.strerror and "feedback not found" in e.strerror.lower():
                    raise IndexError(
                        f"Feedback not found at index {feedback_index} for thread '{thread_id}'."
                    )
            elif e.errno == 422:
                raise IndexError(f"Index {feedback_index} is out of bounds.")
            elif e.errno in (500, 503):
                raise ConnectionError("Error on the backend saving/modifying feedback.")
            raise

    def _construct_path(self, endpoint_name: str) -> str:
        """Constructs the full API path for an endpoint.

        Args:
            endpoint_name (str): Name of the endpoint from CLIMATECLAW_API_ENDPOINTS.

        Returns:
            str: Full path string combining root API path and endpoint path.
        """
        return f"{self._root_api_path}/{CLIMATECLAW_API_ENDPOINTS[endpoint_name]}"


class AsyncClimateClaw(AsyncAPIClient):
    """Asynchronous client for interacting with the ClimateClaw API.

    This class provides an async high-level interface for communicating with
    the ClimateClaw chatbot API, including methods for managing threads, sending
    prompts, and retrieving conversation history.

    Note:
        This is the async version of ClimateClaw. Use this class when working
        with async/await code. For synchronous code, use ClimateClaw instead.

    Attributes:
        thread_id: Current active thread ID.
        model: Selected chatbot model.
    """

    _root_api_path: str = "/api/chatbot"
    _user: str = getpass.getuser()
    _model: str | None = None
    thread_id: str | None = None

    def __init__(
        self,
        *,
        base_url: str | URL,
        store_path: str = "",
        follow_redirects: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
        thread_id: str | None = None,
        model: str | None = None,
        auth_url: str | URL = "",
        interactive_auth: bool = True,
    ):
        """Initializes the AsyncClimateClaw client.

        Args:
            base_url: Base URL for the ClimateClaw API.
            store_path: Path to store authentication tokens.
            follow_redirects: Whether to follow HTTP redirects.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            http_client: Optional pre-configured httpx.AsyncClient.
            thread_id: Optional thread ID for an existing conversation.
            model: Optional chatbot model to use for requests.
            auth_url: Optional url to determine location of auth endpoint.
            interactive_auth: Optional boolean to determine if authentication should be performed interactively.

        """
        super().__init__(
            version=__version__,
            base_url=base_url,
            store_path=store_path,
            follow_redirects=follow_redirects,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            auth_url=auth_url,
            interactive_auth=interactive_auth,
        )
        asyncio.run(self._validate_backend_endpoints())
        self.thread_id = thread_id
        self.model = model

    @cached_property
    def available_models(self) -> List[str]:
        """Gets the list of available chatbot models.

        Returns:
            List of available model names.
        """
        response = asyncio.run(self.get(path=self._construct_path("chatbots"), stream=False))
        response = cast(httpx.Response, response)
        return response.json()

    @property
    def model(self) -> str | None:
        """Get the current model."""
        return self._model

    @model.setter
    def model(self, new_model: str | None) -> None:
        """Set the model.

        Note: Validation is not performed here to avoid async in __init__.
        Validation happens in prompt() and other methods that use the model.
        """
        self._validate_model(new_model)
        self._model = new_model

    def _validate_model(self, model: str | None) -> str | None:
        """Validate a model against available models.
        Sets model to instance model, if model vale is None.

        Args:
            model: The model to validate.

        Returns:
            The validated model.

        Raises:
            ValueError: If model is specified but not in available_models.
        """
        if model:
            if model not in (available := self.available_models):
                raise ValueError(
                    f"Value '{model}' is not a valid selection for param 'model'. "
                    f"Please select from available models: {available} instead."
                )
        else:
            model = self._model
        return model

    async def _validate_backend_endpoints(self) -> None:
        """Validates chatbot endpoints available on the backend against those expected by the client.

        Fetches the OpenAPI spec from the backend and verifies that all expected
        endpoints are available. Also checks for unexpected endpoints and logs warnings.

        Raises:
            KeyError: If required keys ('paths', 'info') are missing from the OpenAPI spec,
                or if expected endpoints are not found in the backend specification.
        """
        r = await self.get(path=f"{self._root_api_path}/{OPENAPI_SPEC_PATH}", stream=False)
        r = cast(httpx.Response, r)
        openapi_spec = r.json()
        if (key := "paths") not in openapi_spec or (key := "info") not in openapi_spec:
            raise KeyError(
                f"Key '{key}' cannot be found in openapi spec file located under "
                f"{self.base_url.join(f'{self._root_api_path}/{OPENAPI_SPEC_PATH}')}. "
                f"Make sure backend is configured correctly."
            )
        if "version" not in openapi_spec["info"]:
            raise KeyError(
                "ClimateClaw backend version information could not be retrieved from "
                "openapi spec file. Make sure backend is configured correctly."
            )
        cc_backend_version = openapi_spec["info"]["version"]
        chatbot_paths = list(
            filter(lambda p: self._root_api_path in p, openapi_spec["paths"].keys())
        )
        for found_path in chatbot_paths:
            if found_path not in map(self._construct_path, CLIMATECLAW_API_ENDPOINTS.keys()):
                logger.warning(
                    f"API endpoint {found_path} not included in client specification. "
                    f"The client (version {self._version}) might not be compatible with "
                    f"the backend (version {cc_backend_version})."
                )
        for endpoint in CLIMATECLAW_API_ENDPOINTS.keys():
            if (expected_endpoint := self._construct_path(endpoint)) not in chatbot_paths:
                raise KeyError(
                    f"ClimateClaw client expected endpoint {expected_endpoint} could not be "
                    f"found in backend specification. Make sure that client (version "
                    f"{self._version}) and backend (version {cc_backend_version}) "
                    f"are up-to-date."
                )

    async def authenticate(self) -> None:
        """Authenticates the client with the ClimateClaw API.

        Triggers the OIDC authentication flow.
        """
        await self._auth._async_authenticate()

    async def newthread(self) -> str:
        """Creates a new conversation thread and updates the current thread id.

        Returns:
            The ID of the newly created thread.
        """
        response = await self.get(path=self._construct_path("newthread"), stream=False)
        response = cast(httpx.Response, response)
        thread_id = response.json()
        self.thread_id = thread_id
        return thread_id

    async def prompt(
        self,
        input: str,
        model: str | None = None,
        thread_id: str | None = None,
        stream: bool = False,
        store_thread: bool = True,
    ) -> Conversation | StreamConversation:
        """Sends a prompt to the chatbot and gets the response.

        Args:
            input (str): The user input/prompt to send.
            model (str | None, optional): Model to use for this request. Falls back to instance attribute if not specified.
            thread_id (str | None, optional): Thread ID for the conversation. Creates a new thread if not specified and no active thread exists. Defaults to None.
            stream (bool): If True, returns a `StreamConversation` for streaming, else returns a complete `Conversation`. Defaults to False.
            store_thread (bool): If True, stores thread to the backend database. Note: only threads stored to the database can be accessed in a later session. Defaults to True.

        Examples:
            Send a prompt to the backend and print the result to get an overview of the response:

            .. code-block:: python

                import asyncio
                from climateclaw_client import AsyncClimateClaw


                async def main():
                    cc = AsyncClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")
                    response = await cc.prompt(
                        input="Please calculate and plot the average yearly temperature over Germany for the years 1990-2020!"
                    )

                    print(response)


                asyncio.run(main())

            After having received the response, you can extract certain information from it, such as code cells (if any are part of the response):

            .. code-block:: python

                import asyncio
                from climateclaw_client import AsyncClimateClaw


                async def main():
                    cc = AsyncClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")
                    response = await cc.prompt(
                        input="Please calculate and plot the average yearly temperature over Germany for the years 1990-2020!"
                    )

                    for cell in response.code_cells:
                        print(cell)


                asyncio.run(main())

            Additionally, if the output includes any (base64-encoded) images, these can be extracted and saved to file:

            .. code-block:: python

                import asyncio
                from climateclaw_client import AsyncClimateClaw


                async def main():
                    cc = AsyncClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")
                    response = await cc.prompt(
                        input="Please calculate and plot the average yearly temperature over Germany for the years 1990-2020!"
                    )

                    for i, image in enumerate(response.images):
                        image.save_to_file(f"plot_average_temperature_germany_{i+1}.png")


                asyncio.run(main())

            If ``stream=True``, the response can be streamed incrementally, either as a string ready for markdown-rendering:

            .. code-block:: python

                import asyncio
                from climateclaw_client import AsyncClimateClaw


                async def main():
                    cc = AsyncClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                    async with cc.prompt(
                        input="Please explain the phenomenon knows as the ENSO to me!", stream=True
                    ) as stream:
                        async for variant, md in stream.aiter_for_markdown():
                            print(variant, md)


                asyncio.run(main())



            Alternatively the raw json-like response can be streamed as dicts:

            .. code-block:: python

                import asyncio
                from climateclaw_client import AsyncClimateClaw


                async def main():
                    cc = AsyncClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                    async with cc.prompt(
                        input="Please explain the phenomenon knows as the ENSO to me!", stream=True
                    ) as stream:
                        async for part in stream.aiter_raw():
                            print(part)


                asyncio.run(main())

            Once a streamed response has been consumed, the entire response can be accessed as a ``Conversation`` object:

            .. code-block:: python

                import asyncio
                from climateclaw_client import AsyncClimateClaw


                async def main():
                    cc = AsyncClimateClaw(base_url="https://nextgems.dkrz.de", model="gpt-4.1")

                    stream = await cc.prompt(
                        input="Please explain the phenomenon knows as the ENSO to me!", stream=True
                    )
                    async for part in stream.aiter_raw():
                        pass
                    print(stream.conversation)


                asyncio.run(main())

        Returns:
            Conversation | StreamConversation: Conversation containing all messages or StreamConversation, allowing for incremental streaming.

            * If ``stream=False`` a ``Conversation`` instance is returned, containing entire response for a given prompt.
            * If ``stream=False`` a ``StreamConversation`` instance is returned, which can be used to stream the response as a markdown-ready string or in its raw json-like form.

        Raises:
            ValueError: If model is specified but not in available_models.
            TypeError: If model is not specified and instance has no model set.
        """
        model = self._validate_model(model)
        if not model:
            raise TypeError(
                f"Argument 'model' has to be specified, unless instance attribute '{self.__class__.__name__}.model' is set."
            )

        if not (self.thread_id or thread_id):
            thread_id = await self.newthread()
        elif not thread_id:
            thread_id = self.thread_id

        try:
            response: httpx.Response | StreamResponse = await self.post(
                path=self._construct_path("streamresponse"),
                json={
                    "input": input,
                    "thread_id": thread_id,
                    "chatbot": model,
                    "store_thread": store_thread,
                },
                stream=stream,
            )
        except KeyboardInterrupt:
            logger.debug("Registered keyboard-interrupt. Stopping thread.")
            await self.stop(thread_id=thread_id)
            raise

        if not stream:
            response = cast(httpx.Response, response)
            response_text = response.text
            messages = [
                MessageModel(message=json.loads(el)) for el in response_text.split("\n") if el
            ]
            return Conversation(raw_messages=messages)
        else:
            response = cast(StreamResponse, response)
            return StreamConversation(
                stream=response,
                on_exit_callback=lambda: self.stop(thread_id),
            )

    async def getthread(self, thread_id: str | None = None) -> Conversation:
        """Retrieves a conversation thread by ID.

        Args:
            thread_id: The ID of the thread to retrieve. If not specified,
                uses the current active thread ID.

        Returns:
            Conversation containing all messages in the thread.

        Raises:
            TypeError: If thread_id is not specified and no active thread exists.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        response = await self.post(
            path=self._construct_path("getthread"),
            json={"thread_id": thread_id},
            stream=False,
        )
        response = cast(httpx.Response, response)
        messages = [MessageModel(message=m) for m in response.json()]
        return Conversation(raw_messages=messages)

    async def getuserthreads(
        self, num_threads: int = 20
    ) -> Tuple[int, List[Dict[str, str | Conversation]]]:
        """Retrieve the most recent conversation threads of the authenticated user.

        Args:
            num_threads: The maximum number of recent threads to return. Defaults to 20.

        Returns:
            A tuple with: 1. the total number of threads, 2. a list containing thread info.

        Raises:
            ValueError: If num_threads is 0 or smaller.
        """
        if num_threads <= 0:
            raise ValueError("Value 'num_threads' has to be at least 1.")
        response = await self.post(
            path=self._construct_path("getuserthreads"),
            json={
                "num_threads": num_threads,
                "page": 0,  # currently hardcoded to be 0 (other values seem to always return an empty list)
            },
            stream=False,
        )
        response = cast(httpx.Response, response)
        data = response.json()
        user_threads: List[Dict[str, Any]] = data[0]
        n_threads: int = data[1]

        def map_key_value(k: str, v: Any) -> Any:
            return (
                Conversation(raw_messages=[MessageModel(message=m) for m in v])
                if k.lower() == "content"
                else str(v)
            )

        thread_data = [
            {key: map_key_value(key, value) for key, value in ut.items()} for ut in user_threads
        ]
        return n_threads, thread_data

    async def deletethread(self, thread_id: str | None = None) -> None:
        """Delete a given thread by the authenticated user.

        Args:
            thread_id: The ID of the thread to be deleted. If not specified, uses the current active thread ID.

        Raises:
            TypeError: If thread_id is not specified and no active thread exists.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        await self.post(
            path=self._construct_path("deletethread"),
            json={"thread_id": thread_id},
            stream=False,
        )
        self.thread_id = None if self.thread_id == thread_id else self.thread_id

    async def setthreadtopic(self, new_topic: str, thread_id: str | None = None) -> str:
        """Sets the topic of a given thread.

        Args:
            new_topic: String describing the new thread topic.
            thread_id: The ID of the thread whose topic should be changed. Uses current thread if not specified.

        Raises:
            TypeError: If thread_id is not specified and no previous conversation was started.

        Returns:
            The new thread topic.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        await self.post(
            path=self._construct_path("setthreadtopic"),
            json={"thread_id": thread_id, "topic": new_topic},
            stream=False,
        )
        return new_topic

    async def searchthreads(
        self, query: str, num_threads: int = 20
    ) -> Tuple[int, List[Dict[str, str | Conversation]]]:
        """Search the authenticated user's conversation threads using a query string.

        Args:
            query: The search query string.
            num_threads: The maximum number of results to return. Defaults to 20.

        Returns:
            A tuple with: 1. the total number of threads, 2. a list containing thread info.

        Raises:
            ValueError: If num_threads is 0 or smaller.
        """
        if num_threads <= 0:
            raise ValueError("Value 'num_threads' has to be at least 1.")
        response = await self.post(
            path=self._construct_path("searchthreads"),
            json={"query": query, "num_threads": num_threads},
            stream=False,
        )
        response = cast(httpx.Response, response)
        user_threads, n_threads = response.json()

        def map_key_value(k: str, v: Any) -> Any:
            return (
                Conversation(raw_messages=[MessageModel(message=m) for m in v])
                if k.lower() == "content"
                else str(v)
            )

        thread_data = [
            {key: map_key_value(key, value) for key, value in ut.items()} for ut in user_threads
        ]
        return n_threads, thread_data

    async def stop(self, thread_id: str | None = None) -> bool:
        """Stop an active streaming conversation, cancels any in-flight tool executions.

        Args:
            thread_id: The ID of the thread to stop. Uses current thread if not specified.

        Raises:
            TypeError: If thread_id is not specified and no active thread exists.
            ConnectionError: If the request results in an internal server error.

        Returns:
            True if thread was stopped successfully.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        try:
            await self.post(
                path=self._construct_path("stop"),
                json={"thread_id": thread_id},
                stream=False,
            )
            return True
        except ConnectionError as e:
            if e.errno == 404:
                logger.warning(f"No active thread could be found under thread_id {thread_id}.")
                return True
            elif e.errno == 505:
                raise ConnectionError(
                    "Could not stop thread due to an internal server error."
                ) from e
            raise

    async def editthread(
        self, user_index: int, source_thread_id: str | None = None
    ) -> Tuple[str, Conversation]:
        """Fork an existing conversation thread at a given message index.

        Args:
            user_index: The (zero-based) index to fork from.
            source_thread_id: The ID of the thread to fork. Uses current thread if not specified.

        Raises:
            TypeError: If thread_id is not specified and no active thread exists.
            ValueError: If the source thread cannot be found.
            IndexError: If the user index is out of bounds.
            ConnectionError: If there's an internal server error.
            KeyError: If the response is malformed.

        Returns:
            Tuple of (new_thread_id, Conversation) with the forked thread's starting history.
        """
        if not source_thread_id and self.thread_id:
            source_thread_id = self.thread_id
        elif not (source_thread_id or self.thread_id):
            raise TypeError(
                "Argument 'source_thread_id' has to be specified, if no conversation was started previously."
            )
        try:
            response = await self.post(
                path=self._construct_path("editthread"),
                json={"source_thread_id": source_thread_id, "user_index": user_index},
                stream=False,
            )
            response = cast(httpx.Response, response)
        except ConnectionError as e:
            if e.errno == 404:
                raise ValueError(f"No thread found for id '{source_thread_id}'!")
            elif e.errno == 422:
                raise IndexError(f"User message index {user_index} out of bounds!")
            else:
                raise ConnectionError(
                    "Editing thread failed due to an internal server error."
                ) from e
        response_dict: Dict[str, Any] = response.json()
        if not response_dict.keys() >= (expected_keys := {"new_thread_id", "history"}):
            raise KeyError(
                f"The response to editing thread '{source_thread_id}' did not include keys {expected_keys}."
            )
        new_thread_id = response_dict["new_thread_id"]
        history = Conversation(
            raw_messages=[MessageModel(message=m) for m in response_dict["history"]]
        )
        return new_thread_id, history

    async def userfeedback(
        self,
        feedback_index: int,
        feedback: Literal["up", "down", "remove"],
        thread_id: str | None = None,
    ) -> str:
        """Submit or modify feedback for a specific message within a thread.

        Args:
            feedback_index: The (zero-based) index of the (Code, Assistant) message.
            feedback: Must be 'up', 'down', or 'remove'.
            thread_id: The ID of the thread. Uses current thread if not specified.

        Raises:
            TypeError: If thread_id is not specified and no active thread exists.
            ValueError: If feedback is invalid or thread not found.
            IndexError: If feedback index is out of bounds or not found.
            ConnectionError: If there's an internal server error.

        Returns:
            Detail message from backend.
        """
        if not thread_id and self.thread_id:
            thread_id = self.thread_id
        elif not (thread_id or self.thread_id):
            raise TypeError(
                "Argument 'thread_id' has to be specified, if no conversation was started previously."
            )
        if feedback not in (allowed_feedback := ["up", "down", "remove"]):
            raise ValueError(f"Feedback string must be one of {allowed_feedback}.")
        try:
            response = await self.post(
                path=self._construct_path("userfeedback"),
                json={
                    "thread_id": thread_id,
                    "feedback_index": feedback_index,
                    "feedback": feedback,
                },
                stream=False,
            )
            response = cast(httpx.Response, response)
            response_dict: Dict[str, str] = response.json()
            message: str = response_dict.get(
                "detail",
                "Empty message was returned. User feedback was possibly not correctly processed by the backend.",
            )
            return message
        except ConnectionError as e:
            if e.errno == 404:
                err_str = str(e.strerror or "")
                if "thread not found" in err_str.lower():
                    raise ValueError(f"No thread found for id '{thread_id}'.")
                elif "feedback not found" in err_str.lower():
                    raise IndexError(
                        f"Feedback not found at index {feedback_index} for thread '{thread_id}'."
                    )
            elif e.errno == 422:
                raise IndexError(f"Index {feedback_index} is out of bounds.")
            elif e.errno in (500, 503):
                raise ConnectionError("Error on the backend saving/modifying feedback.")
            raise

    def _construct_path(self, endpoint_name: str) -> str:
        """Constructs the full API path for an endpoint.

        Args:
            endpoint_name: Name of the endpoint from CLIMATECLAW_API_ENDPOINTS.

        Returns:
            Full path string combining root API path and endpoint path.
        """
        return f"{self._root_api_path}/{CLIMATECLAW_API_ENDPOINTS[endpoint_name]}"
