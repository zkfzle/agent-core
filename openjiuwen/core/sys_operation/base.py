import asyncio
import os
from email.policy import default
from functools import wraps
from pathlib import Path
from shlex import shlex
from typing import Optional

from e2b_code_interpreter import Sandbox

from openjiuwen.core.sys_operation.constants import LOGS_DIR, _SANDBOX_CREATE_FUNC, E2B_API_KEY, \
    _SANDBOX_SHOULD_USE_CREATE, _SANDBOX_CREATE_SUPPORTS_API_KEY, _SANDBOX_CREATE_SUPPORTS_TIMEOUT, DEFAULT_TIMEOUT, \
    DEFAULT_TEMPLATE_ID, _SANDBOX_SUPPORTS_API_KEY, _SANDBOX_INIT_SUPPORTS_TIMEOUT, _SANDBOX_TEMPLATE_PARAM, \
    _SANDBOX_CREATE_TEMPLATE_PARAM, _SANDBOX_CONNECT_SUPPORTS_API_KEY


def singleton(cls):
    cache = {}
    @wraps(cls)
    def wrapper(*a, **k):
        if cls not in cache:
            cache[cls] = cls(*a, **k)
        return cache[cls]
    return wrapper


@singleton
class SysOperation:
    def __init__(self):
        self.sandboxes = {}

    @staticmethod
    def _instantiate_sandbox():
        """Create a sandbox instance compatible with both old and new SDKs."""
        if _SANDBOX_SHOULD_USE_CREATE and _SANDBOX_CREATE_FUNC:
            sandbox_kwargs = {}
            if E2B_API_KEY and _SANDBOX_CREATE_SUPPORTS_API_KEY:
                sandbox_kwargs["api_key"] = E2B_API_KEY
            if _SANDBOX_CREATE_SUPPORTS_TIMEOUT:
                sandbox_kwargs["timeout"] = DEFAULT_TIMEOUT
            if (
                    DEFAULT_TEMPLATE_ID
                    and _SANDBOX_CREATE_TEMPLATE_PARAM
                    and _SANDBOX_CREATE_TEMPLATE_PARAM not in ("cls",)
            ):
                sandbox_kwargs[_SANDBOX_CREATE_TEMPLATE_PARAM] = DEFAULT_TEMPLATE_ID
            return Sandbox.create(**sandbox_kwargs)

        sandbox_kwargs = {}
        if E2B_API_KEY and _SANDBOX_SUPPORTS_API_KEY:
            sandbox_kwargs["api_key"] = E2B_API_KEY
        if _SANDBOX_INIT_SUPPORTS_TIMEOUT:
            sandbox_kwargs["timeout"] = DEFAULT_TIMEOUT
        if _SANDBOX_TEMPLATE_PARAM and DEFAULT_TEMPLATE_ID:
            sandbox_kwargs[_SANDBOX_TEMPLATE_PARAM] = DEFAULT_TEMPLATE_ID
        return Sandbox(**sandbox_kwargs)

    @staticmethod
    def _safe_set_timeout(sandbox, timeout: int = DEFAULT_TIMEOUT) -> None:
        """Set the sandbox timeout only if the method exists."""
        if sandbox is None:
            return

        set_timeout = getattr(sandbox, "set_timeout", None)
        if not callable(set_timeout):
            return
        try:
            set_timeout(timeout)
        except Exception as e:
            pass

    async def _create_sandbox(self, sandbox_id: str=None):
        """Create a linux sandbox and get the `sandbox_id` for safely executing commands and running python code. Note that the `sandbox_id` can only be assigned and cannot be manually specified.

        The sandbox may timeout and automatically shutdown. If so, you will need to create a new sandbox.

        IMPORTANT: Do not execute `create_sandbox` and other sandbox tools in the same message. You must wait for `create_sandbox` to return the `sandbox_id`, then use that `sandbox_id` to specify the working sandbox in subsequent messages.

        Returns:
            The `sandbox_id` of the newly created sandbox. You should use this `sandbox_id` to run other tools in the sandbox.
        """
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        max_retries = 1
        sandbox = None
        for attempt_index in range(1, max_retries + 1):
            try:
                sandbox = self._instantiate_sandbox()
                info = sandbox.get_info()
                tmp_files_dir = os.path.join(LOGS_DIR, "e2b_tmp_files")
                os.makedirs(tmp_files_dir, exist_ok=True)
                self.sandboxes[sandbox_id] = info.sandbox_id
                return info.sandbox_id
            except Exception as e:
                if attempt_index == max_retries:
                    raise TimeoutError(
                        f"Failed to create sandbox after {max_retries} attempts: {e}, please retry later.")
                await asyncio.sleep(attempt_index * 2)  # Exponential backoff # ""Exponential"" backoff
            finally:
                self._safe_set_timeout(sandbox)
        return None

    def get_sandbox(self, sandbox_id: str = None):
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        if sandbox_id not in self.sandboxes:
            return self._create_sandbox(sandbox_id)
        else:
            return self.sandboxes[sandbox_id]

    def _connect_to_sandbox(self, sandbox_id: str = None):
        """Connect to a sandbox, passing api_key only when supported."""
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        if sandbox_id in self.sandboxes:
            connect_kwargs = {}
            if _SANDBOX_CONNECT_SUPPORTS_API_KEY:
                connect_kwargs["api_key"] = E2B_API_KEY
            return Sandbox.connect(self.sandboxes[sandbox_id], **connect_kwargs)
        else:
            return None

    async def read_file(self, path: Path, file_format: str = "text", sandbox_id: str = None) -> Optional[str]:
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        try:
            sandbox_client = self._connect_to_sandbox(sandbox_id)
        except Exception as e:
            return (f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is "
                    f"created and the id is correct.")

        max_retries = 5
        for attempt_index in range(1, max_retries + 1):
            try:
                self._safe_set_timeout(sandbox_client)
                content = sandbox_client.read(str(path), file_format)
                return content
            except Exception as e:
                if attempt_index == max_retries:
                    return (f"[ERROR]: Failed to run code in sandbox {sandbox_id} after {max_retries} attempts. "
                            f"Exception type: {type(e).__name__}, Details: {e}.")
                await asyncio.sleep(attempt_index * 2)
            finally:
                self._safe_set_timeout(sandbox_client)
        return None

    async def upload_file(
            self,
            local_file_path: str,
            sandbox_file_path: str,
            sandbox_id: str = None
    ) -> str:
        """Upload a local file to the `/home/user` dir of the sandbox.

        Args:
            sandbox_id: The id of the existing sandbox to update files in. To have a sandbox, use tool `create_sandbox`.
            local_file_path: The local path of the file to upload.
            sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

        Returns:
            The path of the uploaded file in the sandbox if the upload is successful.
        """
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        try:
            sandbox_client = self._connect_to_sandbox(sandbox_id)
        except Exception:
            return f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created and the id is correct."

        try:
            self._safe_set_timeout(
                sandbox_client
            )  # refresh the timeout for each command execution

            # Get the uploaded file path
            uploaded_file_path = os.path.join(
                sandbox_file_path, os.path.basename(local_file_path)
            )

            # Upload the file
            with open(local_file_path, "rb") as f:
                sandbox_client.files.write(uploaded_file_path, f)

            return f"File uploaded to {uploaded_file_path}\n\n[INFO]: For directly reading local files without uploading to sandbox, consider using the `read_file` tool which can read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) directly from local paths or URLs. Note that `read_file` doesn't support files already in the sandbox."
        except Exception as e:
            return f"[ERROR]: Failed to upload file {local_file_path} to sandbox {sandbox_id}: {e}\n\n[INFO]: This tool is for uploading local files to the sandbox. For security reasons, downloading files from sandbox to local system is not supported. Alternatively, consider using the `read_file` tool which can directly read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) from local paths or URLs without uploading to sandbox."
        finally:
            # Set timeout before exit to prevent timeout after function exits
            self._safe_set_timeout(sandbox_client)

    async def run_command(self, command: str, sandbox_id: str = None) -> Optional[str]:
        """Execute a shell command in the linux sandbox.
        The sandbox is already installed with common system packages for the task.

        Args:
            command: The shell command to execute.
            sandbox_id: The id of the existing sandbox to execute the command in. (must be created first via `create_sandbox`).

        Returns:
            A result of the command execution, format like (stderr=..., stdout=..., exit_code=..., error=...)
        """
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        try:
            sandbox_client = self._connect_to_sandbox(sandbox_id)
        except Exception as e:
            return (f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created "
                    f"and the id is correct.")
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                self._safe_set_timeout(
                    sandbox_client,
                )  # refresh the timeout for each command execution
                result = sandbox_client.commands.run(command)

                # Check if command contains package installation commands
                result_str = str(result)
                if "pip install" in command or "apt-get" in command:
                    result_str += "\n\n[PACKAGE INSTALL STATUS]: The system packages and Python packages required for the task have been installed. No need to install them again unless a missing package error occurs during execution."

                return result_str
            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"[ERROR]: Failed to run command after {max_retries} attempts. Exception type: {type(e).__name__}, Details: {e}. \n\n[HINT]: Shell commands can be error-prone. Consider using the `run_python_code` tool instead to accomplish the same task with Python code, which often provides better error handling and more detailed error messages.\n\n[PERMISSION HINT]: You are running as user, not root. If you encounter permission issues, use `sudo` for commands that require administrator privileges (e.g., `sudo apt-get install`, `sudo systemctl`, etc.)."

                    # Add package install status note for failed install commands too
                    if "pip install" in command or "apt-get" in command:
                        error_msg += "\n\n[PACKAGE INSTALL STATUS]: The system packages and Python packages required for the task have been installed. No need to install them again unless a missing package error occurs during execution."

                    return error_msg
                await asyncio.sleep(attempt * 2)  # Exponential backoff
            finally:
                # Set timeout before exit to prevent timeout after function exits
                self._safe_set_timeout(sandbox_client)
        return None

    async def run_python_code(self, code_block: str, sandbox_id: str = None) -> Optional[str]:
        """Run python code in the sandbox and return the execution result.
        The sandbox is already installed with common python packages for the task.

        Args:
            code_block: The python code to run.
            sandbox_id: The id of the existing sandbox to run the code in. (must be created first via `create_sandbox`).

        Returns:
            A result of the command execution, format like (stderr=..., stdout=..., exit_code=..., error=...)
        """
        if sandbox_id is None:
            sandbox_id = "_default_sandbox_id"
        try:
            sandbox = self._connect_to_sandbox(sandbox_id)
        except Exception:
            return (f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created "
                    f"and the id is correct.")

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                self._safe_set_timeout(sandbox)
                command = f'python3 -c {shlex.quote(code_block)}'
                result = sandbox.commands.run(command)
                return str(result)
            except Exception as e:
                if attempt == max_retries:
                    return f"[ERROR]: Failed to run code in sandbox {sandbox_id} after {max_retries} attempts. Exception type: {type(e).__name__}, Details: {e}."
                await asyncio.sleep(attempt * 2)
            finally:
                self._safe_set_timeout(sandbox)
        return None