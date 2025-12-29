"""
Odoo shell tool utilities for executing Python code in an Odoo environment.

This module provides utilities for Odoo shell execution including
output filtering to remove initialization noise.
"""

import re
from typing import Optional

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


def filter_odoo_output(output: str, ssh_mode: bool = False) -> str:
    """Filter Odoo shell initialization noise from output.

    The Odoo shell produces a lot of initialization output before running
    user code. This function detects when actual command output begins
    and returns only that portion.

    Strategy:
    1. For SSH mode, strip any STDERR section first (logs go to stderr)
       and return the cleaned stdout directly
    2. Always show errors (Traceback, exceptions, ODOO_ERROR marker)
    3. Skip all initialization noise (logs, warnings, banners)
    4. For Docker mode, look for shell prompt (In [N]: or >>>) to know shell is ready
    5. After prompt, capture actual command output

    Args:
        output: Raw output from Odoo shell execution
        ssh_mode: If True, output is from SSH where stderr is separate

    Returns:
        Filtered output containing only the relevant command results
    """
    # For SSH mode, the Odoo shell properly separates stdout (user output)
    # from stderr (initialization logs). We just need to strip STDERR section.
    if ssh_mode or "\n\nSTDERR:\n" in output:
        # Split at STDERR marker and take only stdout portion
        if "\n\nSTDERR:\n" in output:
            output = output.split("\n\nSTDERR:\n")[0]
        # Also strip any trailing "STDERR:" at the end
        if output.endswith("\nSTDERR:"):
            output = output[:-8]

        # For SSH mode, the stdout should be clean - just return it
        # (after checking for errors)
        clean_output = output.strip()

        # Check for error marker in the clean output
        if 'ODOO_ERROR:' in clean_output:
            error_idx = clean_output.index('ODOO_ERROR:')
            error_part = clean_output[error_idx + 11:].strip()
            return f"Error: {error_part}"

        # Check for Python exceptions
        if re.match(r'^Traceback \(most recent call last\):', clean_output):
            return clean_output

        # Return the clean stdout
        if clean_output:
            return clean_output

        # If stdout is empty but no error, the command likely succeeded with no output
        return ""

    # Docker mode: need to parse through initialization noise to find user output
    lines = output.split("\n")
    result_lines = []
    found_prompt = False
    capturing_output = False

    # Patterns that indicate shell is ready (user command being entered)
    prompt_patterns = [
        r'^In \[\d+\]:',      # IPython prompt
        r'^>>>',              # Standard Python prompt
        r'^\.\.\.',           # Continuation prompt
    ]

    # Patterns that indicate end of Odoo initialization (Docker mode - no prompts)
    registry_loaded_patterns = [
        r'.*Registry loaded in \d+\.\d+s',
        r'.*Modules loaded\.',
    ]

    # Patterns to always skip (initialization noise)
    noise_patterns = [
        # Odoo log lines with timestamps
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \d+ INFO',
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \d+ WARNING',
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \d+ DEBUG',
        # Python warnings
        r'^/.*\.py:\d+: UserWarning:',
        r'^\s*import pkg_resources',
        r'^\s*The pkg_resources package',
        r'^\s*Refrain from using this package',
        # Profiling errors
        r'^profiling:.*Cannot open',
        r'^profiling:/tmp/.*\.gcda:Cannot open',
        # Custom banners and separators
        r'^Entering Odoo shell',
        r'^-{20,}$',  # Long separator lines (20+ dashes)
        r'^Command:.*odoo.*shell',
        r'^Command:.*odoo-bin',
        # Shell version banners
        r'^Python \d+\.\d+\.\d+',
        r'^IPython \d+\.\d+',
        r'^Type .* for help',
        r'^Tip:',
        # Object display lines from shell
        r'^env:',
        r'^odoo:',
        r'^openerp:',
        r'^self:',
        r'^werkzeug:',
        # Module loading
        r'^.*modules loaded in \d+\.\d+s',
        r'^loading \d+ modules',
        r'^Initiating shutdown',
        r'^Hit CTRL-C',
        r'^Will use the Wkhtmltopdf binary',
        r'^addons paths:',
        r'^database:.*@',
        r'^Odoo version \d+',
    ]

    for line in lines:
        line_stripped = line.rstrip()

        # Check for our custom error marker - always capture
        if 'ODOO_ERROR:' in line_stripped:
            error_part = line_stripped[line_stripped.index('ODOO_ERROR:') + 11:].strip()
            return f"Error: {error_part}"

        # Check for Python exceptions - always capture
        if any(re.match(pattern, line_stripped) for pattern in [
            r'^Traceback \(most recent call last\):',
            r'^\w+Error:',
            r'^\w+Exception:',
        ]):
            result_lines.append(line_stripped)
            capturing_output = True
            continue

        # If we're already capturing output due to an error, continue capturing
        if capturing_output and result_lines and any(
            re.match(pattern, result_lines[0]) for pattern in [
                r'^Traceback \(most recent call last\):',
                r'^\w+Error:',
                r'^\w+Exception:',
            ]
        ):
            # Continue capturing traceback lines
            if line_stripped.startswith('  ') or re.match(r'^\w+Error:', line_stripped) or re.match(r'^\w+Exception:', line_stripped):
                result_lines.append(line_stripped)
                continue
            # End of traceback
            if not line_stripped:
                continue

        # Skip noise patterns always
        if any(re.match(pattern, line_stripped) for pattern in noise_patterns):
            continue

        # Skip empty lines during initialization
        if not found_prompt and not line_stripped:
            continue

        # Look for shell prompt - indicates shell is ready
        if not found_prompt:
            if any(re.match(pattern, line_stripped) for pattern in prompt_patterns):
                found_prompt = True
                # Skip the prompt line itself, but start capturing after
                continue
            # Still in initialization, skip everything
            continue

        # After finding prompt, look for start of actual output
        if found_prompt and not capturing_output:
            # Skip empty lines and continuation prompts
            if not line_stripped or any(re.match(pattern, line_stripped) for pattern in prompt_patterns):
                continue
            # Skip Out[N]: prefix but capture remaining content
            if re.match(r'^Out\[\d+\]:', line_stripped):
                remaining = re.sub(r'^Out\[\d+\]:\s*', '', line_stripped)
                if remaining:
                    result_lines.append(remaining)
                capturing_output = True
                continue
            # This looks like actual output
            capturing_output = True
            result_lines.append(line_stripped)
            continue

        # Once we're capturing, include everything except profiling noise
        if capturing_output:
            # Skip profiling errors
            if re.match(r'^profiling:.*Cannot open', line_stripped):
                continue
            result_lines.append(line_stripped)

    # Docker mode fallback: if no prompts found, look for content after "Registry loaded"
    if not result_lines and not found_prompt:
        registry_loaded_idx = None
        for i, line in enumerate(lines):
            line_stripped = line.rstrip()
            if any(re.match(pattern, line_stripped) for pattern in registry_loaded_patterns):
                registry_loaded_idx = i
                # Don't break - take the LAST occurrence of registry loaded

        if registry_loaded_idx is not None:
            # Capture all non-empty, non-noise lines after "Registry loaded"
            for line in lines[registry_loaded_idx + 1:]:
                line_stripped = line.rstrip()
                if not line_stripped:
                    continue
                # Skip noise patterns
                if any(re.match(pattern, line_stripped) for pattern in noise_patterns):
                    continue
                # Skip profiling errors
                if re.match(r'^profiling:.*Cannot open', line_stripped):
                    continue
                result_lines.append(line_stripped)

    result = "\n".join(result_lines).strip()

    # Clean up any remaining shell artifacts
    result = re.sub(r'^In \[\d+\]:\s*', '', result, flags=re.MULTILINE)
    result = re.sub(r'^Out\[\d+\]:\s*', '', result, flags=re.MULTILINE)
    result = re.sub(r'^\.\.\.:?\s*', '', result, flags=re.MULTILINE)

    return result


def build_wrapped_code(code: str) -> str:
    """Wrap user code with env setup and error handling.

    Args:
        code: User's Python code to execute in Odoo shell

    Returns:
        Wrapped code with proper indentation and error handling
    """
    # Indent each line of user code by 4 spaces
    indented_code = "\n".join("    " + line for line in code.strip().split("\n"))

    wrapped_code = f'''
env = self.env
try:
{indented_code}
except Exception as e:
    print(f"ODOO_ERROR: {{type(e).__name__}}: {{e}}")
'''
    return wrapped_code


def build_local_shell_command(
    python_bin: str,
    odoo_bin: str,
    database: str,
    config_path: Optional[str] = None
) -> list[str]:
    """Build command for local Odoo shell execution.

    Args:
        python_bin: Path to Python binary
        odoo_bin: Path to Odoo binary
        database: Database name
        config_path: Optional path to Odoo config file

    Returns:
        Command list for subprocess execution
    """
    cmd = [
        python_bin, odoo_bin, "shell",
        "-d", database,
        "--no-http",
        "--shell-interface=ipython",
    ]
    if config_path:
        cmd.extend(["-c", config_path])
    return cmd


def build_ssh_shell_command(
    ssh_host: str,
    ssh_user: str,
    ssh_key_path: str,
    remote_python: str,
    remote_odoo_bin: str,
    database: str,
    config_path: Optional[str] = None,
    ssh_port: int = 22
) -> list[str]:
    """Build command for remote Odoo shell execution via SSH.

    Args:
        ssh_host: SSH host address
        ssh_user: SSH username
        ssh_key_path: Path to SSH private key
        remote_python: Path to Python binary on remote host
        remote_odoo_bin: Path to Odoo binary on remote host
        database: Database name
        config_path: Optional path to Odoo config file on remote host
        ssh_port: SSH port (default 22)

    Returns:
        Command list for subprocess execution
    """
    # Build remote odoo command
    odoo_cmd = f"{remote_python} {remote_odoo_bin} shell -d {database} --no-http --shell-interface=ipython"
    if config_path:
        odoo_cmd += f" -c {config_path}"

    cmd = [
        "ssh",
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        f"{ssh_user}@{ssh_host}",
        odoo_cmd
    ]
    return cmd
