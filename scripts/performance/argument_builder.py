import argparse
import os
from pathlib import Path

from argument_parser import parse_cli_args


def list_of_strings(arg):
    """Split a comma-separated string into a list of substrings."""
    return arg.split(",")


def normalize_arg_name(arg_name: str) -> str:
    """
    Normalizes a command-line argument name (e.g., '--model_family_name' or '-m')
    into a suitable environment variable name (e.g., 'MODEL_FAMILY_NAME').
    """
    name = arg_name.lstrip("-")
    name = name.upper()
    name = name.replace("-", "_")
    return name


def build_cli_args_from_env_vars(parser: argparse.ArgumentParser) -> str:
    """
    Inspects an argparse.ArgumentParser, checks for corresponding environment
    variables, and constructs a CLI argument string from them.
    """
    cli_arg_string = []

    for action in parser._actions:
        if action.option_strings:
            long_arg_name = action.option_strings[-1]
            env_var_name = normalize_arg_name(long_arg_name)
            env_value = os.getenv(env_var_name)

            if env_value is not None:
                if isinstance(action, argparse._StoreTrueAction):
                    is_true = env_value.lower() in ("true", "1", "yes", "on")
                    if is_true:
                        cli_arg_string.append(long_arg_name)
                    continue
                elif action.type is list_of_strings:
                    if env_value:
                        cli_arg_string.append(long_arg_name)
                        cli_arg_string.append(env_value)
                    continue
                else:
                    cli_arg_string.append(long_arg_name)
                    cli_arg_string.append(env_value)

    return " ".join(cli_arg_string)


if __name__ == "__main__":
    cli_args_string = build_cli_args_from_env_vars(parse_cli_args())
    print(cli_args_string)
