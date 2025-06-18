#!/usr/bin/env python3
"""
Import checker script for megatron.hub package.

This script recursively discovers all Python modules in the specified package
and attempts to import them, reporting any import errors.
"""

import os
import sys
import importlib
import pkgutil
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import argparse
from megatron.hub.utils.import_utils import UnavailableError

class ImportChecker:
    """Check imports for all modules in a package."""

    def __init__(self, package_name: str = "megatron.hub", verbose: bool = False):
        self.package_name = package_name
        self.verbose = verbose
        self.success_count = 0
        self.failure_count = 0
        self.skipped_count = 0
        self.failures: Dict[str, str] = {}
        self.successes: List[str] = []
        self.skipped: List[str] = []

        # Modules to skip (known problematic ones)
        self.skip_patterns = {
            "__pycache__",
            ".pytest_cache",
            ".git",
            "test_",
            "_test",
        }

        # Add current directory to Python path if not already there
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

    def should_skip_module(self, module_name: str) -> bool:
        """Check if a module should be skipped."""
        for pattern in self.skip_patterns:
            if pattern in module_name:
                return True
        return False

    def discover_modules(self, package_path: str) -> List[str]:
        """Discover all Python modules in the given package path."""
        modules = []

        try:
            # Import the main package first
            package = importlib.import_module(self.package_name)
            package_path = package.__path__[0]

            # Walk through all Python files
            for root, dirs, files in os.walk(package_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

                for file in files:
                    if file.endswith(".py") and not file.startswith("."):
                        # Convert file path to module name
                        rel_path = os.path.relpath(os.path.join(root, file), package_path)
                        module_parts = rel_path.replace(os.sep, ".").replace(".py", "")

                        # Handle __init__.py files
                        if module_parts.endswith(".__init__"):
                            module_parts = module_parts[:-9]  # Remove .__init__

                        full_module_name = f"{self.package_name}.{module_parts}" if module_parts else self.package_name

                        if not self.should_skip_module(full_module_name):
                            modules.append(full_module_name)

            # Also try to discover using pkgutil
            try:
                for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                    if not self.should_skip_module(modname):
                        modules.append(modname)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: pkgutil discovery failed: {e}")

            # Remove duplicates and sort
            modules = sorted(list(set(modules)))

        except ImportError as e:
            print(f"Error: Could not import package '{self.package_name}': {e}")
            return []

        return modules

    def import_module(self, module_name: str) -> Tuple[bool, str]:
        """
        Try to import a module and return success status and error message.

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Clear any existing module from sys.modules to force reimport
            if module_name in sys.modules:
                del sys.modules[module_name]

            importlib.import_module(module_name)
            return True, ""
        except UnavailableError as e:
            return True, f"Missing but gracefully handled: {str(e)}"
        except ImportError as e:
            return False, f"ImportError: {str(e)}"
        except ModuleNotFoundError as e:
            return False, f"ModuleNotFoundError: {str(e)}"
        except Exception as e:
            # Capture full traceback for other errors
            tb = traceback.format_exc()
            return False, f"{type(e).__name__}: {str(e)}\n{tb}"

    def check_all_imports(self) -> None:
        """Check imports for all discovered modules."""
        print(f"Discovering modules in package '{self.package_name}'...")
        modules = self.discover_modules(self.package_name)

        if not modules:
            print("No modules found!")
            return

        print(f"Found {len(modules)} modules to check")
        print("=" * 60)

        for i, module_name in enumerate(modules, 1):
            if self.verbose:
                print(f"[{i}/{len(modules)}] Checking {module_name}...", end=" ")

            success, error_msg = self.import_module(module_name)

            if success:
                self.success_count += 1
                self.successes.append(module_name)
                if self.verbose:
                    print("✓ OK")
            else:
                self.failure_count += 1
                self.failures[module_name] = error_msg
                if self.verbose:
                    print("✗ FAILED")
                    print(f"    Error: {error_msg.split(chr(10))[0]}")  # First line only

    def print_summary(self) -> None:
        """Print a summary of the import check results."""
        total = self.success_count + self.failure_count + self.skipped_count

        print("\n" + "=" * 60)
        print("IMPORT CHECK SUMMARY")
        print("=" * 60)
        print(f"Total modules checked: {total}")
        print(f"Successful imports:    {self.success_count} ({self.success_count / total * 100:.1f}%)")
        print(f"Failed imports:        {self.failure_count} ({self.failure_count / total * 100:.1f}%)")
        if self.skipped_count > 0:
            print(f"Skipped modules:       {self.skipped_count} ({self.skipped_count / total * 100:.1f}%)")

        if self.failures:
            print(f"\n❌ FAILED IMPORTS ({len(self.failures)}):")
            print("-" * 40)
            for module_name, error_msg in self.failures.items():
                print(f"\n• {module_name}")
                # Show only the first few lines of error to keep output manageable
                error_lines = error_msg.split("\n")[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"  {line}")
                if len(error_msg.split("\n")) > 3:
                    print("  ...")

        if self.successes and self.verbose:
            print(f"\n✅ SUCCESSFUL IMPORTS ({len(self.successes)}):")
            print("-" * 40)
            for module_name in self.successes:
                print(f"• {module_name}")

    def get_exit_code(self) -> int:
        """Return appropriate exit code based on results."""
        return 1 if self.failure_count > 0 else 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check that all modules in a package can be imported")
    parser.add_argument(
        "--package", "-p", default="megatron.hub", help="Package name to check (default: megatron.hub)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show summary (overrides verbose)")

    args = parser.parse_args()

    # Set verbosity
    verbose = args.verbose and not args.quiet

    try:
        checker = ImportChecker(package_name=args.package, verbose=verbose)
        checker.check_all_imports()
        checker.print_summary()

        return checker.get_exit_code()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
