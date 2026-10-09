# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Converter errors and the exit code each one maps to."""

from __future__ import annotations

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNSUPPORTED = 3
EXIT_EXPORT = 4
EXIT_VALIDATION = 5
EXIT_DOWNLOAD = 6


class ConverterError(Exception):
    exit_code = EXIT_EXPORT


class UsageError(ConverterError):
    """Bad arguments or inputs the user can fix (exit 2)."""

    exit_code = EXIT_USAGE


class UnsupportedModelError(ConverterError):
    """The model cannot be converted (exit 3)."""

    exit_code = EXIT_UNSUPPORTED


class ExportError(ConverterError):
    """The export or the post-export privacy checks failed (exit 4)."""

    exit_code = EXIT_EXPORT


class ValidationError(ConverterError):
    """The written pack failed its own validation (exit 5)."""

    exit_code = EXIT_VALIDATION


class DownloadError(ConverterError):
    """A download failed, was refused or did not match its published hash (exit 6)."""

    exit_code = EXIT_DOWNLOAD
