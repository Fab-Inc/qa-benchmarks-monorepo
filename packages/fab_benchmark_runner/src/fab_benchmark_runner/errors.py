"""Exceptions shared across the runner."""


class ConfigError(ValueError):
    """Raised when a run config, its inputs or its model registry are invalid."""
