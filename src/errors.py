"""Project-specific exceptions so error messages are clear for beginners."""


class ProjectError(Exception):
    """Base class for all expected, user-fixable errors."""


class ConfigError(ProjectError):
    """The configuration file is missing, malformed or inconsistent."""


class DataError(ProjectError):
    """An input data file is missing, malformed or too short."""
