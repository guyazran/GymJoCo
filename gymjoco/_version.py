# The version is placed here instead of directly in __init__.py to avoid import errors on installation.
# The errors are due to dynamic version loading in pyproject.toml.
# Importing __init__.py directly fails on importing gymnasium, which is not installed yet.
__version__ = '0.0.0'
