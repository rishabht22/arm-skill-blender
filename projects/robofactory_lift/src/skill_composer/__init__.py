"""Learn coordination from RoboFactory demonstrations, keeping execution separate."""

from pkgutil import extend_path

# The cell and goal-conditioned experiments retain their public submodule names.
__path__ = extend_path(__path__, __name__)
