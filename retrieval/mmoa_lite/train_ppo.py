"""Deprecated CLI wrapper; use ``python -m affectagent.train_ppo``."""

from affectagent.train_ppo import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
