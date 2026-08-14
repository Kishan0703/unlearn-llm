from .config import UnlearnConfig


def unlearn(config: UnlearnConfig):
    from .pipeline import unlearn as run_unlearn

    return run_unlearn(config)
