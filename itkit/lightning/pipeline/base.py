class BaseTransform:
    def __call__(self, sample: dict) -> dict|list[dict]:
        raise NotImplementedError("Subclasses should implement this method.")
