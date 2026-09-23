import hashlib


def compute_short_ref_hash(ref: str) -> str:
    """Short hash identifying a config/prompt source ref for cache-key scoping.

    Used by ``ConfigReader`` and ``PromptStore`` so cache keys are scoped by
    source ref (e.g. mid-rollout, where a pod's config path still points at a
    stale ref) and never read or write another ref's cached content under the
    same key -- see BAI-720.
    """
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()[:12]
