from engine.llm.cache import CacheContext, cache_key


def test_cache_key_changes_with_batch_size() -> None:
    text = "same comment"
    ctx5 = CacheContext(prompt_version="moderation-v1", batch_size=5, model_chain="m1|m2")
    ctx15 = CacheContext(prompt_version="moderation-v1", batch_size=15, model_chain="m1|m2")
    assert cache_key(text, ctx5) != cache_key(text, ctx15)


def test_cache_key_changes_with_model_chain() -> None:
    text = "same comment"
    ctx_a = CacheContext(prompt_version="moderation-v1", batch_size=15, model_chain="a")
    ctx_b = CacheContext(prompt_version="moderation-v1", batch_size=15, model_chain="b")
    assert cache_key(text, ctx_a) != cache_key(text, ctx_b)


def test_sweep_tag_isolates_batch_sweep() -> None:
    text = "same comment"
    prod = CacheContext(prompt_version="moderation-v1", batch_size=15, model_chain="m1")
    sweep5 = CacheContext(
        prompt_version="moderation-v1", batch_size=15, model_chain="m1", sweep_tag="sweep-bs-5"
    )
    sweep25 = CacheContext(
        prompt_version="moderation-v1", batch_size=15, model_chain="m1", sweep_tag="sweep-bs-25"
    )
    assert cache_key(text, prod) != cache_key(text, sweep5)
    assert cache_key(text, sweep5) != cache_key(text, sweep25)


def test_configured_batch_size_not_chunk_length_in_key() -> None:
    """Sweep chunk size must not appear in key when batch_size is fixed at configured 15."""
    text = "c"
    a = CacheContext(prompt_version="v", batch_size=15, model_chain="m", sweep_tag="sweep-bs-5")
    b = CacheContext(prompt_version="v", batch_size=15, model_chain="m", sweep_tag="sweep-bs-10")
    assert a.batch_size == b.batch_size == 15
    assert cache_key(text, a) != cache_key(text, b)