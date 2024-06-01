from hypothesis import settings, HealthCheck

settings.register_profile(
    "default",
    deadline=1000,
    max_examples=1000,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "crunch",
    parent=settings.default,
    max_examples=5000,
)
