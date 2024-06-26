from hypothesis import settings, HealthCheck

settings.register_profile(
    "default",
    deadline=1000,
    max_examples=1000,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "crunch",
    deadline=1000,
    max_examples=10000,
    suppress_health_check=[HealthCheck.too_slow],
)
