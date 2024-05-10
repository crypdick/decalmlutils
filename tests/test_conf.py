from unittest.mock import patch

from decalmlutils.conf import Settings


def test_default_settings():
    settings = Settings()
    assert (
        settings.AWS_EVENT_BUS_ARN
        == "arn:aws:events:us-east-1:123:event-bus/my-event-bus"
    )


def test_override_settings_with_env_vars():
    new_region = "us-west-2"
    with patch.dict("os.environ", {"AWS_REGION": new_region}):
        settings = Settings()
        assert settings.AWS_REGION == new_region
