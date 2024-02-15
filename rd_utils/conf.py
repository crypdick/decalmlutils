from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Container for default settings.

    Override specific keys using environment variables.

    Usage:
    ```
    from .conf import settings

    print(settings.AWS_REGION)
    ```
    """

    AWS_REGION: str = "us-east-1"
    """
    APP.
    """
    APP_BASEURL: str = Field(pattern=r"https://.+/$", default="https://app.mysite.io/")
    ML_BUCKET: str = "ml-bucket-example"
    """
    EVENT BUS.
    """
    AWS_EVENT_BUS_REGION: str = "us-east-1"
    AWS_EVENT_BUS_ARN: str = "arn:aws:events:us-east-1:123:event-bus/my-event-bus"
    AWS_EVENT_BUS_SOURCE: str = "ml.richard"
    """
    SLACK ALERTS.
    """
    AWS_SNS_REGION: str = "us-east-1"
    SNS_TOPIC_ARN: str = "arn:aws:sns:us-east-1:123:Slacker"
    DEFAULT_SLACK_CHANNEL: str = "ml-microservice-errors"
    ML_ALERTS_CHANNEL: str = "ml-alerts"


settings = Settings()
