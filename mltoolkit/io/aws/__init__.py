import boto3

from mltoolkit.conf import settings


def get_aws_client(resource: str, **kwargs):
    REGION_MAP = {"sns": settings.AWS_SNS_REGION, "lambda": settings.AWS_LAMBDA_REGION}

    kwargs.setdefault("region_name", REGION_MAP.get(resource, settings.AWS_REGION))
    return boto3.client(resource, **kwargs)
