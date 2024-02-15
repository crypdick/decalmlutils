# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/

import base64
import json
import re

import boto3
from beartype import beartype
from beartype.typing import Dict, Union
from botocore.exceptions import ClientError, NoCredentialsError

from rd_utils.conf import settings

SECRET_NAME_REGEX = re.compile(r"[-/_+=.@!a-zA-Z0-9]+")


@beartype
def get_secret(secret_name: str, jsonify: bool = False) -> Union[str, bytes, Dict]:
    assert bool(re.fullmatch(pattern=SECRET_NAME_REGEX, string=secret_name))

    region_name = settings.AWS_REGION

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        # note: using spinner() here caused problems with Streamlit Dashboards
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except (ClientError, NoCredentialsError) as e:
        raise RuntimeError(
            "Failed to connect to AWS SecretsManager. If this error is thrown in the unit test suite or the "
            "integration test suite, you likely forgot to mock out a class which makes API calls"
        ) from e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
        elif "SecretBinary" in get_secret_value_response:
            secret = base64.b64decode(get_secret_value_response["SecretBinary"])
        else:
            raise ValueError(f"Unexpected response: {get_secret_value_response}")
        if jsonify:
            secret = json.loads(secret)
        return secret
