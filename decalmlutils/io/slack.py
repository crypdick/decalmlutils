import tempfile
from unittest.mock import MagicMock

from beartype import beartype
from beartype.typing import TYPE_CHECKING, Dict, Optional
from slack_sdk import WebClient

from decalmlutils.io.aws.secretsmanager import get_secret
from decalmlutils.io.sort import natural_sort

if TYPE_CHECKING:
    import matplotlib

BOT_TOKEN_SECRET_NAME = "ml-slack-token"
ML_ALERTS_CHANNEL = "ml-alerts"
ML_INFERENCE_ALERTS_CHANNEL = "ml-inference-alerts"
ML_TRAINING_ALERTS_CHANNEL = "ml-training-alerts"
PRED_FREQ_ALERTS_CHANNEL = "ml-pred-frequency-alerts"
ML_MODEL_STATS_CHANNEL = "ml-model-stats"
TAXONOMY_ERRORS_CHANNEL = "ml-taxonomy-errors"
METRICS_ERRORS_CHANNEL = "ml-cloudwatch-metrics-errors"
ML_SLACKER_TEST_CHANNEL = "ml-slacker-test"
BOT_TEST_CHANNEL = "bot-test-channel"
# info on how to mention user groups in slack:
# https://api.slack.com/reference/surfaces/formatting#mentioning-groups
MENTION_ML_DEVELOPERS = "<!subteam^FOOBAR>"


# This is a class instead of a function so that you can re-use the token and client
# since those take a few seconds to fetch, and the secret costs AWS money
class Slacker:
    """
    Send plots as "ML AUTOMATION BOT" in Slack.

    Attributes:
        client: slack sdk web client for sending files and messages to slack
    """

    def __init__(self, is_prod: bool = True):
        """
        Fetch the token and create the client right away and save them to save time when sending multiple plots.
        """
        if is_prod:
            _token = get_secret(BOT_TOKEN_SECRET_NAME)
            self.client = WebClient(_token)
        else:
            self.client = MagicMock()
            self.client.chat_postMessage = MagicMock()
            self.client.files_upload = MagicMock()

    @beartype
    def send_message(self, channel: str, message: str):
        """
        Send a message to slack.

        Args:
            channel (string): Name of a slack channel to send to.
            message (string): text to appear in the message
        """

        self.client.chat_postMessage(channel=channel, text=f"{message}")
        print(f"#{channel}: {message}")

    @beartype
    def send_file(
        self,
        channel: str,
        fig: Optional["matplotlib.figure.Figure"] = None,
        fpath: Optional[str] = None,
        fig_bytes: Optional[bytes] = None,
        message: str = "",
    ):
        """
        Send a file to slack.

        Args:
            fig (Matplotib Figure): Figure to send to slack.
            fpath (string): Path to a file that will be sent.
            fig_bytes (bytes): Bytes of a file that will be sent.
            channel (string): Name of a slack channel to send to.
            message (string): text to appear before the image.
        """
        if fig is None and fpath is None and fig_bytes is None:
            raise ValueError(
                "Must provide either a figure or a file path to send to slack."
            )
        if fig is not None and fpath is not None and fig_bytes is not None:
            raise ValueError(
                "Must provide either a figure or a file path to send to slack, not both."
            )

        if fpath is not None:
            try:
                file_upload_response = self.client.files_upload(
                    channel=channel, file=fpath
                )
            except Exception as e:
                print(f"Error sending file {fpath} to slack: {e}")
                return
        elif fig is not None:
            with tempfile.NamedTemporaryFile() as tmp:
                fig.savefig(tmp, format="png")

                # This creates a warning for us to use files_upload_v2.
                # However, v2 gives SlackApiError with error 'channel_not_found'
                file_upload_response = self.client.files_upload(
                    channel=channel, file=tmp.name
                )
        elif fig_bytes is not None:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(fig_bytes)
                tmp.seek(0)
                file_upload_response = self.client.files_upload(
                    channel=channel, file=tmp.name
                )

        image_url = file_upload_response["file"]["permalink"]
        self.client.chat_postMessage(channel=channel, text=f"{message}\n{image_url}")
        print(f"#{channel}: {message}\n{image_url}")

    @beartype
    def send_multiple_files(
        self,
        channel: str,
        file_info: list[Dict],
        initial_comment: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """
        Send multiple files to slack at once.

        See https://github.com/slackapi/python-slack-sdk/releases/tag/v3.19.0 for more info

        Args:
            channel: the slack channel
            file_info: a list of dicts with the info for each file. Each dict must have a 'file' key with the file path.
                Other keys are optional, including: `title`, `content`
            initial_comment: a message to send with the files
        """
        print(
            "WARNING: this method runs to completion and an `ok` response, but does not appear to actually upload "
            "any files to Slack"
        )

        # validate files
        for fi in file_info:
            assert (
                "file" in fi.keys()
            ), "each file_info must have a 'file' key with the file path"
            if "title" in fi.keys():
                assert isinstance(
                    fi["title"], str
                ), f"title must be a string, got {fi['title']}"

        # files_upload_v2 does not support channel names. You have to use the channel ID, which is insane and slow.
        # https://github.com/slackapi/python-slack-sdk/issues/1326
        # return all non-archived channels. set high limit to avoid pagination
        resp = self.client.conversations_list(exclude_artchived=True, limit=9999999)
        channel_id = None
        channels_found = []
        for channel_info in resp.data["channels"]:
            if channel_info["name"] == channel:
                channel_id = channel_info["id"]
                break
            else:
                channels_found.append(channel_info["name"])
        if channel_id is None:
            raise ValueError(
                f"Could not find channel {channel}. Available channels: {natural_sort(channels_found)}"
            )

        # upload files
        resp = self.client.files_upload_v2(
            channel=channel_id,
            initial_comment=initial_comment,
            title=title,
            file_uploads=file_info,
        )
        print(f"Uploaded {len(file_info)} files to #{channel}. Response: {resp['ok']}")
