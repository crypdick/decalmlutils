import matplotlib.pyplot as plt
import pytest


@pytest.mark.skip(
    reason="run manually only since this actually sends requests to slack"
)
def test_send_plot():
    from decalmlutils.slack.slack import BOT_TEST_CHANNEL, Slacker

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(range(10), "b-")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(range(20), "r^")

    bot = Slacker()
    bot.send_file(fig, BOT_TEST_CHANNEL, "a plot with a caption")
