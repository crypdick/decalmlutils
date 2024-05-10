from decalmlutils.io import git_


def mock_git(mocker):
    git_ls_mock = mocker.patch.object(git_.git, "cmd")
    git_ls_mock.Git().ls_remote.return_value = "\n".join(
        [f"cmt_{i*1234567890}\trefs/heads/branch_{i}" for i in range(1, 10)]
    )
    return git_ls_mock


def test_get_current_commit():
    commit = git_.get_current_local_commit()
    assert len(commit) == 12

    commit = git_.get_current_local_commit(length=15)
    assert len(commit) == 15


def test_get_current_branch():
    git_.get_current_local_branch()
