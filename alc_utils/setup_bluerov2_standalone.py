import os
import distutils
import distutils.dir_util
import shutil
from alc_utils import setup_repo
from pathlib import Path

src = os.path.expandvars('$ALC_HOME/bluerov2_standalone')
repo_info = 'bluerov2_standalone'
r = setup_repo.RepoSetup()
r.create_new_repo(repo_info, src, '', True)  # create repo with content
