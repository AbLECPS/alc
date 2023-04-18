import os
import distutils
import distutils.dir_util
import shutil
from alc_utils import setup_repo
from pathlib import Path

src = ''
repo_info = 'alc_core'
r = setup_repo.RepoSetup()
r.create_new_repo(repo_info, src, '', True)  # create repo
