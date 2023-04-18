import os
import distutils
import distutils.dir_util
import shutil
from alc_utils import setup_repo
from pathlib import Path

repo_info = 'alc_core'
clone_dst_root = os.path.expandvars('$ALC_WORKING_DIR/.exec')
clone_dst = Path(os.path.join(clone_dst_root, repo_info))
r = setup_repo.RepoSetup()
r.clone_repo(clone_dst, repo_info, 'master', '', '', True)  # clone
r.build_repo(str(clone_dst), repo_info, '')
