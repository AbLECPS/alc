import os
import distutils
import distutils.dir_util
import shutil
from alc_utils import setup_repo
from pathlib import Path
archive_repo_info = 'bluerov2_standalone'
root_path = os.path.expandvars('$ALC_WORKING_DIR/archive/')
root_path = os.path.join(root_path, archive_repo_info)
archive_dst = Path(root_path)
archive_filename = os.path.join(root_path, 'repo.tar.gz')
r = setup_repo.RepoSetup()
r.archive_repo(archive_filename, archive_dst,
               archive_repo_info, 'master', '', '')
