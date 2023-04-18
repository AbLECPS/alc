import sys
import subprocess
import time
from pathlib import Path


slurmrestd_exec_file_path = Path("/usr/sbin/slurmrestd")
unix_socket_file_path = Path("/tmp/slurm.sock")

current_dir_path = Path("/alc/webgme")

log_file_path = Path(current_dir_path, "slurmrestd.log")

max_unix_socket_iterations = 10


def monitor_slurmrestd():

    with log_file_path.open("a") as log_file_fp:
        print("slurmrestd_monitor:  starting slurmrestd ...", file=log_file_fp)

    while True:

        if unix_socket_file_path.exists():
            unix_socket_file_path.unlink()

        with log_file_path.open("a") as log_file_fp:
            slurmrestd_process = subprocess.Popen(
                [str(slurmrestd_exec_file_path), "unix:{0}".format(str(unix_socket_file_path))],
                cwd=str(current_dir_path),
                stdout=log_file_fp,
                stderr=subprocess.STDOUT
            )

        unix_socket_iterations = 0
        while not unix_socket_file_path.exists() and unix_socket_iterations < max_unix_socket_iterations:
            unix_socket_iterations += 1
            time.sleep(1)

        if not unix_socket_file_path.exists():
            print(
                "slurmrestd_monitor:  unix socket \"{0}\" for slurmrestd ".format(unix_socket_file_path) +
                "should have been created by slurmrestd but wasn't.  Terminating.",
                file=log_file_fp
            )
            slurmrestd_process.terminate()
            sys.exit(1)

        while True:
            if slurmrestd_process.poll():
                with log_file_path.open("a") as log_file_fp:
                    print(
                        "slurmrestd_monitor:  ERROR:  slurmrestd has unexpectedly terminated!  Restarting slurmrestd",
                        file=log_file_fp
                    )
                break

            if not unix_socket_file_path.exists():
                with log_file_path.open("a") as log_file_fp:
                    print(
                        "slurmrestd_monitor:  ERROR:  unix socket \"{0}\" ".format(unix_socket_file_path) +
                        "for slurmrestd has been removed!  Restarting slurmrestd",
                        file=log_file_fp
                    )
                slurmrestd_process.terminate()
                slurmrestd_process.wait()
                break

            time.sleep(1)


monitor_slurmrestd()
