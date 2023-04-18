import unittest
import tempfile
import sys
import shutil
import atexit
from pathlib import Path

sys.path.append("../../main/python")

tempDir = tempfile.mkdtemp()
historyDataPath = Path(tempDir, "HistoryData")
shutil.copytree("../resources/HistoryData", str(historyDataPath))

sys.argv = [sys.argv[0], "--__workflow_dir__", str(historyDataPath)]
from HistoryData import parameter_output, history_data, _job_info_key

assert len(sys.argv) == 1


class MyTestCase(unittest.TestCase):

    def test_something(self):

        tempDir = tempfile.mkdtemp()

        shutil.rmtree(tempDir)

        raw_history_data = history_data.history_data
        self.assertEqual({"Job-1", "Job-2", "Job-3"}, set(raw_history_data.keys()))


        job1_loop_history = raw_history_data.get("Job-1")
        self.assertEqual(1, len(job1_loop_history))

        job1_loop_data_1 = job1_loop_history[0]
        self.assertEqual(2, len(job1_loop_data_1))

        for iteration_data in job1_loop_data_1:
            self.assertEqual({"Job-1"}, set(iteration_data[_job_info_key].keys()))


        job2_loop_history = raw_history_data.get("Job-2")
        self.assertEqual(2, len(job2_loop_history))

        job2_loop_data_1 = job2_loop_history[0]
        self.assertEqual(2, len(job2_loop_data_1))

        for iteration_data in job2_loop_data_1:
            self.assertEqual({"Job-1", "Job-2"}, set(iteration_data[_job_info_key].keys()))

        job2_loop_data_2 = job2_loop_history[1]
        self.assertEqual(2, len(job2_loop_data_2))

        for iteration_data in job2_loop_data_2:
            self.assertEqual({"Job-1", "Job-2"}, set(iteration_data[_job_info_key].keys()))


        job3_loop_history = raw_history_data.get("Job-3")
        self.assertEqual(1, len(job3_loop_history))

        job3_loop_data_1 = job3_loop_history[0]
        self.assertEqual(1, len(job3_loop_data_1))

        for iteration_data in job3_loop_data_1:
            self.assertEqual({"Job-1", "Job-2", "Job-3"}, set(iteration_data[_job_info_key].keys()))


        self.assertEqual({"Job-1", "Job-2", "Job-3"}, history_data.get_job_names())

        job1_loop_history = history_data.get_loop_history_for_job("Job-1")
        self.assertEqual(1, job1_loop_history.get_loop_history_size())

        job1_loop_data_1 = job1_loop_history.get_loop(0)
        self.assertEqual(2, job1_loop_data_1.get_num_iterations())

        for ix in range(0, job1_loop_data_1.get_num_iterations()):
            job1_iteration_data_1_1 = job1_loop_data_1.get_iteration_data(ix)
            self.assertEqual({"Job-1"}, job1_iteration_data_1_1.get_job_names())


        job2_loop_history = history_data.get_loop_history_for_job("Job-2")
        self.assertEqual(2, job2_loop_history.get_loop_history_size())

        job2_loop_data_1 = job2_loop_history.get_loop(1)
        self.assertEqual(2, job2_loop_data_1.get_num_iterations())

        for ix in range(1, job2_loop_data_1.get_num_iterations()):
            job2_iteration_data_1_1 = job2_loop_data_1.get_iteration_data(ix)
            self.assertEqual({"Job-1", "Job-2"}, job2_iteration_data_1_1.get_job_names())

        job2_loop_data_2 = job2_loop_history.get_loop(1)
        self.assertEqual(2, job2_loop_data_2.get_num_iterations())

        for ix in range(0, job2_loop_data_2.get_num_iterations()):
            job2_iteration_data_2_1 = job2_loop_data_2.get_iteration_data(ix)
            self.assertEqual({"Job-1", "Job-2"}, job2_iteration_data_2_1.get_job_names())


        job3_loop_history = history_data.get_loop_history_for_job("Job-3")
        self.assertEqual(1, job3_loop_history.get_loop_history_size())

        job3_loop_data_1 = job3_loop_history.get_loop(0)
        self.assertEqual(1, job3_loop_data_1.get_num_iterations())

        for ix in range(0, job3_loop_data_1.get_num_iterations()):
            job3_iteration_data_1_1 = job3_loop_data_1.get_iteration_data(ix)
            self.assertEqual({"Job-1", "Job-2", "Job-3"}, job3_iteration_data_1_1.get_job_names())


if __name__ == '__main__':
    unittest.main()
    shutil.rmtree(tempDir)
