/* globals define, $ */
define([
    'deepforge/ExecutionEnv',
    'q',
    'deepforge/viz/Utils',
    'deepforge/api/JobOriginClient',
    'text!./WorkerModal.html',
    'text!./WorkerTemplate.html.ejs',
    'text!./WorkerJobItem.html',
    'css!./WorkerModal.css'
], function(
    ExecutionEnv,
    Q,
    utils,
    JobOriginClient,
    WorkerHtml,
    WorkerTemplate,
    WorkerJobItem
) {
    'use strict';

    var WorkerDialog = function(logger) {
        this.workerDict = {};
        this.workers = {};
        this.runningWorkers = [];
        this.jobsDict = {};
        this.jobs = {};
        this.active = false;
        this.logger = logger.fork('WorkerDialog');
        this.originManager = new JobOriginClient({
            logger: this.logger
        });
    };

    WorkerDialog.prototype.initialize = function() {
        this._dialog = $(WorkerHtml);
        this._table = this._dialog.find('.worker-list');
        this.$noJobs = this._dialog.find('.no-jobs-msg');
        this.$noWorkers = this._dialog.find('.no-workers-msg');
        this._isShowingJobs = false;
        this._isShowingWorkers = true;
        this._queue = this._dialog.find('.job-queue-list');
        this._dialog.modal('show');
        this._dialog.on('hidden.bs.modal', () => this.active = false);
    };

    WorkerDialog.prototype.show = function() {
        this.active = true;
        this.update();
        this.initialize();
    };

    WorkerDialog.prototype.update = function() {
        // Poll the workers
        return Q.all([
            ExecutionEnv.getWorkers().then(workers => this.updateWorkers(workers)),
            ExecutionEnv.getJobs().then(jobs => this.updateJobs(jobs))
        ]).then(() => {
            if (this.active) {
                setTimeout(this.update.bind(this), 1000);
            }
        })
        .catch(err => this.logger.error('Update failed:', err));
    };

    WorkerDialog.prototype.updateWorkers = function(workerDict) {
        var ids = Object.keys(workerDict),
            oldWorkerIds,
            visibleWorkers = false,
            i;

        this.runningWorkers = [];
        for (i = ids.length; i--;) {
            this.updateWorker(workerDict[ids[i]]);
            visibleWorkers = true;
            delete this.workerDict[ids[i]];
        }
        this.toggleNoWorkersMsg(!visibleWorkers);

        // Clear old workers
        oldWorkerIds = Object.keys(this.workerDict);
        for (i = oldWorkerIds.length; i--;) {
            this.removeWorker(oldWorkerIds[i]);
        }

        this.workerDict = workerDict;
    };

    WorkerDialog.prototype.updateWorker = function(worker) {
        var row = this.workers[worker.clientId] || $(WorkerTemplate),
            clazz;

        worker.lastSeen = utils.getDisplayTime(worker.lastSeen*1000);
        worker.status = worker.jobs.length ? 'RUNNING' : 'READY';

        clazz = worker.status === 'RUNNING' ? 'warning' : 'success';
        row[0].className = clazz;

        row.find('.lastSeen').text(worker.lastSeen);
        row.find('.clientId').text(worker.clientId);
        row.find('.status').text(worker.status);
        if (!this.workers[worker.clientId]) {
            this._table.append(row);
            this.workers[worker.clientId] = row;
        }

        if (worker.status === 'RUNNING') {
            this.runningWorkers.push(worker);
        }
    };

    WorkerDialog.prototype.removeWorker = function(workerId) {
        this.workers[workerId].remove();
        delete this.workers[workerId];
    };

    WorkerDialog.prototype.updateJobs = function(jobsDict) {
        var allJobIds = Object.keys(jobsDict),
            hasJobs = false,
            id;

        this.jobsDict = jobsDict;
        for (var i = allJobIds.length; i--;) {
            id = allJobIds[i];
            if (this.jobs[id] || !this.isFinished(id)) {
                hasJobs = this.updateJobItem(id) || hasJobs;
            }
        }
        this.setNoJobsMessage(!hasJobs);  // hide if no queue
    };

    WorkerDialog.prototype.setNoJobsMessage = function(visible) {
        var visibility = visible ? 'inherit' : 'none',
            wasVisible = !this._isShowingJobs;

        if (visible !== wasVisible) {
            this.$noJobs.css('display', visibility);
            this._isShowingJobs = !visible;
        }
    };

    WorkerDialog.prototype.toggleNoWorkersMsg = function(visible) {
        var visibility = visible ? 'inherit' : 'none';

        if (visible !== this._isShowingWorkers) {
            this.$noWorkers.css('display', visibility);
            this._isShowingWorkers = visible;
        }
    };

    WorkerDialog.prototype.isFinished = function(jobId) {
        return this.jobsDict[jobId].status === 'FAILED_TO_EXECUTE' ||
            this.jobsDict[jobId].status === 'SUCCESS' ||
            this.jobsDict[jobId].status === 'CANCELED';
    };

    WorkerDialog.prototype.updateJobItemName = function(jobId) {
        return this.originManager.getOrigin(jobId)
            .then(info => {
                var job = this.jobs[jobId],
                    project = info.project.replace(/^guest\+/, '');

                if (job && this.active) {
                    if (info.branch !== 'master') {
                        project += ' (' + info.branch + ')';
                    }
                    job.find('.job-id').text(info.job);
                    job.find('.execution').text(info.execution);
                    job.find('.project').text(project);
                }
            });
    };

    WorkerDialog.prototype.getWorkerWithJob = function(jobId) {
        var jobs;

        for (var i = this.runningWorkers.length; i--;) {
            jobs = this.runningWorkers[i].jobs;
            for (var j = jobs.length; j--;) {
                if (jobs[j].hash === jobId) {
                    return this.runningWorkers[i].clientId;
                }
            }
        }

        return 'unknown';
    };

    WorkerDialog.prototype.updateJobItem = function(jobId) {
        var job = this.jobs[jobId] || $(WorkerJobItem),
            info = this.jobsDict[jobId],
            createdTime = new Date(info.createTime).getTime(),
            clazz = utils.ClassForJobStatus[info.status.toLowerCase()],
            status = info.status;

        job[0].className = `job-tag ${clazz}`;

        // Add the worker id if running
        if (info.status.toLowerCase() === 'running') {
            var workerId = this.getWorkerWithJob(jobId);
            status += ' (' + workerId + ')';
        }
        job.find('.status').text(status);

        if (!this.jobs[jobId]) {
            job.find('.job-id').text('Loading');
            job.find('.createdAt').text(utils.getDisplayTime(createdTime));
            this.updateJobItemName(jobId);
            this._queue.append(job);
            this.jobs[jobId] = job;
        }

        if (this.isFinished(jobId)) {
            job.remove();
            delete this.jobs[jobId];
            return false;
        }
        return true;
    };

    return WorkerDialog;
});
