/*globals define*/
/*jshint node:true, browser:true*/

/**
 * Generated by PluginGenerator 0.14.0 from webgme on Thu Apr 07 2016 06:13:30 GMT-0500 (CDT).
 */

define([
    'plugin/PluginBase',
    'executor/ExecutorClient',
    'deepforge/api/ExecPulseClient',
    'text!./metadata.json',
	'underscore',
	'q',
	'plugin/util'],
	function (
        PluginBase,
        ExecutorClient,
        ExecPulseClient,
		metadata,
		_,
		Q,
		PluginUtils) {
		'use strict';

        /**
         * Initializes a new instance of ExecuteExpt.
         * @class
         * @augments {PluginBase}
         * @classdesc This class represents the plugin ExecuteExpt.
         * @constructor
         */
        var StopExpt = function () {
            // Call base class' constructor.
            PluginBase.call(this);
            this.metaTypes = '';
            this.pluginMetadata = StopExpt.metadata;
            
		};

        StopExpt.metadata = JSON.parse(metadata);

        // Prototypal inheritance from PluginBase.
        StopExpt.prototype = Object.create(PluginBase.prototype);
        StopExpt.prototype.constructor = StopExpt;

               
        

        /**
         * Main function for the plugin to execute. This will perform the execution.
         * Notes:
         * - Always log with the provided logger.[error,warning,info,debug].
         * - Do NOT put any user interaction logic UI, etc. inside this method.
         * - callback always has to be called even if error happened.
         *
         * @param {function(string, plugin.PluginResult)} callback - the result callback
         */
        StopExpt.prototype.main = function (callback) {
            // Use self to access core, project, result, logger etc from PluginBase.
            // These are all instantiated at this point.
            var self = this;
            //var jobId = self.activeNode.getId();
            var job = self.activeNode;
            var jobId = self.core.getPath(self.activeNode);
        

            // Using the coreAPI to make changes.
            if (self.core.getPath(self.activeNode) === ' ' ||
                !(self.isMetaTypeOf(self.activeNode, self.META.Job))) {
                self.result.setSuccess(false);
                self.result.setError('Node should be an active Job');
                callback(null, self.result);
                return;
            }

            if (!self.isRunningJob(job))
            {
                self.result.setSuccess(false);
                self.result.setError( 'It is not an active Job');
                callback(null, self.result);
                return;
                
            }

            this.pulseClient = new ExecPulseClient({
                logger: this.logger
            });

            this._executor = new ExecutorClient({
                logger: this.logger.fork('ExecutorClient'),
                serverPort: WebGMEGlobal.gmeConfig.server.port,
                httpsecure: window.location.protocol === 'https:'
            });

            
    
            if (this.silentStopJob(job))
            {
                this._setJobStopped(jobId);
                self.result.setSuccess(true);
                callback(null, self.result);
            }

                        
        };

        StopExpt.prototype.isRunningJob = function(job) {
            var self = this;
            var status = self.core.getAttribute(job,'status');
            var secret = self.core.getAttribute(job,'secret');
            var jobId = self.core.getAttribute(job,'jobId');
    
            return (status === 'running' || status === 'pending') && (secret && jobId);
        };
    
        StopExpt.prototype.silentStopJob = function(job) {
            var jobHash,
                secret;
            var self = this;

            secret = self.core.getAttribute(job,'secret');
            jobHash = self.core.getAttribute(job,'jobId');
    
            this._executor.cancelJob(jobHash, secret)
            .then(() => {
                this.logger.info(`${jobHash} has been cancelled!`)
                return 1;
            })
            .fail(err => {
                this.logger.error(`Job cancel failed: ${err}`);
                return 0;
            }
           );

        };
    
        StopExpt.prototype._setJobStopped = function(jobId) {
            this.client.delAttribute(jobId, 'jobId');
            this.client.delAttribute(jobId, 'secret');
            this.client.setAttribute(jobId, 'status', 'canceled');
    
        };

        return StopExpt;
    
        
    });