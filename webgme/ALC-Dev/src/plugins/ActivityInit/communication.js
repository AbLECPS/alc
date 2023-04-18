/*globals define, WebGMEGlobal*/
/**
 * Example of basic two-way communicaiton. We need to inject our code into a custom config widget...
 * @author kecso / https://github.com/kecso
 */

 define(['js/Dialogs/PluginConfig/PluginConfigDialog', 'js/Constants'], function (BaseConfig, CONSTANTS) {
    'use strict';

    /**
     * This class overrides the PluginConfigDialog class.
     * The purpose is that it can provide a form template and so allow all kinds of user input.
     * Just like in the plugin config, it can take string, number, multiple choice, ordering, color, and asset.
     * For an example form, check the main function of the actual plugin! 
     */
    function SimpleQuestionDialog (client) {
        BaseConfig.call(this, {client: client});
    }

    SimpleQuestionDialog.prototype = Object.create(BaseConfig.prototype);
    SimpleQuestionDialog.prototype.constructor = BaseConfig;

    SimpleQuestionDialog.prototype.show = function (form, callback) {
        this._mockMeta = {
            "id": "Ask user",
            "name": form.title || "Please submit the form to continue",
            "version": "42.42.42",
            "description": form.description || "Form questionarre from the executing plugin",
            "icon": {
              "class": "glyphicon glyphicon-list-cog"
            },
            "disableServerSideExecution": false,
            "disableBrowserSideExecution": false,
            "writeAccessRequired": false,
            "configStructure": form.fields
          };
        
        BaseConfig.prototype.show.call(this,[],this._mockMeta,{},(global, options, save) => {
            callback(options);
        });
    };

    SimpleQuestionDialog.prototype._initDialog = function() {
        BaseConfig.prototype._initDialog.call(this,false);
        this._dialog.find('.btn-default').hide();
        this._dialog.find('.save-configuration').hide();
        this._dialog.find('.close').hide();
        this._title.text(this._mockMeta.name);
        this._btnSave.text('Submit');
    };

    /**
     * This is the 'config widget' class.
     * It actually calls the regular plugin config dialog.
     * This means that this module can be used without any change if the way
     * how this handles messages is acceptable for the user.
     */
    function Communication(params) {
        this._client = params.client;
        this._logger = params.logger.fork('Communication');
        this._eid = null; // TODO - execution id missing from regular notificaiton!
        this._id = null; //this will represent the execution
        this._name = null //this will be the name of the plugin to make identification easier

        this.onStartPlugin = this.onStartPlugin.bind(this);
        this.onMessageFromPlugin = this.onMessageFromPlugin.bind(this);
        this.onEndPlugin = this.onEndPlugin.bind(this);
    }

    /**
     * Called by the InterpreterManager if pointed to by metadata.configWidget.
     * You can reuse the default config by including it from 'js/Dialogs/PluginConfig/PluginConfigDialog'.
     *
     * @param {object[]} globalConfigStructure - Array of global options descriptions (e.g. runOnServer, namespace)
     * @param {object} pluginMetadata - The metadata.json of the the plugin.
     * @param {object} prevPluginConfig - The config at the previous (could be stored) execution of the plugin.
     * @param {function} callback
     * @param {object|boolean} callback.globalConfig - Set to true to abort execution otherwise resolved global-config.
     * @param {object} callback.pluginConfig - Resolved plugin-config.
     * @param {boolean} callback.storeInUser - If true the pluginConfig will be stored in the user for upcoming execs.
     *
     */
    Communication.prototype.show = function (globalConfigStructure, pluginMetadata, prevPluginConfig, callback) {
        const self = this;
        this._name = pluginMetadata.id;
        const base = new BaseConfig({client: this._client});
        base.show(globalConfigStructure, pluginMetadata, prevPluginConfig, (global, config, store) => {
            //inject the initiating code here as this get only executed if no abort happened so far
            this._client.addEventListener(CONSTANTS.CLIENT.PLUGIN_INITIATED, self.onStartPlugin);
            this._client.addEventListener(CONSTANTS.CLIENT.PLUGIN_FINISHED, self.onEndPlugin);
            this._client.addEventListener(CONSTANTS.CLIENT.PLUGIN_NOTIFICATION, self.onMessageFromPlugin);
            callback(global, config, store);
        });       
    };

    /**
     * This function is where any functionality should happen.
     * This basic instance will pop-up a plugin config like dialog fed by the message content.
     * @param {*} sender 
     * @param {*} event 
     */
    Communication.prototype.onMessageFromPlugin = function(sender, event) {
        if (event.socketId === this._id) {
            let dialog = new SimpleQuestionDialog(this._client);
            dialog.show(event.notification.content, (form) => {
                this.sendMessage('message', {id:event.notification.id, content:form});
            });
        }
    };

    /**
     * This function is necessary to get the execution id so that sending message will be possible.
     * @param {object} sender - the initiator of the event
     * @param {object} event - the event data
     * @param {string} event.executionId - the unique id that represents the execution
     * @param {string} event.name - the name of the plugin
     * @param {boolean} event.clientSide - flag that shows if the execution happens on the client side
     */
    Communication.prototype.onStartPlugin = function(sender, event) {
        if (event.name === this._name) {
            this._eid = event.executionId;
            this._id = event.socketId; // TODO - why we not recieving the executionId
            this._client.removeEventListener(CONSTANTS.CLIENT.PLUGIN_INITIATED, this.onStartPlugin);
        }
    };

    /**
     * This function is necessary to close the communication channels and event handling.
     * @param {object} sender - the initiator of the event
     * @param {object} event - the event data 
     */
    Communication.prototype.onEndPlugin = function(sender, event) {
        if(event.executionId === this._eid) {
            // This is our stop.
            this._client.removeEventListener(CONSTANTS.CLIENT.PLUGIN_FINISHED, this.onEndPlugin);
            this._client.removeEventListener(CONSTANTS.CLIENT.PLUGIN_NOTIFICATION, this.onMessageFromPlugin);
        }
    };

    /**
     * This is a generic message sender function. Both parties should be aware of the id and content!
     * @param {*} messageId 
     * @param {*} content 
     */
    Communication.prototype.sendMessage = function(messageId, content) {
        this._client.sendMessageToPlugin(this._eid, messageId, content);
    };

    return Communication;
});