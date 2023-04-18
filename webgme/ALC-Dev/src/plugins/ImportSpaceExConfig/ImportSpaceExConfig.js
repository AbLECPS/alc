/*globals define*/
/*eslint-env node, browser*/

/**
 * Generated by PluginGenerator 2.20.5 from webgme on Wed Dec 12 2018 16:14:35 GMT-0600 (Central Standard Time).
 * A plugin that inherits from the PluginBase. To see source code documentation about available
 * properties and methods visit %host%/docs/source/PluginBase.html.
 */

define([
    'plugin/PluginConfig',
    'text!./metadata.json',
    'plugin/PluginBase',
    'hysteditor/HyST',
    'q'
], function (PluginConfig,
             pluginMetadata,
             PluginBase,
             HyST,
             Q) {
    'use strict';

    pluginMetadata = JSON.parse(pluginMetadata);

    /**
     * Initializes a new instance of ImportSpaceExConfig.
     * @class
     * @augments {PluginBase}
     * @classdesc This class represents the plugin ImportSpaceExConfig.
     * @constructor
     */
    function ImportSpaceExConfig() {
        // Call base class' constructor.
        PluginBase.call(this);
        this.pluginMetadata = pluginMetadata;
    }

    /**
     * Metadata associated with the plugin. Contains id, name, version, description, icon, configStructure etc.
     * This is also available at the instance at this.pluginMetadata.
     * @type {object}
     */
    ImportSpaceExConfig.metadata = pluginMetadata;

    // Prototypical inheritance from PluginBase.
    ImportSpaceExConfig.prototype = Object.create(PluginBase.prototype);
    ImportSpaceExConfig.prototype.constructor = ImportSpaceExConfig;

    /**
     * Main function for the plugin to execute. This will perform the execution.
     * Notes:
     * - Always log with the provided logger.[error,warning,info,debug].
     * - Do NOT put any user interaction logic UI, etc. inside this method.
     * - callback always has to be called even if error happened.
     *
     * @param {function(Error|null, plugin.PluginResult)} callback - the result callback
     */
    ImportSpaceExConfig.prototype.main = function (callback) {
        var self = this,
            core = this.core,
            regExp = new RegExp(/(\w+) = (["\w]\w+["\w])/),
            model = this.activeNode,
            bc = this.blobClient,
            config = this.getCurrentConfig(),
            system,
            outputVariables = [],
            configNode = core.createNode({parent: model, base: self.META["SpaceEx.Configuration"]});

        Q.ninvoke(bc, 'getObjectAsString', config.configFile)
            .then(function (configString) {
                var dimensions = HyST.getTextDimensions(configString, false);
                // TODO - once we figure out what options there might be we should model it
                core.setAttribute(configNode, 'file', config.configFile);
                core.setAttribute(configNode, 'content', configString);

                core.setRegistry(configNode, 'position', {x: 20, y: 40});
                core.setRegistry(configNode, 'decoratorHeight', dimensions.height);
                core.setRegistry(configNode, 'decoratorWidth', dimensions.width);

                return Q.ninvoke(self, 'save', 'Imported config from file');
            })
            .then(function () {
                self.result.setSuccess(true);
                callback(null, self.result);
            })
            .catch(function (err) {
                callback(err, self.result);
            });
    };

    return ImportSpaceExConfig;
});
