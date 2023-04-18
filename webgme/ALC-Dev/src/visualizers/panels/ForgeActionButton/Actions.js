/*globals define, WebGMEGlobal*/
// These are actions defined for specific meta types. They are evaluated from
// the context of the ForgeActionButton
define([
    './LibraryDialog',
    'panel/FloatingActionButton/styles/Materialize',
    'q',
    'js/RegistryKeys',
    'deepforge/globals',
    'deepforge/viz/TextPrompter'
], function(
    LibraryDialog,
    Materialize,
    Q,
    REGISTRY_KEYS,
    DeepForge,
    TextPrompter
) {
    ////////////// Downloading files //////////////
    var downloadAttrs = [
            'data',
            'execFiles'
        ],
        download = {};

    downloadAttrs.forEach(attr => {
        download[attr] = function() {
            return downloadButton.call(this, attr);
        };
    });

    // Add download model button
    var downloadButton = function(attr) {
        var id = this._currentNodeId,
            node = this.client.getNode(id),
            hash = node.getAttribute(attr);

        if (hash) {
            return '/rest/blob/download/' + hash;
        }
        return null;
    };

    var returnToLast = (place) => {
        var returnId = DeepForge.last[place];
        WebGMEGlobal.State.registerActiveObject(returnId);
    };

    var prototypeButtons = function(type, fromType) {
        return [      
        ];
    };

    
    var makeRestartButton = function(name, pluginId, hotkeys) {
        return {
            name: 'Restart ' + name,
            icon: 'replay',
            priority: 1000,
            color: 'red',
            hotkey: hotkeys && 'shift enter',
            filter: function() {
                // Only show if stopped!
                return !this.isRunning();
            },
            action: function(event) {
                this.runExecutionPlugin(pluginId, {useSecondary: event.shiftKey});
            }
        };
    };

    return {
        // Instances
        Job: [
            
            // Stop execution button
            {
                name: 'Stop Current Job',
                icon: 'stop',
                priority: 1001,
                filter: function() {
                    return this.isRunning();
                },
                action: function() {
                    this.stopJob();
                }
            }
        ]
    };
});
