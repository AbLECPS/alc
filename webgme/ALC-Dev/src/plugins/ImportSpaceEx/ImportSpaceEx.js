/*globals define*/
/*eslint-env node, browser*/

/**
 * Generated by PluginGenerator 2.20.5 from webgme on Tue Dec 11 2018 12:39:14 GMT-0600 (Central Standard Time).
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
     * Initializes a new instance of ImportSpaceEx.
     * @class
     * @augments {PluginBase}
     * @classdesc This class represents the plugin ImportSpaceEx.
     * @constructor
     */
    function ImportSpaceEx() {
        // Call base class' constructor.
        PluginBase.call(this);
        this.pluginMetadata = pluginMetadata;
    }

    /**
     * Metadata associated with the plugin. Contains id, name, version, description, icon, configStructure etc.
     * This is also available at the instance at this.pluginMetadata.
     * @type {object}
     */
    ImportSpaceEx.metadata = pluginMetadata;

    // Prototypical inheritance from PluginBase.
    ImportSpaceEx.prototype = Object.create(PluginBase.prototype);
    ImportSpaceEx.prototype.constructor = ImportSpaceEx;

    /**
     * Main function for the plugin to execute. This will perform the execution.
     * Notes:
     * - Always log with the provided logger.[error,warning,info,debug].
     * - Do NOT put any user interaction logic UI, etc. inside this method.
     * - callback always has to be called even if error happened.
     *
     * @param {function(Error|null, plugin.PluginResult)} callback - the result callback
     */
    ImportSpaceEx.prototype.main = function (callback) {
        // Use this to access core, project, result, logger etc from PluginBase.
        var self = this,
            core = this.core,
            modelContainer = this.activeNode,
            bc = this.blobClient,
            config = this.getCurrentConfig();

        Q.ninvoke(bc, 'getObjectAsString', config.modelFile)
            .then(function (xmlString) {
                //TODO some compatibility check maybe
                var modelSnipet = HyST.spaceExToJson(xmlString),
                    model = core.createNode({parent: modelContainer, base: self.META.PlantModel}),
                    bases = {},
                    networkComponentSnipets = [],
                    componentArray = modelSnipet.component || [];

                if (Array.isArray(componentArray) === false) {
                    componentArray = [componentArray];
                }

                componentArray.forEach(function (componentSnipet) {
                    var component;

                    if (componentSnipet.location) {
                        //BaseComponent
                        component = self._addComponent(componentSnipet, model, false);
                        bases[core.getAttribute(component, 'name')] = component;
                    } else {
                        component = self._addComponent(componentSnipet, model, true);
                        bases[core.getAttribute(component, 'name')] = component;
                        networkComponentSnipets.push(componentSnipet);
                    }
                });

                networkComponentSnipets.forEach(function (snipet) {
                    self._addNetworkComponent(snipet, bases);
                });

                return Q.ninvoke(self, 'save', 'Imported model from file');
            })
            .then(function () {
                self.result.setSuccess(true);
                callback(null, self.result);
            })
            .catch(function (err) {
                callback(err, self.result);
            });
    };

    ImportSpaceEx.prototype._addComponent = function (snipet, parent, isNetwork) {
        var locations = {},
            self = this,
            core = this.core,
            component = core.createNode({
                parent: parent,
                base: isNetwork ? this.META["SpaceEx.NetworkComponent"] : this.META["SpaceEx.BaseComponent"]
            }),
            paramArray = snipet.param || [],
            locationArray = snipet.location || [],
            transitionArray = snipet.transition || [];

        core.setAttribute(component, 'name', HyST.getAttribute(snipet, 'id') || '');

        // As the id of the location is used as relid, these types must be the first ones to process!
        if (Array.isArray(locationArray) === false) {
            locationArray = [locationArray];
        }

        locationArray.forEach(function (locationSnipet) {
            var location = self._addLocation(locationSnipet, component);
            locations[core.getRelid(location)] = location;
        });

        if (Array.isArray(paramArray) === false) {
            paramArray = [paramArray];
        }
        paramArray.forEach(function (paramSnipet) {
            self._addParameter(paramSnipet, component);
        });

        if (Array.isArray(transitionArray) === false) {
            transitionArray = [transitionArray];
        }

        transitionArray.forEach(function (transitionSnipet) {
            self._addTransition(transitionSnipet, locations, component);
        });

        return component;

    };

    ImportSpaceEx.prototype._addParameter = function (snipet, parent) {
        var core = this.core,
            value,
            parameter = core.createNode({parent: parent, base: this.META["SpaceEx.Parameter"]});

        // {
        //     "@name": "x",
        //     "@type": "real",
        //     "@local": "false",
        //     "@d1": "1", - optional
        //     "@d2": "1", - optional
        //     "@dynamics": "any", - optional
        //     "@controlled": "true" - optional
        // }
        core.setAttribute(parameter, 'name', HyST.getAttribute(snipet, 'name') || '');
        core.setAttribute(parameter, 'type', HyST.getAttribute(snipet, 'type') || 'real');
        core.setAttribute(parameter, 'local', HyST.getAttribute(snipet, 'local') === 'true');
        value = HyST.getAttribute(snipet, 'd1');
        if (value !== undefined && value !== null) {
            value = Number(value);
            if (value !== 'NaN') {
                core.setAttribute(parameter, 'd1', value);
            }
        }
        value = HyST.getAttribute(snipet, 'd2');
        if (value !== undefined && value !== null) {
            value = Number(value);
            if (value !== 'NaN') {
                core.setAttribute(parameter, 'd2', value);
            }
        }
        value = HyST.getAttribute(snipet, 'dynamics');
        if (typeof value === 'string') {
            core.setAttribute(parameter, 'dynamics', value);
        }
        value = HyST.getAttribute(snipet, 'controlled');
        if (typeof value === 'string') {
            core.setAttribute(parameter, 'controlled', value === 'true');
        }
        return parameter;
    };

    ImportSpaceEx.prototype._addLocation = function (snipet, parent) {
        var core = this.core,
            location = core.createNode({
                parent: parent,
                base: this.META["SpaceEx.Location"],
                relid: HyST.getAttribute(snipet, 'id')
            });

        core.setAttribute(location, 'name', HyST.getAttribute(snipet, 'name') || '');
        core.setRegistry(location, 'position', {
            x: HyST.getAttribute(snipet, 'x') || 0,
            y: HyST.getAttribute(snipet, 'y') || 0
        });
        core.setRegistry(location, 'decoratorHeight', HyST.getAttribute(snipet, 'height') || 100);
        core.setRegistry(location, 'decoratorWidth', HyST.getAttribute(snipet, 'width') || 100);
        core.setAttribute(location, 'invariant', HyST.getContent(snipet.invariant || {'#text': ''}));
        core.setAttribute(location, 'flow', HyST.getContent(snipet.flow || {'#text': ''}));
        core.setAttribute(location, 'initial', HyST.getContent(snipet.initial || {'#text': ''}));

        return location;
    };

    ImportSpaceEx.prototype._addTransition = function (snipet, locations, parent) {
        var core = this.core,
            transition = core.createNode({
                parent: parent,
                base: this.META["SpaceEx.Transition"]
            });

        core.setPointer(transition, 'src', locations[HyST.getAttribute(snipet, 'source')]);
        core.setPointer(transition, 'dst', locations[HyST.getAttribute(snipet, 'target')]);
        core.setAttribute(transition, 'name',
            HyST.getContent(snipet.label || {'#text': ''}));
        core.setAttribute(transition, 'guard',
            HyST.getContent(snipet.guard || {'#text': ''}));
        core.setAttribute(transition, 'assignment',
            HyST.getContent(snipet.assignment || {'#text': ''}));

        core.setRegistry(transition, 'position', {
            x: HyST.getAttribute(snipet.labelposition || {}, 'x') || 0,
            y: HyST.getAttribute(snipet.labelposition || {}, 'y') || 0
        });

        return transition;
    };

    ImportSpaceEx.prototype._addNetworkComponent = function (snipet, bases) {
        var self = this,
            component = bases[HyST.getAttribute(snipet, 'id') || ''],
            bindArray = snipet.bind || [];

        //TODO what about note? is it applies here only, is it a single one?

        if (Array.isArray(bindArray) === false) {
            bindArray = [bindArray];
        }

        bindArray.forEach(function (bindSnipet) {
            self._addBind(bindSnipet, bases, component);
        });

    };

    ImportSpaceEx.prototype._addBind = function (snipet, bases, parent) {
        var self = this,
            core = self.core,
            bind,
            mappingArray = snipet.map || [],
            component = bases[HyST.getAttribute(snipet, 'component')];

        if (!component)
            return; //TODO - we should start giving errors?

        bind = core.createNode({parent: parent, base: component});
        core.setAttribute(bind, 'name', HyST.getAttribute(snipet, 'as'));
        core.setRegistry(bind, 'position', {
            x: HyST.getAttribute(snipet, 'x') || 0,
            y: HyST.getAttribute(snipet, 'y') || 0
        });
        // core.setRegistry(bind, 'decoratorHeight', HyST.getAttribute(snipet, 'height') || 100);
        // core.setRegistry(bind, 'decoratorWidth', HyST.getAttribute(snipet, 'width') || 100);

        if (Array.isArray(mappingArray) === false) {
            mappingArray = [mappingArray];
        }

        self._addMappings(mappingArray, bind, parent);

        return bind;
    };

    ImportSpaceEx.prototype._addMappings = function (snipetArray, binding, networkComponent) {
        var self = this,
            core = self.core,
            componentParams = {},
            componentParamIds = core.getChildrenRelids(binding),
            networkParamIds = core.getChildrenRelids(networkComponent),
            networkParams = {},
            constant;


        componentParamIds.forEach(function (componentChildRelid) {
            var componentChild = core.getChild(binding, componentChildRelid);
            if (core.isInstanceOf(componentChild, self.META["SpaceEx.Parameter"])) {
                if (core.getAttribute(componentChild, 'local') === false) {
                    componentParams[core.getAttribute(componentChild, 'name')] = componentChild;
                }
            }
        });

        networkParamIds.forEach(function (networkParamRelid) {
            var networkChild = core.getChild(networkComponent, networkParamRelid);
            if (core.isInstanceOf(networkChild, self.META["SpaceEx.Parameter"])) {
                networkParams[core.getAttribute(networkChild, 'name')] = networkChild;
            }
        });

        snipetArray.forEach(function (snipet) {
            var mapping = core.createNode({parent: networkComponent, base: self.META["SpaceEx.Mapping"]});

            core.setPointer(mapping, 'src', componentParams[HyST.getAttribute(snipet, 'key')] || null);
            core.setPointer(mapping, 'dst', networkParams[HyST.getContent(snipet)] || null);
            if (core.getPointerPath(mapping, 'dst') === null) {
                constant = core.createNode({parent: networkComponent, base: self.META["SpaceEx.Constant"]});
                core.setAttribute(constant, 'value', HyST.getContent(snipet) || '');
                core.setAttribute(constant, 'name', HyST.getAttribute(snipet, 'key'));
                core.setPointer(mapping, 'dst', constant);
            }
        });


    };

    ImportSpaceEx.prototype._layoutNetworkComponent = function (networkComponent) {
        var core = this.core,
            childrenRelids = core.getChildrenRelids(networkComponent),
            binds = {},
            mappings = {};
    };

    ImportSpaceEx.prototype._layoutBaseComponent = function (baseComponent) {
        //TODO how on earth to do this
    };

    ImportSpaceEx.prototype._layoutModel = function (model) {
        var core = this.core,
            childrenRelIds = core.getChildrenRelids(model);

        childrenRelIds.forEach(function (relId) {
            var child = core.getChild(model, relId);

            if (core.isTypeOf(child, this.META["SpaceEx.BaseComponent"])) {
                this._layoutBaseComponent(child);
            } else if (core.isTypeOf(child, this.META["SpaceEx.NetworkComponent"])) {
                this._layoutNetworkComponent(child);
            }
        });
    };

    return ImportSpaceEx;
});
