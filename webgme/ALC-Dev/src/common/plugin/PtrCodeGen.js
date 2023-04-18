/*globals define, requirejs*/
define([
    'plugin/util',
    'q'
], function(
    PluginUtils,
    Q
) {

    var CodeGen = {
        Operation: {
            pluginId: 'GenerateJob',
            namespace: 'pipeline'
        }
    };

    var PtrCodeGen = function() {
    };

    PtrCodeGen.prototype.getCodeGenPluginIdFor = function(node) {
        var base = this.core.getBase(node),
            name = this.core.getAttribute(node, 'name'),
            namespace = this.core.getNamespace(node),
            pluginId;

        //this.logger.debug(`loaded pointer target of ${ptrId}: ${ptrNode}`);
        pluginId = (this.core.getOwnRegistry(node, 'validPlugins') || '').split(' ').shift();
        //this.logger.info(`generating code for ${this.core.getAttribute(ptrNode, 'name')} using ${pluginId}`);

        if (this.core.isMetaNode(node) && CodeGen[name]) {
            pluginId = CodeGen[name].pluginId || CodeGen[name];
            //namespace = CodeGen[name].namespace;
        }

        if (pluginId) {
            return {
                namespace: namespace,
                pluginId: pluginId
            };
        } else if (base) {
            return this.getCodeGenPluginIdFor(base);
        } else {
            return null;
        }
    };

    PtrCodeGen.prototype.getPtrCodeHash = function(ptrId, execname='') {
        return this.core.loadByPath(this.rootNode, ptrId)
            .then(ptrNode => {
                // Look up the plugin to use
                var genInfo = this.getCodeGenPluginIdFor(ptrNode);
                this.logger.debug('+++++++++++ getPtrCodeHash execname '+ execname);

                if (genInfo.pluginId) {
                    var context = {
                        namespace: genInfo.namespace,
                        activeNode: this.core.getPath(ptrNode)
                    };
                    if (execname)
                    {
                        context.name=execname;
                         this.logger.debug('+++++++++++ getPtrCodeHash context.name '+ context.name)
                    }

                    // Load and run the plugin
                    return this.executePlugin(genInfo.pluginId, context);
                } else {
                    var metanode = this.core.getMetaType(ptrNode),
                        type = this.core.getAttribute(metanode, 'name');
                    this.logger.warn(`Could not find plugin for ${type}. Will try to proceed anyway`);
                    return null;
                }
            })
            .then(hashes => hashes[0]);  // Grab the first asset for now
    };

    PtrCodeGen.prototype.getPtrCode = function(ptrId) {
        return this.getPtrCodeHash(ptrId)
            .then(hash => this.blobClient.getObjectAsString(hash));
    };

    PtrCodeGen.prototype.createPlugin = function(pluginId) {
        var deferred = Q.defer(),
            pluginPath = [
                'plugin',
                pluginId,
                pluginId,
                pluginId
            ].join('/');

        requirejs([pluginPath], Plugin => {
            var plugin = new Plugin();
            deferred.resolve(plugin);
        }, err => {
            this.logger.error(`Could not load ${pluginId}: ${err}`);
            deferred.reject(err);
        });
        return deferred.promise;
    };

    PtrCodeGen.prototype.configurePlugin = function(plugin, opts) {
        var logger = this.logger.fork(plugin.getName());
         var k0 = Object.keys(opts);

            this.logger.debug('++++++++++++++++ k0'+k0);
        return PluginUtils.loadNodesAtCommitHash(
            this.project,
            this.core,
            this.currentHash,
            this.logger,
            opts
        ).then(config => {
            plugin.initialize(logger, this.blobClient, this.gmeConfig);
            config.core = this.core;
            config.project = this.project;
            config.projectId = this.projectId;
            config.branchName = this.branchName;
            plugin.configure(config);
            
            var x = plugin.getCurrentConfig();
            var k = Object.keys(x);
            var k1 = Object.keys(opts);
            this.logger.debug('++++++++++++++++ k'+k);
            this.logger.debug('++++++++++++++++ k1'+k1);
            if (k1.indexOf('name') > -1)
            {
                x.name = opts.name;
                plugin.setCurrentConfig(x);
                x=plugin.getCurrentConfig();
                k = Object.keys(x);
                this.logger.debug('++++++++++++++++ x3'+k);
            }
            
            return plugin;
        });
    };

    PtrCodeGen.prototype.executePlugin = function(pluginId, config) {
        return this.createPlugin(pluginId)
            .then(plugin => this.configurePlugin(plugin, config))
            .then(plugin => {
                return Q.ninvoke(plugin, 'main');
            })
            .then(result => {
                this.logger.info('Finished calling ' + pluginId);
                return result.artifacts;
            });
    };

    return PtrCodeGen;
});
