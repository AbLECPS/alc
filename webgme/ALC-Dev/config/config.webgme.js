// DO NOT EDIT THIS FILE
// This file is automatically generated from the webgme-setup-tool.
'use strict';


var config = require('webgme/config/config.default'),
    validateConfig = require('webgme/config/validator');

// The paths can be loaded from the webgme-setup.json
config.plugin.basePaths.push(__dirname + '/../src/plugins');
config.plugin.basePaths.push(__dirname + '/../node_modules/webgme-simple-nodes/src/plugins');
config.plugin.basePaths.push(__dirname + '/../node_modules/deepforge-keras/src/plugins');
config.visualization.layout.basePaths.push(__dirname + '/../src/layouts');
config.visualization.layout.basePaths.push(__dirname + '/../node_modules/webgme-chflayout/src/layouts');
config.visualization.decoratorPaths.push(__dirname + '/../src/decorators');
config.visualization.decoratorPaths.push(__dirname + '/../node_modules/webgme-easydag/src/decorators');
config.visualization.decoratorPaths.push(__dirname + '/../node_modules/deepforge-keras/src/decorators');
config.visualization.decoratorPaths.push(__dirname + '/../node_modules/webgme-hfsm/src/decorators');
config.seedProjects.basePaths.push(__dirname + '/../src/seeds/pipeline');
config.seedProjects.basePaths.push(__dirname + '/../src/seeds/project');
config.seedProjects.basePaths.push(__dirname + '/../node_modules/deepforge-keras/src/seeds/dev');
config.seedProjects.basePaths.push(__dirname + '/../node_modules/deepforge-keras/src/seeds/base');
config.seedProjects.basePaths.push(__dirname + '/../node_modules/deepforge-keras/src/seeds/keras');
config.seedProjects.basePaths.push(__dirname + '/../node_modules/deepforge-keras/src/seeds/tests');
config.seedProjects.basePaths.push(__dirname + '/../node_modules/webgme-easydag/src/seeds/Example');
config.seedProjects.basePaths.push(__dirname + '/../node_modules/webgme-simple-nodes/src/seeds/ExampleModel');



config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-fab/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-breadcrumbheader/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-autoviz/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-easydag/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/deepforge-keras/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-asset-manager-viz/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-codeeditor/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../node_modules/webgme-hfsm/src/visualizers/panels');
config.visualization.panelPaths.push(__dirname + '/../src/visualizers/panels');


config.rest.components['JobLogsAPI'] = {
  src: __dirname + '/../src/routers/JobLogsAPI/JobLogsAPI.js',
  mount: 'execution/logs',
  options: {}
};
config.rest.components['JobOriginAPI'] = {
  src: __dirname + '/../src/routers/JobOriginAPI/JobOriginAPI.js',
  mount: 'job/origins',
  options: {}
};
config.rest.components['ExecPulse'] = {
  src: __dirname + '/../src/routers/ExecPulse/ExecPulse.js',
  mount: 'execution/pulse',
  options: {}
};
config.rest.components['SlurmRouter'] = {
  src: __dirname + '/../src/routers/SlurmRouter/SlurmRouter.js',
  mount: 'slurm',
  options: {}
};
config.rest.components['ALCModelUpdater'] = {
  src: __dirname + '/../src/routers/ALCModelUpdater/ALCModelUpdater.js',
  mount: 'alcmodelupdater',
  options: {}
};
config.rest.components['KerasAnalysis'] = {
  src: __dirname + '/../node_modules/deepforge-keras/src/routers/KerasAnalysis/KerasAnalysis.js',
  mount: 'routers/KerasAnalysis',
  options: {}
};
config.rest.components['BindingsDocs'] = {
  src: __dirname + '/../node_modules/webgme-bindings/src/routers/BindingsDocs/BindingsDocs.js',
  mount: 'bindings-docs',
  options: {}
};

// Visualizer descriptors
config.visualization.visualizerDescriptors.push(__dirname + '/../src/visualizers/Visualizers.json');
// Add requirejs paths
config.requirejsPaths = {
  'BindingsDocs': 'node_modules/webgme-bindings/src/routers/BindingsDocs',
  'KerasAnalysis': 'node_modules/deepforge-keras/src/routers/KerasAnalysis',
  'ExampleModel': 'node_modules/webgme-simple-nodes/src/seeds/ExampleModel',
  'Example': 'node_modules/webgme-easydag/src/seeds/Example',
  'tests': 'node_modules/deepforge-keras/src/seeds/tests',
  'keras': 'node_modules/deepforge-keras/src/seeds/keras',
  'base': 'node_modules/deepforge-keras/src/seeds/base',
  'dev': 'node_modules/deepforge-keras/src/seeds/dev',
  'LayerDecorator': 'node_modules/deepforge-keras/src/decorators/LayerDecorator',
  'EllipseDecorator': 'node_modules/webgme-easydag/src/decorators/EllipseDecorator',
  'HFSMViz': 'panels/HFSMViz/HFSMVizPanel',
  'CodeEditor': 'panels/CodeEditor/CodeEditorPanel',
  'AssetManager': 'panels/AssetManager/AssetManagerPanel',
  'GenericAttributeEditor': 'panels/GenericAttributeEditor/GenericAttributeEditorPanel',
  'KerasArchEditor': 'panels/KerasArchEditor/KerasArchEditorPanel',
  'EasyDAG': 'panels/EasyDAG/EasyDAGPanel',
  'AutoViz': 'panels/AutoViz/AutoVizPanel',
  'BreadcrumbHeader': 'panels/BreadcrumbHeader/BreadcrumbHeaderPanel',
  'FloatingActionButton': 'panels/FloatingActionButton/FloatingActionButtonPanel',
  'CHFLayout': 'node_modules/webgme-chflayout/src/layouts/CHFLayout',
  'CreateKerasMeta': 'node_modules/deepforge-keras/src/plugins/CreateKerasMeta',
  'MinimalExample': 'node_modules/webgme-simple-nodes/src/plugins/MinimalExample',
  'ExamplePlugin': 'node_modules/webgme-simple-nodes/src/plugins/ExamplePlugin',
  'ValidateKeras': 'node_modules/deepforge-keras/src/plugins/ValidateKeras',
  'GenerateKeras': 'node_modules/deepforge-keras/src/plugins/GenerateKeras',
  'GenerateKerasMeta': 'node_modules/deepforge-keras/src/plugins/GenerateKerasMeta',
  'SimpleNodes': 'node_modules/webgme-simple-nodes/src/plugins/SimpleNodes',
  'panels': './src/visualizers/panels',
  'widgets': './src/visualizers/widgets',
  'panels/HFSMViz': './node_modules/webgme-hfsm/src/visualizers/panels/HFSMViz',
  'widgets/HFSMViz': './node_modules/webgme-hfsm/src/visualizers/widgets/HFSMViz',
  'panels/CodeEditor': './node_modules/webgme-codeeditor/src/visualizers/panels/CodeEditor',
  'widgets/CodeEditor': './node_modules/webgme-codeeditor/src/visualizers/widgets/CodeEditor',
  'panels/AssetManager': './node_modules/webgme-asset-manager-viz/src/visualizers/panels/AssetManager',
  'widgets/AssetManager': './node_modules/webgme-asset-manager-viz/src/visualizers/widgets/AssetManager',
  'panels/GenericAttributeEditor': './node_modules/deepforge-keras/src/visualizers/panels/GenericAttributeEditor',
  'widgets/GenericAttributeEditor': './node_modules/deepforge-keras/src/visualizers/widgets/GenericAttributeEditor',
  'panels/KerasArchEditor': './node_modules/deepforge-keras/src/visualizers/panels/KerasArchEditor',
  'widgets/KerasArchEditor': './node_modules/deepforge-keras/src/visualizers/widgets/KerasArchEditor',
  'panels/EasyDAG': './node_modules/webgme-easydag/src/visualizers/panels/EasyDAG',
  'widgets/EasyDAG': './node_modules/webgme-easydag/src/visualizers/widgets/EasyDAG',
  'panels/AutoViz': './node_modules/webgme-autoviz/src/visualizers/panels/AutoViz',
  'widgets/AutoViz': './node_modules/webgme-autoviz/src/visualizers/widgets/AutoViz',
  'panels/BreadcrumbHeader': './node_modules/webgme-breadcrumbheader/src/visualizers/panels/BreadcrumbHeader',
  'widgets/BreadcrumbHeader': './node_modules/webgme-breadcrumbheader/',
  'panels/FloatingActionButton': './node_modules/webgme-fab/src/visualizers/panels/FloatingActionButton',
  'widgets/FloatingActionButton': './node_modules/webgme-fab/src/visualizers/widgets/FloatingActionButton',
  'webgme-simple-nodes': './node_modules/webgme-simple-nodes/src/common',
  'deepforge-keras': './node_modules/deepforge-keras/src/common',
  'webgme-chflayout': './node_modules/webgme-chflayout/src/common',
  'webgme-fab': './node_modules/webgme-fab/src/common',
  'webgme-breadcrumbheader': './node_modules/webgme-breadcrumbheader/src/common',
  'webgme-autoviz': './node_modules/webgme-autoviz/src/common',
  'webgme-easydag': './node_modules/webgme-easydag/src/common',
  'webgme-asset-manager-viz': './node_modules/webgme-asset-manager-viz/src/common',
  'webgme-codeeditor': './node_modules/webgme-codeeditor/src/common',
  'webgme-hfsm': './node_modules/webgme-hfsm/src/common',
  'webgme-bindings': './node_modules/webgme-bindings/src/common',
  'alcdev': './src/common'
};


config.mongo.uri = 'mongodb://127.0.0.1:27017/alcdev';
//config.mongo.uri = 'mongodb://127.0.0.1:27017/alc';
validateConfig(config);
module.exports = config;
