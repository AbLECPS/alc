/*globals require, module, process*/
'use strict';

var config = require('./config.webgme'),
	path = require('path'),
    validateConfig = require('webgme/config/validator'),
	userManPath = path.join(__dirname, '..', 'node_modules', 'webgme-registration-user-management-page', 'src', 'server');

require('dotenv').load({silent: true});

// Add/overwrite any additional settings here
config.server.port = 9091;

// Executors
config.executor.enable = true;

config.core.enableCustomConstraints = true;

// Plugins
config.plugin.allowServerExecution = true;
config.plugin.allowBrowserExecution = true;

// Seeds
config.seedProjects.enable = true;
config.seedProjects.basePaths.push("./src/seeds");
//config.seedProjects.basePaths.push("./src/seeds/pipeline");
//config.seedProjects.basePaths.push("src/seeds/project");
//config.seedProjects.defaultProject = 'project';

//authentication
config.authentication.enable = true;
config.authentication.jwt.privateKey = path.join(__dirname, '..', '..', 'token_keys', 'private_key');
config.authentication.jwt.publicKey = path.join(__dirname, '..', '..', 'token_keys', 'public_key');
config.authentication.allowGuests = true;
config.authentication.guestAccount = 'guest';
config.authentication.allowUserRegistration = true; //path.join(userManPath, 'registrationEndPoint');
//config.authentication.userManagementPage =  path.join(userManPath, 'usermanagement');
config.authentication.logInUrl = '/profile/login';
config.authentication.logOutUrl = '/profile/login';

//visualization svg
config.visualization.svgDirs.push(path.join(__dirname, '..', "./src/svgs"));
config.visualization.svgDirs.push(path.join(__dirname, '..', "./src/svg"));
config.visualization.svgDirs.push(path.join(__dirname, '..', "./node_modules/webgme-hfsm/src/svgs"));
config.visualization.svgDirs.push(path.join(__dirname, '..', "./node_modules/webgme-easydag/src/visualizers/widgets/EasyDAG/lib/open-iconic"));
config.visualization.decoratorPaths.push(__dirname + '/../node_modules/ui-components/src/decorators');
config.visualization.extraCss.push('deepforge/styles/global.css');

// Merging config
config.storage.autoMerge.enable = true;

// Add/overwrite any additional settings here
config.server.port = +process.env.PORT || config.server.port;
config.server.timeout = 0;
config.mongo.uri = process.env.MONGO_URI || config.mongo.uri;
config.blob.fsDir = process.env.DEEPFORGE_BLOB_DIR || config.blob.fsDir;

//requirejs paths
config.requirejsPaths.deepforge = './src/common';
config.requirejsPaths.ace = './src/visualizers/widgets/TextEditor/lib/ace';
config.requirejsPaths.rosmod = "./src/common/";
config.requirejsPaths.cytoscape = "./node_modules/cytoscape/dist";
config.requirejsPaths.plottable = "./node_modules/plottable/";
config.requirejsPaths.handlebars = "./node_modules/handlebars/";
config.requirejsPaths['cytoscape-cose-bilkent'] = "./node_modules/cytoscape-cose-bilkent/";
config.requirejsPaths['webgme-to-json'] = "./node_modules/webgme-to-json/";
config.requirejsPaths['remote-utils'] = "./node_modules/remote-utils/";
config.requirejsPaths['plotly-js'] = "./node_modules/plotly.js/dist/";
config.requirejsPaths['showdown'] = "./node_modules/showdown/";
config.requirejsPaths['blob-util'] = "./node_modules/blob-util/";
config.requirejsPaths['hfsm'] = './node_modules/webgme-hfsm/src/common/';
config.requirejsPaths['hfsm-library'] = './node_modules/webgme-hfsm/';
config.requirejsPaths['bower'] = "./node_modules/webgme-hfsm/bower_components/";
config.requirejsPaths['cytoscape-edgehandles'] = "./node_modules/webgme-hfsm/bower_components/cytoscape-edgehandles/cytoscape-edgehandles";
config.requirejsPaths['cytoscape-context-menus'] = "./node_modules/webgme-hfsm/bower_components/cytoscape-context-menus/cytoscape-context-menus";
config.requirejsPaths['cytoscape-panzoom'] = "./node_modules/webgme-hfsm/bower_components/cytoscape-panzoom/cytoscape-panzoom";
config.requirejsPaths['select2'] = "./node_modules/select2/dist";
config.requirejsPaths["jszip"] = "./node_modules/jszip";
config.requirejsPaths["MultilineAttributeDecorator"]="./node_modules/ui-components/src/decorators/MultilineAttributeDecorator";
config.requirejsPaths["ui-components"]="./node_modules/ui-components/src/common";
config.requirejsPaths["hysteditor"]="./src/common";

//client log level
//config.client.log.level = 'info';

validateConfig(config);
module.exports = config;