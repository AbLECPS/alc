/******* WebGME config for running ALC in docker images ********/
'use strict';

// Use 'config.default.js' as a starting point
var config = require('./config.default.js'),
    path = require('path'),
    JSZip = require("jszip"),
    validateConfig = require('webgme/config/validator'),
    userManPath = path.join(__dirname, '..', 'node_modules', 'webgme-registration-user-management-page', 'src', 'server');


// Set paths as configured within docker images
config.authentication.jwt.privateKey = '/token_keys/private_key';
config.authentication.jwt.publicKey = '/token_keys/public_key';
config.blob.fsDir = '/blob-local-storage';

// This is the exposed port from the docker container.
config.server.port = 8888;

// Connect to the linked mongo container N.B. container must be named mongo
//config.mongo.uri = 'mongodb://' + process.env.MONGO_PORT_27017_TCP_ADDR + ':' + process.env.MONGO_PORT_27017_TCP_PORT + '/webgme-seaml';
config.mongo.uri = 'mongodb://mongo:27017/alc';


validateConfig(config);
module.exports = config;
