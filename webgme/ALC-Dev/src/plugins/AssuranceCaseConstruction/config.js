'use strict';

var config = require('../../../config/config.base'),
	path = require('path'),
  JSZip = require("jszip"),
    validateConfig = require('webgme/config/validator'),
	userManPath = path.join(__dirname, '..', 'node_modules', 'webgme-registration-user-management-page', 'src', 'server');
 

config.authentication.jwt.privateKey = '/token_keys/private_key';
config.authentication.jwt.publicKey = '/token_keys/public_key';
config.blob.fsDir = '/blob-local-storage';

// Configure default admin username and password
config.authentication.enable = true
config.authentication.adminAccount = 'admin:vanderbilt'
config.authentication.allowUserRegistration = true
config.authentication.publicOrganizations = ["ALC"]

config.blob.fsDir = '/blob-local-storage';

// This is the exposed port from the docker container.
config.server.port = 18888;

// Connect to the linked mongo container N.B. container must be named mongo
config.mongo.uri = 'mongodb://mongo:27017/alc';



validateConfig(config);
module.exports = config;
