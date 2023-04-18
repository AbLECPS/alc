'use strict';

var config = require('./config.base'),
	path = require('path'),
  JSZip = require("jszip"),
    validateConfig = require('webgme/config/validator'),
	userManPath = path.join(__dirname, '..', 'node_modules', 'webgme-registration-user-management-page', 'src', 'server');
 

config.authentication.jwt.privateKey = '/home/ninehs/ALC/alc_dockerfiles/token_keys/private_key';
config.authentication.jwt.publicKey = '/home/ninehs/ALC/alc_dockerfiles/token_keys/public_key';

// Configure default admin username and password
config.authentication.enable = true
config.authentication.adminAccount = 'admin:vanderbilt'
config.authentication.allowUserRegistration = true
config.authentication.publicOrganizations = ["ALC"]

config.blob.fsDir = '/blob-local-storage';

// This is the exposed port from the docker container.
config.server.port = 8888;

// Connect to the linked mongo container N.B. container must be named mongo
//config.mongo.uri = 'mongodb://' + process.env.MONGO_PORT_27017_TCP_ADDR + ':' + process.env.MONGO_PORT_27017_TCP_PORT + '/webgme-seaml';
config.mongo.uri = 'mongodb://172.26.0.4:27017/alc';

// Setup project seeds
config.seedProjects.createAtStartup= [
{
  seedId: 'SEAM',
  projectName: 'SEAM',
  rights: {
	  ALC: { read: true, write: false, delete: false }
  }
},
{
  seedId: 'ROSMOD',
  projectName: 'ROSMOD',
  rights: {
	  ALC: { read: true, write: false, delete: false }
  }
},
{
  seedId: 'pipeline',
  projectName: 'pipeline',
  rights: {
	  ALC: { read: true, write: false, delete: false }
  }
},
{
  seedId: 'keras',
  projectName: 'keras',
  rights: {
      ALC: {read: true, write: false, delete: false}
  }
},
{
  seedId: 'DEEPFORGE',
  projectName: 'DEEPFORGE',
  rights: {
      ALC: {read: true, write: false, delete: false}
  }
},
{
  seedId: 'ALC_Meta',
  projectName: 'ALC_Meta',
  rights: {
	  ALC: { read: true, write: false, delete: false }
  }
},
{
  seedId: 'ALC_Template',
  projectName: 'ALC_Template',
  rights: {
	  ALC: { read: true, write: false, delete: false }
  }
}
];


validateConfig(config);
module.exports = config;
