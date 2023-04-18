// jshint node: true
'use strict';

var config = require('./config.base'),
    path = require('path'),
    JSZip = require("jszip"),
    validateConfig = require('webgme/config/validator'),
    userManPath = path.join(__dirname, '..', 'node_modules', 'webgme-registration-user-management-page', 'src', 'server');

config.mongo.uri = 'mongodb://127.0.0.1:27017/alc';

// Configure default admin username and password
config.authentication.enable = true
config.authentication.adminAccount = 'admin:vanderbilt'
config.authentication.allowUserRegistration = true
config.authentication.publicOrganizations = ["ALC"]

// Setup project seeds
config.seedProjects.createAtStartup = [
    {
        seedId: 'SEAM',
        projectName: 'SEAM',
        rights: {
            ALC: {read: true, write: false, delete: false}
        }
    },
    {
        seedId: 'ROSMOD',
        projectName: 'ROSMOD',
        rights: {
            ALC: {read: true, write: false, delete: false}
        }
    },
    {
        seedId: 'pipeline',
        projectName: 'pipeline',
        rights: {
            ALC: {read: true, write: false, delete: false}
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
        seedId: 'alc_ep_meta',
        projectName: 'ep_meta',
        rights: {
            ALC: {read: true, write: false, delete: false}
        }
    },
    {
        seedId: 'BlueROVActivity',
        projectName: 'BlueROV',
        rights: {
            ALC: {read: true, write: false, delete: false}
        }
    }

   
];

validateConfig(config);
module.exports = config;
