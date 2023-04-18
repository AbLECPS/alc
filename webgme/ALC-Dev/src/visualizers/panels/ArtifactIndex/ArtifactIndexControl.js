/*globals define, WebGMEGlobal*/
/*jshint browser: true*/

define([
    'blob/BlobClient',
    'js/Constants'
], function (
    BlobClient,
    CONSTANTS
) {

    'use strict';

    var ArtifactIndexControl;

    ArtifactIndexControl = function (options) {

        this._logger = options.logger.fork('Control');
        this.blobClient = new BlobClient({
            logger: this._logger.fork('BlobClient')
        });

        this._client = options.client;
        this._embedded = options.embedded;

        // Initialize core collections and variables
        this._widget = options.widget;

        this._currentNodeId = null;
        this._initWidgetEventHandlers();

        this._logger.debug('ctor finished');
    };

    ArtifactIndexControl.prototype._initWidgetEventHandlers = function () {
        this._widget.onNodeClick = (/*id*/) => {
            // Change the current active object
            // This is currently disabled as there are not any good
            // visualizers for the data types
            // WebGMEGlobal.State.registerActiveObject(id);
        };

        this._widget.onNodeDeleteClicked = id => {
            var name = this._client.getNode(id).getAttribute('name'),
                msg = `Deleted "${name}" artifact (${id}) --`;

            this._client.startTransaction(msg);
            this._client.deleteNode(id);
            this._client.completeTransaction();
        };

        this._widget.onNameChange = (id, newName) => {
            var name = this._client.getNode(id).getAttribute('name'),
                msg = `Renamed "${name}" artifact to "${newName}"`;

            this._client.startTransaction(msg);
            this._client.setAttribute(id, 'name', newName);
            this._client.completeTransaction();
        };

        this._widget.onAddDataHash = (id, hash) => {
            var name = this._client.getNode(id).getAttribute('name'),
                msg = `Added hash to  "${name}`;

            this._client.startTransaction(msg);
            this._client.setAttribute(id, 'data', hash);
            this._client.completeTransaction();
        };
    };

    /* * * * * * * * Visualizer content update callbacks * * * * * * * */
    // One major concept here is with managing the territory. The territory
    // defines the parts of the project that the visualizer is interested in
    // (this allows the browser to then only load those relevant parts).
    ArtifactIndexControl.prototype.selectedObjectChanged = function (nodeId) {
        this._logger.debug('activeObject nodeId \'' + nodeId + '\'');

        // Remove current territory patterns
        if (this._currentNodeId) {
            this._client.removeUI(this._territoryId);
        }

        this._currentNodeId = nodeId;

        if (typeof this._currentNodeId === 'string') {
            // Put new node's info into territory rules
            this._widget.currentNode = this._currentNodeId;
            this._selfPatterns = {};

            this._territoryId = this._client.addUI(this, events => {
                this._eventCallback(events);
            });

            this._selfPatterns[nodeId] = {children: 2};
            this._client.updateTerritory(this._territoryId, this._selfPatterns);
        }
    };

    // This next function retrieves the relevant node information for the widget
    ArtifactIndexControl.prototype._getObjectDescriptor = function (nodeId) {
        var self = this;
        var node = self._client.getNode(nodeId),
            type,
            hash,
            objDescriptor = {
                id: nodeId,
                metaName: undefined,
                type: undefined,
                name: undefined,
                createdAt: undefined,
                dataURL: undefined,
                parentId: undefined,
                resultDir: undefined,
                status:undefined
            };


        return new Promise(function (resolve, reject) {
        
            if (node) {
                var metaObj = self._client.getNode(node.getMetaTypeId()),
                    metaName = undefined;
                objDescriptor = {
                        id: node.getId(),
                        metaName: undefined,
                        type: undefined,
                        name: node.getAttribute('name'),
                        createdAt: undefined,
                        dataURL: undefined,
                        parentId: node.getParentId(),
                        resultURL: undefined,
                        logURL: undefined,
                        viewURL: undefined,
                        jobstatus: undefined,
                        hashnew: false,
                        hash: undefined
                };

                if (metaObj) {
                    metaName = metaObj.getAttribute('name');
                    objDescriptor.metaName = metaName;
                    var pid = node.getParentId();
                    var pobj = self._client.getNode(pid);
                    var gpid = pobj.getParentId();
                    var gpobj = self._client.getNode(gpid);
                    var pmid = gpobj.getMetaTypeId();
                    var pmeta = self._client.getNode(pmid);
                    var pmetaname = pmeta.getAttribute('name');
                    if (objDescriptor.metaName.indexOf('Data')>-1)
                    {
                    
                        type = 'Data';
                        if ((pmetaname.indexOf('SLTraining') >-1) || (pmetaname.indexOf('RLTraining') >-1))
                        {
                          type = 'LEC';
                        }
                        if (pmetaname.indexOf('Assurance') >-1 )
                        {
                          type = 'AM';
                        }
                        
                        if (pmetaname.indexOf('SystemID') >-1 )
                        {
                          type = 'SystemID';
                        }
                        
                        if (pmetaname.indexOf('Verification') >-1 )
                        {
                          type = 'Verification';
                        }
                        
                        if (pmetaname.indexOf('Validation') >-1 )
                        {
                          type = 'Validation';
                        }
                        
                        hash = node.getAttribute('data');
                        var datainfo = node.getAttribute('datainfo');
                        var loginfo = node.getAttribute('resultDir')
                        
                        objDescriptor = {
                            id: node.getId(),
                            metaName: metaName,
                            type: type,
                            name: node.getAttribute('name'),
                            createdAt: node.getAttribute('createdAt'),
                            dataURL: self.blobClient.getDownloadURL(hash),
                            parentId: node.getParentId(),
                            resultURL: undefined,
                            logURL: undefined,
                            jobstatus: node.getAttribute('jobstatus')
                        };

                        if (loginfo)
                        {
                            var h = window.location.hostname;
                            var p = window.location.port;
                            var pos = loginfo.indexOf('jupyter');
                            if (pos > -1)
                            {
                                var logstr = "http://"+ h +":"+p+"/ipython/edit/"
                                if (p)
                                {
                                    logstr = "http://"+ h +":"+p+"/ipython/edit/"
                                }
                                else{
                                    logstr = "https://"+ h +"/ipython/edit/"
                                }
                                var folder = loginfo.substring(pos+8)
                                logstr += folder
                                logstr += "/slurm_job_log.txt"
                                objDescriptor.logURL = logstr;

                                if (type == 'Data')
                                {
                                    objDescriptor.viewURL = "http://"+ h +":15001"

                                }

                            }
                        }
                        
                        
                        if (datainfo)
                        {
                          try{
                                datainfo = datainfo.replace(/\bNaN\b/g, "null")
                              	var datainfos =JSON.parse(datainfo);
                                var keys = Object.keys(datainfos);
                                var h = window.location.hostname;
                                var p = window.location.port;
                                var result_url = "http://"+ h +":"+p+'/';
                                if (p)
                                {
                                    result_url = "http://"+ h +":"+p+'/';
                                }
                                else{
                                    result_url = "https://"+ h +'/';
                                }
                                var resultinfo =''
                                if (keys.indexOf('result_url') >-1)
                                {
                                    resultinfo = datainfos['result_url'];
                                }
                                else if   (keys.indexOf('url') >-1)
                                {
                                    resultinfo = datainfos['url'];
                                    if (resultinfo)
                                    {
                                        objDescriptor.logURL = undefined;
                                        objDescriptor.jobstatus = 'Finished';
                                    }
                                }
                                if (resultinfo)
                                {
                                    if (resultinfo.indexOf('ipython')==-1 && resultinfo.indexOf('matlab'==-1))
                                    {
                                        result_url += "ipython/notebooks/"+resultinfo;
                                        if ((type == 'Validation') || (type == 'Verification') || (type == 'SystemID'))
                                        {
                                            result_url = "http://"+ h +":"+p+'/' + "matlab/notebooks/"+resultinfo;
                                        }
                                    }
                                    else {
                                        result_url += resultinfo;
                                    }
                                    objDescriptor.resultURL = result_url;
                                }
                          }
                          catch(e){               					
                						
                					}
                        }
                        

                        resolve(objDescriptor);
                        
                        if (hash)
                        {
                            /*self.blobClient.getMetadata(hash)
                            .then(metadata => {
                                objDescriptor.size = self._humanFileSize(metadata.size);
                                resolve(objDescriptor);
                            });*/
                            resolve(objDescriptor);
                        }
                        else {
                            if (datainfo)
                            {
                                self.blobClient.putFile('result-metadata',datainfo)
                                .then(metadata => {
                                    hash = metadata;
                                    //self._widget.onAddDataHash(objDescriptor.id,hash);
                                    objDescriptor.dataURL = self.blobClient.getDownloadURL(hash);
                                    objDescriptor.hashnew = true;
                                    objDescriptor.hash = hash
                                    resolve(objDescriptor);

                                });
                            }
                            else
                            { 
                                resolve(objDescriptor);
                            }
                        }
                    }
                    else{
                        resolve(objDescriptor);
                    }
                }
                else {
                    resolve(objDescriptor);
                }
            }
            else {
                resolve(objDescriptor);
            }

            //resolve(objDescriptor);
            
        });

    };

    ArtifactIndexControl.prototype._humanFileSize = function (bytes, si) {
        var thresh = si ? 1000 : 1024,
            units = si ?
                ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'] :
                ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'],
            u = -1;

        if (bytes < thresh) {
            return bytes + ' B';
        }

        do {
            bytes = bytes / thresh;
            u += 1;
        } while (bytes >= thresh);

        return bytes.toFixed(1) + ' ' + units[u];
    };

    /* * * * * * * * Node Event Handling * * * * * * * */
    ArtifactIndexControl.prototype._eventCallback = function (events) {
        events = events.filter(event => event.eid !== this._currentNodeId);
        let i = events ? events.length : 0;
        this._logger.debug('_eventCallback \'' + i + '\' items');

        let event;
        while (i--) {
            event = events[i];
            switch (event.etype) {

            case CONSTANTS.TERRITORY_EVENT_LOAD:
                this._onLoad(event.eid,i);
                break;
            case CONSTANTS.TERRITORY_EVENT_UPDATE:
                this._onUpdate(event.eid);
                break;
            case CONSTANTS.TERRITORY_EVENT_UNLOAD:
                this._onUnload(event.eid);
                break;
            default:
                break;
            }
        }

        this._logger.debug('_eventCallback \'' + events.length + '\' items - DONE');
    };

    ArtifactIndexControl.prototype._onLoad = function (gmeId, count=-1) {
        var self=this;
        this._getObjectDescriptor(gmeId)
          .then(function(description) {
              description.last=-1;
              if (count==1){
                  description.last=1;
              }
              //self._logger.debug('metaname  _onLoad= '+ description.faultlabel);
              self._widget.addNode(description);
          });

        //this._getObjectDescriptor(gmeId).then(desc => this._widget.addNode(desc));
    };

    ArtifactIndexControl.prototype._onUpdate = function (gmeId) {
        this._getObjectDescriptor(gmeId).then(desc => this._widget.updateNode(desc));
    };

    ArtifactIndexControl.prototype._onUnload = function (gmeId) {
        this._widget.removeNode(gmeId);
    };

    ArtifactIndexControl.prototype._stateActiveObjectChanged = function (model, activeObjectId) {
        if (this._currentNodeId === activeObjectId) {
            // The same node selected as before - do not trigger
        } else {
            this.selectedObjectChanged(activeObjectId);
        }
    };

    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
    ArtifactIndexControl.prototype.destroy = function () {
        this._detachClientEventListeners();
    };

    ArtifactIndexControl.prototype._attachClientEventListeners = function () {
        this._detachClientEventListeners();
        if (!this._embedded) {
            WebGMEGlobal.State.on('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged, this);
        }
    };

    ArtifactIndexControl.prototype._detachClientEventListeners = function () {
        if (!this._embedded) {
            WebGMEGlobal.State.off('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged);
        }
    };

    ArtifactIndexControl.prototype.onActivate = function () {
        this._attachClientEventListeners();

        if (typeof this._currentNodeId === 'string') {
            WebGMEGlobal.State.registerSuppressVisualizerFromNode(true);
            WebGMEGlobal.State.registerActiveObject(this._currentNodeId);
            WebGMEGlobal.State.registerSuppressVisualizerFromNode(false);
        }
    };

    ArtifactIndexControl.prototype.onDeactivate = function () {
        this._detachClientEventListeners();
    };

    return ArtifactIndexControl;
});
