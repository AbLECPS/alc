/*globals define, WebGMEGlobal*/
/*jshint browser: true*/

/**
 * Generated by VisualizerGenerator 0.1.0 from webgme on Wed Mar 16 2016 12:18:29 GMT-0700 (PDT).
 */

define([
    'text!./RootViz.html',
    'text!./DefaultIcon.svg',
    'js/DragDrop/DropTarget',
    'js/DragDrop/DragConstants',
    'decorators/DocumentDecorator/DiagramDesigner/DocumentEditorDialog',
    // showdown
    'showdown/dist/showdown.min',
    'ejs',
    './Templates',
    'js/Controls/ContextMenu',
    'js/Utils/GMEConcepts',
    'js/NodePropertyNames',
    'css!./styles/RootVizWidget.css'
], function (
    RootVizHtml,
    DefaultIcon,
    dropTarget,
    DROP_CONSTANTS,
    DocumentEditorDialog,
    showdown,
    ejs,
    TEMPLATES,
    ContextMenu,
    GMEConcepts,
    nodePropertyNames) {
    'use strict';

    var RootVizWidget,
        WIDGET_CLASS = 'root-viz';

    RootVizWidget = function (logger, container, client) {
        this._logger = logger.fork('Widget');

        this.$el = container;

	this._client = client;

        this.nodes = {};

        this._initialize();
	this._makeDroppable();

        this._logger.debug('ctor finished');
    };

    RootVizWidget.prototype._initialize = function () {
	var self = this;
        // set widget class
        self.$el.addClass(WIDGET_CLASS);
        self.$el.append(RootVizHtml);
        
        // html context menu
        var cm;
        var onEmptyItems = [
            {
                name: 'New Project',
                icon: '',
                doNotHide: false,
                callback: self.newProject.bind(self)
            },
        ];
	// default handlers for the rest of the viz space
	self.$el.on('contextmenu', (event) => {
	    var posX = event.clientX;
	    var posY = event.clientY;

            cm = new ContextMenu({
                items: onEmptyItems,
                callback: function(key) {
                    onEmptyItems[key].callback();
                    cm.destroy();
                }
            });

            cm.show({x: posX, y: posY});

	    event.stopPropagation();
	    event.preventDefault();
	});

	self.$table = self.$el.find('#rootVizTable');
	self._tableSetup = false;
    };

    RootVizWidget.prototype.setupTable = function() {
	var self = this;
	var sizeOfElement = 300;
        var width = self.$el.width(),
            height = self.$el.height();
	self.$table.empty();
	self._tableSetup = true;
    };

    RootVizWidget.prototype.createNodeEntry = function (desc) {
	var self = this;
	if (!self._tableSetup)
	    self.setupTable();

	self.updateNodeEntry(desc);
    };

    var converter = new showdown.Converter();

    RootVizWidget.prototype.updateNodeEntry = function(desc) {
	var self = this;
	var projectHtml,
	    gmeId,
	    panelId,
	    title,
	    icon,
	    authors,
	    brief,
	    detailed,
	    svg,
	    htmlId,
	    html;
	
	title = desc.name;
	gmeId = desc.id;
	panelId = gmeId.replace(/\//g,'-');
	icon = desc.icon || DefaultIcon;
	authors = converter.makeHtml(desc.authors);
	brief = converter.makeHtml(desc.brief);
	detailed = converter.makeHtml(desc.detailed);
	projectHtml = ejs.render(TEMPLATES['Project.html.ejs'], {
	    id: panelId,
	    title: title,
	    icon: icon,
	    authors: authors,
	    brief: brief,
	    detailed: detailed
	});

        self.$table.find('#'+panelId+'-node-panel').first().remove();
        self.$table.append(projectHtml);
	// sort alphabetically
	var items = $(self.$table).children('div');
	items.detach().sort(function(a, b) {
	    var contentA =$(a).data('sort');
	    var contentB =$(b).data('sort');
	    return (contentA < contentB) ? -1 : (contentA > contentB) ? 1 : 0;	    
	});
	self.$table.append(items);

        function dialogPopup(gmeId, attrName) {
            var node = self._client.getNode(gmeId);
            var attr = node.getAttribute(attrName);
            var nodeName = node.getAttribute('name');
            var editorDialog = new DocumentEditorDialog("Edit " + attrName + " for " + nodeName);

            editorDialog.initialize(attr, function (text) {
                try {
                    self._client.setAttribute(gmeId, attrName, text, 'updated '+attrName+' for ' + gmeId);
                } catch (e) {
                    console.error('Could not save '+attrName+': ');
                    console.error(e);
                }
            });

            editorDialog.show();
        }

	if (!self._client.isProjectReadOnly()) {
	    // editable authors area
	    self.$el.find('#'+panelId+'-authors').on('click', _.bind(dialogPopup, self, gmeId, 'Authors'));
	    // editable brief area
	    self.$el.find('#'+panelId+'-brief').on('click', _.bind(dialogPopup, self, gmeId, 'Brief Description'));
	    // editable detailed area
	    self.$el.find('#'+panelId+'-detailed').on('click', _.bind(dialogPopup, self, gmeId, 'Detailed Description'));
	}

	htmlId = panelId + '-node-panel';
	html = self.$el.find('#' + htmlId);

	svg = html.find('svg');
	svg.css('height', '120px');
	svg.css('width', 'auto');

	html.addClass('panel-info');
	html.on('mouseenter', (event) => {
	    html.addClass('panel-primary');
	    html.removeClass('panel-info');
	});
	html.on('mouseleave', (event) => {
	    html.addClass('panel-info');
	    html.removeClass('panel-primary');
	});
	html.on('click', (event) => {
	    // hide context menu

	    self.onNodeSelect(desc.id);
	    //event.stopPropagation();
	    //event.preventDefault();
	});
	html.on('dblclick', (event) => {
	    self.onNodeClick(desc.id);
	    event.stopPropagation();
	    event.preventDefault();
	});
        var cm;
        // context menu:
        var onProjectItems = [
            {
                name: 'Copy ' + desc.name,
                icon: '',
                doNotHide: false,
                callback: self.copyProject.bind(self)
            },
            {
                name: 'Delete ' + desc.name,
                icon: '',
                doNotHide: false,
                callback: self.deleteProject.bind(self)
            },
            {
                name: 'New Project',
                icon: '',
                doNotHide: false,
                callback: self.newProject.bind(self)
            },
        ];
        
	html.on('contextmenu', (event) => {
	    // handle right click here
	    // make menu with copy option
	    // make menu with delete option
	    var posX = event.clientX;
	    var posY = event.clientY;

            cm = new ContextMenu({
                items: onProjectItems,
                callback: function(key) {
                    onProjectItems[key].callback(desc.id);
                    cm.destroy();
                }
            });

            cm.show({x: posX, y: posY});

	    self.onNodeSelect(desc.id);
	    event.stopPropagation();
	    event.preventDefault();
	});

	self.nodes[desc.id] = desc;
    };

    // CONTEXT MENU FUNCTIONS

    RootVizWidget.prototype.deleteProject = function( gmeId ) {
        var self = this;
        if (gmeId)
            self._client.deleteNode( gmeId, 'Deleting project: ' + gmeId);
    };

    RootVizWidget.prototype.copyProject = function( gmeId ) {
        var self = this;
        if (gmeId)
            self.createProject( gmeId );
    };

    RootVizWidget.prototype.newProject = function( ) {
        var self = this;
        var projectMetaId = '/3/h';
        self.createProject( projectMetaId );
    };

    // RESIZE

    RootVizWidget.prototype.onWidgetContainerResize = function (width, height) {
	var self = this;
	self.setupTable();
	for (var id in self.nodes) {
            self.addNode(self.nodes[id]);
	}
    };

    // Adding/Removing/Updating items
    var NODE_WHITELIST = {
        Project: true
    };
    RootVizWidget.prototype.addNode = function (desc) {
	var self = this;
        if (desc) {
	    var isValid = NODE_WHITELIST[desc.meta];

            if (isValid) {
		//self._nodes.push(desc);
		self.createNodeEntry(desc);
            }
        }
    };

    RootVizWidget.prototype.removeNode = function (gmeId) {
	var self = this;
	if (self.nodes[gmeId]) {
            delete self.nodes[gmeId];
	    self.setupTable();
	    for (var id in self.nodes) {
		self.addNode(self.nodes[id]);
	    }
	}
    };

    RootVizWidget.prototype.updateNode = function (desc) {
	var self = this;
	if (desc) {
	    var isValid = NODE_WHITELIST[desc.meta];
	    if (isValid) {
		self.updateNodeEntry(desc);
	    }
	}
    };

    RootVizWidget.prototype._isValidDrop = function (dragInfo) {
	var self = this;
	if (!dragInfo)
	    return false;
	var self = this;
        var result = false,
            draggedNodePath,
	    nodeObj,
	    nodeName,
	    metaObj,
	    metaName;

        if (dragInfo[DROP_CONSTANTS.DRAG_ITEMS].length === 1) {
            draggedNodePath = dragInfo[DROP_CONSTANTS.DRAG_ITEMS][0];
	    nodeObj = self._client.getNode(draggedNodePath);
	    nodeName = nodeObj.getAttribute('name');
	    metaObj = self._client.getNode(nodeObj.getMetaTypeId());
	    if (metaObj) {
		metaName = metaObj.getAttribute('name');
	    }
            result = metaName && metaName == 'Project';
        }

        return result;
    };

    RootVizWidget.prototype.createProject = function(nodePath) {
	var self = this;
	var client = self._client;
	var parentId = '/v', // for our seeds, /v is always 'Projects'
	    node = client.getNode(nodePath),
	    nodeId = node.getId(),
	    baseId = node.getBaseId(),
	    baseNode = client.getNode(baseId),
	    baseName = baseNode.getAttribute('name');

	if (baseName == 'FCO') { // 
	    var childCreationParams = {
		parentId: parentId,  // Should be Projects (/v)
		baseId: nodeId,    // should be META:Project
	    };
	    client.createChild(childCreationParams, 'Creating new Project');
	}
	else if (baseName == 'Project') {
            var params = {parentId: parentId};
	    params[nodeId] = {
                'attributes': {
                    'name': node.getAttribute('name') + ' Copy'
                }
            };
            self._client.startTransaction();
            self._client.copyMoreNodes(params);
            self._client.completeTransaction();
	}
    };

    /* * * * * * * * Visualizer event handlers * * * * * * * */

    RootVizWidget.prototype._makeDroppable = function () {
	var self = this,
	    desc;
        self.$el.addClass('drop-area');
        //self._div.append(self.__iconAssignNullPointer);

        dropTarget.makeDroppable(self.$el, {
            over: function (event, dragInfo) {
                if (self._isValidDrop(dragInfo)) {
                    self.$el.addClass('accept-drop');
                } else {
                    self.$el.addClass('reject-drop');
                }
            },
            out: function (/*event, dragInfo*/) {
                self.$el.removeClass('accept-drop reject-drop');
            },
            drop: function (event, dragInfo) {
                if (self._isValidDrop(dragInfo)) {
		    self.createProject(dragInfo[DROP_CONSTANTS.DRAG_ITEMS][0]);
                }
                self.$el.removeClass('accept-drop reject-drop');
            }
        });
    };

    RootVizWidget.prototype.onNodeSelect = function (id) {
	if (id)
	    WebGMEGlobal.State.registerActiveSelection([id]);
    };

    RootVizWidget.prototype.onNodeClick = function (id) {
        // This currently changes the active node to the given id and
        // this is overridden in the controller.
    };

    RootVizWidget.prototype.onBackgroundDblClick = function () {
    };

    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
    RootVizWidget.prototype.destroy = function () {
    };

    RootVizWidget.prototype.onActivate = function () {
    };

    RootVizWidget.prototype.onDeactivate = function () {
    };

    return RootVizWidget;
});
