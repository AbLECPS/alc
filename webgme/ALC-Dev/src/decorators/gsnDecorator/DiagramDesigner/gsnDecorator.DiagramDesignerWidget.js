/*globals define, _, $*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
	'bower_components/EpicEditor/epiceditor/js/epiceditor.min.js',
    'js/Constants',
    'js/NodePropertyNames',
    'js/Widgets/DiagramDesigner/DiagramDesignerWidget.DecoratorBase',
	'./gsnTextEditorDialog',
	'decorators/DocumentDecorator/DiagramDesigner/DocumentEditorDialog',
    'text!./gsnDecorator.DiagramDesignerWidget.html',
    'css!./gsnDecorator.DiagramDesignerWidget.css',
	
], function (marked,CONSTANTS, nodePropertyNames, DiagramDesignerWidgetDecoratorBase, gsnTextEditorDialog, DocumentEditorDialog, gsnDecoratorTemplate) {

    'use strict';

    var gsnDecorator,
        __parent__ = DiagramDesignerWidgetDecoratorBase,
        __parent_proto__ = DiagramDesignerWidgetDecoratorBase.prototype,
        DECORATOR_ID = 'gsnDecorator',
		EXCLUDED_POINTERS = [CONSTANTS.POINTER_BASE, CONSTANTS.POINTER_SOURCE, CONSTANTS.POINTER_TARGET],
		TEXT_EDIT_BTN_BASE = $('<i class="glyphicon glyphicon-cog text-gsn" title="Edit"/>'),
        DROPDOWN_EDIT_BTN_BASE = $('<div class="dropdown text-gsn"><a role="button" data-toggle="dropdown" href="#"><i class="glyphicon glyphicon-cog text-gsn" data-toggle="dropdown" title="Edit"/></a></div>'),
		STATUS_BTN_BASE = $('<i class="glyphicon glyphicon-chevron-left gsnstatus"/>'),
		STATUS_BTN_BASE2 = $('<i class="glyphicon glyphicon-chevron-right gsnstatus2"/>'),
		REF_BTN_BASE = $('<i class="glyphicon glyphicon-share text-gsn"/>');

    gsnDecorator = function (options) {
        var opts = _.extend({}, options);

        __parent__.apply(this, [opts]);

        this.name = '';
		this.description= '';
		this.defines ='';
		this.requires = '';
		this.gsnid='';
		this.isRef=false;
		this.hasLink = false;
		this.refobj='';
		this.status=false;
		this.refID = '';
		this.isReq = 0;
		this.reqtext ='';
		this.reqid ='';
		this.reqrisk ='';
		this.sets = {};
        this.references = {};
        this.referenceList = {};
		
		marked.setOptions({
            gfm: true,
            tables: true,
            breaks: true,
            pedantic: true,
            sanitize: false,
            smartLists: true,
            smartypants: true
        });

        this.logger.debug('gsnDecorator ctor');
    };

    _.extend(gsnDecorator.prototype, __parent_proto__);
    gsnDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    gsnDecorator.prototype.$DOMBase = $(gsnDecoratorTemplate);

    gsnDecorator.prototype.on_addTo = function () {
        let self = this,
			client = this._control._client,
        	nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            metaname;
        
        self.logger.debug('hostname ' + window.location.hostname);
        self.logger.debug('port ' + window.location.port);
        var id = '/y/f/W';
        var parentid = id.substring(0,id.lastIndexOf('/'));

        var address = window.location.origin + '/?project=' + encodeURIComponent(client.getActiveProjectId());
                    address += '&branch=' + encodeURIComponent(client.getActiveBranchName());
                    address += '&node='+ encodeURIComponent(parentid);
                    address += '&selection='+encodeURIComponent(id);

        self.logger.debug('address ' + address);            

        // Check node exists
		if (!nodeObj)
		{
			return;
		}

		metaname = this.getMetaName(nodeObj);

		// Check metaname and perform any type-specific initialization steps
		if (metaname === 'SupportRef' || metaname === 'InContextRef')
		{
			this.isRef=true;
			self._updatePointer("Ref");
			if (this.refobj)
			{
				metaname = this.getMetaName(this.refobj);
			}
		}
		else if (metaname === 'Hazard') {
			//this.hasLink=true;
            //self._updatePointer("Effect");
            this.skinParts.$metaname = this.$el.find('.metaname');
		}
        else if (metaname === 'BowtieEvent') {
            this.skinParts.$metaname = this.$el.find('.metaname');
        }
        else if (metaname === 'Mitigation') {
            this.skinParts.$metaname = this.$el.find('.metaname');
        }
		else if (metaname === 'Requirement')
		{
			this.isReq = 1;
            this.skinParts.$metaname = this.$el.find('.metaname');
		}

        // Check if node contains any sets and load all linked nodes if so
        let setNames = nodeObj.getSetNames();
        for (let i = 0; i < setNames.length; i++){
            let setName = setNames[i],
                setIds = nodeObj.getMemberIds(setName),
                set = {},
                patterns = {};
            for (let j = 0; j < setIds.length; j++) {
                let setNode = client.getNode(setIds[j]);
                // If node has been loaded, store it in funcSet. Otherwise add to patterns.
                if (setNode) {
                    set[setIds[j]] = setNode;
                }
                else {
                    set[setIds[j]] = null;
                    patterns[setIds[j]] = {children: 0};
                }
            }
            this.sets[setName] = set;

            // If some members of the set have not yet been loaded, add them to territory and attach eventHandler
            let patternKeys = Object.keys(patterns);
            if (patternKeys.length > 0) {
                let userId = client.addUI(null, function (events) {
                    self.setEventHandler(self, setName, events);
                });
                client.updateTerritory(userId, patterns);
            }
        }

        // repeat above for nodeobj children of type modelref.
        let childnodes = nodeObj.getChildrenIds();
        let patterns = {};
        for (let i = 0; i < childnodes.length; i++){
            let childID = childnodes[i];
            let childnode = client.getNode(childID);
            let childMetaName = self.getMetaName(childnode);
            if (childMetaName.indexOf('ModelRef')==-1)
            {
                continue;
            }

            let model_type = childnode.getAttribute('element_type');
            let ptrid= childnode.getPointerId('Ref');
            if ((!ptrid) || (ptrid == ''))
            {
                continue;

            }
            let ptrNode = client.getNode(ptrid);
            let refKeys = Object.keys(self.references);
            if (refKeys.indexOf(model_type)==-1)
            {
                self.references[model_type]=[];
            }
            self.references[model_type].push(ptrid);
            if (ptrNode)
            {
                self.referenceList[ptrid] = ptrNode;
            }
            else{
                self.referenceList[ptrid] = null;
                patterns[ptrid] = {children: 0};
            }
        }

        // If some members of the set have not yet been loaded, add them to territory and attach eventHandler
        let patternKeys = Object.keys(patterns);
        if (patternKeys.length > 0) {
            let userId = client.addUI(null, function (events) {
                self.setEventHandler(self, 'ModelRef', events);
            });
            client.updateTerritory(userId, patterns);
        }
        

		// The metaname part may be used for any object which has a link
        // Part is used to display info about the linked object
		if (this.hasLink) {
            this.skinParts.$metaname = this.$el.find('.metaname');
        }
		

        // set title editable on double-click
        /*this.skinParts.$name.on('dblclick.editOnDblClick', null, function (event) {
            if (self.hostDesignerItem.canvas.getIsReadOnlyMode() !== true) {
                $(this).editInPlace({
                    class: '',
                    onChange: function (oldValue, newValue) {
                        self._onNodeTitleChanged(oldValue, newValue);
                    }
                });
            }
            event.stopPropagation();
            event.preventDefault();
        });*/

        // Add name and description to skinParts
        this.skinParts.$name = this.$el.find('.name');
        this.skinParts.$description = this.$el.find('.description');
		
		//render text-editor based GSN editing UI piece
		if (!this.isRef){
			// Construct dropdown menu for selecting which attribute to edit (GOAL nodes only)
            if (metaname === "Goal") {
                this.skinParts.$textGSNEditorBtn = DROPDOWN_EDIT_BTN_BASE.clone();
                this.skinParts.$editorDropdown = document.createElement("ul");
                let dropdown = this.skinParts.$editorDropdown,
                    items = ["description", "defines", "requires"];
                dropdown.className = "dropdown-menu dropdown-menu-right";
                for (let i = 0; i < items.length; i++) {
                    let item = document.createElement("li"),
                        a = document.createElement("a");
                    a.className = "dropdown-item";
                    a.text = items[i];
                    a.onclick = function () {
                        self._showGSNTextEditorDialog(items[i]);
                    };
                    item.appendChild(a);
                    dropdown.appendChild(item);
                }

                this.skinParts.$textGSNEditorBtn.append(dropdown);
            }

            // Non-Goal nodes use simplified button
            else{
                // set description editable on double-click
                this.skinParts.$description.on('dblclick.editOnDblClick', null, function (event) {
                    if (self.hostDesignerItem.canvas.getIsReadOnlyMode() !== true) {
                        self._showGSNTextEditorDialog("description");
                    }
                    event.stopPropagation();
                    event.preventDefault();
                });
            }

            if (this.hasLink){
                this.skinParts.$textGSNEditorBtn= REF_BTN_BASE.clone();
			}
		}
		else
			this.skinParts.$textGSNEditorBtn= REF_BTN_BASE.clone();
		
		this.$el.append(this.skinParts.$textGSNEditorBtn);
		if (!this.isReq)
		{
			this.skinParts.$gsnstatus= STATUS_BTN_BASE.clone();
			this.$el.append(this.skinParts.$gsnstatus);
			this.skinParts.$gsnstatus2= STATUS_BTN_BASE2.clone();
			this.$el.append(this.skinParts.$gsnstatus2);
		}


		if (!this.isRef && !this.hasLink)
		{
            // Goal nodes should show the dropdown selection on click
            if (metaname === "Goal") {
                this.skinParts.$textGSNEditorBtn.on('click', function (event) {
                    // Toggle icon between Cog and Remove/Close button
                    let icon = self.skinParts.$textGSNEditorBtn[0].childNodes[0].childNodes[0];
                    if (icon.className.includes("glyphicon-remove")) {
                        icon.className = "glyphicon glyphicon-cog text-gsn";
                        icon.style = "";
                    }
                    else {
                        icon.className = "glyphicon glyphicon-remove text-gsn";
                        icon.style = "color:red";
                    }

                    // Toggle dropdown
                    if (self.hostDesignerItem.canvas.getIsReadOnlyMode() !== true) {
                        self.skinParts.$editorDropdown.classList.toggle("show");
                    }
                    event.stopPropagation();
                    event.preventDefault();
                });
            }

            // // Other nodes should immediately open the editor
            // else{
            //     this.skinParts.$textGSNEditorBtn.on('click', function (event) {
            //
            //         if (self.hostDesignerItem.canvas.getIsReadOnlyMode() !== true) {
            //             self._showGSNTextEditorDialog("description");
            //         }
            //         event.stopPropagation();
            //         event.preventDefault();
            //     });
            // }
		}
		else {
			this.skinParts.$textGSNEditorBtn.on('click', function (event) {
				self.logger.debug('is ref');
				if (self.refobj)
					self._navigateToPointerTarget({x: event.clientX, y: event.clientY});
				else
					self.logger.debug('null ref object');
				event.stopPropagation();
				event.preventDefault();
        	});
		}

		// Set block background color
		let bgColor = this.getBackgroundColor(metaname);
		this.$el.css({'background-color': bgColor});

        //let the parent decorator class do its job first
        __parent_proto__.on_addTo.apply(this, arguments);
        // setTimeout(this.onRenderGetLayoutInfo.bind(this), 5);
        // var autorouter = this.hostDesignerItem.canvas.connectionRouteManager;
        // var connections = this.hostDesignerItem.canvas.connectionIds.slice(0);
        // setTimeout(autorouter.redrawConnections.bind(autorouter, connections), 6);

		this.update();
    };

    gsnDecorator.prototype.destroy = function () {
        let nodeId = this._metaInfo[CONSTANTS.GME_ID],
            client = this._control._client,
            nodeObj = client.getNode(nodeId);
        debugger;
    };
	
	gsnDecorator.prototype._updatePointer = function (pointerName) {
		let self = this,
			client = this._control._client,
            nodeId = this._metaInfo[CONSTANTS.GME_ID],
            nodeObj = client.getNode(nodeId),
            ptrid;
		if (!nodeObj)
		{
			self.refobj= '';
			return;
		}
        
		self.logger.debug('in updatepointer');
		ptrid= nodeObj.getPointerId(pointerName);
		self.refobj='';
		if (ptrid)
		{
			self.refID = ptrid;
			self.logger.debug('ptr id' + ptrid);
			self.refobj = client.getNode(ptrid);
		}

		if (self.refobj) {
		    // Referenced object has already
            self.logger.debug('ref obj');

            // If this is the original definition Hazard (not a child instance), then update corresponding Effect pointer
            // Determine if this is the original Hazard by checking if it is contained within a "Hazards" block
            let parentNode = client.getNode(nodeObj.getParentId()),
                parentMetaName = this.getMetaName(parentNode);
            if (parentMetaName === "Hazards") {
                client.setPointer(ptrid, 'Hazard', nodeId, 'GSN-Decorator');
            }
        } else if ((self.isRef || self.hasLink) && ptrid) {
            let patterns = {};
            patterns[''] = {children: 0};
            patterns[ptrid] = {children: 0};
            var userId = client.addUI(null, function (events) {
                self.eventHandler(self, events)
            });

            client.updateTerritory(userId, patterns);

            /*self.logger.debug('ptr id' + ptrid);
            self.refobj=client.getNode(ptrid);
            if (self.refobj)
                self.logger.debug('ref obj');
            else
                self.logger.debug('no ref obj');*/
        } else {
            self.refobj = '';
            self.logger.debug('no ref obj');
        }
    };

    gsnDecorator.prototype.setEventHandler = function(context, setName, events) {
        let nodeObj,
            self = context,
            selfId = self._metaInfo[CONSTANTS.GME_ID],
            client = self._control._client,
            selfObj = client.getNode(selfId);

        self.logger.debug('in set event handler');

        // Verify set exists
        if (setName != 'ModelRef')
        {
            let sets = Object.keys(self.sets);
            if (sets.indexOf(setName) > -1) {
                let setDict = self.sets[setName],
                    setKeys = Object.keys(setDict);
                // Loop over current events
                for (let i = 0; i < events.length; i += 1) {
                    // Verify node ID of this event is a member of set
                    if (setKeys.indexOf(events[i].eid) > -1) {
                        // If this node was loaded or updated, update function dictionary entry
                        if (events[i].etype === 'load' || events[i].etype === 'update') {
                            setDict[events[i].eid] = client.getNode(events[i].eid);
                        }
                        // If this node was unloaded, clear function dictionary entry
                        else if (events[i].etype === 'unload') {
                            setDict[events[i].eid] = null;
                        }
                        else {
                            // "Technical events" not used.
                        }
                    }
                }
                // Update self.sets
                self.sets[setName] = setDict;
            }
        }
        else {
            let referencekeys = Object.keys(self.referenceList);
            if (referencekeys.length >0)
            {
                for (let i = 0; i < events.length; i += 1) {
                    // Verify node ID of this event is a member of set
                    if (referencekeys.indexOf(events[i].eid) > -1) {
                        // If this node was loaded or updated, update function dictionary entry
                        if (events[i].etype === 'load' || events[i].etype === 'update') {
                            self.referenceList[events[i].eid] = client.getNode(events[i].eid);
                        }
                        // If this node was unloaded, clear function dictionary entry
                        else if (events[i].etype === 'unload') {
                            self.referenceList[events[i].eid] = null;
                        }
                        else {
                            // "Technical events" not used.
                        }
                    }
                }
            }

        }

        // Update
        self.update();
    };
	
	gsnDecorator.prototype.eventHandler = function(context,events) {
        let i,
            nodeObj,
            self = context,
            selfId = self._metaInfo[CONSTANTS.GME_ID],
            client = self._control._client,
            selfObj = client.getNode(selfId);

        self.logger.debug('in event handler');

        if (self.isRef || self.hasLink) {
            for (i = 0; i < events.length; i += 1) {
                self.logger.debug('eventhandler eid ' + events[i].eid + ' refid ' + self.refID);
                if (self.refID && events[i].eid !== self.refID) {
                    self.logger.debug('eventhandler eid ' + events[i].eid);
                    continue;
                }

                nodeObj = client.getNode(events[i].eid);
                if (!nodeObj) {
                    if (events[i].etype === 'unload') {
                        // The node was removed from the model (we can no longer access it).
                        // We still get the path/id via events[i].eid
                        if (self.refID) {
                            self.refobj = '';
                        }
                    }
                    continue;
                }

                if (events[i].etype === 'load') {
                    // The node is loaded and we have access to it.
                    // It was either just created or this is the initial
                    // updateTerritory we invoked.
                    if (self.refID) {
                        self.refobj = nodeObj;

                        // If this is the original definition Hazard (not a child instance), then update corresponding Effect pointer
                        // Determine if this is the original Hazard by checking if it is contained within a "Hazards" block
                        let parentNode = client.getNode(nodeObj.getParentId()),
                            parentMetaName = this.getMetaName(parentNode);
                        if (parentMetaName === "Hazards") {
                            client.setPointer(self.refID, 'Hazard', selfId, 'GSN-Decorator');
                        }
                    }
                    // The nodeObj contains methods for querying the node, see below.

                } else if (events[i].etype === 'update') {
                    // There were changes to the node (some might not apply to your application).
                    // The node is still loaded and we have access to it.
                    nodeObj = client.getNode(events[i].eid);
                    if (self.refID && nodeObj) {
                        self.refobj = nodeObj;
                    }
                    else
                        self.refObj = '';

                } else if (events[i].etype === 'unload') {
                    // The node was removed from the model (we can no longer access it).
                    // We still get the path/id via events[i].eid
                    if (self.refID) {
                        self.refobj = '';
                    }

                } else {
                    // "Technical events" not used.
                }
            }
        }

        self.update();
	};

	/*
	gsnDecoratorDiagramDesignerWidget.prototype._getNodeColorsFromRegistry = function () {
        var objID = this._metaInfo[CONSTANTS.GME_ID];

        this.fillColor = this.preferencesHelper.getRegistry(objID, REGISTRY_KEYS.COLOR, false);
        this.borderColor = this.preferencesHelper.getRegistry(objID, REGISTRY_KEYS.BORDER_COLOR, false);
        this.textColor = this.preferencesHelper.getRegistry(objID, REGISTRY_KEYS.TEXT_COLOR, false);
    };

	
    gsnDecoratorDiagramDesignerWidget.prototype.updateColors = function () {
        this._getNodeColorsFromRegistry();

        if (this.fillColor) {
            this.$el.css({'background-color': this.fillColor});
        } else {
            this.$el.css({'background-color': ''});
        }

        if (this.borderColor) {
            this.$el.css({'border-color': this.borderColor});
            this._skinParts.$name.css({'border-color': this.borderColor});
        } else {
            this.$el.css({
                'border-color': '',
                'box-shadow': ''
            });
            this._skinParts.$name.css({'border-color': ''});
        }

        if (this.textColor) {
            this.$el.css({color: this.textColor});
        } else {
            this.$el.css({color: ''});
        }
    };
	*/

    gsnDecorator.prototype._renderName = function () {
        let client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
			obj = nodeObj,
			prefix = '',
            suffix = '';

        //render GME-ID in the DOM, for debugging
        this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});

        // Check node exists
		if(!nodeObj) {
            return;
        }

        // Check if node is a reference node and add prefix
		if (this.isRef)
		{
			if (this.refobj)
			{
				obj=this.refobj;
				prefix='Ref - ';
			}
		}

		// Check if this node is an instance of a type node
        let baseName = this.getBaseName(nodeObj),
            metaName = this.getMetaName(nodeObj);
		if (baseName !== metaName) {
            prefix = baseName + '::';
        }

		// Get name and GSN ID
        if (obj) {
            this.name = obj.getAttribute(nodePropertyNames.Attributes.name) || '';
			this.gsnid = obj.getAttribute('gsnid') || '';

			// Show metaname for requirements
			if (this.isReq)
			{
				this.skinParts.$metaname.text('<<Requirement>>');
				this.skinParts.$metaname.css({color: '#000000'});
			}
        }

        // Fill name text
		if (this.gsnid === '')
			this.skinParts.$name.text(prefix + this.name + suffix);
		else
			this.skinParts.$name.text(prefix + this.name + ':'+this.gsnid + suffix);

		// Set name color
		this.skinParts.$name.css({color: '#000000'});
		if (this.isRef && this.refobj)
		{
			this.skinParts.$name.css({color: '#aa0000'});
		}
    };

    gsnDecorator.prototype.checkReferenceList = function () {
        let client = this._control._client;
        let self = this;
        let keys = Object.keys(self.referenceList);
        let newDesc = '';
        if (keys.length == 0)
        {
            return newDesc;
        }

        
        let modelkeys = Object.keys(self.references);
        modelkeys.sort();
        for (let i = 0; i < modelkeys.length; i++) {
            let modeltype = modelkeys[i];
            let links = [];
            let linkdict = {};

            for (let j = 0; j < self.references[modeltype].length; j++) {
                let linknodeid  = self.references[modeltype][j];
                if (keys.indexOf(linknodeid) ==-1)
                {
                    continue
                }
                let linknode = self.referenceList[linknodeid];
                if (linknode)
                {
                    let nodename = linknode.getAttribute('name');
                    let parentid = linknodeid.substr(0,linknodeid.lastIndexOf('/'));
                    var address = window.location.origin + '/?project=' + encodeURIComponent(client.getActiveProjectId());
                        address += '&branch=' + encodeURIComponent(client.getActiveBranchName());
                        address += '&node='+ encodeURIComponent(parentid);
                        address += '&selection='+encodeURIComponent(linknodeid);
                    let linkstr = '<a href="'+address+'" target="_blank">'+nodename+'</a>';
                    //links.push(linkstr);
                    linkdict[nodename]=linkstr;
                }
            }
            let linkkeys = Object.keys(linkdict);
            

            if (linkkeys.length > 0)
            {
                linkkeys.sort();
                if (newDesc == '')
                {
                    newDesc += '\n\n --- \n';
		            newDesc += 'References : <br>';
                }

                newDesc += '\n\n*'+modeltype + '*\n'
                for (let j = 0; j < linkkeys.length; j++) {
                    newDesc += '\n- '+linkdict[linkkeys[j]];
                }

            }
            
        }
        self.logger.debug(newDesc);
        return newDesc;
        

    };
	
	gsnDecorator.prototype._renderDescription = function () {
        let client = this._control._client,
			nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
			obj = nodeObj;

        //render GME-ID in the DOM, for debugging
        this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});

        // Check node exists
        if (!nodeObj){
        	return;
		}

        // For reference objects, render attributes of the referenced node
        if (this.isRef)
        {
			if (this.refobj)
				obj=this.refobj;
        }

        // Fill description
		let newDesc = '';
		if (this.isReq)
		{
			this.reqtext = obj.getAttribute('Text') || '';
			this.reqid = obj.getAttribute('Id') || '';
			this.reqrisk = obj.getAttribute('Risk') || '';

			if (this.reqid !== '')
				newDesc = 'Id&nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;**'+this.reqid + '**<br><br>';
			else
				newDesc = 'Id&nbsp;&nbsp;&nbsp;&nbsp;:<br><br>';
			if (this.reqtext !== '')
				newDesc += 'Text:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**'+this.reqtext + '**';
			else
				newDesc += 'Text:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;';

			if (this.reqrisk !== '')
				newDesc += '<br><br>Risk:&nbsp;**'+this.reqrisk+'**';
		}
		else
		{
			newDesc = obj.getAttribute('description') || '';
			this.defines = obj.getAttribute('defines') || '';
			if (this.defines !== '')
			{
				newDesc += '\n\n --- \n\n';
				newDesc += 'Defines : <br>';
				newDesc +=  this.defines;
			}
			this.requires = obj.getAttribute('requires') || '';
			if (this.requires !== '')
			{
				newDesc += '\n\n --- \n\n';
				newDesc += 'Requires : <br>';
				newDesc +=  this.requires;
			}

            newDesc += this.checkReferenceList();

			// Mitigations define a set containing any number of required functions as well as a set for failure effects
            // Check if sets exists and display the members if so
            let sets = Object.keys(this.sets);
			if (sets.indexOf('RequiredFunctions') > -1) {
                let funcDict = this.sets['RequiredFunctions'],
                    funcKeys = Object.keys(funcDict),
                    funcDesc = '',
                    validCnt = 0;
                funcDesc += '\n\n --- \n\n';
                funcDesc += 'Required Functions : <br>';
                for (let i = 0; i < funcKeys.length; i++) {
                    let funcNode = funcDict[funcKeys[i]];
                    if (funcNode) {
                        validCnt++;
                        funcDesc += funcNode.getAttribute('name');
                        // TODO: Find out how to do hyperlinks in WebGME. Normal Markdown style doesn't seem to work
                        //funcDesc += '[' + funcNode.getAttribute('name') + '](' + funcKeys[i] + ')';
                    }
                }

                // If at least 1 valid function node, add required functions to node description
                if (validCnt > 0){
                    newDesc += funcDesc;
                }
            }

            if (sets.indexOf('InhibitingFailures') > -1) {
                let setDict = this.sets['InhibitingFailures'],
                    setKeys = Object.keys(setDict),
                    setDesc = '',
                    validCnt = 0;
                setDesc += '\n\n --- \n\n';
                setDesc += 'Inhibiting Failures: <br>';
                for (let i = 0; i < setKeys.length; i++) {
                    let setNode = setDict[setKeys[i]];
                    if (setNode) {
                        validCnt++;
                        setDesc += setNode.getAttribute('name');
                        // TODO: Find out how to do hyperlinks in WebGME. Normal Markdown style doesn't seem to work
                        //funcDesc += '[' + funcNode.getAttribute('name') + '](' + funcKeys[i] + ')';
                    }
                }

                // If at least 1 valid function node, add required functions to node description
                if (validCnt > 0){
                    newDesc += setDesc;
                }
            }
		}

		// If new description is empty, add placeholder text
		// Document Editor Dialog appends newline when no text is entered. Have to catch this case.
        if(newDesc === '' || newDesc === '\n'){
            newDesc = '*Edit Description*';
        }

		// Update description if changed
		if (this.description !== newDesc) {
			this.description = newDesc;
			this.skinParts.$description.empty();
			this.skinParts.$description.append($(marked(this.description)));
		}

		// Show or hide "in development" icon
		if (!this.isReq)
		{
			this.status = obj.getAttribute('In Development') || false;
			if (this.status)
			{
				this.skinParts.$gsnstatus.css({'visibility': 'visible'});
				this.skinParts.$gsnstatus2.css({'visibility': 'visible'});
			}
			else
			{
				this.skinParts.$gsnstatus.css({'visibility': 'hidden'});
				this.skinParts.$gsnstatus2.css({'visibility': 'hidden'});
			}
		}
    };
	
	
    gsnDecorator.prototype.update = function () {
        let client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            // newName = '',
			// newgsnid= '',
			// status=false,
			baseid=null,
			obj = nodeObj,
			metaObj,
			metaname = '',
			bgColor;

        // Check node exists
        if (!nodeObj)
        {
			return;
        }

        baseid = obj.getBaseId();

        // Render name and description
		this._renderName();
		this._renderDescription();

		// Get meta name
        metaObj = client.getNode(baseid);
        if (metaObj)
            metaname = metaObj.getAttribute(nodePropertyNames.Attributes.name) || '';

        // Set background color
        if (this.isRef){
            // References should take the color of the node they point to
            let refMetaName = '';
            if (this.refobj) {
                let refBaseID = this.refobj.getBaseId(),
                    refMetaObj = client.getNode(refBaseID);
                if(refMetaObj) {
                    refMetaName = refMetaObj.getAttribute(nodePropertyNames.Attributes.name) || '';
                }
            }
			bgColor = this.getBackgroundColor(refMetaName);
		}
		// Background color/type of Bowtie Events based on number or incoming/outgoing connections
		else if (metaname === "BowtieEvent") {
            // FIXME: Inefficient to call getNodeConnections every update.
            // Better to call this once, then keep the list of connections updated as connections are modified
            // Not sure if that is possible to do purely within the decorator
            let nodeConns = this.getNodeConnections(this._metaInfo[CONSTANTS.GME_ID]);
            if (nodeConns.outConns.length > 0 && nodeConns.inConns.length > 0){
                bgColor = this.getBackgroundColor(metaname, 'TopEvent');
                this.skinParts.$metaname.text('<<Top Event>>');
                this.skinParts.$metaname.css({color: '#000000'});
            }
            else if (nodeConns.outConns.length > 0) {
                bgColor = this.getBackgroundColor(metaname, 'Threat');
                this.skinParts.$metaname.text('<<Threat>>');
                this.skinParts.$metaname.css({color: '#000000'});
            }
            else if (nodeConns.inConns.length > 0) {
                bgColor = this.getBackgroundColor(metaname, 'Consequence');
                this.skinParts.$metaname.text('<<Consequence>>');
                this.skinParts.$metaname.css({color: '#000000'});
            }
            else {
                bgColor = this.getBackgroundColor(metaname);
                this.skinParts.$metaname.text('');
            }
        }
		else if (metaname === "Hazard") {
            this.skinParts.$metaname.text('<<Hazard>>');
        }
		else if (metaname === "Mitigation") {
            this.skinParts.$metaname.text('<<Barrier>>');
        }
        // For objects with a link to another node (but that are not direct references), display link name on object
        else if (this.hasLink) {
            if (this.refobj) {
                let refName = this.refobj.getAttribute("name"),
                    refMetaType = this.getMetaName(this.refobj);
                if (refMetaType === "Effect") {
                    this.skinParts.$metaname.text('<<Effect - ' + refName + '>>');
                }
                else {
                    this.skinParts.$metaname.text('<<' + refName + '>>');
                }
                this.skinParts.$metaname.css({color: '#000000'});
            }
        }
		else {
            // All other nodes colored based on meta type
            bgColor = this.getBackgroundColor(metaname);
		}
        this.$el.css({'background-color': bgColor});
    };

    gsnDecorator.prototype.getConnectionAreas = function (id /*, isEnd, connectionMetaInfo*/) {
        var result = [],
            edge = 10,
            LEN = 20;

        //by default return the bounding box edge's midpoints

        if (id === undefined || id === this.hostDesignerItem.id) {
            //NORTH
            result.push({
                id: '0',
                x1: edge,
                y1: 0,
                x2: this.hostDesignerItem.getWidth() - edge,
                y2: 0,
                angle1: 270,
                angle2: 270,
                len: LEN
            });

            //EAST
            result.push({
                id: '1',
                x1: this.hostDesignerItem.getWidth(),
                y1: edge,
                x2: this.hostDesignerItem.getWidth(),
                y2: this.hostDesignerItem.getHeight() - edge,
                angle1: 0,
                angle2: 0,
                len: LEN
            });

            //SOUTH
            result.push({
                id: '2',
                x1: edge,
                y1: this.hostDesignerItem.getHeight(),
                x2: this.hostDesignerItem.getWidth() - edge,
                y2: this.hostDesignerItem.getHeight(),
                angle1: 90,
                angle2: 90,
                len: LEN
            });

            //WEST
            result.push({
                id: '3',
                x1: 0,
                y1: edge,
                x2: 0,
                y2: this.hostDesignerItem.getHeight() - edge,
                angle1: 180,
                angle2: 180,
                len: LEN
            });
        }

        return result;
    };

    /**************** EDIT NODE TITLE ************************/

    gsnDecorator.prototype._onNodeTitleChanged = function (oldValue, newValue) {
        var client = this._control._client;

        client.setAttributes(this._metaInfo[CONSTANTS.GME_ID], nodePropertyNames.Attributes.name, newValue);
    };
	
	gsnDecorator.prototype._onNodeDescriptionChanged = function (oldValue, newValue) {
        var client = this._control._client;

		if (!this.isReq)
			client.setAttributes(this._metaInfo[CONSTANTS.GME_ID], 'description', newValue);
		else
			client.setAttributes(this._metaInfo[CONSTANTS.GME_ID], 'Text', newValue);
			
    };
	
	gsnDecorator.prototype._showGSNTextEditorDialog = function (attrName) {
        let client = this._control._client,
			dialog = new DocumentEditorDialog(),
			//dialog = new gsnTextEditorDialog(),
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
			descriptionStr = nodeObj.getAttribute(attrName) || '',
			self = this;

		if (this.isReq)
		{
			descriptionStr = nodeObj.getAttribute('Text') || '';
		}

		dialog.initialize(descriptionStr,
		function (text) {
			try {
				if (!self.isReq)
					client.setAttributes(self._metaInfo[CONSTANTS.GME_ID], attrName, text);
				else
					client.setAttributes(self._metaInfo[CONSTANTS.GME_ID], 'Text', text);
				// These lines don't seem to be necessary. Decorator updates correctly without them.
				// self.skinParts.$description.empty();
				// self.skinParts.$description.append($(marked(text)));
			} catch (e) {
				self.logger.error('Saving META failed... Either not JSON object or something else went wrong...');
			}
		});

		var dheader = dialog._dialog.find('.modal-header').first();
		dheader.html('<h3>Edit ' + attrName + '</h3>');

		dialog.show();
	};
		
	gsnDecorator.prototype._navigateToPointerTarget = function (mousePos) {
		var self= this;
        var targetNodeObj = this.refobj;
        if (targetNodeObj) {
            if (targetNodeObj.getParentId() || targetNodeObj.getParentId() === CONSTANTS.PROJECT_ROOT_ID) {
                WebGMEGlobal.State.registerActiveObject(targetNodeObj.getParentId());
                WebGMEGlobal.State.registerActiveSelection([targetNodeObj._id]);
            } else {
                WebGMEGlobal.State.registerActiveObject(CONSTANTS.PROJECT_ROOT_ID);
                WebGMEGlobal.State.registerActiveSelection([targetNodeObj._id]);
            }
        }
		
    };

    /**************** END OF - EDIT NODE TITLE ************************/

    gsnDecorator.prototype.doSearch = function (searchDesc) {
        var searchText = searchDesc.toString();
        if (this.name && this.name.toLowerCase().indexOf(searchText.toLowerCase()) !== -1) {
            return true;
        }

        return false;
    };

    gsnDecorator.prototype.getBackgroundColor = function (metaname, subtype = '') {
        let bgColor = '#dedede';

        if (metaname === 'Goal' || metaname === "Mitigation") {
            bgColor = '#99ccff';
        }
        else if (metaname === 'Strategy') {
            bgColor = '#ccff99';
        }
        else if (metaname === 'Solution') {
            bgColor = '#ffb266';
        }
        else if (metaname === 'Context') {
            bgColor = '#ffffbb';
        }
        else if (metaname === 'Assumption') {
            bgColor = '#f8f8f8';
        }
        else if (metaname === 'Justification' || metaname === "BowtieEvent") {
        	if (subtype === 'Threat'){
                bgColor = '#ccffe5';
            }
            else if (subtype === 'TopEvent') {
                bgColor = '#ffa500';
            }
            else if (subtype === 'Consequence') {
                bgColor = '#CD5c5c';
            }
            else {
                bgColor = '#ccffe5';
            }
        }
        else if (metaname === 'Requirement') {
            bgColor = '#ffffff';
        }
        else if (metaname === 'Hazard') {
        	bgColor = '#ffff00';
		}

        return bgColor;
    };

    gsnDecorator.prototype.getMetaName = function(nodeObj) {
        if (!nodeObj) {
            return '';
        }

        let client = this._control._client,
            metaId = nodeObj.getMetaTypeId(),
            metaObj = client.getNode(metaId);

        if (!metaObj)
        {
            return '';
        }
        return metaObj.getAttribute(nodePropertyNames.Attributes.name) || '';
    };

    gsnDecorator.prototype.getBaseName = function(nodeObj) {
        if (!nodeObj) {
            return '';
        }

        let client = this._control._client,
            baseId = nodeObj.getBaseId(),
            baseObj = client.getNode(baseId);

        if (!baseObj)
        {
            return '';
        }
        return baseObj.getAttribute(nodePropertyNames.Attributes.name) || '';
    };

    // Find all connections (in and out) to a given node.
    // This loops over all siblings of the given node, identifies connections, then checks 'src' and 'dst' of each connection
    gsnDecorator.prototype.getNodeConnections = function (nodeId) {
        let rv = {'inConns': [], 'outConns': []},
            client = this._control._client,
            nodeObj,
            parentPath,
            parentObj,
            childPaths;

        // Get all sibling nodes (get node, then get parent node, then get parent's children)
        nodeObj = client.getNode(nodeId);
        if (!nodeObj) {
            return rv;
        }
        parentPath = nodeObj.getParentId();
        if (!parentPath) {
            return rv;
        }
        parentObj = client.getNode(parentPath);
        if (!parentObj) {
            return rv;
        }
        childPaths = parentObj.getChildrenIds();

        // Loop over all siblings looking for connection nodes.
        for (let i = 0; i < childPaths.length; i++) {
            let childNode = client.getNode(childPaths[i]),
                metaname = this.getMetaName(childNode);

            // For connection nodes, see if they connect to node in question
            if (metaname === "BowtieConnection" || metaname === "BowtieConnection") {
                let srcId = childNode.getPointerId('src'),
                    dstId = childNode.getPointerId('dst');

                if (srcId === nodeId){
                    rv.outConns.push(childNode);
                }
                if (dstId === nodeId) {
                    rv.inConns.push(childNode);
                }
            }
        }

        return rv;
    };

    return gsnDecorator;
});