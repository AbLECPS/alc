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
    'text!./PropDecorator.DiagramDesignerWidget.html',
    'css!./PropDecorator.DiagramDesignerWidget.css'
	
], function (marked,CONSTANTS, nodePropertyNames, DiagramDesignerWidgetDecoratorBase, PropDecoratorTemplate) {

    'use strict';

    var PropDecorator,
        __parent__ = DiagramDesignerWidgetDecoratorBase,
        __parent_proto__ = DiagramDesignerWidgetDecoratorBase.prototype,
        DECORATOR_ID = 'PropDecorator',
		EXCLUDED_POINTERS = [CONSTANTS.POINTER_BASE, CONSTANTS.POINTER_SOURCE, CONSTANTS.POINTER_TARGET],
		REF_BTN_BASE = $('<i class="glyphicon glyphicon-share text-gsn"/>');
		

    PropDecorator = function (options) {
        var opts = _.extend({}, options);

        __parent__.apply(this, [opts]);

        this.name = '';
		this.description= '';
		this.metaname = '';
		this.min = 9999999;
		this.max =-9999999;
		this.changeIn=' - ';
		this.newValue=' - ';
		this.refobj='';
		this.refID='';
		this.varname='';
		this.modeStatus='';
		this.parentName ='';
		this.isTriggerCondition=0;
		this.cids = [];
		this.cidobjs = {};
		this.faultList=[];
		
		
		marked.setOptions({
            gfm: true,
            tables: true,
            breaks: true,
            pedantic: true,
            sanitize: false,
            smartLists: true,
            smartypants: true
        });

        this.logger.debug('PropDecorator ctor');
    };

    _.extend(PropDecorator.prototype, __parent_proto__);
    PropDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    PropDecorator.prototype.$DOMBase = $(PropDecoratorTemplate);

    PropDecorator.prototype.on_addTo = function () {
        var self = this;
		var client = this._control._client;
        var nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
		if (!nodeObj)
		{
			return;
		}
		
		var baseid= nodeObj.getBaseId();
		var metaObj = client.getNode(baseid);
		if (!metaObj)
		{
			return;
		}
		
		this.metaname = metaObj.getAttribute(nodePropertyNames.Attributes.name) || '';
		if (this.metaname == 'Trigger_Condition')
		{
			this.isTriggerCondition=1;
			self.updateTrigger();
		}
		
		if (this.metaname == 'Requirements' || this.metaname == 'OperationalImpact' || this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements')
		{
			this.skinParts.$textGSNEditorBtn= REF_BTN_BASE.clone();
			this.$el.append(this.skinParts.$textGSNEditorBtn);
			self._updatePointer();
			if (this.refobj)
			{
				self.varname= this.refobj.getAttribute(nodePropertyNames.Attributes.name) || '';
				if (this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements')
					self.getParentName();
			}
			else
				self.varname='';
			
			this.skinParts.$textGSNEditorBtn.on('dblclick.ptrDblClick', function (event) {
				if (!($(this).hasClass('ptr-nonset')))
					self._navigateToPointerTarget({x: event.clientX, y: event.clientY});
				event.stopPropagation();
				event.preventDefault();
        	});
		}
		
		
		
		
		
		
		
		this._renderName();
		this._renderDescription();
		        
		
		this.$el.css({'background-color': '#dedede'});
		
		
        //let the parent decorator class do its job first
        __parent_proto__.on_addTo.apply(this, arguments);
        
		this.update();
	

    };
	
	PropDecorator.prototype.updateTrigger = function () {
		var self = this,
			client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
		
		if (!nodeObj)
		{
			self.faultList= '';
			return;
		}
        
		self.cids = nodeObj.getMemberIds('fmset');
		self.logger.debug('member ids '+ self.cids);
		self.cidobjs = {};
		var c=0;
		// self.logger.debug('updating terroritory');
		
		 var patterns = {};
			patterns[''] = {children:0};
		for(c=0; c!= self.cids.length; c +=1)
		{
		    
			self.cidobjs[self.cids[c]]='';
			
			
			patterns[self.cids[c]]={children:0};
			
		
			
		}
		var userId = client.addUI(null, function(events) { self.eventHandler(self, events)});
		client.updateTerritory(userId, patterns);
			
    };
	
	PropDecorator.prototype._updatePointer = function () {
		var self = this,
			client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
		if (!nodeObj)
		{
			self.refobj= '';
			return;
		}
        
		self.logger.debug('in updatepointer');
		var ptrid= nodeObj.getPointerId('Ref');
		if (this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements')
			ptrid = nodeObj.getPointerId('ref');
		self.refobj='';
		if (ptrid)
		{
			self.logger.debug('ptr id '+ptrid);
			self.refID = ptrid;
			self.refobj=client.getNode(ptrid);
			if (this.skinParts.$textGSNEditorBtn.hasClass('ptr-nonset'))
			{
				this.skinParts.$textGSNEditorBtn.removeClass('ptr-nonset');
			}
			
		}
		else
		{
			self.logger.debug('no ptr id ');
			if (!this.skinParts.$textGSNEditorBtn.hasClass('ptr-nonset'))
				this.skinParts.$textGSNEditorBtn.addClass('ptr-nonset');
		}
		
		if (self.refobj)
			self.logger.debug('ref obj');
		else if ((this.metaname == 'Requirements' || this.metaname == 'OperationalImpact' || this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements') && ptrid)
		{
			self.logger.debug('updating terroritory');
			var patterns = {};
			patterns[''] = {children:0};
			patterns[ptrid]={children:0};
			var userId = client.addUI(null, function(events) { self.eventHandler(self, events)});
			client.updateTerritory(userId, patterns);
			
		}
		else
		{
			self.refobj='';
			self.logger.debug('no ref obj');
			
		}
			
    };
	
	PropDecorator.prototype.eventHandler= function(context,events) {
	  var i,
		  nodeObj,
		  self=context,
		  client = self._control._client;
		  

	  self.logger.debug('in event handler');
	  
	  for (i = 0; i < events.length; i += 1) {
	     self.logger.debug('eventhandler eid ' +events[i].eid + ' refid ' + self.refID);
	     if (self.refID && events[i].eid != self.refID)
		 {
			self.logger.debug('eventhandler eid ' +events[i].eid);
			 continue;
		 }
		 
		 nodeObj = client.getNode(events[i].eid);
		if (!nodeObj)
		{
			if (events[i].etype === 'unload') {
			  if (self.refID)
			  {
				self.refobj = '';
			  }
			  if (self.isTriggerCondition)
			  {
				self.cids=[];
				self.cidobjs = {};
			  }
		  }
		  continue;
		}
		 
		if (events[i].etype === 'load') {
		  // The node is loaded and we have access to it.
		  // It was either just created or this is the initial
		  // updateTerritory we invoked.
		  
		  
		  if (self.refID )
		  {
		    self.refobj = nodeObj;
		  }
		  
		  if (self.isTriggerCondition)
		  {
			  if (self.cids.indexOf(events[i].eid) >-1)
			   {
				 self.cidobjs[events[i].eid]=nodeObj;
			   }
		 }

		  // The nodeObj contains methods for querying the node, see below.
		} else if (events[i].etype === 'update') {
		  // There were changes to the node (some might not apply to your application).
		  // The node is still loaded and we have access to it.
		  nodeObj = client.getNode(events[i].eid);
		  if (self.refID && nodeObj)
		  {
		    self.refobj = nodeObj;
		  }
		  else if (self.isTriggerCondition && self.cids.indexOf(events[i].eid) >-1)
		  {
			self.cidobjs[events[i].eid]=nodeObj;
		   }
		  else 
		        self.refObj = '';
			  
		} else if (events[i].etype === 'unload') {
		  // The node was removed from the model (we can no longer access it).
		  // We still get the path/id via events[i].eid
		  if (self.refID)
		  {
		    self.refobj = '';
		  }
		  // else if (self.isTriggerCondition && self.cids.indexOf(events[i].eid) >-1)
		  // {
			// self.cidobjs[events[i].eid]='';
			// delete self.cidobjs[events[i].eid];
			// var idx = self.cids.indexOf(events[i].etype);
			// self.cids.splice(idx,1);
		  //}
		 else {
		  // "Technical events" not used.
		 }
	  }
	}
	
	self._renderDescription();
	  

	};

    PropDecorator.prototype._renderName = function () {
        var self = this,
			client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);

        //render GME-ID in the DOM, for debugging
        this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});
		
		var obj= nodeObj;
		
        if (obj) {
            this.name = obj.getAttribute(nodePropertyNames.Attributes.name) || '';
			
        }

        //find name placeholder
        this.skinParts.$name = this.$el.find('.name');
		this.skinParts.$name.text(this.name);
		this.skinParts.$name.css({color: '#000000'});
		
    };
	
	PropDecorator.prototype.getParentName = function () {
		var self = this,
			client = this._control._client;
		if (!this.refobj)
		{
			self.parentName = '';
			return;
		}
		var pid = this.refobj.getParentId();
		var pnode = client.getNode(pid);
		var baseid= pnode.getBaseId();
		var metaObj = client.getNode(baseid);
		if (!metaObj)
		{
			return;
		}
		
		var pmetaname = metaObj.getAttribute(nodePropertyNames.Attributes.name) || '';
		if (pmetaname =='System_Model')
			return;
		
		self.parentName = pnode.getAttribute('name');
		
	};
	
	PropDecorator.prototype.getFaultList = function () {
		var self = this,
			client = this._control._client;
			
		var c=0;
		var cnames= [];
		var fullname;
		
		for(c=0; c!= self.cids.length; c +=1)
		{
			if (!self.cidobjs[self.cids[c]])
				return cnames;
			
			var obj = self.cidobjs[self.cids[c]];
			var name =  obj.getAttribute('name');
			var pid =  obj.getParentId();
			var pnode = client.getNode(pid);
			var pName = pnode.getAttribute('name');
			var baseid= pnode.getBaseId();
			var metaObj = client.getNode(baseid);
			if (!metaObj)
			{
				continue;
			}
			var pmetaname = metaObj.getAttribute(nodePropertyNames.Attributes.name) || '';
			if (pmetaname =='System_Model')
			{
				pName = '';
				fullname = name;
			}
			else
			    fullname = pName + '.' + name;
			
			cnames.push(fullname);	
		}
		
		cnames.sort();
		return cnames;
		
		
	};
	
	PropDecorator.prototype._renderDescription = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
			self = this;
			
	    
        //render GME-ID in the DOM, for debugging
        this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});

		var obj= nodeObj;
		
		var name= '';
		var min = '';
		var max = '';
		
		self.description='';
		
        if (obj) {
		
			if (this.metaname == 'Requirements' || this.metaname == 'OperationalImpact' || this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements')
			{
				if (this.refobj)
				{
					self.varname= this.refobj.getAttribute(nodePropertyNames.Attributes.name) || '';
					if (this.metaname == 'ModeTriggered' || this.metaname == 'ModeRequirements')
						self.getParentName();
					
				}
				else
					self.varname = '';
					
				if (this.metaname == 'Requirements' || this.metaname == 'OperationalImpact' )	
				{
					this.description += 'Variable : ';
					if (this.varname == '')
						this.description += ' - ';
					else
						this.description += this.varname;
				}
				
				if (this.metaname== 'ModeTriggered')
					this.description += 'New Mode : ';
				if (this.metaname== 'ModeRequirements')
					this.description += ' Mode : ';
				
				
				//this.description += '<br>';
			}
			
			if (this.metaname == 'Trigger_Condition')
			{
				self.faultList = self.getFaultList();
				if (self.faultList != '')
				{
					var c=0;
					this.description = 'Fault List: <br>';
					for (c=0; c!= self.faultList.length; c+=1)
					{
						this.description += self.faultList[c] + '<br>';
					}
				}
				else{
					this.description = 'Fault List : <empty> <br>';
				}
			}
			
			if (this.metaname == 'Requirements' || this.metaname == 'Variable')
			{
				self.min = obj.getAttribute('Minimum') ||  9999999;
				self.max = obj.getAttribute('Maximum') || -9999999;
				this.description += '<br> Minimum : ';
				if (self.min < 9999)
					this.description += self.min.toString();
				else
				   this. description += ' - ';
				 
				this.description += '<br> Maximum : ';
				if (self.max  > -9999)
					this.description += self.max.toString();
				else
				   this. description += ' - ';
			}
			
			if (this.metaname == 'OperationalImpact'){
			    self.changeIn = obj.getAttribute('ChangeIn') ||  ' - ';
				self.newValue = obj.getAttribute('NewValue') || ' - ';
				this.description += '<br> Change In : ';
				this.description += self.changeIn.toString();
				this.description += '<br> NewValue : ';
				this.description += self.newValue.toString();
			}
			
			
			if (this.metaname == 'ModeTriggered' || this.metaname == 'ModeRequirements'){
			    self.modeStatus = obj.getAttribute('Status') ||  '';
				if (self.modeStatus=='Disabled')
					self.modeStatus = '!';
				else
					self.modeStatus = '';
				
				if (this.parentName)
					this.description += self.modeStatus + this.parentName+ '.'+this.varname;
				else
					this.description += self.modeStatus + this.varname;
				
			}
			
			
        }
		

        //find name placeholder
        this.skinParts.$description = this.$el.find('.description');
		this.skinParts.$description.empty();
        this.skinParts.$description.append(this.description);
		this.$el.css({'background-color': '#dedede'});
    };
	
	
    PropDecorator.prototype.update = function () {
        var self = this,
			client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
			newName = '',
			newMin = '',
			newMax = '',
			newStatus='',
			newchangeIn=' - ',
			newnewValue=' - ',
			newvarname ='',
			newfaultlist=[];
		if (!nodeObj)
		{
			return;
		}
		
		var obj= nodeObj;
			
		
        if (obj) {
		
			if (this.metaname == 'Requirements' || this.metaname == 'OperationalImpact' || this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements')
			{
				self._updatePointer();
				if (this.refobj)
				{
					newvarname= this.refobj.getAttribute(nodePropertyNames.Attributes.name) || '';
				}
				
			}
			
			
			if (this.metaname == 'Trigger_Condition')
			{
				self.updateTrigger();
				newfaultlist = self.getFaultList();
				
			}
			
		
            newName = obj.getAttribute(nodePropertyNames.Attributes.name) || '';
			
			if (this.metaname == 'Requirements' || this.metaname == 'Variable')
			{
				newMin = obj.getAttribute('Minimum') || '';
				newMax = obj.getAttribute('Maximum') || '';
			}
			
			if (this.metaname == 'OperationalImpact'){
				newchangeIn = obj.getAttribute('ChangeIn') ||  '';
				newnewValue = obj.getAttribute('NewValue') || '';
			}
			
			if (this.metaname== 'ModeTriggered' || this.metaname == 'ModeRequirements')
			{
				newStatus = obj.getAttribute('Status') ||  '';
				if (newStatus == 'Disabled')
					newStatus = '!';
				else
					newStatus = '';
			}
			

            if (this.name !== newName) {
                this._renderName();
            }
       
		    
            if (this.min != newMin || this.max != newMax || this.name != newName || this.changeIn != newchangeIn || this.newValue != newnewValue || this.varname != newvarname ||this.modeStatus != newStatus || self.isTriggerCondition) {
				this._renderDescription();
            }
			
			
        }
		
		//this._updateDropArea();
		
		
		
    };
	
	PropDecorator.prototype._updateDropArea = function () {
        var inverseClass = 'inverse-on-hover';

        if (this.metaname == 'Requirements' || this.metaname == 'OperationalImpact') {
            this._enableDragEvents();
        } else {
            this._disableDragEvents();
        }

        this._setPointerTerritory(this._getPointerTargets());
    };

    
	PropDecorator.prototype._navigateToPointerTarget = function (mousePos) {
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


    /**************** EDIT NODE TITLE ************************/

    PropDecorator.prototype._onNodeTitleChanged = function (oldValue, newValue) {
        var self = this,
			client = this._control._client;

        client.setAttributes(this._metaInfo[CONSTANTS.GME_ID], nodePropertyNames.Attributes.name, newValue);
    };
	
	
	
    /**************** END OF - EDIT NODE TITLE ************************/

    PropDecorator.prototype.doSearch = function (searchDesc) {
        var self = this,
			searchText = searchDesc.toString();
        if (this.name && this.name.toLowerCase().indexOf(searchText.toLowerCase()) !== -1) {
            return true;
        }

        return false;
    };
	
	
    


    return PropDecorator;
});