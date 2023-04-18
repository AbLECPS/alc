/*globals define, _, $*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Constants',
    'js/NodePropertyNames',
    'js/RegistryKeys',
    './BlkPort',
    'js/Utils/DisplayFormat',
	'decorators/ModelDecorator/Core/ModelDecorator.Core'
], function (CONSTANTS,
             nodePropertyNames,
             REGISTRY_KEYS,
             BlkPort,
             displayFormat,
			 ModelDecoratorCore) {

    'use strict';

    var BlkDecoratorCore,
        BASEDIR = '/' + CONSTANTS.ASSETS_DECORATOR_SVG_FOLDER,
        PORT_DOM_HEIGHT = 15;


    BlkDecoratorCore = function (params) {
        ModelDecoratorCore.apply(this, []);
		this.isTemplate = 0;
                this.stereotype = '';

    };

    _.extend(BlkDecoratorCore.prototype, ModelDecoratorCore.prototype);

    BlkDecoratorCore.prototype._initializeVariables = function (params) {

        ModelDecoratorCore.prototype._initializeVariables.apply(this, [params]);
        this.skinParts.$metaname = undefined;
		
            /*$name: undefined,
            $portsContainer: undefined,
            $portsContainerLeft: undefined,
            $portsContainerRight: undefined,
            $portsContainerCenter: undefined,
            $ptr: undefined,
            $imgSVG: undefined*/
        //};


    };


    BlkDecoratorCore.prototype._renderContent = function () {
        //render GME-ID in the DOM, for debugging

        //this.$el.attr({ 'data-id': this._metaInfo[CONSTANTS.GME_ID] });

        /* BUILD UI*/
        //find placeholders
        //this.skinParts.$name = this.$el.find('.name');
        
		
        this.skinParts.$metaname = this.$el.find('.metaname');
        ModelDecoratorCore.prototype._renderContent.apply(this);
        this._update();
        ModelDecoratorCore.prototype._renderContent.apply(this);
		
		this.skinParts.$portsContainer = this.$el.find('.ports');
        this.skinParts.$portsContainerLeft = this.skinParts.$portsContainer.find('.left');
        this.skinParts.$portsContainerRight = this.skinParts.$portsContainer.find('.right');
        this.skinParts.$portsContainerCenter = this.skinParts.$portsContainer.find('.center');

       
        
        this._updatePortContainerSize();

    };


    BlkDecoratorCore.prototype._updatePortContainerSize = function () {
        var maxports = Math.max(this.skinParts.$portsContainerLeft.children().length, this.skinParts.$portsContainerRight.children().length);
        this.skinParts.$portsContainer.css('height', maxports * PORT_DOM_HEIGHT);
    };


    /***** UPDATE THE NAME OF THE NODE *****/
    BlkDecoratorCore.prototype._updateName = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            noName = '(N/A)',
			typeval = '';

        if (nodeObj) {
            this.name = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
            this.formattedName = displayFormat.resolve(nodeObj);
			      typeval = nodeObj.getAttribute('Type');
			      if (typeval == 'Template')
				      this.isTemplate =1;

            var metaID = nodeObj.getMetaTypeId();
            if (metaID) {
                var metaObj = client.getNode(metaID);
                if (metaObj) {
						this.metaname = '<< ' + metaObj.getAttribute(nodePropertyNames.Attributes.name) + ' >>' || '';
                }
            }

            var baseid = nodeObj.getBaseId();
            var baseObj = client.getNode(baseid);
            var basename = baseObj.getAttribute(nodePropertyNames.Attributes.name) || '';
            var stereotype = '';
            if (this.metaname == '<< Block >>') 
            {
                 stereotype = nodeObj.getAttribute('Role')||'';
                 if ((stereotype == 'Other') || (stereotype == ''))
                 {
                    stereotype = nodeObj.getAttribute('Role_Other')||stereotype;

                 }
            }
            else if ((this.metaname == '<< File >>') ||(this.metaname == '<< Code >>')) 
            {
                 stereotype = nodeObj.getAttribute('OtherCategory') ||  nodeObj.getAttribute('Category') || '';
            }
            else if (this.metaname == '<< LEC_Model >>') 
            {
                 stereotype = nodeObj.getAttribute('Category') || '';
            }
            else if (this.metaname == '<< Verification_Model >>') 
            {
                stereotype = nodeObj.getAttribute('ProblemType') || '';
            }
            else if (this.metaname == '<< EvaluationSetup >>') 
            {
                stereotype = nodeObj.getAttribute('Labels') || '';
                if (stereotype == 'NA')
                {
                    stereotype = '';
                }
            }

            
          
            if (stereotype != '')
             {
                this.metaname = '<<' + stereotype + '>>'
             }
             
            if ((!this.isTemplate) && (this.metaname == '<<Block>>') && (baseid != metaID))
				        this.formattedName = this.formattedName + '::' + basename;
      			if (this.isTemplate)
      			{
      				this.formattedName = this.formattedName;
      				if (baseid != metaID)
      					this.metaname = '<< Template - ' + basename + ' >>';
      				else
      					this.metaname = '<< Template >>';
      			}
            

        } else {
            this.name = '';
            this.formattedName = noName;
        }

        this.skinParts.$metaname.text(this.metaname);
        this.skinParts.$metaname.attr('title', this.metaname);
        this.skinParts.$name.text(this.metaname);//formattedName);
        this.skinParts.$name.attr('title', this.metaname);//formattedName);

        var isimpl = nodeObj.getAttribute('IsImplementation');
        var isactive = nodeObj.getAttribute('IsActive');
        if (isimpl && !isactive)
        {
            //this.$el.css({'style': "color: #cccccc;"});
            this.$el.css({'color': "#cccccc"});
        }

        
    };



    BlkDecoratorCore.prototype.renderPort = function (portId) {
        var client = this._control._client,
            portNode = client.getNode(portId),
            portTitle = displayFormat.resolve(portNode),
            svgFile;

		if (!portNode)
		{
			return;
		}
		
        var portdir = portNode.getAttribute('Direction');
		var portmetaid = portNode.getMetaTypeId();
		var portmeta = client.getNode(portmetaid);
		var portmetaname = portmeta.getAttribute(nodePropertyNames.Attributes.name) || '';

        if (portdir == 'Output') {
            svgFile = 'svg/powerPortLeft.svg';
			if (portmetaname =='SignalPort')
			{
				svgFile='svg/signalPortLeft.svg';
			}
			else if (portmetaname =='MaterialPort')
			{
				svgFile='svg/materialPortLeft.svg';
			}
        }
        if (portdir == 'InputOutput') {
            svgFile='svg/powerPortInOut.svg';
			if (portmetaname =='SignalPort')
			{
				svgFile='svg/signalPortInOut.svg';
			}
			else if (portmetaname =='MaterialPort')
			{
				svgFile='svg/materialPortInOut.svg';
			}
	    
        }

        var portInstance = new BlkPort(portId, {
            title: portTitle,
            decorator: this,
            svg: svgFile ? BASEDIR + svgFile : WebGMEGlobal.SvgManager.getSvgUri(portNode, REGISTRY_KEYS.PORT_SVG_ICON)
        });

        this._addPortToContainer(portNode, portInstance);

        return portInstance;
    };


    BlkDecoratorCore.prototype._updatePort = function (portId) {
        var idx = this.portIDs.indexOf(portId),
            client = this._control._client,
            portNode = client.getNode(portId),
            isPort = this.isPort(portId),
            portTitle;
		
		if (!portNode)
		{
			return;
		}
        var portdir = portNode.getAttribute('Direction');
        var svgfile = WebGMEGlobal.SvgManager.getSvgUri(portNode, REGISTRY_KEYS.PORT_SVG_ICON);//portNode.getRegistry(REGISTRY_KEYS.PORT_SVG_ICON);
		var portmetaid = portNode.getMetaTypeId();
		var portmeta = client.getNode(portmetaid);
		var portmetaname = portmeta.getAttribute(nodePropertyNames.Attributes.name) || '';
		if (portdir == 'Output') {
            svgfile = BASEDIR +'svg/powerPortLeft.svg';
			if (portmetaname =='SignalPort')
			{
				svgfile=BASEDIR +'svg/signalPortLeft.svg';
			}
			else if (portmetaname =='MaterialPort')
			{
				svgfile=BASEDIR +'svg/materialPortLeft.svg';
			}
        }
	
        if (portdir == 'InputOutput') {
            svgfile=BASEDIR +'svg/powerPortInOut.svg';
			if (portmetaname =='SignalPort')
			{
				svgfile=BASEDIR +'svg/signalPortInOut.svg';
			}
			else if (portmetaname =='MaterialPort')
			{
				svgfile=BASEDIR +'svg/materialPortInOut.svg';
			}
        } 
 
        //check if it is already displayed as port
        if (idx !== -1) {
            //port already, should it stay one?
            if (isPort === true) {
                portTitle = displayFormat.resolve(portNode);
                this.ports[portId].update({
                    title: portTitle,
                    svg: svgfile,
                });
                this._updatePortPosition(portId);
            } else {
                this.removePort(portId);
            }
        } else {
            this.addPort(portId);
        }
    };





    return BlkDecoratorCore;
});
