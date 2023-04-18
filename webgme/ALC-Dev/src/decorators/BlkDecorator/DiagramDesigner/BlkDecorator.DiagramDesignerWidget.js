/*globals define, _, $, WebGMEGlobal*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Constants',
    'js/NodePropertyNames',
    'text!./BlkDecorator.html',
	'decorators/ModelDecorator/DiagramDesigner/ModelDecorator.DiagramDesignerWidget',
    './BlkDecorator.Core',
    'css!./BlkDecorator.DiagramDesignerWidget.css'
], function (CONSTANTS,
             nodePropertyNames,
             BlkDecoratorTemplate,
			 ModelDecoratorWidget,
             BlkDecoratorCore) {

    'use strict';

    var BlkDecoratorDiagramDesignerWidget,
        DECORATOR_ID = 'BlkDecoratorDiagramDesignerWidget',
        PORT_CONTAINER_OFFSET_Y = 15,
        ACCEPT_DROPPABLE_CLASS = 'accept-droppable',
        DRAGGABLE_MOUSE = 'DRAGGABLE';

    BlkDecoratorDiagramDesignerWidget = function (options) {
        var opts = _.extend({}, options);

        ModelDecoratorWidget.apply(this, [opts]);
        BlkDecoratorCore.apply(this, [opts]);

        this._initializeVariables({ connectors: true });

        this._selfPatterns = {};

        this.logger.debug('BlkDecoratorDiagramDesignerWidget ctor');
    };

    /************************ INHERITANCE *********************/
    _.extend(BlkDecoratorDiagramDesignerWidget.prototype, ModelDecoratorWidget.prototype);
    _.extend(BlkDecoratorDiagramDesignerWidget.prototype, BlkDecoratorCore.prototype);

    /**************** OVERRIDE INHERITED / EXTEND ****************/

    /**** Override from DiagramDesignerWidgetDecoratorBase ****/
    BlkDecoratorDiagramDesignerWidget.prototype.DECORATORID = DECORATOR_ID;


    /**** Override from DiagramDesignerWidgetDecoratorBase ****/
    BlkDecoratorDiagramDesignerWidget.prototype.$DOMBase = $(BlkDecoratorTemplate);

    /**** Override from DiagramDesignerWidgetDecoratorBase ****/
    //jshint camelcase: false

	/**** Override from ModelDecoratorDiagramDesignerWidget ****/
	 BlkDecoratorDiagramDesignerWidget.prototype.update = function () {
		 ModelDecoratorWidget.prototype.update.call(this);
		 this._updatePortContainerSize();
        
    };



    /**** Override from BlkDecoratorCore ****/
    BlkDecoratorDiagramDesignerWidget.prototype.renderPort = function (portId) {
        this.__registerAsSubcomponent(portId);

        return BlkDecoratorCore.prototype.renderPort.call(this, portId);
    };


    /**** Override from BlkDecoratorCore ****/
    BlkDecoratorDiagramDesignerWidget.prototype.removePort = function (portId) {
        var idx = this.portIDs.indexOf(portId);

        if (idx !== -1) {
            this.__unregisterAsSubcomponent(portId);
        }

        BlkDecoratorCore.prototype.removePort.call(this, portId);
    };






    return BlkDecoratorDiagramDesignerWidget;
});