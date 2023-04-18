/*globals define, _, DEBUG, $*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */


define([
    'js/Constants',
    'js/NodePropertyNames',
    'decorators/ModelDecorator/PartBrowser/ModelDecorator.PartBrowserWidget',
    'js/Widgets/DiagramDesigner/DiagramDesignerWidget.Constants',
    'text!../DiagramDesigner/BlkDecorator.DiagramDesignerWidget.html',
    'css!../DiagramDesigner/BlkDecorator.DiagramDesignerWidget.css'
], function (CONSTANTS,
             nodePropertyNames,
             PartBrowserWidgetDecoratorBase,
             DiagramDesignerWidgetConstants,
             BlkDecoratorDiagramDesignerWidgetTemplate) {

    'use strict';

    var BlkDecoratorPartBrowserWidget,
        DECORATOR_ID = 'BlkDecoratorPartBrowserWidget';

    BlkDecoratorPartBrowserWidget = function (options) {
        var opts = _.extend({}, options);

        PartBrowserWidgetDecoratorBase.apply(this, [opts]);

        this.logger.debug('BlkDecoratorPartBrowserWidget ctor');
    };

    _.extend(BlkDecoratorPartBrowserWidget.prototype, PartBrowserWidgetDecoratorBase.prototype);
    BlkDecoratorPartBrowserWidget.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    // BlkDecoratorPartBrowserWidget.prototype.$DOMBase = (function () {
        // var el = $(BlkDecoratorDiagramDesignerWidgetTemplate);
        // //use the same HTML template as the BlkDecorator.DiagramDesignerWidget
        // //but remove the connector DOM elements since they are not needed in the PartBrowser
        // el.find('.' + DiagramDesignerWidgetConstants.CONNECTOR_CLASS).remove();
        // return el;
    // })();
	
	

    BlkDecoratorPartBrowserWidget.prototype.beforeAppend = function () {
		PartBrowserWidgetDecoratorBase.prototype.beforeAppend.apply(this, []);
		this._updateName();
    };

    // BlkDecoratorPartBrowserWidget.prototype.afterAppend = function () {
    // };


    BlkDecoratorPartBrowserWidget.prototype.update = function () {
		PartBrowserWidgetDecoratorBase.prototype.update.apply(this, []);
		this._updateName();
    };
	
	BlkDecoratorPartBrowserWidget.prototype._updateName = function () {
		var client = this._control._client,
			nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
		var name = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
		this.skinParts.$name.text(name);
	};

    return BlkDecoratorPartBrowserWidget;
});