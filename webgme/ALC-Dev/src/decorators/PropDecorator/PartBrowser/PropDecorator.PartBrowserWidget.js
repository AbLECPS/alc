/*globals define, _, DEBUG, $*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */


define([
    'js/Constants',
    'js/NodePropertyNames',
    'js/Widgets/PartBrowser/PartBrowserWidget.DecoratorBase',
    'js/Widgets/DiagramDesigner/DiagramDesignerWidget.Constants',
    'text!../DiagramDesigner/PropDecorator.DiagramDesignerWidget.html',
    'css!./PropDecorator.PartBrowserWidget.css'
], function (CONSTANTS,
             nodePropertyNames,
             PartBrowserWidgetDecoratorBase,
             DiagramDesignerWidgetConstants,
             PropDecoratorDiagramDesignerWidgetTemplate) {

    'use strict';

    var PropDecoratorPartBrowserWidget,
        __parent__ = PartBrowserWidgetDecoratorBase,
        DECORATOR_ID = 'PropDecoratorPartBrowserWidget';

    PropDecoratorPartBrowserWidget = function (options) {
        var opts = _.extend({}, options);

        __parent__.apply(this, [opts]);

        this.logger.debug('PropDecoratorPartBrowserWidget ctor');
    };

    _.extend(PropDecoratorPartBrowserWidget.prototype, __parent__.prototype);
    PropDecoratorPartBrowserWidget.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    PropDecoratorPartBrowserWidget.prototype.$DOMBase = (function () {
        var el = $(PropDecoratorDiagramDesignerWidgetTemplate);
        //use the same HTML template as the PropDecorator.DiagramDesignerWidget
        //but remove the connector DOM elements since they are not needed in the PartBrowser
        el.find('.' + DiagramDesignerWidgetConstants.CONNECTOR_CLASS).remove();
        return el;
    })();

    PropDecoratorPartBrowserWidget.prototype.beforeAppend = function () {
        this.$el = this.$DOMBase.clone();

        //find name placeholder
        this.skinParts.$name = this.$el.find('.name');

        this._renderContent();
    };

    PropDecoratorPartBrowserWidget.prototype.afterAppend = function () {
    };

    PropDecoratorPartBrowserWidget.prototype._renderContent = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);

        //render GME-ID in the DOM, for debugging
        if (DEBUG) {
            this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});
        }

        if (nodeObj) {
            this.skinParts.$name.text(nodeObj.getAttribute(nodePropertyNames.Attributes.name) || '');
        }
    };

    PropDecoratorPartBrowserWidget.prototype.update = function () {
	var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
	var metaname = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
        this._renderContent();
	
	this.$el.css({'background-color': '#dedede'});
	this.$el.css({'width': '100px'});
		
		if (metaname == 'Goal')
		{
			this.$el.css({'background-color': '#99ccff'});
		}
		if (metaname == 'Strategy')
		{
			this.$el.css({'background-color': '#ccff99'});
		}
		if (metaname == 'Solution')
		{
			this.$el.css({'background-color': '#ffb266'});
		}
		if (metaname == 'Context')
		{
			this.$el.css({'background-color': '#ffffbb'});
		}
		if (metaname == 'Assumption')
		{
			this.$el.css({'background-color': '#f8f8f8'});
		}
		if (metaname == 'Justification')
		{
			this.$el.css({'background-color': '#ccffe5'});
		}
    };

    return PropDecoratorPartBrowserWidget;
});