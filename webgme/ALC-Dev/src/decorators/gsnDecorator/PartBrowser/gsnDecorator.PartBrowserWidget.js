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
    'text!../DiagramDesigner/gsnDecorator.DiagramDesignerWidget.html',
    'css!./gsnDecorator.PartBrowserWidget.css'
], function (CONSTANTS,
             nodePropertyNames,
             PartBrowserWidgetDecoratorBase,
             DiagramDesignerWidgetConstants,
             gsnDecoratorDiagramDesignerWidgetTemplate) {

    'use strict';

    var gsnDecoratorPartBrowserWidget,
        __parent__ = PartBrowserWidgetDecoratorBase,
        DECORATOR_ID = 'gsnDecoratorPartBrowserWidget';

    gsnDecoratorPartBrowserWidget = function (options) {
        var opts = _.extend({}, options);

        __parent__.apply(this, [opts]);

        this.logger.debug('gsnDecoratorPartBrowserWidget ctor');
    };

    _.extend(gsnDecoratorPartBrowserWidget.prototype, __parent__.prototype);
    gsnDecoratorPartBrowserWidget.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    gsnDecoratorPartBrowserWidget.prototype.$DOMBase = (function () {
        var el = $(gsnDecoratorDiagramDesignerWidgetTemplate);
        //use the same HTML template as the gsnDecorator.DiagramDesignerWidget
        //but remove the connector DOM elements since they are not needed in the PartBrowser
        el.find('.' + DiagramDesignerWidgetConstants.CONNECTOR_CLASS).remove();
        return el;
    })();

    gsnDecoratorPartBrowserWidget.prototype.beforeAppend = function () {
        this.$el = this.$DOMBase.clone();

        //find name placeholder
        this.skinParts.$name = this.$el.find('.name');

        this._renderContent();
    };

    gsnDecoratorPartBrowserWidget.prototype.afterAppend = function () {
    };

    gsnDecoratorPartBrowserWidget.prototype._renderContent = function () {
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

    gsnDecoratorPartBrowserWidget.prototype.update = function () {
        let client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            metaname = nodeObj.getAttribute(nodePropertyNames.Attributes.name),
            bgColor = this.getBackgroundColor(metaname);

        this._renderContent();

        this.$el.css({'background-color': bgColor});
        this.$el.css({'width': '100px'});
    };

    gsnDecoratorPartBrowserWidget.prototype.getBackgroundColor = function (metaname) {
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
            bgColor = '#ccffe5';
        }
        else if (metaname === 'Requirement') {
            bgColor = '#ffffff';
        }
        else if (metaname === 'Hazard') {
            bgColor = '#ffff00';
        }

        return bgColor;
    };

    return gsnDecoratorPartBrowserWidget;
});