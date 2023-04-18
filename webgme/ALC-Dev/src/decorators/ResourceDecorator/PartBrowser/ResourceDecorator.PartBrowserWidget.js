/*globals define, _, DEBUG, $, WebGMEGlobal*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */


define([
    'js/Constants',
    'js/RegistryKeys',
    'js/NodePropertyNames',
    'js/Widgets/PartBrowser/PartBrowserWidget.DecoratorBase',
    '../Core/ResourceDecorator.Core',
    'js/Widgets/DiagramDesigner/DiagramDesignerWidget.Constants',
    'text!../DiagramDesigner/ResourceDecorator.DiagramDesignerWidget.html',
    'css!../DiagramDesigner/ResourceDecorator.DiagramDesignerWidget.css',
    'css!./ResourceDecorator.PartBrowserWidget.css'
], function (CONSTANTS,
             REGISTRY_KEYS,
             nodePropertyNames,
             PartBrowserWidgetDecoratorBase,
             ResourceDecoratorCore,
             DiagramDesignerWidgetConstants,
             ResourceDecoratorDiagramDesignerWidgetTemplate) {

    'use strict';

    var ResourceDecoratorPartBrowserWidget,
        DECORATOR_ID = 'ResourceDecoratorPartBrowserWidget',
        EMBEDDED_SVG_IMG_BASE = $('<img>', {class: 'embeddedsvg'});

    ResourceDecoratorPartBrowserWidget = function (options) {
        var opts = _.extend({}, options);

        PartBrowserWidgetDecoratorBase.apply(this, [opts]);
        ResourceDecoratorCore.apply(this, [opts]);

        this.logger.debug('ResourceDecoratorPartBrowserWidget ctor');
    };

    _.extend(ResourceDecoratorPartBrowserWidget.prototype, PartBrowserWidgetDecoratorBase.prototype);
    _.extend(ResourceDecoratorPartBrowserWidget.prototype, ResourceDecoratorCore.prototype);

    ResourceDecoratorPartBrowserWidget.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    ResourceDecoratorPartBrowserWidget.prototype.$DOMBase = (function () {
        var el = $(ResourceDecoratorDiagramDesignerWidgetTemplate);
        //use the same HTML template as the ResourceDecorator.DiagramDesignerWidget
        //but remove the connector DOM elements since they are not needed in the PartBrowser
        el.find('.' + DiagramDesignerWidgetConstants.CONNECTOR_CLASS).remove();
        return el;
    })();

    // Public API
    ResourceDecoratorPartBrowserWidget.prototype.beforeAppend = function () {
        this.$el = this.$DOMBase.clone();

        //find name placeholder
        this.skinParts.$name = this.$el.find('.name');

        this._renderContent();
    };

    ResourceDecoratorPartBrowserWidget.prototype.afterAppend = function () {
    };

    ResourceDecoratorPartBrowserWidget.prototype.update = function () {
        this._renderContent();
    };

    // Helper methods

    ResourceDecoratorPartBrowserWidget.prototype._renderContent = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);

        //render GME-ID in the DOM, for debugging
        if (DEBUG) {
            this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});
        }

        if (nodeObj) {
            this.skinParts.$name.text(nodeObj.getAttribute(nodePropertyNames.Attributes.name) || '');
        }

        this._updateColors(true);
        this._updateSVG();
    };

    ResourceDecoratorPartBrowserWidget.prototype._updateSVG = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            svgURL,
            self = this;

        if (nodeObj) {
            svgURL = WebGMEGlobal.SvgManager.getSvgUri(nodeObj, REGISTRY_KEYS.SVG_ICON);
        }

        if (svgURL) {
            // get the svg from the server in SYNC mode, may take some time
            if (!this.skinParts.$imgSVG) {
                this.skinParts.$imgSVG = EMBEDDED_SVG_IMG_BASE.clone();
                this.$el.append(this.skinParts.$imgSVG);
            }
            if (this.skinParts.$imgSVG.attr('src') !== svgURL) {
                this.skinParts.$imgSVG.on('load', function (/*event*/) {
                    self.skinParts.$imgSVG.css('margin-top', '5px');
                    self.skinParts.$imgSVG.off('load');
                    self.skinParts.$imgSVG.off('error');
                });
                this.skinParts.$imgSVG.on('error', function (/*event*/) {
                    self.skinParts.$imgSVG.css('margin-top', '5px');
                    self.skinParts.$imgSVG.off('load');
                    self.skinParts.$imgSVG.off('error');
                });
                this.skinParts.$imgSVG.attr('src', svgURL);
            }
        } else {
            if (this.skinParts.$imgSVG) {
                this.skinParts.$imgSVG.remove();
                this.skinParts.$imgSVG = undefined;
            }
        }
    };

    return ResourceDecoratorPartBrowserWidget;
});