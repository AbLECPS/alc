/*globals define, _*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
    './DiagramDesigner/DynamicSVGDecorator.DiagramDesignerWidget',
    '../SEAM_SVGDecorator/PartBrowser/SEAM_SVGDecorator.PartBrowserWidget.js'
], function (DecoratorBase,
             DynamicSVGDecoratorDiagramDesignerWidget,
             DynamicSVGDecoratorPartBrowserWidget) {

    'use strict';

    var DynamicSVGDecorator,
        DECORATOR_ID = 'DynamicSVGDecorator';

    DynamicSVGDecorator = function (params) {
        var opts = _.extend({ loggerName: this.DECORATORID }, params);

        DecoratorBase.apply(this, [opts]);

        this.logger.debug('DynamicSVGDecorator ctor');
    };

    _.extend(DynamicSVGDecorator.prototype, DecoratorBase.prototype);
    DynamicSVGDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    DynamicSVGDecorator.prototype.initializeSupportedWidgetMap = function () {

        this.supportedWidgetMap = {
            DiagramDesigner: DynamicSVGDecoratorDiagramDesignerWidget,
            PartBrowser: DynamicSVGDecoratorPartBrowserWidget
        };

    };

    return DynamicSVGDecorator;
});
