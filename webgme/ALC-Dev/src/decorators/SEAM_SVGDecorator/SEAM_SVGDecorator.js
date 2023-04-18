/*globals define, _*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
    'decorators/SVGDecorator/DiagramDesigner/SVGDecorator.DiagramDesignerWidget',
    './PartBrowser/SEAM_SVGDecorator.PartBrowserWidget'
], function (DecoratorBase, SEAM_SVGDecoratorDiagramDesignerWidget, SEAM_SVGDecoratorPartBrowserWidget) {

    'use strict';

    var SEAM_SVGDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'SEAM_SVGDecorator';

    SEAM_SVGDecorator = function (params) {
        var opts = _.extend({loggerName: this.DECORATORID}, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('SEAM_SVGDecorator ctor');
    };

    _.extend(SEAM_SVGDecorator.prototype, __parent_proto__);
    SEAM_SVGDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    SEAM_SVGDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: SEAM_SVGDecoratorDiagramDesignerWidget,
            PartBrowser: SEAM_SVGDecoratorPartBrowserWidget
        };
    };

    return SEAM_SVGDecorator;
});