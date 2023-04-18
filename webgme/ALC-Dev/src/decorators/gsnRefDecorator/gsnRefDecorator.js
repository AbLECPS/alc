/*globals define, _*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
	'../gsnDecorator/DiagramDesigner/gsnDecorator.DiagramDesignerWidget',
	'../SEAM_ModelDecorator/PartBrowser/SEAM_ModelDecorator.PartBrowserWidget.js'
], function (DecoratorBase, gsnRefDecoratorDiagramDesignerWidget, gsnRefDecoratorPartBrowserWidget) {

    'use strict';

    var gsnRefDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'gsnRefDecorator';

    gsnRefDecorator = function (params) {
        var opts = _.extend({ loggerName: this.DECORATORID }, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('gsnRefDecorator ctor');
    };

    _.extend(gsnRefDecorator.prototype, __parent_proto__);
    gsnRefDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    gsnRefDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: gsnRefDecoratorDiagramDesignerWidget,
            PartBrowser: gsnRefDecoratorPartBrowserWidget
        };
    };

    return gsnRefDecorator;
});