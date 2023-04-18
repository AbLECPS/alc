/*globals define, _*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
    './DiagramDesigner/gsnDecorator.DiagramDesignerWidget',
    './PartBrowser/gsnDecorator.PartBrowserWidget'
], function (DecoratorBase, gsnDecoratorDiagramDesignerWidget, gsnDecoratorPartBrowserWidget) {

    'use strict';

    var gsnDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'gsnDecorator';

    gsnDecorator = function (params) {
        var opts = _.extend({ loggerName: this.DECORATORID }, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('gsnDecorator ctor');
    };

    _.extend(gsnDecorator.prototype, __parent_proto__);
    gsnDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    gsnDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: gsnDecoratorDiagramDesignerWidget,
            PartBrowser: gsnDecoratorPartBrowserWidget
        };
    };

    return gsnDecorator;
});
