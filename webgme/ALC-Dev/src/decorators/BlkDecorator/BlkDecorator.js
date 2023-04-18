/*globals define, _*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
    './DiagramDesigner/BlkDecorator.DiagramDesignerWidget',
    './PartBrowser/BlkDecorator.PartBrowserWidget'
    ], function (DecoratorBase, BlkDecoratorDiagramDesignerWidget, BlkDecoratorPartBrowserWidget) {

    'use strict';

    var BlkDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'BlkDecorator';

    BlkDecorator = function (params) {
        var opts = _.extend({ loggerName: this.DECORATORID }, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('BlkDecorator ctor');
    };

    _.extend(BlkDecorator.prototype, __parent_proto__);
    BlkDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    BlkDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: BlkDecoratorDiagramDesignerWidget,
            PartBrowser: BlkDecoratorPartBrowserWidget
        };
    };

    return BlkDecorator;
});