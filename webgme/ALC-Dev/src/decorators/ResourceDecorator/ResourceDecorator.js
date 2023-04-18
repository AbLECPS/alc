/*globals define, _*/
/*jshint browser: true, camelcase: false*/



define([
    'js/Decorators/DecoratorBase',
    './DiagramDesigner/ResourceDecorator.DiagramDesignerWidget',
    './PartBrowser/ResourceDecorator.PartBrowserWidget'
], function (DecoratorBase, ResourceDecoratorDiagramDesignerWidget, ResourceDecoratorPartBrowserWidget) {

    'use strict';

    var ResourceDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'ResourceDecorator';

    ResourceDecorator = function (params) {
        var opts = _.extend({loggerName: this.DECORATORID}, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('ResourceDecorator ctor');
    };

    _.extend(ResourceDecorator.prototype, __parent_proto__);
    ResourceDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    ResourceDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: ResourceDecoratorDiagramDesignerWidget,
            PartBrowser: ResourceDecoratorPartBrowserWidget
        };
    };

    return ResourceDecorator;
});