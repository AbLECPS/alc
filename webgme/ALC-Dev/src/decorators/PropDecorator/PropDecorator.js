/*globals define, _*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
    './DiagramDesigner/PropDecorator.DiagramDesignerWidget',
    './PartBrowser/PropDecorator.PartBrowserWidget'
], function (DecoratorBase, PropDecoratorDiagramDesignerWidget, PropDecoratorPartBrowserWidget) {

    'use strict';

    var PropDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'PropDecorator';

    PropDecorator = function (params) {
        var opts = _.extend({ loggerName: this.DECORATORID }, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('PropDecorator ctor');
    };

    _.extend(PropDecorator.prototype, __parent_proto__);
    PropDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    PropDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: PropDecoratorDiagramDesignerWidget,
            PartBrowser: PropDecoratorPartBrowserWidget
        };
    };

    return PropDecorator;
});
