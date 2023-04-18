/*globals define, _*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Decorators/DecoratorBase',
    'decorators/ModelDecorator/DiagramDesigner/ModelDecorator.DiagramDesignerWidget',
    './PartBrowser/SEAM_ModelDecorator.PartBrowserWidget'
], function (DecoratorBase, SEAM_ModelDecoratorDiagramDesignerWidget, SEAM_ModelDecoratorPartBrowserWidget) {

    'use strict';

    var SEAM_ModelDecorator,
        __parent__ = DecoratorBase,
        __parent_proto__ = DecoratorBase.prototype,
        DECORATOR_ID = 'SEAM_ModelDecorator';

    SEAM_ModelDecorator = function (params) {
        var opts = _.extend({loggerName: this.DECORATORID}, params);

        __parent__.apply(this, [opts]);

        this.logger.debug('SEAM_ModelDecorator ctor');
    };

    _.extend(SEAM_ModelDecorator.prototype, __parent_proto__);
    SEAM_ModelDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DecoratorBase MEMBERS **************************/

    SEAM_ModelDecorator.prototype.initializeSupportedWidgetMap = function () {
        this.supportedWidgetMap = {
            DiagramDesigner: SEAM_ModelDecoratorDiagramDesignerWidget,
            PartBrowser: SEAM_ModelDecoratorPartBrowserWidget
        };
    };

    return SEAM_ModelDecorator;
});