/*globals define, _, $*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 */

define([
    'js/Constants',
    'js/NodePropertyNames',
    'js/Widgets/PartBrowser/PartBrowserWidget.DecoratorBase',
    'decorators/ModelDecorator/PartBrowser/ModelDecorator.PartBrowserWidget'
], function (CONSTANTS,
             nodePropertyNames,
             PartBrowserWidgetDecoratorBase,
             ModelDecoratorPartBrowser) {

    'use strict';

    var SEAM_ModelDecoratorPartBrowserWidget,
        DECORATOR_ID = 'SEAM_ModelDecoratorPartBrowserWidget';


    SEAM_ModelDecoratorPartBrowserWidget = function (options) {
        var opts = _.extend({}, options);

        ModelDecoratorPartBrowser.apply(this, [opts]);

        //this._initializeVariables({connectors: false});

        this.logger.debug('SEAM_ModelDecoratorPartBrowserWidget ctor');
    };


    /************************ INHERITANCE *********************/
    _.extend(SEAM_ModelDecoratorPartBrowserWidget.prototype, ModelDecoratorPartBrowser.prototype);


    /**************** OVERRIDE INHERITED / EXTEND ****************/

    /**** Override from PartBrowserWidgetDecoratorBase ****/
    SEAM_ModelDecoratorPartBrowserWidget.prototype.DECORATORID = DECORATOR_ID;


   
    
    /**** Override from ModelDecoratorPartBrowser ****/
    SEAM_ModelDecoratorPartBrowserWidget.prototype._updateName = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            noName = '(N/A)';

        if (nodeObj) {
            this.name = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
            //this.name = nodeObj.getFullyQualifiedName();
            this.formattedName = this.name;
        } else {
            this.name = '';
            this.formattedName = noName;
        }

        this.skinParts.$name.text(this.formattedName);
        this.skinParts.$name.attr('title', this.formattedName);
    };


  

    return SEAM_ModelDecoratorPartBrowserWidget;
});