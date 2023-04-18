/*globals define, _, $*/
/*jshint browser: true*/

/**
 * @author brollb / https://github/brollb
 */

define([
    'js/Constants',
    'js/NodePropertyNames',
    'js/Widgets/PartBrowser/PartBrowserWidget.DecoratorBase',
    'decorators/SVGDecorator/PartBrowser/SVGDecorator.PartBrowserWidget'
], function (CONSTANTS,
             nodePropertyNames,
             PartBrowserWidgetDecoratorBase,
             SVGDecoratorPartBrowser) {

    'use strict';

    var SEAM_SVGDecoratorPartBrowserWidget,
        DECORATOR_ID = 'SEAM_SVGDecoratorPartBrowserWidget';


    SEAM_SVGDecoratorPartBrowserWidget = function (options) {
        var opts = _.extend({}, options);

        SVGDecoratorPartBrowser.apply(this, [opts]);

        //this._initializeVariables({connectors: false});

        this.logger.debug('SEAM_SVGDecoratorPartBrowserWidget ctor');
    };


    /************************ INHERITANCE *********************/
    _.extend(SEAM_SVGDecoratorPartBrowserWidget.prototype, SVGDecoratorPartBrowser.prototype);


    /**************** OVERRIDE INHERITED / EXTEND ****************/

    /**** Override from PartBrowserWidgetDecoratorBase ****/
    SEAM_SVGDecoratorPartBrowserWidget.prototype.DECORATORID = DECORATOR_ID;

  
    
    /**** Override from SVGDecoratorPartBrowser ****/
    SEAM_SVGDecoratorPartBrowserWidget.prototype._updateName = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            noName = '(N/A)';

        if (nodeObj) {
            this.name = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
            //this.name = nodeObj.getFullyQualifiedName();
            this.formattedName = this.name;
            if ((this.formattedName == 'BlockPackage') || (this.formattedName == 'MessagePackage'))
            {
                this.formattedName='Package'
            }
        } else {
            this.name = '';
            this.formattedName = noName;
        }

        this.$name.text(this.formattedName);
        this.$name.attr('title', this.formattedName);

        if(this.$svgElement.data('hidename') === true){
            this.$name.hide();
        } else {
            this.$name.show();
        }
    };

   
    return SEAM_SVGDecoratorPartBrowserWidget;
});
