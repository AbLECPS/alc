/*globals define, $, _*/
/*jshint browser: true*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 * @author nabana / https://github.com/nabana
 */


define([
    'js/Constants',
    'js/NodePropertyNames',
    'js/Widgets/DiagramDesigner/DiagramDesignerWidget.Constants',
    'text!decorators/SVGDecorator/Core/SVGDecorator.html',
    'decorators/SVGDecorator/DiagramDesigner/SVGDecorator.DiagramDesignerWidget',
    'js/RegistryKeys',
    'text!./default.svg'
], function (CONSTANTS,
             nodePropertyNames,
             DiagramDesignerWidgetConstants,
             DynamicSVGDecoratorTemplate,
             DynamicSVGDecoratorBase,
             REGISTRY_KEYS,
             defaultSVG) {

    'use strict';

    var DynamicSVGDecoratorDiagramDesignerWidget,
        DECORATOR_ID = 'DynamicSVGDecoratorDiagramDesignerWidget',
        SVG_DIR = CONSTANTS.ASSETS_DECORATOR_SVG_FOLDER,
        SVG_CACHE = {};

    DynamicSVGDecoratorDiagramDesignerWidget = function (options) {
        var opts = _.extend({}, options);

        DynamicSVGDecoratorBase.apply(this, [opts]);

        this.svgtext = '';
        this._initializeVariables({connectors: true});

        this._selfPatterns = {};
        this._svgCache = SVG_CACHE;

        this.logger.debug('DynamicSVGDecoratorDiagramDesignerWidget ctor');
    };

    /************************ INHERITANCE *********************/
    _.extend(DynamicSVGDecoratorDiagramDesignerWidget.prototype, DynamicSVGDecoratorBase.prototype);

    /**************** OVERRIDE INHERITED / EXTEND ****************/

    /**** Override from DiagramDesignerWidgetDecoratorBase ****/
    DynamicSVGDecoratorDiagramDesignerWidget.prototype.DECORATORID = DECORATOR_ID;


    /**** Override from DiagramDesignerWidgetDecoratorBase ****/
    DynamicSVGDecoratorDiagramDesignerWidget.prototype.$DOMBase = $(DynamicSVGDecoratorTemplate);

    /**** Override from DynamicSVGDecoratorBase ****/
    //jshint camelcase: false

    DynamicSVGDecoratorDiagramDesignerWidget.prototype._updateSVGContent = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            self = this,
            logger = this.logger,
            svgURL,
            svgFile,
            svgContent;

        this.logger.debug('DynamicSVGDecoratorDiagramDesignerWidget _updateSVGContent');

        if (nodeObj) {
            //set new content
            this.$svgContent.empty();
            this.$svgContent.removeClass();
            this.$svgContent.addClass('svg-content');

            //remove existing connectors (if any)
            this.$el.find('> .' + DiagramDesignerWidgetConstants.CONNECTOR_CLASS).remove();

            svgFile = this.getMetaDependentSvgFile();

            if (svgFile) {
                // This clause is added to the default class.
                if (this.svgCache[svgFile]) {
                    svgContent = this.svgCache[svgFile].clone();
                } else {
                    svgURL = SVG_DIR + svgFile;
                    $.ajax(svgURL, {async: false})
                        .done(function (data) {
                            var svgElements = $(data).find('svg');
                            if (svgElements.length > 0) {
                                self.svgCache[svgFile] = $(data).find('svg').first();
                                svgContent = self.svgCache[svgFile].clone();
                            }
                        })
                        .fail(function () {
                            // download failed for this type
                            logger.error('Failed to download SVG file: ' + svgFile);
                        });
                }
            } else {
                svgContent = WebGMEGlobal.SvgManager.getSvgElement(nodeObj, REGISTRY_KEYS.SVG_ICON);
            }

            if (svgContent) {
                this.$svgElement = svgContent;
				if (this.svgtext) {
					this.$svgElement.find('text').html(this.svgtext);
				}
                this._discoverCustomConnectionAreas(this.$svgElement);

            } else {
                delete this._customConnectionAreas;
                this.$svgElement = $(defaultSVG);

            }
        } else {
            delete this._customConnectionAreas;
            this.$svgElement = $(defaultSVG);
        }

        this._generateConnectors();
        this.$svgContent.append(this.$svgElement);
    };

    DynamicSVGDecoratorDiagramDesignerWidget.prototype.getMetaDependentSvgFile = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            metaID,
            metaObj,
            connType,
            flowType,
			anomalyType,
            metaName,
            svgFile,
			isAD,
            num;

        this.svgtext = '';
		svgFile = defaultSVG;
		
		if (!nodeObj)
		{
			return svgFile;
		}
		
		metaID = nodeObj.getMetaTypeId();

        if (metaID) {
            metaObj = client.getNode(metaID);
            if (metaObj) {
                metaName = metaObj.getAttribute(nodePropertyNames.Attributes.name) || '';
            }
        }

        if (metaName == 'ConnectorFn') {
            connType = nodeObj.getAttribute('Type') || '';
            if (connType == 'Or') {
                svgFile = 'svg/Or.svg';
            }

            if (connType == 'And') {
                svgFile = 'svg/And.svg';
            }
        } else if (metaName == 'ChoiceJn') {
			svgFile = 'svg/choice.svg';
            num = nodeObj.getAttribute('Minimum Required') || '';
            this.svgtext = num.toString();

            if (this.svgtext == '-1') {
                this.svgtext = 'all';
            } else if (this.svgtext == '1') {
                this.svgtext = 'or';
            }

        } else if (metaName == 'Port') {
            flowType = nodeObj.getAttribute('FlowType');
            if (flowType == 'Energy') {
                svgFile = 'svg/energyPort.svg';
            } else if (flowType == 'Information') {
                svgFile = 'svg/signalPort.svg';
            } else if (flowType == 'Material') {
                svgFile = 'svg/materialPort.svg';
            }
        } else if (metaName == 'Anomaly') {
            anomalyType = nodeObj.getAttribute('Type');
			if (anomalyType == 'OR') {
                svgFile = 'svg/AnomalyOR.svg';
            } else if (anomalyType == 'AND') {
                svgFile = 'svg/AnomalyAND.svg';
            }
			
        } else if (metaName == 'FTLogic') {
            anomalyType = nodeObj.getAttribute('Type');
            if (anomalyType == 'OR') {
                svgFile = 'svg/FTOR.svg';
            } else if (anomalyType == 'AND') {
                svgFile = 'svg/FTAnd.svg';
            }
        }else if (metaName == 'Test') {
		    svgFile = 'svg/Test.svg';
            isAD = nodeObj.getAttribute('IsActiveDiagnostic');
            if (isAD) {
                svgFile = 'svg/Test_AD.svg';
            }
        }

        return svgFile;
    };


    return DynamicSVGDecoratorDiagramDesignerWidget;
});
