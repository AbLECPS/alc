/*globals define, _, $*/
/*jshint browser: true, camelcase: false*/

/**
 * @author rkereskenyi / https://github.com/rkereskenyi
 * @author Qishen Zhang / https://github.com/VictorCoder123
 */

define([
    'epiceditor',
    'js/Constants',
    'js/Utils/ComponentSettings',
    'js/NodePropertyNames',
    'js/Utils/DisplayFormat',
    'js/Widgets/DiagramDesigner/DiagramDesignerWidget.DecoratorBase',
    '../Core/ResourceDecorator.Core',
    'text!./ResourceDecorator.DiagramDesignerWidget.html',
    'css!./ResourceDecorator.DiagramDesignerWidget.css'
], function (marked,
             CONSTANTS,
             ComponentSettings,
             nodePropertyNames,
             displayFormat,
             DiagramDesignerWidgetDecoratorBase,
             ResourceDecoratorCore,
             ResourceDecoratorTemplate) {

    'use strict';

    var ResourceDecorator,
        DECORATOR_ID = 'ResourceDecorator';

    ResourceDecorator = function (options) {
        var opts = _.extend({}, options);

        DiagramDesignerWidgetDecoratorBase.apply(this, [opts]);
        ResourceDecoratorCore.apply(this, [opts]);

        this._config = ResourceDecorator.getDefaultConfig();
        ComponentSettings.resolveWithWebGMEGlobal(this._config, ResourceDecorator.getComponentId());

        this.name = '';
        this.metaname = '';
        this.isRef = false;
        this.refobj = '';
        this.refID = '';

        this._skinParts = {};

        this.$doc = this.$el.find('.doc').first();

        // Use default marked options
        marked.setOptions(this._config.parserOptions);

        this.logger.debug('ResourceDecorator ctor');
    };

    ResourceDecorator.getDefaultConfig = function () {
        return {
            parserOptions: {
                // See https://github.com/chjj/marked
                gfm: true,
                tables: true,
                breaks: false,
                pedantic: false,
                sanitize: false, // Set to false if you want to enable html.
                smartLists: true,
                smartypants: false
            }
        };
    };

    ResourceDecorator.getComponentId = function () {
        return 'ResourceDecorator';
    };

    _.extend(ResourceDecorator.prototype, DiagramDesignerWidgetDecoratorBase.prototype);
    _.extend(ResourceDecorator.prototype, ResourceDecoratorCore.prototype);

    ResourceDecorator.prototype.DECORATORID = DECORATOR_ID;

    /*********************** OVERRIDE DiagramDesignerWidgetDecoratorBase MEMBERS **************************/

    ResourceDecorator.prototype.$DOMBase = $(ResourceDecoratorTemplate);

    ResourceDecorator.prototype.on_addTo = function () {
        var self = this;
        var client = this._control._client;
        var nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
        this._renderName();


        if (this.metaname == '<< ResourceRef >>') {
            this.isRef = true;
            self._updatePointer();
        }


        // Show Popover when click on name
        this.skinParts.$name.on('click', function (event) {
            self.skinParts.$name.popover({});
            self.skinParts.$name.popover('show');
            self.logger.debug(self.skinParts.$name.popover);
            event.stopPropagation();
            event.preventDefault();
        });

        // Let the parent decorator class do its job.
        DiagramDesignerWidgetDecoratorBase.prototype.on_addTo.apply(this, arguments);

        // Finally invoke the update too.
        self.update();
    };

    ResourceDecorator.prototype._renderName = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
        //render GME-ID in the DOM, for debugging

        this.$el.attr({'data-id': this._metaInfo[CONSTANTS.GME_ID]});
        if (nodeObj) {
            this.name = displayFormat.resolve(nodeObj);
            var metaID = nodeObj.getMetaTypeId();
            if (metaID) {
                var metaObj = client.getNode(metaID);
                if (metaObj) {
                    this.metaname = '<< ' + metaObj.getAttribute(nodePropertyNames.Attributes.name) + ' >>' || '';
                }
            }
        }
        //find name placeholder
        this.skinParts.$metaname = this.$el.find('.metaname');
        this.skinParts.$metaname.text(this.metaname);

        //find name placeholder
        this.skinParts.$name = this.$el.find('.name');
        this.skinParts.$name.text(this.name);
    };

    ResourceDecorator.prototype.getAttributeName = function () {

        if ((this.metaname == '<< Resource >>') || (this.metaname == '<< ResourceRef >>'))
            return "Info";

        if (this.metaname == '<< Params >>')
            return "Definition";

        if (this.metaname == '<< Campaign >>')
            return "Definition";

        return '';

    };

    ResourceDecorator.prototype.isDict = function (v) {
        return typeof v === 'object' && v !== null && !(v instanceof Array) && !(v instanceof Date);
    };


    ResourceDecorator.prototype.buildTable = function (name, values) {
        var self = this;

        var keys = Object.keys(values);
        if (keys.length === 0)
            return ' ';
        var ret = `<html> 
<head> 
<style> 
table#t01 { 
    font-family: arial, sans-serif; 
    border-collapse: collapse; 
    width: 100%; 
 font-weight: bold;
} 
td, th { 
    border: 1px solid #ff1111; 
    text-align: left; 
    padding: 8px; 
} 
tr:nth-child(even) { 
    background-color: #88b4e1
} 
</style> 
</head>`;


        var doc = [], i = 0;
        if ((self.metaname == '<< Resource >>') || (self.metaname == '<< ResourceRef >>'))
            doc.push('| Parameter | Default |');
        else if (self.metaname == '<< Params >>')
            doc.push('| Parameter | Value |');
        else if (self.metaname == '<< Campaign >>')
            doc.push('| Parameter | Values |');
        else
            doc.push('| Name | Value |');
        doc.push('|:-----------:|:---------:|');
        for (i = 0; i !== keys.length; i += 1) {
            let name = keys[i],
                value = JSON.stringify(values[name]),
                valuearray = value.split(',');
            if (valuearray.length === 1) {
                let valstr = '| ' + name + ' | ' + value + ' | ';
                doc.push(valstr);
            } else {
                var a1 = 0;
                var valstr = '| ' + name + ' | ';
                var nvalue = name;
                var sep = ',';
                for (a1 = 0; a1 != valuearray.length; a1 += 1) {
                    if (a1 > 0)
                        nvalue = '';
                    if (a1 == valuearray.length - 1)
                        sep = '';

                    valstr += valuearray[a1];
                    if (sep)
                        valstr += ', <br>';

                }
                valstr += ' | ';
                doc.push(valstr);
            }
        }

        var ret1 = doc.join('\r\n');
        ret += '\r\n\r\n' + ret1;
        '\r\n </body> </html> \r\n';
        return ret;
    };

    ResourceDecorator.prototype.composeDocument = function () {
        let client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            attributename = this.getAttributeName(),
            self = this,
            doc = [],
            ret = '';

        // Make sure attribute name is valid before continuing
        if (attributename === '')
            return '';

        if (nodeObj) {
            try {
                if (this.metaname === '<< Campaign >>') {
                    let docjson = '';
                    if (!this.isRef)
                        docjson = nodeObj.getAttribute(attributename);
                    else if (this.refobj)
                        docjson = this.refobj.getAttribute(attributename);

                    // var entry = '**Parameters**';
                    //doc.push(entry);
                    let jsonval = JSON.parse(docjson),
                        tableval = this.buildTable('Parameters', jsonval);
                    doc.push(tableval);
                    ret = doc.join('\r\n\r\n');
                    return ret;
                    //return doc;
                } else if (this.metaname === '<< Params >>') {
                    let jsonval, tableval,
                        childIds = nodeObj.getChildrenIds();

                    // 'Params' block do not use single attribute, but instead have each parameter as a separate
                    // modeling object. Parse each object and construct a dictionary of values.
                    jsonval = childIds.map(childId => client.getNode(childId))
                        .filter(childNode => {
                            // Filter out any nodes which are not 'Parameter' meta objects
                            let metaId = childNode.getMetaTypeId(),
                                metaObj = client.getNode(metaId),
                                metaName = metaObj.getAttribute("name");
                            return metaName.toLowerCase() === "parameter";
                        })
                        .reduce((currentParamDict, nextParamNode) => {
                            // Read parameter names and values and store in json dictionary
                            let paramName = nextParamNode.getAttribute("name"),
                                paramValue = nextParamNode.getAttribute("value");
                            currentParamDict[paramName] = paramValue;
                            return currentParamDict;
                        }, {});

                    // Now that parameters are parsed, call buildTable function as usual
                    tableval = this.buildTable('Parameters', jsonval);
                    doc.push(tableval);
                    ret = doc.join('\r\n\r\n');
                    return ret;
                }

                let keys = Object.keys(jsonval),
                    i = 0;
                for (i = 0; i !== keys.length; i += 1) {
                    var entry = '**' + keys[i] + '**';
                    doc.push(entry);
                    var val = jsonval[keys[i]];
                    if (this.isDict(val)) {
                        //build table
                        var tableval = this.buildTable(keys[i], val);
                        doc.push(tableval);

                    } else {
                        if (val.indexOf('www.') > -1 || val.indexOf('http') > -1) {

                            var urlstr = '<a href=\" ';
                            urlstr += val;
                            urlstr += '\" target=\"_blank\">' + keys[i] + ' Link </a>';
                            doc.push('- ' + urlstr);

                        } else {
                            doc.push('- ' + val);
                        }

                    }

                }
                ret = doc.join('\r\n\r\n');
                return ret;


            } catch (e) {
                let estr = 'Unable to parse JSON input for node: ' + nodeObj["_id"] + '\nException: ' + e.toString();
                self.logger.error(estr);
            }
        }

        return ret;
    };

    ResourceDecorator.prototype.update = function () {
        var client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]),
            newDoc = '',
            renderedEl;

        if (nodeObj) {

            this._renderName();

            newDoc = this.composeDocument();//nodeObj.getAttribute('documentation');
            // Update docs on node when attribute "documentation" changes
            this.$doc.empty();
            // Show error message if documentation attribute is not defined
            if (newDoc === undefined) {
                this.$doc.append('Editor is disabled because attribute "documentation" is not found in Meta-Model');
                //this._skinParts.$EditorBtn.addClass('not-activated');
            } else {
                /*if (this.hostDesignerItem.canvas.getIsReadOnlyMode() === false) {
                    this._skinParts.$EditorBtn.removeClass('not-activated');
                } else {
                    this._skinParts.$EditorBtn.addClass('not-activated');
                }*/

                try {
                    renderedEl = $(marked(newDoc));
                    this.$doc.append(renderedEl);
                } catch (e) {
                    this.logger.error('Markdown parsing failed html', e);
                    this.logger.error('Stored text:', newDoc);
                    this.$doc.empty();
                    this.$doc.append('Stored markdown is invalid!');
                }
            }


        }

        this._updateColors();
    };

    ResourceDecorator.prototype.getConnectionAreas = function (id /*, isEnd, connectionMetaInfo*/) {
        var result = [],
            edge = 10,
            LEN = 20;

        //by default return the bounding box edge's midpoints

        if (id === undefined || id === this.hostDesignerItem.id) {
            //NORTH
            result.push({
                id: '0',
                x1: edge,
                y1: 0,
                x2: this.hostDesignerItem.getWidth() - edge,
                y2: 0,
                angle1: 270,
                angle2: 270,
                len: LEN
            });

            //EAST
            result.push({
                id: '1',
                x1: this.hostDesignerItem.getWidth(),
                y1: edge,
                x2: this.hostDesignerItem.getWidth(),
                y2: this.hostDesignerItem.getHeight() - edge,
                angle1: 0,
                angle2: 0,
                len: LEN
            });

            //SOUTH
            result.push({
                id: '2',
                x1: edge,
                y1: this.hostDesignerItem.getHeight(),
                x2: this.hostDesignerItem.getWidth() - edge,
                y2: this.hostDesignerItem.getHeight(),
                angle1: 90,
                angle2: 90,
                len: LEN
            });

            //WEST
            result.push({
                id: '3',
                x1: 0,
                y1: edge,
                x2: 0,
                y2: this.hostDesignerItem.getHeight() - edge,
                angle1: 180,
                angle2: 180,
                len: LEN
            });
        }

        return result;
    };

    /**************** EDIT NODE TITLE ************************/

    ResourceDecorator.prototype._onNodeTitleChanged = function (oldValue, newValue) {
        var client = this._control._client;

        client.setAttribute(this._metaInfo[CONSTANTS.GME_ID], nodePropertyNames.Attributes.name, newValue);
    };

    /**************** END OF - EDIT NODE TITLE ************************/

    ResourceDecorator.prototype.doSearch = function (searchDesc) {
        var searchText = searchDesc.toString(),
            gmeId = (this._metaInfo && this._metaInfo[CONSTANTS.GME_ID]) || '';

        return (this.name && this.name.toLowerCase().indexOf(searchText.toLowerCase()) !== -1) ||
            (gmeId.indexOf(searchText) > -1);
    };

    ResourceDecorator.prototype._updatePointer = function () {
        var self = this,
            client = this._control._client,
            nodeObj = client.getNode(this._metaInfo[CONSTANTS.GME_ID]);
        if (!nodeObj) {
            self.refobj = '';
            return;
        }

        self.logger.debug('in updatepointer');
        var ptrid = nodeObj.getPointerId('LibraryRef');
        self.refobj = '';
        if (ptrid) {
            self.refID = ptrid;
            self.logger.debug('ptr id' + ptrid);
            self.refobj = client.getNode(ptrid);
        }
        if (self.refobj)
            self.logger.debug('ref obj');
        else if (self.isRef && ptrid) {
            var patterns = {};
            patterns[''] = {children: 0};
            patterns[ptrid] = {children: 0};
            var userId = client.addUI(null, function (events) {
                self.eventHandler(self, events)
            });

            client.updateTerritory(userId, patterns);

            /*self.logger.debug('ptr id' + ptrid);
            self.refobj=client.getNode(ptrid);
            if (self.refobj)
                self.logger.debug('ref obj');
            else
                self.logger.debug('no ref obj');*/
        } else {
            self.refobj = '';
            self.logger.debug('no ref obj');
        }

    };

    ResourceDecorator.prototype.eventHandler = function (context, events) {
        var i,
            nodeObj,
            self = context,
            client = self._control._client;


        self.logger.debug('in event handler');
        for (i = 0; i < events.length; i += 1) {
            self.logger.debug('eventhandler eid ' + events[i].eid + ' refid ' + self.refID);
            if (self.refID && events[i].eid != self.refID) {
                self.logger.debug('eventhandler eid ' + events[i].eid);
                continue;
            }

            nodeObj = client.getNode(events[i].eid);
            if (!nodeObj) {
                if (events[i].etype === 'unload') {
                    // The node was removed from the model (we can no longer access it).
                    // We still get the path/id via events[i].eid
                    if (self.refID) {
                        self.refobj = '';
                    }
                }
                continue;
            }

            if (events[i].etype === 'load') {
                // The node is loaded and we have access to it.
                // It was either just created or this is the initial
                // updateTerritory we invoked.


                if (self.refID) {
                    self.refobj = nodeObj;
                }

                // The nodeObj contains methods for querying the node, see below.
            } else if (events[i].etype === 'update') {
                // There were changes to the node (some might not apply to your application).
                // The node is still loaded and we have access to it.
                nodeObj = client.getNode(events[i].eid);
                if (self.refID && nodeObj) {
                    self.refobj = nodeObj;
                } else
                    self.refobj = '';

            } else if (events[i].etype === 'unload') {
                // The node was removed from the model (we can no longer access it).
                // We still get the path/id via events[i].eid
                if (self.refID) {
                    self.refobj = '';
                }
            } else {
                // "Technical events" not used.
            }
        }
        self.update();

    };


    return ResourceDecorator;
});